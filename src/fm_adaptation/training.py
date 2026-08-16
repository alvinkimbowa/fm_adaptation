import argparse
import csv
import random
import shutil
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import CachedFeatureDataset, NnUNet2DDataset, collate_cases, num_classes
from .losses import DiceCrossEntropyLoss, mean_foreground_dice
from .models import build_model
from .patching import patch_loader as _patch_loader


def _raw_dataset(cfg, preprocess, subset):
    return NnUNet2DDataset(
        cfg.raw_data_dir, cfg.train_dataset, "Tr", cfg.fold, subset, preprocess
    )


def _cache_features(cfg, encoder, dataset, device):
    cfg.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [case_id for case_id in dataset.ids if not (cfg.feature_cache_dir / f"{case_id}.pt").exists()]
    if not missing:
        return
    missing_dataset = torch.utils.data.Subset(dataset, [dataset.ids.index(case_id) for case_id in missing])
    loader = DataLoader(
        missing_dataset,
        batch_size=cfg.encoder_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_cases,
        pin_memory=True,
    )
    encoder.to(device).eval()
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with torch.no_grad():
        progress = tqdm(loader, desc="caching features", total=len(loader))
        for images, masks, metadata in progress:
            with amp:
                features = encoder(images.to(device))
            for feature, mask, meta in zip(features, masks, metadata):
                torch.save(
                    {"feature": feature.half().cpu(), "mask": mask.to(torch.int8)},
                    cfg.feature_cache_dir / f"{meta['case_id']}.pt",
                )
    encoder.cpu()


def _image_loader(cfg, preprocess, subset, shuffle=False):
    """Whole images straight to the model, for runs whose encoder trains and so cannot be cached."""
    dataset = _raw_dataset(cfg, preprocess, subset)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_cases,
        pin_memory=True,
        # UPerHead's pooling branch cannot BatchNorm a batch of one, which an odd case count would leave.
        drop_last=shuffle and len(dataset) % cfg.batch_size == 1,
    )


def _loader(cfg, raw_dataset, shuffle):
    dataset = CachedFeatureDataset(cfg.feature_cache_dir, raw_dataset.ids)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_cases,
        pin_memory=True,
    )


def _rng_state():
    """Enough to keep shuffling and patch sampling on the same trajectory across a restart."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _load_weights(model, state):
    """Copy a checkpoint's trained parts into `model`, leaving the rest as built."""
    model.probe.load_state_dict(state["probe"])
    if "encoder" in state:
        # A finetuned trunk; `adapter` carries no `backbone.*` keys, so it cannot undo this.
        model.encoder.trunk.load_state_dict(state["encoder"])
    if "adapter" in state:
        missing, unexpected = model.encoder.adapter.load_state_dict(state["adapter"], strict=False)
        missing = [key for key in missing if not key.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"adapter mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")


def _initialise_from(cfg, model):
    """Seed the adapter and head from a finished run, the way `LP + FT` seeds itself from its probe."""
    path = (
        cfg.results_dir / cfg.model_name / cfg.train_dataset / cfg.init_from
        / f"fold_{cfg.fold}" / f"{cfg.init_from_checkpoint}.pt"
    )
    if not path.exists():
        raise SystemExit(f"init_from given but there is no {path}")
    _load_weights(model, torch.load(path, map_location="cpu", weights_only=True))
    print(f"initialised from {path}")


def _resume(run_dir, model, optimizer):
    """Pick a run back up from `last.pt`, which a completed run no longer has."""
    path = run_dir / "last.pt"
    if not path.exists():
        raise SystemExit(
            f"--resume given but there is no {path}. A finished run keeps weights only, in final.pt."
        )
    state = torch.load(path, map_location="cpu", weights_only=False)
    _load_weights(model, state)
    optimizer.load_state_dict(state["optimizer"])
    rng = state.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return state


def _truncate_history(path, last_epoch):
    """Drop any rows past the epoch we are resuming from, so the file matches the checkpoint."""
    if not path.exists():
        return
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if r and int(float(r[0])) <= last_epoch]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([header, *body])


def _layer_id(name: str, blocks: int) -> int:
    """Which layer a trunk parameter belongs to, for layer-wise learning-rate decay.

    0 is the embedding, 1..blocks are the transformer blocks and blocks+1 is everything after them --
    the same partition ViT-Adapter's `LayerDecayOptimizerConstructor` uses.
    """
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    embedding = ("patch_embed", "pos_embed", "cls_token", "storage_tokens", "mask_token", "rope_embed")
    return 0 if name.startswith(embedding) else blocks + 1


def _param_groups(model, cfg):
    """Optimiser groups: the head and adapter at the config rate, the trunk layer-decayed beneath it.

    Without `encoder_learning_rate` this is exactly the flat list of trainable parameters it always was.
    """
    if cfg.encoder_learning_rate is None:
        return [p for p in model.parameters() if p.requires_grad]
    trunk = model.encoder.trunk
    trunk_params = {id(p) for p in trunk.parameters()}
    blocks = len(trunk.blocks)
    groups = {}
    for name, param in trunk.named_parameters():
        if not param.requires_grad:
            continue
        layer = _layer_id(name, blocks)
        # `encoder_learning_rate` is the rate of the *last* block; everything below it decays away.
        lr = cfg.encoder_learning_rate * cfg.encoder_layer_decay ** max(0, blocks - layer)
        # Biases and one-dimensional parameters (norms, gammas, tokens) are left undecayed, as upstream.
        decay = 0.0 if param.ndim <= 1 else cfg.weight_decay
        groups.setdefault((lr, decay), []).append(param)
    rest = [
        p for p in model.parameters() if p.requires_grad and id(p) not in trunk_params
    ]
    return [
        *({"params": params, "lr": lr, "weight_decay": decay} for (lr, decay), params in groups.items()),
        {"params": rest, "lr": cfg.learning_rate, "weight_decay": cfg.weight_decay},
    ]


def _run_epoch(module, loader, loss_fn, device, optimizer=None, desc="", forward=None, accumulation_steps=1):
    training = optimizer is not None
    forward = forward or (lambda m, x, y: m(x, y.shape[-2:]))
    module.train(training)
    if getattr(module, "encoder", None) is not None and not module.encoder.trainable:
        # The trunk carries stochastic depth; a frozen encoder must never leave eval mode.
        module.encoder.eval()
    losses, dices = [], []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    progress = tqdm(loader, desc=desc, leave=False)
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, (images, masks, _) in enumerate(progress, start=1):
        images, masks = images.to(device), masks.to(device)
        with amp:
            logits = forward(module, images, masks)
            loss = loss_fn(logits, masks)
        if training:
            # Accumulation keeps the effective batch where the memory does not allow the real one.
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        dices.append(mean_foreground_dice(logits.detach(), masks))
        progress.set_postfix(loss=f"{np.mean(losses):.4f}", dice=f"{np.nanmean(dices):.4f}")
    return float(np.mean(losses)), float(np.nanmean(dices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue this run from its own last.pt instead of training from scratch")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = build_model(
        cfg.model_name,
        cfg.probe_name,
        classes,
        cfg.checkpoint,
        train_encoder=cfg.train_encoder,
        injector=cfg.injector,
        variant=cfg.variant,
    )
    if cfg.init_from and not args.resume:
        # A resume restores this run's own state; seeding on top of it would throw the run away.
        _initialise_from(cfg, model)
    # An encoder with parameters of its own -- the adapter -- produces different features every epoch, so
    # like the patchwise case there is nothing stable to cache.
    encoder_trains = any(p.requires_grad for p in model.encoder.parameters())
    if cfg.patching is not None or encoder_trains:
        # Nothing stable to cache -- patches are cut fresh every epoch, and a training adapter changes
        # the features it produces -- so the whole model is what gets stepped through.
        module = model.to(device)
        forward = lambda m, images, masks: m(images)  # noqa: E731
        if cfg.patching is not None:
            train_loader = _patch_loader(cfg, model.encoder.preprocess, "train", shuffle=True)
            val_loader = None if cfg.fold == "all" else _patch_loader(cfg, model.encoder.preprocess, "val")
        else:
            train_loader = _image_loader(cfg, model.encoder.preprocess, "train", shuffle=True)
            val_loader = (
                None if cfg.fold == "all" else _image_loader(cfg, model.encoder.preprocess, "val")
            )
    else:
        train_raw = _raw_dataset(cfg, model.encoder.preprocess, "train")
        val_raw = None if cfg.fold == "all" else _raw_dataset(cfg, model.encoder.preprocess, "val")
        _cache_features(cfg, model.encoder, train_raw, device)
        if val_raw is not None:
            _cache_features(cfg, model.encoder, val_raw, device)
        module = model.probe.to(device)
        forward = None
        train_loader = _loader(cfg, train_raw, True)
        val_loader = None if val_raw is None else _loader(cfg, val_raw, False)
    probe = model.probe
    # The adapter's parameters live on the encoder, so step everything that asks for a gradient rather
    # than the probe alone. For a frozen encoder this is exactly `probe.parameters()`.
    optimizer = torch.optim.AdamW(
        _param_groups(model, cfg),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        # A trainable trunk is 300M parameters; the fused path would hold a second copy of them.
        foreach=False if cfg.train_encoder else None,
    )
    loss_fn = DiceCrossEntropyLoss()

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, cfg.run_dir / "config.yaml")
    history_path = cfg.run_dir / "history.csv"
    best_dice = -1.0
    stopping_reference_dice = -1.0
    epochs_without_improvement = 0
    start_epoch = 1
    if args.resume:
        resumed = _resume(cfg.run_dir, model, optimizer)
        start_epoch = resumed["epoch"] + 1
        best_dice = resumed["best_dice"]
        stopping_reference_dice = resumed["stopping_reference_dice"]
        epochs_without_improvement = resumed["epochs_without_improvement"]
        _truncate_history(history_path, resumed["epoch"])
        print(f"resumed from epoch {resumed['epoch']} (best_val_dice={best_dice:.4f})")
    with open(history_path, "a" if start_epoch > 1 else "w", newline="") as history_file:
        writer = csv.writer(history_file)
        if start_epoch == 1:
            writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_loss, train_dice = _run_epoch(
                module,
                train_loader,
                loss_fn,
                device,
                optimizer,
                desc=f"epoch {epoch}/{cfg.epochs} train",
                forward=forward,
                accumulation_steps=cfg.accumulation_steps,
            )
            if val_loader is None:
                val_loss, val_dice = float("nan"), float("nan")
            else:
                with torch.no_grad():
                    val_loss, val_dice = _run_epoch(
                        module,
                        val_loader,
                        loss_fn,
                        device,
                        desc=f"epoch {epoch}/{cfg.epochs} val",
                        forward=forward,
                    )
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice])
            history_file.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
            )
            # The bookkeeping happens before the save so a resumed run picks up the counters as they
            # stood at the end of this epoch, not as they were before it.
            improved = val_loader is not None and val_dice > best_dice
            if improved:
                best_dice = val_dice
            if val_loader is not None:
                if val_dice > stopping_reference_dice + cfg.early_stopping_min_delta:
                    stopping_reference_dice = val_dice
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            weights = {"probe": probe.state_dict(), "epoch": epoch, "val_dice": val_dice}
            if cfg.train_encoder:
                # A trunk that trains is no longer recoverable from its pretrained checkpoint, so it is
                # stored whole under the key the finetuning runs already use.
                weights["encoder"] = model.encoder.trunk.state_dict()
            if encoder_trains:
                # Only what trains: the frozen trunk is 300M parameters and is rebuilt from its own
                # checkpoint. `encoder` is reserved for the finetuning runs, which store the whole trunk.
                weights["adapter"] = {
                    key: value
                    for key, value in model.encoder.adapter.state_dict().items()
                    if not key.startswith("backbone.")
                }
            # `last.pt` carries the training state as well, so an interrupted run can be picked up; the
            # checkpoints meant for inference hold weights alone.
            torch.save(
                {
                    **weights,
                    "optimizer": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "stopping_reference_dice": stopping_reference_dice,
                    "epochs_without_improvement": epochs_without_improvement,
                    "rng": _rng_state(),
                },
                cfg.run_dir / "last.pt",
            )
            if improved:
                torch.save(weights, cfg.run_dir / "best.pt")
            if (
                val_loader is not None
                and epoch >= cfg.min_epochs
                and cfg.early_stopping_patience > 0
                and epochs_without_improvement >= cfg.early_stopping_patience
            ):
                print(f"early_stopping epoch={epoch} best_val_dice={best_dice:.4f}")
                break

    # Training finished: `final.pt` keeps the last epoch's weights for inference, and `last.pt` goes --
    # it only exists so an interrupted run can be picked up again.
    torch.save(weights, cfg.run_dir / "final.pt")
    (cfg.run_dir / "last.pt").unlink(missing_ok=True)

    # The cache only feeds probe training; finetuning and prediction recompute features from the images.
    if cfg.patching is None and cfg.feature_cache_dir.exists():
        shutil.rmtree(cfg.feature_cache_dir)
        print(f"removed feature cache {cfg.feature_cache_dir}")


if __name__ == "__main__":
    main()
