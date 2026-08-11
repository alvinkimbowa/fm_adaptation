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


def _resume(checkpoint_path, model, optimizer, encoder_trains):
    """Restore weights, optimiser and bookkeeping from a run's own `final.pt`."""
    if not checkpoint_path.exists():
        raise SystemExit(f"--resume given but there is no checkpoint at {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "optimizer" not in state:
        raise SystemExit(
            f"{checkpoint_path} predates resume support (no optimiser state); rerun without --resume"
        )
    model.probe.load_state_dict(state["probe"])
    if encoder_trains:
        missing, unexpected = model.encoder.adapter.load_state_dict(state["adapter"], strict=False)
        missing = [key for key in missing if not key.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"adapter mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
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


def _run_epoch(module, loader, loss_fn, device, optimizer=None, desc="", forward=None):
    training = optimizer is not None
    forward = forward or (lambda m, x, y: m(x, y.shape[-2:]))
    module.train(training)
    if getattr(module, "encoder", None) is not None and not module.encoder.trainable:
        # The trunk carries stochastic depth; a frozen encoder must never leave eval mode.
        module.encoder.eval()
    losses, dices = [], []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    progress = tqdm(loader, desc=desc, leave=False)
    for images, masks, _ in progress:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with amp:
            logits = forward(module, images, masks)
            loss = loss_fn(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        dices.append(mean_foreground_dice(logits.detach(), masks))
        progress.set_postfix(loss=f"{np.mean(losses):.4f}", dice=f"{np.nanmean(dices):.4f}")
    return float(np.mean(losses)), float(np.nanmean(dices))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="continue this run from its own final.pt instead of training from scratch")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = build_model(cfg.model_name, cfg.probe_name, classes, cfg.checkpoint, injector=cfg.injector)
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
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
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
        resumed = _resume(cfg.run_dir / "final.pt", model, optimizer, encoder_trains)
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
            state = {
                "probe": probe.state_dict(),
                "epoch": epoch,
                "val_dice": val_dice,
                # Everything `--resume` needs to carry on as if the run had never stopped. `predict.py`
                # ignores these keys, so an interrupted run stays usable for inference either way.
                "optimizer": optimizer.state_dict(),
                "best_dice": best_dice,
                "stopping_reference_dice": stopping_reference_dice,
                "epochs_without_improvement": epochs_without_improvement,
                "rng": _rng_state(),
            }
            if encoder_trains:
                # Only what trains: the frozen trunk is 300M parameters and is rebuilt from its own
                # checkpoint. `encoder` is reserved for the finetuning runs, which store the whole trunk.
                state["adapter"] = {
                    key: value
                    for key, value in model.encoder.adapter.state_dict().items()
                    if not key.startswith("backbone.")
                }
            torch.save(state, cfg.run_dir / "final.pt")
            if improved:
                torch.save(state, cfg.run_dir / "best.pt")
            if (
                val_loader is not None
                and epoch >= cfg.min_epochs
                and cfg.early_stopping_patience > 0
                and epochs_without_improvement >= cfg.early_stopping_patience
            ):
                print(f"early_stopping epoch={epoch} best_val_dice={best_dice:.4f}")
                break

    # The cache only feeds probe training; finetuning and prediction recompute features from the images.
    if cfg.patching is None and cfg.feature_cache_dir.exists():
        shutil.rmtree(cfg.feature_cache_dir)
        print(f"removed feature cache {cfg.feature_cache_dir}")


if __name__ == "__main__":
    main()
