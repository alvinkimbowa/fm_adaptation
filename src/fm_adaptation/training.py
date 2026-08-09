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
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = build_model(cfg.model_name, cfg.probe_name, classes, cfg.checkpoint)
    if cfg.patching is not None:
        # Patches are cut fresh every epoch, so there is nothing stable to cache: the frozen encoder
        # runs inline under no_grad and the whole model is what gets stepped through.
        module = model.to(device)
        forward = lambda m, images, masks: m(images)  # noqa: E731
        train_loader = _patch_loader(cfg, model.encoder.preprocess, "train", shuffle=True)
        val_loader = None if cfg.fold == "all" else _patch_loader(cfg, model.encoder.preprocess, "val")
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
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = DiceCrossEntropyLoss()

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, cfg.run_dir / "config.yaml")
    history_path = cfg.run_dir / "history.csv"
    best_dice = -1.0
    stopping_reference_dice = -1.0
    epochs_without_improvement = 0
    with open(history_path, "w", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])
        for epoch in range(1, cfg.epochs + 1):
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
            state = {"probe": probe.state_dict(), "epoch": epoch, "val_dice": val_dice}
            torch.save(state, cfg.run_dir / "final.pt")
            if val_loader is not None and val_dice > best_dice:
                best_dice = val_dice
                torch.save(state, cfg.run_dir / "best.pt")
            if val_loader is not None:
                if val_dice > stopping_reference_dice + cfg.early_stopping_min_delta:
                    stopping_reference_dice = val_dice
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            if (
                val_loader is not None
                and epoch >= cfg.min_epochs
                and cfg.early_stopping_patience > 0
                and epochs_without_improvement >= cfg.early_stopping_patience
            ):
                print(f"early_stopping epoch={epoch} best_val_dice={best_dice:.4f}")
                break


if __name__ == "__main__":
    main()
