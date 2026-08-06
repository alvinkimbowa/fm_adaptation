import argparse
import csv
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import NnUNet2DDataset, collate_cases, num_classes
from .losses import DiceCrossEntropyLoss, mean_foreground_dice
from .models import build_model


def _loader(cfg, preprocess, subset, shuffle):
    dataset = NnUNet2DDataset(
        cfg.raw_data_dir, cfg.train_dataset, "Tr", cfg.fold, subset, preprocess
    )
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_cases,
        pin_memory=True,
    )


def _run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.probe.train(training)
    losses, dices = [], []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with amp:
            logits = model(images)
            loss = loss_fn(logits, masks)
        if training:
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        dices.append(mean_foreground_dice(logits.detach(), masks))
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
    model = build_model(cfg.model_name, cfg.probe_name, classes, cfg.checkpoint).to(device)
    train_loader = _loader(cfg, model.encoder.preprocess, "train", True)
    val_loader = None if cfg.fold == "all" else _loader(cfg, model.encoder.preprocess, "val", False)
    optimizer = torch.optim.AdamW(
        model.probe.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = DiceCrossEntropyLoss()

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, cfg.run_dir / "config.yaml")
    history_path = cfg.run_dir / "history.csv"
    best_dice = -1.0
    with open(history_path, "w", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_dice = _run_epoch(model, train_loader, loss_fn, device, optimizer)
            if val_loader is None:
                val_loss, val_dice = float("nan"), float("nan")
            else:
                with torch.no_grad():
                    val_loss, val_dice = _run_epoch(model, val_loader, loss_fn, device)
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice])
            history_file.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
            )
            state = {"probe": model.probe.state_dict(), "epoch": epoch, "val_dice": val_dice}
            torch.save(state, cfg.run_dir / "final.pt")
            if val_loader is not None and val_dice > best_dice:
                best_dice = val_dice
                torch.save(state, cfg.run_dir / "best.pt")


if __name__ == "__main__":
    main()

