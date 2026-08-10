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
from .data import NnUNet2DDataset, collate_cases, num_classes
from .losses import DiceCrossEntropyLoss, mean_foreground_dice
from .models import build_model
from .patching import patch_loader


def _loader(cfg, preprocess, subset, shuffle):
    if cfg.patching is not None:
        return patch_loader(cfg, preprocess, subset, shuffle)
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


def _run_epoch(model, loader, loss_fn, device, optimizer=None, accumulation_steps=1, desc=""):
    training = optimizer is not None
    model.train(training)
    losses, dices = [], []
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    if training:
        optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=desc, leave=False)
    for step, (images, masks, _) in enumerate(progress, start=1):
        images, masks = images.to(device), masks.to(device)
        with torch.set_grad_enabled(training), amp:
            logits = model(images)
            loss = loss_fn(logits, masks)
        if training:
            (loss / accumulation_steps).backward()
            if step % accumulation_steps == 0 or step == len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        dices.append(mean_foreground_dice(logits.detach(), masks))
        progress.set_postfix(loss=f"{np.mean(losses):.4f}", dice=f"{np.nanmean(dices):.4f}")
    return float(np.mean(losses)), float(np.nanmean(dices))


def _checkpoint(model, epoch, val_dice):
    return {
        "encoder": model.encoder.trunk.state_dict(),
        "probe": model.probe.state_dict(),
        "epoch": epoch,
        "val_dice": val_dice,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    with open(args.config) as f:
        import yaml

        raw_cfg = yaml.safe_load(f)
    finetuning = raw_cfg["finetuning"]

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = build_model(
        cfg.model_name, cfg.probe_name, classes, cfg.checkpoint, train_encoder=True
    )
    probe_checkpoint = (
        cfg.results_dir
        / cfg.model_name
        / cfg.train_dataset
        / finetuning["initial_probe"]
        / f"fold_{cfg.fold}"
        / f"{finetuning.get('initial_probe_checkpoint', 'final')}.pt"
    )
    print(f"initialising probe from {probe_checkpoint}")
    state = torch.load(probe_checkpoint, map_location="cpu", weights_only=True)
    model.probe.load_state_dict(state["probe"])
    model.to(device)
    train_loader = _loader(cfg, model.encoder.preprocess, "train", True)
    val_loader = _loader(cfg, model.encoder.preprocess, "val", False)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": float(finetuning["encoder_learning_rate"])},
            {"params": model.probe.parameters(), "lr": cfg.learning_rate},
        ],
        weight_decay=cfg.weight_decay,
        foreach=False,
    )
    accumulation_steps = int(finetuning.get("accumulation_steps", 1))
    loss_fn = DiceCrossEntropyLoss()

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, cfg.run_dir / "config.yaml")
    best_dice = -1.0
    stopping_reference_dice = -1.0
    epochs_without_improvement = 0
    final_state = None
    with open(cfg.run_dir / "history.csv", "w", newline="") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["epoch", "train_loss", "train_dice", "val_loss", "val_dice"])
        for epoch in range(1, cfg.epochs + 1):
            train_loss, train_dice = _run_epoch(
                model,
                train_loader,
                loss_fn,
                device,
                optimizer,
                accumulation_steps,
                desc=f"epoch {epoch}/{cfg.epochs} train",
            )
            val_loss, val_dice = _run_epoch(
                model, val_loader, loss_fn, device, desc=f"epoch {epoch}/{cfg.epochs} val"
            )
            writer.writerow([epoch, train_loss, train_dice, val_loss, val_dice])
            history_file.flush()
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
                f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}"
            )
            final_state = _checkpoint(model, epoch, val_dice)
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(final_state, cfg.run_dir / "best.pt")
            if val_dice > stopping_reference_dice + cfg.early_stopping_min_delta:
                stopping_reference_dice = val_dice
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if (
                epoch >= cfg.min_epochs
                and cfg.early_stopping_patience > 0
                and epochs_without_improvement >= cfg.early_stopping_patience
            ):
                print(f"early_stopping epoch={epoch} best_val_dice={best_dice:.4f}")
                break
    torch.save(final_state, cfg.run_dir / "final.pt")


if __name__ == "__main__":
    main()
