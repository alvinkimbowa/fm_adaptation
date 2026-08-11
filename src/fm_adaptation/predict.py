import argparse
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import NnUNet2DDataset, collate_cases, num_classes
from .metrics import CaseMetrics, compute_metrics, write_metrics
from .models import build_model, restore_prediction
from .patching import build_index, predict_case


def _predict_patchwise(cfg, model, dataset_name, split, subset, classes, device, amp, output_dir, overwrite):
    """Overlapping-patch inference at native resolution, one case at a time."""
    cases = build_index(
        cfg.raw_data_dir,
        dataset_name,
        split,
        cfg.fold,
        subset,
        cfg.patching,
        cfg.patch_cache_dir(dataset_name),
    )
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in tqdm(cases, desc=f"{output_dir.name} {dataset_name}"):
        prediction = predict_case(
            model, case, cfg.patching, classes, device, amp, cfg.batch_size, model.encoder.preprocess
        )
        output_path = prediction_dir / f"{case.case_id}.png"
        if overwrite or not output_path.exists():
            cv2.imwrite(str(output_path), prediction)
        dice, masd = compute_metrics(prediction, np.asarray(case.label), classes)
        rows.append(CaseMetrics(case.case_id, dice, masd))
    return rows


def _jobs(cfg, datasets):
    """Each evaluation to run, as (dataset, split, subset, validation|test).

    The training dataset contributes its fold's validation cases and, when it ships one, its held-out
    `imagesTs`; every other dataset is evaluated whole on `test_split`.
    """
    jobs = []
    for dataset_name in datasets:
        if dataset_name != cfg.train_dataset:
            jobs.append((dataset_name, cfg.test_split, "eval", "test"))
            continue
        if cfg.fold != "all":
            jobs.append((dataset_name, "Tr", "val", "validation"))
        held_out = cfg.raw_data_dir / dataset_name / "imagesTs"
        if held_out.is_dir() and any(held_out.iterdir()):
            jobs.append((dataset_name, "Ts", "eval", "test"))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(cfg.device)
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = build_model(cfg.model_name, cfg.probe_name, classes, cfg.checkpoint, injector=cfg.injector)
    checkpoint_name = "final" if cfg.fold == "all" else args.checkpoint
    state = torch.load(cfg.run_dir / f"{checkpoint_name}.pt", map_location="cpu", weights_only=True)
    model.probe.load_state_dict(state["probe"])
    if "encoder" in state:
        model.encoder.trunk.load_state_dict(state["encoder"])
    if "adapter" in state:
        # Trained adapter weights only; the frozen trunk came from the DINOv3 checkpoint at build time.
        missing, unexpected = model.encoder.adapter.load_state_dict(state["adapter"], strict=False)
        missing = [key for key in missing if not key.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(f"adapter checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    model.to(device).eval()
    datasets = cfg.test_datasets or (cfg.train_dataset,)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    for dataset_name, split, subset, kind in _jobs(cfg, datasets):
        output_dir = cfg.run_dir / kind / dataset_name
        if cfg.patching is not None:
            rows = _predict_patchwise(
                cfg, model, dataset_name, split, subset, classes, device, amp, output_dir, args.overwrite
            )
            write_metrics(rows, output_dir / "metrics.csv")
            print(f"Wrote {len(rows)} predictions and metrics to {output_dir}")
            continue
        dataset = NnUNet2DDataset(
            cfg.raw_data_dir, dataset_name, split, cfg.fold, subset, model.encoder.preprocess
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_cases,
            pin_memory=True,
        )
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        progress = tqdm(loader, desc=f"{kind} {dataset_name}")
        with torch.no_grad():
            for images, masks, metadata in progress:
                with amp:
                    logits = model(images.to(device))
                predictions = logits.argmax(1).cpu()
                for prediction, padded_mask, meta in zip(predictions, masks, metadata):
                    restored = restore_prediction(prediction, meta)
                    target = restore_prediction(padded_mask, meta)
                    output_path = prediction_dir / f"{meta['case_id']}.png"
                    if args.overwrite or not output_path.exists():
                        cv2.imwrite(str(output_path), restored)
                    dice, masd = compute_metrics(restored, target, classes)
                    rows.append(CaseMetrics(meta["case_id"], dice, masd))
        write_metrics(rows, output_dir / "metrics.csv")
        print(f"Wrote {len(rows)} predictions and metrics to {output_dir}")


if __name__ == "__main__":
    main()
