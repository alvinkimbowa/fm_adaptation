import argparse
from contextlib import nullcontext

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import NnUNet2DDataset, collate_cases, num_classes
from .metrics import CaseMetrics, compute_metrics, write_metrics
from .models import build_model, restore_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    device = torch.device(cfg.device)
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = build_model(cfg.model_name, cfg.probe_name, classes, cfg.checkpoint)
    checkpoint_name = "final" if cfg.fold == "all" else args.checkpoint
    state = torch.load(cfg.run_dir / f"{checkpoint_name}.pt", map_location="cpu", weights_only=True)
    model.probe.load_state_dict(state["probe"])
    if "encoder" in state:
        model.encoder.trunk.load_state_dict(state["encoder"])
    model.to(device).eval()
    datasets = cfg.test_datasets or (cfg.train_dataset,)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    for dataset_name in datasets:
        same_dataset = dataset_name == cfg.train_dataset
        split = "Tr" if same_dataset else cfg.test_split
        subset = "val" if same_dataset else "eval"
        if same_dataset and cfg.fold == "all":
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
        output_dir = cfg.run_dir / ("validation" if same_dataset else "test") / dataset_name
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        progress = tqdm(loader, desc=f"{'validation' if same_dataset else 'test'} {dataset_name}")
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
