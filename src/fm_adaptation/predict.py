import argparse
import json
from contextlib import nullcontext

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import (
    NnUNet2DDataset,
    active_planes,
    collate_cases,
    load_dataset_json,
    num_classes,
    stain_planes,
)
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
        output_path = prediction_dir / f"{case.case_id}.png"
        if not overwrite and output_path.exists():
            prediction = cv2.imread(str(output_path), cv2.IMREAD_GRAYSCALE)
        else:
            prediction = predict_case(
                model, case, cfg.patching, classes, device, amp, cfg.batch_size, model.encoder.preprocess
            )
            cv2.imwrite(str(output_path), prediction)
        dice, masd = compute_metrics(prediction, np.asarray(case.label), classes)
        rows.append(CaseMetrics(case.case_id, dice, masd))
    return rows


def _scored_from_disk(prediction_dir, dataset_dir, split, ending, case_ids, classes):
    """Metrics for cases already predicted, read back rather than recomputed.

    A saved prediction is written at the case's native resolution, which is what the label on disk is
    too, so scoring it needs neither the model nor the preprocessing -- and skipping both is the point:
    a forward pass over a 30-megapixel lesion slide costs seconds, reading two PNGs costs milliseconds.
    """
    rows = []
    for case_id in case_ids:
        prediction = cv2.imread(str(prediction_dir / f"{case_id}.png"), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(str(dataset_dir / f"labels{split}" / f"{case_id}{ending}"), cv2.IMREAD_GRAYSCALE)
        if prediction is None or target is None:
            raise FileNotFoundError(f"cannot score {case_id} from disk")
        dice, masd = compute_metrics(prediction, target, classes)
        rows.append(CaseMetrics(case_id, dice, masd))
    return rows


def _jobs(cfg, datasets):
    """Each evaluation as (source dataset, split, subset, validation|test, output column).

    The training dataset contributes its fold's validation cases and, when it ships one, its held-out
    `imagesTs`; every other dataset is evaluated whole on `test_split`.
    """
    jobs = []
    for dataset_name in datasets:
        if dataset_name != cfg.train_dataset:
            jobs.append((dataset_name, cfg.test_split, "eval", "test", dataset_name))
            continue
        if cfg.fold != "all":
            jobs.append((dataset_name, "Tr", "val", "validation", dataset_name))
        dataset_dir = cfg.raw_data_dir / dataset_name
        ending = load_dataset_json(dataset_dir)["file_ending"]
        for held_out in sorted(dataset_dir.glob("imagesTs*")):
            split = held_out.name[len("images"):]
            labels = dataset_dir / f"labels{split}"
            if not labels.is_dir():
                raise FileNotFoundError(f"held-out images have no matching labels: {held_out}")
            if not any(held_out.glob(f"*_0000{ending}")):
                continue
            column = dataset_name if split == "Ts" else f"{dataset_name}{split[2:]}"
            jobs.append((dataset_name, split, "eval", "test", column))
    return jobs


def _has_labels(dataset_dir, split):
    """Whether this split can be scored at all; Dataset212 ships images with no annotations."""
    label_dir = dataset_dir / f"labels{split}"
    return label_dir.is_dir() and any(label_dir.iterdir())


def _write_source(output_dir, dataset_name, split):
    with open(output_dir / "source.json", "w") as f:
        json.dump({"dataset": dataset_name, "split": split}, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final", "last"), default="best")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
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
    checkpoint_name = "final" if cfg.fold == "all" else args.checkpoint
    checkpoint_path = cfg.run_dir / f"{checkpoint_name}.pt"
    if checkpoint_name == "last":
        # `last.pt` also carries optimiser and RNG state, which the safe loader cannot unpickle.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
    # The planes this model was trained to look at. An evaluation set carrying more stains than the
    # training set is read down to the ones it shares -- a czi_B model sees GFAP in blue and never
    # meets SMI, which it has no weights for.
    keep_planes = active_planes(load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["channel_names"])

    for dataset_name, split, subset, kind, column_name in _jobs(cfg, datasets):
        if cfg.patching is not None:
            declared = stain_planes(load_dataset_json(cfg.raw_data_dir / dataset_name)["channel_names"])
            if declared is not None and len(declared) > 1:
                raise ValueError("patchwise prediction is not supported for multi-stain datasets")
        output_dir = cfg.run_dir / kind / column_name
        scored = _has_labels(cfg.raw_data_dir / dataset_name, split)
        if cfg.patching is not None:
            rows = _predict_patchwise(
                cfg, model, dataset_name, split, subset, classes, device, amp, output_dir, args.overwrite
            )
            write_metrics(rows, output_dir / "metrics.csv")
            _write_source(output_dir, dataset_name, split)
            print(f"Wrote {len(rows)} predictions and metrics to {output_dir}")
            continue
        dataset = NnUNet2DDataset(
            cfg.raw_data_dir, dataset_name, split, cfg.fold, subset, model.encoder.preprocess,
            keep_planes=keep_planes, require_labels=scored,
        )
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        # Without --overwrite, a case that already has a prediction is scored from that saved file and
        # never goes through the model again -- the ids are dropped before the loader is built, so the
        # image is not even decoded. `--overwrite` forces the whole dataset to be predicted afresh.
        rows = []
        if not args.overwrite:
            done = [c for c in dataset.ids if (prediction_dir / f"{c}.png").exists()]
            if done:
                if scored:
                    rows = _scored_from_disk(
                        prediction_dir, dataset.dataset_dir, dataset.split, dataset.ending, done, classes
                    )
                dataset.ids = [c for c in dataset.ids if not (prediction_dir / f"{c}.png").exists()]
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            collate_fn=collate_cases,
            pin_memory=True,
        )
        progress = tqdm(loader, desc=f"{kind} {dataset_name}")
        with torch.no_grad():
            for images, masks, metadata in progress:
                with amp:
                    logits = model(images.to(device))
                predictions = logits.argmax(1).cpu()
                for prediction, padded_mask, meta in zip(predictions, masks, metadata):
                    restored = restore_prediction(prediction, meta)
                    cv2.imwrite(str(prediction_dir / f"{meta['case_id']}.png"), restored)
                    if not meta.get("has_label", True):
                        continue
                    target = restore_prediction(padded_mask, meta)
                    dice, masd = compute_metrics(restored, target, classes)
                    rows.append(CaseMetrics(meta["case_id"], dice, masd))
        _write_source(output_dir, dataset_name, split)
        if not scored:
            # No metrics file at all: an empty one reads in the report as a column of failures rather
            # than as a dataset there is nothing to score against.
            count = len(list(prediction_dir.glob("*.png")))
            print(f"Wrote {count} predictions to {output_dir} (no labels, not scored)")
            continue
        write_metrics(rows, output_dir / "metrics.csv")
        print(f"Wrote {len(rows)} predictions and metrics to {output_dir}")


if __name__ == "__main__":
    main()
