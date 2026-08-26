import argparse
import json
from contextlib import nullcontext

import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import ExperimentConfig
from .data import (
    NnUNet2DDataset,
    _case_ids,
    trained_planes,
    collate_cases,
    load_dataset_json,
    num_classes,
    stain_planes,
)
from .datasets import dataset_dir as resolve_dataset_dir
from .models import load_trained_model, restore_prediction
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
    for case in tqdm(cases, desc=f"{output_dir.name} {dataset_name}"):
        output_path = prediction_dir / f"{case.case_id}.png"
        if not overwrite and output_path.exists():
            continue
        prediction = predict_case(
            model, case, cfg.patching, classes, device, amp, cfg.batch_size, model.encoder.preprocess
        )
        cv2.imwrite(str(output_path), prediction)
    return len(cases)


def _jobs(cfg, datasets):
    """Each evaluation as (source dataset, split, subset, validation|test, output column).

    The training dataset contributes its fold's validation cases and, when it ships one, its held-out
    `imagesTs`; every other dataset is evaluated whole on `test_split`.

    `test_split: all` evaluates every split an evaluation set has, under one column name. Some sets
    ship both `imagesTr` and `imagesTs`, and the boundary between them is a fact about how that
    dataset would be trained on, not about a model that never saw it -- scoring only one silently
    drops the rest. It is opt-in rather than the rule because making it unconditional would move
    columns that already exist.
    """
    jobs = []
    for dataset_name in datasets:
        if dataset_name != cfg.train_dataset:
            # `test_split` is one setting for every evaluation set, but the newer ones ship their
            # cases as `imagesTs` while the older ones keep everything in `imagesTr`. Honour the
            # configured split where it exists, so no existing result moves, and fall back to
            # whichever split the dataset actually has.
            available = [s for s in ("Ts", "Tr") if _has_cases(cfg.raw_data_dir / dataset_name, s)]
            if cfg.test_split == "all":
                splits = available
            else:
                split = cfg.test_split
                if not _has_cases(cfg.raw_data_dir / dataset_name, split):
                    split = next(iter(available), split)
                splits = [split]
            for split in splits:
                jobs.append((dataset_name, split, "eval", "test", dataset_name))
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


def _seen_in_training(cfg):
    """Case ids this run's fold was fitted on, so an evaluation set can exclude them.

    Case ids are unique across these datasets -- every one carries its source as a prefix -- so an id
    appearing in two datasets is the same image, not a coincidence of numbering. Dataset213 is wholly
    contained in Dataset208, and without this a model trained on the combined set would be scored on
    the slides it had already fitted. Only the training split counts: the fold's validation
    cases are reported as the validation column, which is what that column is for.
    """
    dataset_dir = resolve_dataset_dir(cfg.raw_data_dir, cfg.train_dataset)
    if not (dataset_dir / "splits_final.json").exists():
        return set()
    subsets = ("train",) if cfg.fold == "all" else ("train", "val")
    return {case for subset in subsets for case in _case_ids(dataset_dir, "Tr", cfg.fold, subset)}



def _has_cases(dataset_dir, split):
    image_dir = dataset_dir / f"images{split}"
    return image_dir.is_dir() and any(image_dir.iterdir())


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
    model = load_trained_model(cfg, args.checkpoint, device, classes)
    datasets = cfg.test_datasets or (cfg.train_dataset,)
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    # The planes this model was trained to look at. An evaluation set carrying more stains than the
    # training set is read down to the ones it shares -- a czi_B model sees GFAP in blue and never
    # meets SMI, which it has no weights for.
    keep_planes = trained_planes(
        load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["channel_names"], cfg.stains
    )
    seen = _seen_in_training(cfg)

    for dataset_name, split, subset, kind, column_name in _jobs(cfg, datasets):
        if cfg.patching is not None:
            declared = stain_planes(load_dataset_json(cfg.raw_data_dir / dataset_name)["channel_names"])
            if declared is not None and len(declared) > 1:
                raise ValueError("patchwise prediction is not supported for multi-stain datasets")
        output_dir = cfg.run_dir / kind / column_name
        # The loader still needs to know whether a label exists to return one; scoring happens later,
        # in `fm_adaptation.compute_metrics`, against the label on disk rather than the resized copy.
        labelled = _has_labels(cfg.raw_data_dir / dataset_name, split)
        if cfg.patching is not None:
            count = _predict_patchwise(
                cfg, model, dataset_name, split, subset, classes, device, amp, output_dir, args.overwrite
            )
            _write_source(output_dir, dataset_name, split)
            print(f"Wrote {count} predictions to {output_dir}")
            continue
        dataset = NnUNet2DDataset(
            cfg.raw_data_dir, dataset_name, split, cfg.fold, subset, model.encoder.preprocess,
            keep_planes=keep_planes, require_labels=labelled,
        )
        if dataset_name != cfg.train_dataset:
            dataset.ids = [c for c in dataset.ids if c not in seen]
            if not dataset.ids:
                print(f"Skipping {column_name}: every case was seen in training")
                continue
        prediction_dir = output_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        # Without --overwrite, a case that already has a prediction keeps it and never goes through
        # the model again -- the ids are dropped before the loader is built, so the image is not even
        # decoded. `--overwrite` forces the whole dataset to be predicted afresh.
        if not args.overwrite:
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
            for images, _, metadata in progress:
                with amp:
                    logits = model(images.to(device))
                predictions = logits.argmax(1).cpu()
                for prediction, meta in zip(predictions, metadata):
                    # Restored to the case's own height and width, so the file on disk is directly
                    # comparable to the label the annotator drew -- that is what makes scoring a
                    # separate stage possible at all.
                    cv2.imwrite(
                        str(prediction_dir / f"{meta['case_id']}.png"),
                        restore_prediction(prediction, meta),
                    )
        _write_source(output_dir, dataset_name, split)
        count = len(list(prediction_dir.glob("*.png")))
        note = "" if labelled else " (no labels, nothing to score)"
        print(f"Wrote {count} predictions to {output_dir}{note}")


if __name__ == "__main__":
    main()
