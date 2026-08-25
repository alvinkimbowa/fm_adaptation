"""Compute segmentation metrics from a prediction directory and a ground-truth directory.

Prediction and metrics are separate stages on purpose. A prediction is written back at the case's
native resolution, which is the resolution the annotator drew at, so measuring it needs nothing but
two directories of label maps -- no model, no preprocessing, no GPU. Folding the two together is what
produced the bug this replaces: the old inference loop measured against the label it had already
shrunk onto the encoder's square canvas and blown back up, which for a lesion slide is a ~7.5x round
trip that costs about 1.2 dice of boundary before a model is involved at all.

Dice, HD95 and mean average surface distance per case, background excluded. Nothing is resized here:
the predictions must already be in the space of the labels.

    PYTHONPATH=src python -m fm_adaptation.compute_metrics \
        --results-dir models --datasets Dataset209_combined_MYE_smi_gfap --folds 0
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from monai.metrics import (
    compute_average_surface_distance,
    compute_dice,
    compute_hausdorff_distance,
)
from monai.networks.utils import one_hot
from PIL import Image

from .selection import matches

# Paul's widefield strips are 29739x6240 -- 186 megapixels against PIL's 89 megapixel ceiling, which
# exists to catch malicious files rather than legitimately enormous microscopy.
Image.MAX_IMAGE_PIXELS = None

# What a label or prediction map can be stored as.
MASK_SUFFIXES = {".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class CaseMetrics:
    image_id: str
    dice: float
    hd95: float
    masd: float


def _nanmean(values) -> float:
    array = np.array(list(values), dtype=np.float64)
    return float("nan") if np.all(np.isnan(array)) else float(np.nanmean(array))


def read_mask(path) -> np.ndarray:
    """A 2D integer label map."""
    array = np.array(Image.open(path))
    if array.ndim != 2:
        raise ValueError(f"expected a 2D label image at {path}, got shape {array.shape}")
    return array if np.issubdtype(array.dtype, np.integer) else array.astype(np.int64)


def index_by_stem(directory: Path):
    """Case id -> path for every mask in `directory`."""
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES
    }


def compute_case_metrics(prediction, target, num_classes, ignore_empty=True):
    """(dice, hd95, masd) averaged over the foreground classes, background excluded."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    # one_hot expects (batch, channel, ...), so both gain two leading axes.
    prediction = one_hot(torch.as_tensor(prediction, dtype=torch.long)[None, None], num_classes)
    target = one_hot(torch.as_tensor(target, dtype=torch.long)[None, None], num_classes)
    prediction, target = prediction.to(torch.float32), target.to(torch.float32)
    dice = compute_dice(prediction, target, include_background=False, ignore_empty=ignore_empty)
    hd95 = compute_hausdorff_distance(
        prediction, target, include_background=False, percentile=95, directed=False
    )
    masd = compute_average_surface_distance(
        prediction, target, include_background=False, symmetric=True
    )
    return tuple(_nanmean(x.detach().cpu().numpy().reshape(-1)) for x in (dice, hd95, masd))


def write_csv(rows: list[CaseMetrics], path: Path, add_aggregate_row=True):
    """Per-case rows plus a trailing MEAN, so the file carries its own summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value):
        return "" if value is None or np.isnan(value) else f"{value:.6f}"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "dice", "hd95", "masd"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image_id": row.image_id,
                "dice": fmt(row.dice), "hd95": fmt(row.hd95), "masd": fmt(row.masd),
            })
        if add_aggregate_row and rows:
            writer.writerow({
                "image_id": "MEAN",
                "dice": fmt(_nanmean(r.dice for r in rows)),
                "hd95": fmt(_nanmean(r.hd95 for r in rows)),
                "masd": fmt(_nanmean(r.masd for r in rows)),
            })


def _label_index(dataset_dir: Path):
    """Every label this dataset ships, as case id -> path, merged across all of its `labels*` dirs.

    Resolution is per case rather than per directory because one column can hold cases from more than
    one split: an evaluation set that ships both `imagesTr` and `imagesTs` is measured whole under
    `test_split: all`, and picking a single directory for the column would silently drop whatever
    lived in the other one. Case ids are unique within a dataset, so merging is unambiguous.

    An empty result means the dataset ships no labels at all -- Dataset212 is images only.
    """
    index = {}
    for label_dir in sorted(dataset_dir.glob("labels*")):
        if label_dir.is_dir():
            index.update(index_by_stem(label_dir))
    return index


def _num_classes(dataset_dir: Path):
    with open(dataset_dir / "dataset.json") as f:
        return len(json.load(f)["labels"])


def _read_run(fold_dir: Path):
    """(model, run name, trained on, raw data dir) from the config the run was launched with."""
    with open(fold_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    return (
        cfg["model"]["name"],
        cfg["model"].get("run_name", cfg["model"]["probe"]),
        cfg["data"]["train_dataset"],
        Path(cfg["data"]["raw_data_dir"]),
    )


def _columns(results_dir: Path, args):
    """Every (predictions dir, raw data dir) this selection covers.

    A column is named after the dataset it was evaluated on, which is all the pairing needs -- the
    run's own config says only what it was launched with, and runs are routinely evaluated on
    datasets added to the config afterwards.
    """
    selected = []
    for config_path in sorted(results_dir.glob("*/*/*/fold_*/config.yaml")):
        fold_dir = config_path.parent
        model, run_name, trained_on, raw_data_dir = _read_run(fold_dir)
        if not (
            matches(model, args.models)
            and matches(trained_on, args.datasets)
            and matches(run_name, args.experiments)
            and matches(fold_dir.name.removeprefix("fold_"), args.folds)
        ):
            continue
        for prediction_dir in sorted(fold_dir.glob("*/*/predictions")):
            if matches(prediction_dir.parents[1].name, args.splits):
                selected.append((prediction_dir, raw_data_dir))
    return selected


def _is_current(metrics_path: Path, prediction_dir: Path):
    """Whether a metrics file already reflects every prediction beside it."""
    if not metrics_path.exists():
        return False
    predictions = list(prediction_dir.glob("*.png"))
    return bool(predictions) and metrics_path.stat().st_mtime >= max(
        p.stat().st_mtime for p in predictions
    )


def measure(prediction_dir: Path, raw_data_dir: Path, overwrite=False, dry_run=False):
    """Write `metrics.csv` beside `prediction_dir`; returns a one-line report of what happened."""
    output_dir = prediction_dir.parent
    dataset_dir = raw_data_dir / output_dir.name
    metrics_path = output_dir / "metrics.csv"
    label = "/".join(p.name for p in reversed(output_dir.parents[:3])) + f" -> {output_dir.name}"

    predictions = index_by_stem(prediction_dir)
    if not predictions:
        return f"{label}: no predictions"
    ground_truth = _label_index(dataset_dir)
    if not ground_truth:
        # Dataset212 ships images with no annotations. An empty metrics file would read in the report
        # as a column of failures rather than as a dataset there is nothing to measure against.
        return f"{label}: no labels, not measured"
    if not overwrite and _is_current(metrics_path, prediction_dir):
        return f"{label}: up to date"
    case_ids = sorted(set(predictions) & set(ground_truth))
    if not case_ids:
        raise RuntimeError(f"no case matches between {prediction_dir} and {dataset_dir}")
    if dry_run:
        return f"{label}: would measure {len(case_ids)} of {len(predictions)}"

    missing = len(predictions) - len(case_ids)
    classes = _num_classes(dataset_dir)
    rows = [
        CaseMetrics(case_id, *compute_case_metrics(
            read_mask(predictions[case_id]), read_mask(ground_truth[case_id]), classes
        ))
        for case_id in case_ids
    ]
    write_csv(rows, metrics_path)
    return f"{label}: measured {len(rows)}" + (f" ({missing} without a label)" if missing else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--config", help="measure only the run this config names, ignoring filters")
    parser.add_argument("--models", nargs="*", default=[], help="dinov3, sam3; empty keeps every one")
    parser.add_argument("--datasets", nargs="*", default=[], help="what a run was trained on")
    parser.add_argument("--experiments", nargs="*", default=[], help="run names, globs or `_suffix`")
    parser.add_argument("--folds", nargs="*", default=[])
    parser.add_argument("--splits", nargs="*", default=[], help="validation, test; empty keeps both")
    parser.add_argument("--overwrite", action="store_true", help="redo columns already current")
    parser.add_argument("--dry-run", action="store_true", help="list the columns, measure nothing")
    args = parser.parse_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        args.models = [cfg["model"]["name"]]
        args.datasets = [cfg["data"]["train_dataset"]]
        args.experiments = [cfg["model"].get("run_name", cfg["model"]["probe"])]
        args.folds = [str(cfg["data"]["fold"])]

    columns = _columns(Path(args.results_dir), args)
    if not columns:
        raise RuntimeError(f"No predictions under {args.results_dir} for this selection")
    for prediction_dir, raw_data_dir in columns:
        print(measure(prediction_dir, raw_data_dir, args.overwrite, args.dry_run), flush=True)


if __name__ == "__main__":
    main()
