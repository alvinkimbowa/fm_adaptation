"""Compute segmentation metrics from a prediction directory and a ground-truth directory.

Prediction and metrics are separate stages on purpose. A prediction is written back at the case's
native resolution, which is the resolution the annotator drew at, so measuring it needs nothing but
two directories of label maps -- no model, no preprocessing, no GPU. Folding the two together is what
produced the bug this replaces: the old inference loop measured against the label it had already
shrunk onto the encoder's square canvas and blown back up, which for a lesion slide is a ~7.5x round
trip that costs about 1.2 dice of boundary before a model is involved at all.

Dice, centreline Dice, HD95 and mean average surface distance per case, background excluded. Nothing
is resized here: the predictions must already be in the space of the labels.

    PYTHONPATH=src python -m fm_adaptation.compute_metrics \
        --results-dir models --datasets 209 --folds 0
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
from skimage.morphology import skeletonize

from . import agreement
from .datasets import dataset_dir as resolve_dataset_dir, resolve
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
    cldice: float
    hd95: float
    masd: float


def _nanmean(values) -> float:
    """The mean of the cases that produced a number.

    `inf` is masked along with `nan`: a case whose label is empty has no surface for a distance to be
    measured to, and MONAI reports that as `inf` rather than `nan`. Two of Paul's sixteen cases are
    like this, and left in they made every distance column read `inf`.
    """
    array = np.array(list(values), dtype=np.float64)
    finite = np.isfinite(array)
    return float("nan") if not finite.any() else float(array[finite].mean())


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


def _cldice(prediction, target, num_classes):
    """Centreline Dice, averaged over the foreground classes.

    Shit et al.'s clDice: the harmonic mean of how much of the prediction's skeleton lies inside the
    truth and how much of the truth's skeleton lies inside the prediction. Overlap Dice on a structure
    two pixels wide is dominated by whether a trace is placed exactly right; clDice asks instead
    whether the same paths were followed, which is what tracing is judged on.
    """
    scores = []
    for value in range(1, num_classes):
        pred, truth = prediction == value, target == value
        if not pred.any() or not truth.any():
            # No foreground to trace on one side: undefined rather than zero, and `_nanmean` drops it.
            scores.append(np.nan)
            continue
        pred_skeleton, truth_skeleton = skeletonize(pred), skeletonize(truth)
        precision = (pred_skeleton & truth).sum() / pred_skeleton.sum()
        sensitivity = (truth_skeleton & pred).sum() / truth_skeleton.sum()
        scores.append(
            0.0 if precision + sensitivity == 0
            else 2 * precision * sensitivity / (precision + sensitivity)
        )
    return _nanmean(scores)


def compute_case_metrics(prediction, target, num_classes, ignore_empty=True):
    """(dice, cldice, hd95, masd) averaged over the foreground classes, background excluded."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    cldice = _cldice(np.asarray(prediction), np.asarray(target), num_classes)
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
    overlap = tuple(_nanmean(x.detach().cpu().numpy().reshape(-1)) for x in (dice, hd95, masd))
    return (overlap[0], cldice) + overlap[1:]


def write_csv(rows: list[CaseMetrics], path: Path, add_aggregate_row=True):
    """Per-case rows plus a trailing MEAN, so the file carries its own summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    def fmt(value):
        return "" if value is None or np.isnan(value) else f"{value:.6f}"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "dice", "cldice", "hd95", "masd"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image_id": row.image_id,
                "dice": fmt(row.dice), "cldice": fmt(row.cldice),
                "hd95": fmt(row.hd95), "masd": fmt(row.masd),
            })
        if add_aggregate_row and rows:
            writer.writerow({
                "image_id": "MEAN",
                "dice": fmt(_nanmean(r.dice for r in rows)),
                "cldice": fmt(_nanmean(r.cldice for r in rows)),
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
    raw_data_dir = Path(cfg["data"]["raw_data_dir"])
    return (
        cfg["model"]["name"],
        cfg["model"].get("run_name", cfg["model"]["probe"]),
        resolve(raw_data_dir, cfg["data"]["train_dataset"]),
        raw_data_dir,
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


def _nnunet_columns(results_dir: Path, raw_data_dir: Path, args):
    """Every (predictions dir, raw data dir) an nnU-Net results tree holds for this selection.

    nnU-Net runs are trained elsewhere and carry no config of ours, so the selection comes from the
    layout: `nnunet/<trained on>/<trainer>/fold_N/test/<tested on>/preds`. The predictions are written
    at the case's native resolution, which is all this needs.
    """
    selected = []
    for prediction_dir in sorted(results_dir.glob("nnunet/Dataset*/*/fold_*/test/Dataset*/preds")):
        trained_on = prediction_dir.parents[4].name
        fold = prediction_dir.parents[2].name.removeprefix("fold_")
        if (
            matches(trained_on, args.datasets)
            and matches(fold, args.folds)
            and matches(prediction_dir.parent.name, args.tested_on)
        ):
            selected.append((prediction_dir, raw_data_dir))
    return selected


def _is_current(metrics_path: Path, prediction_dir: Path):
    """Whether a metrics file already reflects every prediction beside it."""
    if not metrics_path.exists():
        return False
    predictions = [
        path for path in prediction_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_SUFFIXES
    ]
    return bool(predictions) and metrics_path.stat().st_mtime >= max(
        p.stat().st_mtime for p in predictions
    )


def measure(prediction_dir: Path, raw_data_dir: Path, overwrite=False, dry_run=False):
    """Write `metrics.csv` beside `prediction_dir`; returns a one-line report of what happened."""
    output_dir = prediction_dir.parent
    dataset_dir = resolve_dataset_dir(raw_data_dir, output_dir.name)
    metrics_path = output_dir / "metrics.csv"
    label = "/".join(p.name for p in reversed(output_dir.parents[:4])) + f" -> {output_dir.name}"

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
    parser.add_argument("--tested-on", nargs="*", default=[], help="evaluation sets, nnU-Net only")
    parser.add_argument(
        "--nnunet-results-dir", nargs="*", default=[],
        help="nnU-Net results trees to measure as well, selected by --datasets/--folds/--tested-on",
    )
    parser.add_argument("--raw-data-dir", help="where the nnU-Net trees' datasets live")
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

    columns = _columns(Path(args.results_dir), args) if not args.nnunet_results_dir else []
    for results_dir in args.nnunet_results_dir:
        if not args.raw_data_dir:
            raise RuntimeError("--nnunet-results-dir needs --raw-data-dir")
        found = _nnunet_columns(Path(results_dir), Path(args.raw_data_dir), args)
        if not found:
            raise RuntimeError(f"No nnU-Net predictions under {results_dir} for this selection")
        columns += found
    if not columns:
        raise RuntimeError(f"No predictions under {args.results_dir} for this selection")
    for prediction_dir, raw_data_dir in columns:
        print(measure(prediction_dir, raw_data_dir, args.overwrite, args.dry_run), flush=True)

    # Agreement between annotators is a property of a dataset rather than of any run, so it is
    # measured once for every evaluation set that ships the same image drawn twice.
    agreement_dir = Path(args.results_dir) / "agreement"
    for dataset_dir in sorted({resolve_dataset_dir(raw, d.parent.name) for d, raw in columns}):
        for split in agreement.splits(dataset_dir):
            path = agreement.path_for(agreement_dir, dataset_dir.name, split)
            if path.exists() and not args.overwrite:
                print(f"{dataset_dir.name}{split} agreement: up to date", flush=True)
                continue
            if args.dry_run:
                print(f"{dataset_dir.name}{split} agreement: would measure", flush=True)
                continue
            rows, unpaired = agreement.measure(dataset_dir, split)
            agreement.write(rows, unpaired, path)
            print(f"{dataset_dir.name}{split} agreement: measured {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
