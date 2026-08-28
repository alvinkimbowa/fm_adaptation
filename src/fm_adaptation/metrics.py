import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


@dataclass(frozen=True)
class CaseMetrics:
    case_id: str
    dice: float
    masd: float


def compute_metrics(prediction: np.ndarray, target: np.ndarray, classes: int):
    dice_scores, surface_distances = [], []
    for label in range(1, classes):
        pred = prediction == label
        truth = target == label
        if not truth.any():
            continue
        denominator = pred.sum() + truth.sum()
        dice_scores.append(2.0 * np.logical_and(pred, truth).sum() / denominator)
        if not pred.any():
            surface_distances.append(float("inf"))
            continue
        pred_surface = np.logical_xor(pred, binary_erosion(pred))
        truth_surface = np.logical_xor(truth, binary_erosion(truth))
        pred_to_truth = distance_transform_edt(~truth_surface)[pred_surface]
        truth_to_pred = distance_transform_edt(~pred_surface)[truth_surface]
        surface_distances.append(float(np.concatenate([pred_to_truth, truth_to_pred]).mean()))
    return float(np.mean(dice_scores)), float(np.mean(surface_distances))


def write_metrics(rows: list[CaseMetrics], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "dice", "masd"])
        for row in rows:
            writer.writerow([row.case_id, row.dice, row.masd])


METRIC_FIELDS = ("dice", "cldice", "hd95", "masd")


def read_case_metrics(path: Path) -> list[dict]:
    """Per-case rows from a metrics CSV, whichever tool wrote it, with the numbers made comparable.

    `fm_adaptation.compute_metrics` writes `image_id,dice,hd95,masd` with a trailing `MEAN` row; the
    older files predate it and are `case_id,dice,masd` with no aggregate, as are the baseline CSVs
    read from elsewhere. The aggregate has to go: every consumer here treats a row as one case, so
    leaving it in counts the mean as an extra case and drags the spread of the distribution towards
    it.

    A case whose ground truth has no foreground -- two of the widefield slides -- has no dice to
    report, and a surface distance to an empty set is not a large distance, it is no distance at all.
    MONAI says that with a blank dice and an infinite distance; both become NaN here, so such a case
    is counted as undefined rather than silently dragging a column's mean to infinity.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        case_id = row.get("image_id") or row.get("case_id")
        if case_id == "MEAN":
            continue
        values = {}
        for field in METRIC_FIELDS:
            if field not in row:
                continue
            raw = (row[field] or "").strip()
            values[field] = float(raw) if raw else float("nan")
        if np.isnan(values.get("dice", 0.0)):
            values = dict.fromkeys(values, float("nan"))
        out.append({**row, **values, "case_id": case_id})
    return out
