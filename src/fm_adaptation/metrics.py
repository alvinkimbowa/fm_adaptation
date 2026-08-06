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
