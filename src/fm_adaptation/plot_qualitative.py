"""Qualitative figures: sample images beside the same image with ground truth and prediction drawn on.

Everything is read back from what `predict.py` already wrote, so this never loads a model. How each mask
is drawn is chosen per mask from the command line — no style is tied to a particular dataset.
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

from .config import ExperimentConfig
from .patching import open_image
from .selection import matches as _matches

STYLES = ("contour", "overlay", "centerline")
COLORS = {
    "red": (220, 50, 47),
    "green": (60, 200, 90),
    "blue": (60, 130, 240),
    "yellow": (240, 200, 40),
    "magenta": (220, 60, 200),
    "cyan": (60, 210, 210),
}


def _draw(canvas: np.ndarray, mask: np.ndarray, style: str, color, width: int, alpha: float) -> None:
    """Paint a binary mask onto an RGB canvas in place."""
    if not mask.any():
        return
    color = np.array(color, dtype=np.float32)
    if style == "overlay":
        canvas[mask] = (1.0 - alpha) * canvas[mask] + alpha * color
    elif style == "contour":
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, color.tolist(), max(1, width))
    elif style == "centerline":
        line = skeletonize(mask)
        if width > 1:
            line = binary_dilation(line, iterations=width // 2)
        canvas[line] = color
    else:
        raise ValueError(f"Unknown style: {style} (expected one of {STYLES})")


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Grayscale or BGR input, float RGB output ready to be painted on."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32)


def _crop_window(label: np.ndarray, shape, size: int, rng) -> tuple[slice, slice]:
    """A `size` window centred on a random labelled pixel, or on the image centre when there is none."""
    height, width = shape
    rows, cols = np.nonzero(label)
    if len(rows):
        index = rng.integers(len(rows))
        center_y, center_x = int(rows[index]), int(cols[index])
    else:
        center_y, center_x = height // 2, width // 2
    top = int(np.clip(center_y - size // 2, 0, max(0, height - size)))
    left = int(np.clip(center_x - size // 2, 0, max(0, width - size)))
    return slice(top, top + size), slice(left, left + size)


def _select_cases(metrics_path: Path, count: int):
    """Cases spanning the Dice range, best to worst, evenly spaced by rank."""
    with open(metrics_path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("dice")]
    rows.sort(key=lambda row: float(row["dice"]), reverse=True)
    if not rows:
        return []
    picks = np.unique(np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int))
    return [(rows[i]["case_id"], float(rows[i]["dice"])) for i in picks]


def _case_paths(cfg, dataset_name, case_id, output_dir):
    dataset_dir = cfg.raw_data_dir / dataset_name
    split = "Tr" if dataset_name == cfg.train_dataset else cfg.test_split
    ending = None
    for candidate in (".png", ".tif", ".tiff", ".jpg"):
        if (dataset_dir / f"images{split}" / f"{case_id}_0000{candidate}").exists():
            ending = candidate
            break
    if ending is None:
        raise FileNotFoundError(f"No image for {case_id} in {dataset_dir}/images{split}")
    return (
        dataset_dir / f"images{split}" / f"{case_id}_0000{ending}",
        dataset_dir / f"labels{split}" / f"{case_id}{ending}",
        output_dir / "predictions" / f"{case_id}.png",
    )


def _panel(cfg, dataset_name, case_id, output_dir, crop_size, rng, style):
    """Returns the plain image and a copy with both masks painted on, cropped identically."""
    image_path, label_path, prediction_path = _case_paths(cfg, dataset_name, case_id, output_dir)
    image = open_image(image_path)
    label = open_image(label_path)
    prediction = open_image(prediction_path)
    if crop_size:
        rows, cols = _crop_window(np.asarray(label), image.shape[:2], crop_size, rng)
        image, label, prediction = image[rows, cols], label[rows, cols], prediction[rows, cols]
    plain = _to_rgb(np.asarray(image))
    painted = plain.copy()
    _draw(painted, np.asarray(prediction) > 0, style["pred_style"], style["pred_color"], style["pred_width"], style["alpha"])
    _draw(painted, np.asarray(label) > 0, style["gt_style"], style["gt_color"], style["gt_width"], style["alpha"])
    return plain.astype(np.uint8), painted.clip(0, 255).astype(np.uint8)


def _render(cfg, dataset_name, output_dir, rows, cols, crop_size, seed, style):
    cases = _select_cases(output_dir / "metrics.csv", rows * cols)
    if not cases:
        return None
    rng = np.random.default_rng(seed)
    grid_rows = -(-len(cases) // cols)
    figure, axes = plt.subplots(
        grid_rows, 2 * cols, figsize=(3.2 * 2 * cols, 3.4 * grid_rows), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")
    for index, (case_id, dice) in enumerate(cases):
        plain, painted = _panel(cfg, dataset_name, case_id, output_dir, crop_size, rng, style)
        row, column = divmod(index, cols)
        for ax, panel in zip(axes[row][2 * column : 2 * column + 2], (plain, painted)):
            ax.imshow(panel)
        axes[row][2 * column].set_title(_shorten(case_id), fontsize=8)
        axes[row][2 * column + 1].set_title(f"Dice = {dice:.3f}", fontsize=8)
    handles = [
        mpatches.Patch(color=np.array(style["gt_color"]) / 255, label=f"ground truth ({style['gt_style']})"),
        mpatches.Patch(color=np.array(style["pred_color"]) / 255, label=f"prediction ({style['pred_style']})"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    figure.suptitle(f"{dataset_name} — {output_dir.parents[1].parent.name}/{output_dir.parents[1].name}")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    out_path = output_dir / "qualitative.png"
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return out_path


def _shorten(case_id: str, limit: int = 34) -> str:
    return case_id if len(case_id) <= limit else f"{case_id[: limit - 3]}..."


def _resolve_crop(cfg, crop: str) -> int | None:
    if crop == "full":
        return None
    if crop == "auto":
        return cfg.patching.patch_size if cfg.patching is not None else None
    return int(crop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--splits", nargs="*", default=["validation", "test"])
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=2, help="image pairs per row")
    parser.add_argument("--crop", default="auto", help="auto | full | pixels")
    parser.add_argument("--gt-style", choices=STYLES, default="contour")
    parser.add_argument("--pred-style", choices=STYLES, default="overlay")
    parser.add_argument("--gt-color", default="green")
    parser.add_argument("--pred-color", default="red")
    parser.add_argument("--gt-width", type=int, default=2)
    parser.add_argument("--pred-width", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    style = {
        "gt_style": args.gt_style,
        "pred_style": args.pred_style,
        "gt_color": COLORS.get(args.gt_color, COLORS["green"]),
        "pred_color": COLORS.get(args.pred_color, COLORS["red"]),
        "gt_width": args.gt_width,
        "pred_width": args.pred_width,
        "alpha": args.alpha,
    }
    for metrics_path in sorted(Path(args.results_dir).glob("*/Dataset*/*/fold_*/*/*/metrics.csv")):
        output_dir = metrics_path.parent
        fold_dir = output_dir.parents[1]
        dataset_name = output_dir.name
        if not _matches(fold_dir.parents[1].name, args.datasets):
            continue
        if not _matches(fold_dir.parent.name, args.experiments):
            continue
        if not _matches(output_dir.parent.name, args.splits):
            continue
        config_path = fold_dir / "config.yaml"
        if not config_path.exists():
            print(f"skipped {output_dir} (no config.yaml)")
            continue
        cfg = ExperimentConfig.from_yaml(config_path)
        out_path = _render(
            cfg,
            dataset_name,
            output_dir,
            args.rows,
            args.cols,
            _resolve_crop(cfg, args.crop),
            args.seed,
            style,
        )
        print(f"wrote {out_path}" if out_path else f"skipped {output_dir} (no metrics)")


if __name__ == "__main__":
    main()
