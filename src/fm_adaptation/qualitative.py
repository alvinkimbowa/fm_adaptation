"""Qualitative segmentation figures: sample images beside their ground truth and prediction.

Self-contained on purpose — it imports nothing from this package, so the file can be dropped into any
project that has images, labels and predictions sitting in three directories:

    python -m fm_adaptation.qualitative \
        --images IMAGES --labels LABELS --predictions PREDICTIONS --output figure.png \
        --layout pair --rows 3 --cols 2 [--metrics metrics.csv] [--crop 1008]

Layout decides which panels each sample gets; the per-mask styles decide how a mask is painted, and the two
are independent. Nothing here is specific to a dataset or a modality.
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

STYLES = ("contour", "overlay", "centerline")
# Panels each sample occupies, in order.
LAYOUTS = {
    "overlay": ("both",),
    "pair": ("image", "both"),
    "split": ("image", "gt", "prediction"),
    "masks": ("image", "gt_mask", "prediction_mask"),
}
COLORS = {
    "red": (220, 50, 47),
    "green": (60, 200, 90),
    "blue": (60, 130, 240),
    "yellow": (240, 200, 40),
    "magenta": (220, 60, 200),
    "cyan": (60, 210, 210),
    "white": (245, 245, 245),
}
# Multiclass masks colour by *class*, so ground truth and prediction can no longer be told apart by
# colour; the styles do that instead (a contour against a fill). Indexed by class value - 1.
CLASS_PALETTE = (
    COLORS["red"], COLORS["blue"], COLORS["yellow"], COLORS["magenta"], COLORS["cyan"],
    COLORS["green"], COLORS["white"],
)
IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


# --------------------------------------------------------------------------------------- drawing

def draw(canvas, mask, style, color, width=2, alpha=0.4):
    """Paint a binary mask onto an RGB float canvas in place."""
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


def mask_colors(classes, style):
    """The colour each class is painted in, as (ground truth, prediction) maps of value -> RGB.

    One foreground class is the binary case the styles were designed around, and keeps its two
    configured colours. Beyond that, colour has to carry the class, so both maps share a palette and
    ground truth and prediction are told apart by their styles alone.
    """
    classes = [int(value) for value in classes if int(value) != 0]
    if len(classes) <= 1:
        value = classes[0] if classes else 1
        return {value: style["gt_color"]}, {value: style["pred_color"]}
    palette = {value: CLASS_PALETTE[(value - 1) % len(CLASS_PALETTE)] for value in classes}
    return palette, palette


def _panels(image, gt, prediction, layout, style, gt_colors, pred_colors):
    """The panels for one sample, as (title suffix, RGB uint8 array) pairs.

    `gt` and `prediction` are integer label maps; 0 is background and every other value is painted in
    its own colour from the maps.
    """
    plain = _to_rgb(image)
    out = []
    for slot in LAYOUTS[layout]:
        if slot == "image":
            panel = plain.copy()
        elif slot in ("gt_mask", "prediction_mask"):
            source, colors = (gt, gt_colors) if slot == "gt_mask" else (prediction, pred_colors)
            panel = np.zeros_like(plain)
            for value, color in colors.items():
                panel[source == value] = color
        else:
            panel = plain.copy()
            # Prediction first so a ground-truth contour stays legible on top of a filled overlay.
            if slot in ("both", "prediction"):
                for value, color in pred_colors.items():
                    draw(panel, prediction == value, style["pred_style"], color,
                         style["pred_width"], style["alpha"])
            if slot in ("both", "gt"):
                for value, color in gt_colors.items():
                    draw(panel, gt == value, style["gt_style"], color,
                         style["gt_width"], style["alpha"])
        out.append((slot, panel.clip(0, 255).astype(np.uint8)))
    return out


def _to_rgb(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32)


# ------------------------------------------------------------------------------------ input side

def read_image(path):
    """Memory-mapped for TIFF so a crop out of a whole slide stays cheap; decoded otherwise."""
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            import tifffile

            try:
                return tifffile.memmap(path, mode="r")
            except (ValueError, MemoryError):
                return tifffile.imread(path)
        except ImportError:
            pass
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _find(directory, stem):
    for suffix in IMAGE_SUFFIXES:
        for candidate in (f"{stem}{suffix}", f"{stem}_0000{suffix}"):
            path = Path(directory) / candidate
            if path.exists():
                return path
    raise FileNotFoundError(f"No file for {stem} in {directory}")


def select_cases(prediction_dir, count, metrics=None, rng=None):
    """Cases spanning the Dice range when metrics are available, else the first `count` by name.

    The ranking is cut into `count` equal-size bins, best to worst, and one case is drawn from each.
    Every band of quality is therefore represented, but which of a band's cases stands for it varies
    with the seed -- picking the bin edges themselves would show the same handful of cases forever,
    and a flat sample over all cases would over-represent whatever the bulk of the distribution is.
    """
    if metrics and Path(metrics).exists():
        with open(metrics, newline="") as f:
            # A case whose Dice is nan -- no foreground in either the truth or the prediction -- has no
            # place on the ranking; nan compares false against everything and would sort arbitrarily.
            rows = [row for row in csv.DictReader(f)
                    if row.get("dice") and not np.isnan(float(row["dice"]))]
        rows.sort(key=lambda row: float(row["dice"]), reverse=True)
        if rows:
            edges = np.linspace(0, len(rows), min(count, len(rows)) + 1).round().astype(int)
            picks = [
                int(rng.integers(low, high)) if rng is not None and high > low else int(low)
                for low, high in zip(edges[:-1], edges[1:])
            ]
            return [(rows[i]["case_id"], float(rows[i]["dice"])) for i in picks]
    stems = sorted(p.stem for p in Path(prediction_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return [(stem, None) for stem in stems[:count]]


def crop_window(label, shape, size, rng):
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


# ---------------------------------------------------------------------------------------- render

def default_style(**overrides):
    style = {
        "gt_style": "contour",
        "pred_style": "overlay",
        "gt_color": COLORS["green"],
        "pred_color": COLORS["red"],
        "gt_width": 2,
        "pred_width": 2,
        "alpha": 0.4,
    }
    style.update({key: value for key, value in overrides.items() if value is not None})
    return style


def _classes_in(labels, predictions, case_id):
    """The foreground values one case uses, for figures whose class set was not declared up front."""
    found = set()
    for directory in (labels, predictions):
        found |= set(np.unique(np.asarray(read_image(_find(directory, case_id)))).tolist())
    return sorted(value for value in found if value != 0)


def render(images, labels, predictions, output, layout="pair", rows=3, cols=2, metrics=None,
           crop=None, seed=0, style=None, title=None, classes=None, class_names=None):
    """Write one figure of `rows` x `cols` samples, each drawn as `layout` prescribes."""
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout: {layout} (expected one of {sorted(LAYOUTS)})")
    style = style or default_style()
    # One generator for both the case draw and the crop windows, so a seed reproduces a figure whole.
    rng = np.random.default_rng(seed)
    cases = select_cases(predictions, rows * cols, metrics, rng)
    if not cases:
        return None

    # Fixed for the whole figure: a class must keep its colour even in a sample that does not contain
    # it. When the caller does not declare the set, the first sample stands in for the dataset.
    if classes is None:
        classes = _classes_in(labels, predictions, cases[0][0])
    gt_colors, pred_colors = mask_colors(classes, style)

    per_sample = len(LAYOUTS[layout])
    grid_rows = -(-len(cases) // cols)
    figure, axes = plt.subplots(
        grid_rows, per_sample * cols, figsize=(3.2 * per_sample * cols, 3.4 * grid_rows), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")

    for index, (case_id, dice) in enumerate(cases):
        image = read_image(_find(images, case_id))
        label = read_image(_find(labels, case_id))
        prediction = read_image(_find(predictions, case_id))
        if crop:
            window = crop_window(np.asarray(label), image.shape[:2], crop, rng)
            image, label, prediction = image[window], label[window], prediction[window]
        panels = _panels(
            np.asarray(image), np.asarray(label), np.asarray(prediction),
            layout, style, gt_colors, pred_colors,
        )
        row, column = divmod(index, cols)
        base = per_sample * column
        for offset, (slot, panel) in enumerate(panels):
            axes[row][base + offset].imshow(panel)
        score = "" if dice is None else f"Dice = {dice:.3f}"
        if per_sample == 1:
            # Nowhere to put a second title, so the one panel carries both.
            axes[row][base].set_title(" — ".join(filter(None, (_shorten(case_id), score))), fontsize=8)
        else:
            axes[row][base].set_title(_shorten(case_id), fontsize=8)
            if score:
                axes[row][base + per_sample - 1].set_title(score, fontsize=8)

    # `masks` paints the masks flat, so naming a style there would be misleading.
    labels_ = ("ground truth", "prediction")
    if layout != "masks":
        labels_ = (f"ground truth ({style['gt_style']})", f"prediction ({style['pred_style']})")
    if len(pred_colors) > 1:
        # Multiclass: the colours name the classes, so the roles move into the labels.
        names = class_names or {}
        handles = [
            mpatches.Patch(color=np.array(color) / 255, label=names.get(value, f"class {value}"))
            for value, color in sorted(pred_colors.items())
        ]
        if len(set(gt_colors.values())) == 1:
            handles.append(mpatches.Patch(
                color=np.array(next(iter(gt_colors.values()))) / 255, label=labels_[0]))
        else:
            handles.append(mpatches.Patch(color="none", label=f"{labels_[0]} vs {labels_[1]}"))
    else:
        handles = [
            mpatches.Patch(color=np.array(style["gt_color"]) / 255, label=labels_[0]),
            mpatches.Patch(color=np.array(style["pred_color"]) / 255, label=labels_[1]),
        ]
    figure.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
                  frameon=False, fontsize=9)
    if title:
        figure.suptitle(title)
    # Reserve a fixed strip for the legend rather than a fraction, so short figures are not crowded.
    height = figure.get_size_inches()[1]
    figure.tight_layout(rect=(0, 0.45 / height, 1, 1 - (0.4 / height if title else 0)), h_pad=2.0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return Path(output)


def _shorten(text, limit=34):
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def add_style_arguments(parser):
    """Shared by this script and any wrapper that wants the same flags."""
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="pair")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=2, help="samples per row")
    parser.add_argument("--gt-style", choices=STYLES, default="contour")
    parser.add_argument("--pred-style", choices=STYLES, default="overlay")
    parser.add_argument("--gt-color", default="green")
    parser.add_argument("--pred-color", default="red")
    parser.add_argument("--gt-width", type=int, default=2)
    parser.add_argument("--pred-width", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def style_from_arguments(args):
    return default_style(
        gt_style=args.gt_style,
        pred_style=args.pred_style,
        gt_color=COLORS.get(args.gt_color, COLORS["green"]),
        pred_color=COLORS.get(args.pred_color, COLORS["red"]),
        gt_width=args.gt_width,
        pred_width=args.pred_width,
        alpha=args.alpha,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", help="metrics.csv with case_id,dice for Dice-ranked selection")
    parser.add_argument("--crop", type=int, help="crop a window of this many pixels around the label")
    parser.add_argument("--title")
    add_style_arguments(parser)
    args = parser.parse_args()

    path = render(
        args.images,
        args.labels,
        args.predictions,
        args.output,
        layout=args.layout,
        rows=args.rows,
        cols=args.cols,
        metrics=args.metrics,
        crop=args.crop,
        seed=args.seed,
        style=style_from_arguments(args),
        title=args.title,
    )
    print(f"wrote {path}" if path else "nothing to render")


if __name__ == "__main__":
    main()
