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
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

from .metrics import read_case_metrics
from skimage.morphology import skeletonize

STYLES = ("contour", "overlay", "centerline")
# Panels each sample occupies, in order.
LAYOUTS = {
    "overlay": ("both",),
    "pair": ("image", "both"),
    "mask_pair": ("image", "both_mask"),
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
# colour; the styles do that instead (a contour against a fill). Indexed by class value - 1. Brighter
# than COLORS, which is tuned for lines over a bright image rather than fills over a dark one.
CLASS_PALETTE = (
    (255, 75, 70),     # red
    (90, 165, 255),    # blue
    (255, 215, 65),    # yellow
    (255, 100, 225),   # magenta
    (80, 240, 240),    # cyan
    (95, 240, 130),    # green
    (245, 245, 245),   # white
)
IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")
# Panels that exist only to show the ground truth, dropped from a layout when there is none.
GT_ONLY_SLOTS = ("gt", "gt_mask")


def _slots(layout, has_labels=True):
    """The panels one sample occupies. Without labels the ground-truth-only panels fall away."""
    slots = LAYOUTS[layout]
    return slots if has_labels else tuple(s for s in slots if s not in GT_ONLY_SLOTS)


# --------------------------------------------------------------------------------------- drawing

def coverage(mask, style, width=2.0, alpha=0.4, size=None):
    """How much of each pixel a mask covers under one style, in [0, 1].

    The style is applied at the mask's own resolution and only then resampled to `size`. That order
    is the point: a skeleton of a shrunk mask is the medial axis of the wrong shape, and a contour of
    it is the wrong contour -- both have to be found on the pixels the annotator drew. `width` is in
    panel pixels, so it is scaled up to match while drawing and comes back out at the size asked for.

    Resampling the coverage rather than the mask is what keeps a one-pixel line visible: averaging
    turns it into a fainter line instead of dropping it the way nearest neighbour would.
    """
    if not mask.any():
        return None
    height, wide = mask.shape[:2]
    ratio = 1.0 if size is None else max(1.0, wide / max(size[0], 1))
    if style == "overlay":
        painted = np.where(mask, np.float32(alpha), np.float32(0.0))
    elif style == "contour":
        painted = _contour_coverage(mask, width * ratio)
    elif style == "centerline":
        line = skeletonize(mask)
        iterations = int(round(width * ratio)) // 2
        if iterations:
            line = binary_dilation(line, iterations=iterations)
        painted = line.astype(np.float32)
    else:
        raise ValueError(f"Unknown style: {style} (expected one of {STYLES})")
    if size is not None and (wide, height) != tuple(size):
        painted = cv2.resize(painted, tuple(size), interpolation=cv2.INTER_AREA)
    return painted[..., None]


def _contour_coverage(mask, width):
    """The mask's outline as coverage. OpenCV strokes in whole pixels, so a fractional width is drawn
    at double scale and averaged back down; the coverage that falls out is the line's antialiasing."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    height, wide = mask.shape[:2]
    scale = 1 if width == int(width) else 2
    canvas = np.zeros((height * scale, wide * scale), dtype=np.float32)
    cv2.drawContours(canvas, [c * scale for c in contours] if scale > 1 else contours, -1, 1.0,
                     max(1, round(width * scale)), lineType=cv2.LINE_AA if scale > 1 else cv2.LINE_8)
    if scale == 1:
        return canvas
    return cv2.resize(canvas, (wide, height), interpolation=cv2.INTER_AREA)


def draw(canvas, mask, style, color, width=2, alpha=0.4, size=None):
    """Paint a binary mask onto an RGB float canvas in place.

    `canvas` is already at panel size; `mask` may be larger, and `size` says what to reduce it to
    once the style has been applied to the pixels it was drawn on.
    """
    painted = coverage(mask, style, width, alpha, size)
    if painted is None:
        return
    color = np.array(color, dtype=np.float32)
    canvas[:] = (1.0 - painted) * canvas + painted * color


def mask_colors(classes, style):
    """The colour each class is painted in, as (ground truth, prediction) maps of value -> RGB.

    A binary dataset keeps the two configured colours. With more than one class the prediction's
    fill carries the class, so ground truth drawn as a line takes one colour for all of them --
    an outline in the same colour as the fill beneath it is invisible wherever the two agree, which
    is most of the mask. `gt_color="auto"` asks for the class colours there regardless.
    """
    classes = [int(value) for value in classes if int(value) != 0] or [1]
    palette = {value: CLASS_PALETTE[(value - 1) % len(CLASS_PALETTE)] for value in classes}
    auto = style["gt_color"] == "auto"
    if len(classes) == 1:
        value = classes[0]
        return {value: palette[value] if auto else style["gt_color"]}, {value: style["pred_color"]}
    if auto or style["gt_style"] not in ("contour", "centerline"):
        return palette, palette
    return {value: style["gt_color"] for value in classes}, palette


def _panels(image, gt, prediction, layout, style, gt_colors, pred_colors, size=None):
    """The panels for one sample, as (title suffix, RGB uint8 array) pairs.

    `gt` and `prediction` are integer label maps; 0 is background and every other value is painted in
    its own colour from the maps. `gt` may be None -- a dataset can ship images to predict without
    annotations to compare against -- and then the ground-truth panels simply do not appear.
    """
    plain = _to_rgb(image)
    out = []
    for slot in _slots(layout, gt is not None):
        if slot == "image":
            panel = plain.copy()
        elif slot in ("gt_mask", "prediction_mask"):
            # A mask slot differs from its overlay only in what sits underneath: black rather than
            # the image. How a mask is painted is `gt_style`/`pred_style`, the same as everywhere.
            source, colors = (gt, gt_colors) if slot == "gt_mask" else (prediction, pred_colors)
            painted = style["gt_style"] if slot == "gt_mask" else style["pred_style"]
            width = style["gt_width"] if slot == "gt_mask" else style["pred_width"]
            panel = np.zeros_like(plain)
            for value, color in colors.items():
                draw(panel, source == value, painted, color, width, style["alpha"], size)
        else:
            # `both_mask` is the same drawing without the image under it, so the masks are read on
            # their own. Alpha applies here too, against the black rather than against the image.
            panel = np.zeros_like(plain) if slot == "both_mask" else plain.copy()
            # Prediction first so a ground-truth contour stays legible on top of a filled overlay.
            if slot in ("both", "both_mask", "prediction"):
                for value, color in pred_colors.items():
                    draw(panel, prediction == value, style["pred_style"], color,
                         style["pred_width"], style["alpha"], size)
            if gt is not None and slot in ("both", "both_mask", "gt"):
                for value, color in gt_colors.items():
                    draw(panel, gt == value, style["gt_style"], color,
                         style["gt_width"], style["alpha"], size)
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


def _find(directory, stem, channel=0):
    """The file for one case, searching the sibling splits when it is not in `directory`.

    A results column can span both `imagesTr` and `imagesTs` -- an evaluation set is scored whole,
    since its own split boundary means nothing to a model that never trained on it -- while the
    recorded source names a single split. Rather than fail on the half that lives elsewhere, fall
    back to the dataset's other splits of the same kind.
    """
    directory = Path(directory)
    prefix = "labels" if directory.name.startswith("labels") else "images"
    candidates = [directory] + sorted(
        d for d in directory.parent.glob(f"{prefix}*") if d.is_dir() and d != directory
    )
    for candidate_dir in candidates:
        for suffix in IMAGE_SUFFIXES:
            for candidate in (f"{stem}{suffix}", f"{stem}_{channel:04d}{suffix}"):
                path = candidate_dir / candidate
                if path.exists():
                    return path
    raise FileNotFoundError(f"No file for {stem} in {directory}")


def read_channel_planes(images, case_id, channel_planes):
    """Compose separate greyscale files into the RGB planes used by the model.

    The returned array is BGR because the drawing path calls ``_to_rgb`` exactly once for every
    source image, including ordinary OpenCV images.
    """
    composed = None
    for stored, rgb_plane in sorted(channel_planes.items()):
        plane = np.asarray(read_image(_find(images, case_id, stored)))
        if plane.ndim == 3:
            plane = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)
        if composed is None:
            composed = np.zeros((*plane.shape, 3), dtype=plane.dtype)
        elif plane.shape != composed.shape[:2]:
            raise ValueError(f"stain planes have different shapes for {case_id}")
        composed[..., 2 - int(rgb_plane)] = plane
    if composed is None:
        raise ValueError("channel_planes cannot be empty")
    return composed


def select_cases(prediction_dir, count, metrics=None, rng=None):
    """Cases spanning the Dice range when metrics are available, else the first `count` by name.

    The ranking is cut into `count` equal-size bins, best to worst, and one case is drawn from each.
    Every band of quality is therefore represented, but which of a band's cases stands for it varies
    with the seed -- picking the bin edges themselves would show the same handful of cases forever,
    and a flat sample over all cases would over-represent whatever the bulk of the distribution is.
    """
    if metrics and Path(metrics).exists():
        # A case whose Dice is nan -- no foreground in either the truth or the prediction -- has no
        # place on the ranking; nan compares false against everything and would sort arbitrarily. A
        # Dice of 0.0 is a real score and stays.
        rows = [row for row in read_case_metrics(Path(metrics))
                if row.get("dice") is not None and not np.isnan(row["dice"])]
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


def _window_overlap(a, b, size):
    """Fraction of a `size` window at `a` that a window at `b` covers."""
    down = size - abs(a[0] - b[0])
    across = size - abs(a[1] - b[1])
    return max(down, 0) * max(across, 0) / float(size * size)


def crop_window(label, shape, size, rng, avoid=(), tolerance=0.25, attempts=32):
    """A `size` window centred on a random labelled pixel, or on the image centre when there is none.

    `avoid` names windows already taken from this image, as (top, left). A candidate is accepted once
    it covers less than `tolerance` of every one of them; failing that the least overlapping of
    `attempts` tries is returned, so an image whose annotation occupies one small region still yields
    a window rather than raising.
    """
    height, width = shape
    rows, cols = np.nonzero(label)
    best = best_overlap = None
    for _ in range(attempts if avoid else 1):
        if len(rows):
            index = rng.integers(len(rows))
            center_y, center_x = int(rows[index]), int(cols[index])
        else:
            center_y, center_x = height // 2, width // 2
        top = int(np.clip(center_y - size // 2, 0, max(0, height - size)))
        left = int(np.clip(center_x - size // 2, 0, max(0, width - size)))
        overlap = max((_window_overlap((top, left), taken, size) for taken in avoid), default=0.0)
        if best is None or overlap < best_overlap:
            best, best_overlap = (top, left), overlap
        if overlap < tolerance:
            break
    top, left = best
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
        if directory is None:
            continue
        found |= set(np.unique(np.asarray(read_image(_find(directory, case_id)))).tolist())
    return sorted(value for value in found if value != 0)


# A cell is drawn at the shape of what goes in it, and the figure is then held inside this envelope.
# Without the first, a portrait slide -- the lesion sections are nearly 1:3 -- collapses to a sliver in
# a square cell and the figure is mostly white margin; without the second, shaping the cells correctly
# is what makes the file enormous.
MAX_FIGURE_INCHES = 24.0
MAX_FIGURE_PIXELS = 2600


def panel_aspect(images, case_id, crop, channel_planes=None):
    """Height / width of what one panel will show, read cheaply from the source image."""
    if crop:
        # `crop_window` cuts a square, so the panels are square whatever the slide looks like.
        return 1.0
    channel = min(channel_planes) if channel_planes else 0
    path = _find(images, case_id, channel)
    if path is None:
        return 1.0
    # An eighth-size decode: the aspect is all that is wanted here, not the pixels.
    probe = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if probe is None or not probe.shape[0] or not probe.shape[1]:
        return 1.0
    # Clamped because a pathological aspect would otherwise leave the other axis too small to read.
    return float(np.clip(probe.shape[0] / probe.shape[1], 0.35, 3.0))


def _to_panel(image, size):
    """The image reduced to the pixels the panel actually has, and the size the masks reduce to.

    Only the image is resampled here. The masks stay at the resolution they were drawn at until their
    style has been applied to them -- see `coverage` -- because a skeleton or a contour found on a
    shrunk mask describes a different shape. Their coverage is what gets resized, so a line survives
    the reduction as a fainter line rather than being dropped.
    """
    height, width = np.asarray(image).shape[:2]
    scale = min(size[0] / width, size[1] / height)
    if scale >= 1.0:
        return image, None
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(np.asarray(image), target, interpolation=cv2.INTER_AREA), target


def render(images, labels, predictions, output, layout="pair", rows=3, cols=2, metrics=None,
           crop=None, seed=0, style=None, title=None, classes=None, class_names=None,
           channel_planes=None):
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

    per_sample = len(_slots(layout, labels is not None))
    grid_rows = -(-len(cases) // cols)
    cell_width = 3.2
    cell_height = cell_width * panel_aspect(images, cases[0][0], crop, channel_planes)
    width = cell_width * per_sample * cols
    height = cell_height * grid_rows
    scale = min(1.0, MAX_FIGURE_INCHES / max(width, height))
    width, height = width * scale, height * scale
    # Trade resolution for extent once a figure is large, so a tall grid stays a readable file rather
    # than a 30-megapixel one.
    dpi = min(140, MAX_FIGURE_PIXELS / max(width, height))
    # What one panel is worth in pixels once the figure is sized and rasterised. Everything is
    # reduced to this before it is drawn on.
    panel_size = (max(1, int(cell_width * scale * dpi)), max(1, int(cell_height * scale * dpi)))
    figure, axes = plt.subplots(
        grid_rows, per_sample * cols, figsize=(width, height), squeeze=False
    )
    for ax in axes.ravel():
        ax.axis("off")

    for index, (case_id, dice) in enumerate(cases):
        image = (
            read_channel_planes(images, case_id, channel_planes)
            if channel_planes
            else read_image(_find(images, case_id))
        )
        label = None if labels is None else read_image(_find(labels, case_id))
        prediction = read_image(_find(predictions, case_id))
        if crop:
            # Without a ground truth the prediction is what the crop is centred on; it is the only
            # mask there is, and a window drawn blind would usually miss the lesion entirely.
            window = crop_window(np.asarray(label if label is not None else prediction),
                                 image.shape[:2], crop, rng)
            image, prediction = image[window], prediction[window]
            label = None if label is None else label[window]
        image, target = _to_panel(image, panel_size)
        panels = _panels(
            np.asarray(image), None if label is None else np.asarray(label),
            np.asarray(prediction), layout, style, gt_colors, pred_colors, target,
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
        # The resolved colours, not the configured ones: `gt_color` may be "auto", which only
        # `mask_colors` knows how to turn into an RGB. One entry each, on this branch.
        (gt_color,), (pred_color,) = gt_colors.values(), pred_colors.values()
        handles = [
            mpatches.Patch(color=np.array(gt_color) / 255, label=labels_[0]),
            mpatches.Patch(color=np.array(pred_color) / 255, label=labels_[1]),
        ]
    figure.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
                  frameon=False, fontsize=9)
    if title:
        figure.suptitle(title)
    # Reserve a fixed strip for the legend rather than a fraction, so short figures are not crowded.
    height = figure.get_size_inches()[1]
    figure.tight_layout(rect=(0, 0.45 / height, 1, 1 - (0.4 / height if title else 0)), h_pad=2.0)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
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
    parser.add_argument("--gt-color", default="green",
                        help=f"{'|'.join(COLORS)}, or auto to follow each class's own colour")
    parser.add_argument("--pred-color", default="red", help="|".join(COLORS))
    parser.add_argument("--gt-width", type=float, default=2)
    parser.add_argument("--pred-width", type=float, default=2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def style_from_arguments(args):
    return default_style(
        gt_style=args.gt_style,
        pred_style=args.pred_style,
        # "auto" is not a colour but a request to follow the class palette, so it passes through.
        gt_color="auto" if args.gt_color == "auto" else COLORS.get(args.gt_color, COLORS["green"]),
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
