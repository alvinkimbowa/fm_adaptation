"""One figure comparing several models on the same cases: a row per case, a column per model.

`qualitative.py` draws one run at a time, which answers "how does this model do?" but not "which model
does this case better?" -- reading that off separate figures fails because each run picks its own cases.
Here the cases are chosen once and every model is drawn on the same ones, so a row is a like-for-like
comparison across the columns.

`--experiments` is a flat list of runs, each named by its path under `--results-dir`, and the order
given is the column order. A run is a model, a training set, a configuration and a fold, so any
combination of them can be put side by side -- they need not share a training set or a configuration.
`--datasets` says which evaluation sets to draw.

    python -m fm_adaptation.compare_qualitative \
        --experiments dinov3/Dataset208_lesion_MYKE_smi_gfap/upernet_inj_ft_balanced_aug_ours/fold_0 \
                      dinov3/Dataset219_lesion_MYK_smi_gfap/upernet_inj_ft_balanced_aug_gfap_ours/fold_0 \
        --datasets Dataset211_lesion_paul_widefield_smi_gfap --splits test --rows 5

Drawing is `qualitative.py`'s: the same styles, the same colours, the same channel handling. Only the
arrangement is new.
"""

import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from .config import ExperimentConfig
from .data import active_planes, load_dataset_json, rgb_planes
from .naming import MODEL_NAMES, dataset_tag, describe_run
from .metrics import read_case_metrics
from .qualitative import (
    MAX_FIGURE_INCHES,
    MAX_FIGURE_PIXELS,
    _classes_in,
    _find,
    _shorten,
    _to_rgb,
    add_style_arguments,
    crop_window,
    draw,
    mask_colors,
    panel_aspect,
    read_channel_planes,
    read_image,
    select_cases,
    style_from_arguments,
)
from .datasets import dataset_dir as resolve_dataset_dir
from .selection import matches as _matches, resolve_runs


def _columns(run_dirs, kind, tested_on):
    """(label, run_dir, prediction_dir, metrics) per run, or the first run that has no predictions.

    A run only predicts the sets its own config names, so a row's own training set is one the other
    rows never evaluated. That is the one case where a comparison cannot be drawn at all, and the
    caller reports which row was missing rather than silently dropping a column.
    """
    columns = []
    for run_dir in run_dirs:
        prediction_dir = run_dir / kind / tested_on / "predictions"
        if not prediction_dir.is_dir():
            return None, run_dir
        # Everything that tells this run from another: what it is, and what it was trained on. Two
        # columns can differ in either alone, so both are always written and `_distinct` drops back
        # whichever part they turn out to share.
        model = run_dir.parents[2].name
        label = " + ".join((
            MODEL_NAMES.get(model, model),
            describe_run(run_dir.parent.name),
            f"trained on {dataset_tag(run_dir.parents[1].name)}",
        ))
        columns.append((label, run_dir, prediction_dir, prediction_dir.parent / "metrics.csv"))
    return columns, None


def _evaluation_sets(run_dirs, kind):
    """Evaluation sets every selected run has predictions for, so a figure has every column."""
    shared = None
    for run_dir in run_dirs:
        here = {p.parent.name for p in (run_dir / kind).glob("*/predictions")}
        shared = here if shared is None else (shared & here)
    return sorted(shared or ())


def _common_cases(runs, count, reference, rng):
    """Cases every run predicted, sampled across the reference run's Dice range."""
    shared = None
    for _, _, prediction_dir, _ in runs:
        stems = {p.stem for p in prediction_dir.iterdir() if p.suffix.lower() == ".png"}
        shared = stems if shared is None else (shared & stems)
    shared = shared or set()
    # `select_cases` spans the quality range rather than taking the first few, which is what makes a
    # comparison figure worth looking at -- the interesting rows are the ones the reference finds hard.
    metrics = reference[3] if Path(reference[3]).exists() else None
    picked = select_cases(reference[2], count * 3, metrics, rng)
    ordered = [case for case in picked if case[0] in shared]
    if len(ordered) < count:
        # Not enough of the reference's sample is shared; fall back to whatever every run has.
        extra = [(stem, None) for stem in sorted(shared) if stem not in {c for c, _ in ordered}]
        ordered += extra
    # Keep the spread: cut the surviving ranking into `count` equal bands, best to worst, and draw one
    # case from each rather than taking the top `count`. Drawing within the band rather than at its
    # edge is what lets the seed move the sample: picking edges deterministically showed the same
    # cases forever on any dataset small enough that `select_cases` returned all of them, which is
    # every evaluation set here except Eric's 73.
    if len(ordered) > count:
        edges = np.linspace(0, len(ordered), count + 1).round().astype(int)
        picks = [
            int(rng.integers(low, high)) if rng is not None and high > low else int(low)
            for low, high in zip(edges[:-1], edges[1:])
        ]
        ordered = [ordered[i] for i in dict.fromkeys(picks)]
    return ordered[:count]


def _case_dice(metrics_path, case_id):
    if not Path(metrics_path).exists():
        return None
    for row in read_case_metrics(Path(metrics_path)):
        # A Dice of 0.0 is a real score, so test for presence rather than truthiness.
        if row["case_id"] == case_id and row.get("dice") is not None:
            value = float(row["dice"])
            return None if np.isnan(value) else value
    return None


def _fit(array, target_width, nearest):
    """Scale to roughly the pixels the panel will occupy, before anything is drawn onto it.

    Drawing at native resolution and letting the figure scale it down is the expensive way round: a
    lesion slide is tens of megapixels and the cell it lands in is a few hundred pixels wide, so all
    but a fraction of the contouring and blending is discarded. Labels and predictions take nearest
    neighbour, which keeps class values as values rather than blending them into new ones.
    """
    array = np.asarray(array)
    height, width = array.shape[:2]
    if target_width <= 0 or width <= target_width:
        return array
    scale = target_width / width
    return cv2.resize(
        array, (int(round(width * scale)), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA,
    )


def _distinct(labels):
    """Column headers cut down to the part that tells the runs apart.

    Every column here is the same architecture on the same dataset, so the labels share a long
    prefix and a header truncated from the right reads identically in all of them -- which is the
    one thing a comparison figure must not do. Drop the shared leading terms instead.
    """
    parts = [label.split(" + ") for label in labels]
    if len(parts) < 2:
        return list(labels)
    common = 0
    while all(len(p) > common + 1 for p in parts) and len({p[common] for p in parts}) == 1:
        common += 1
    if not common:
        return list(labels)
    return ["… + " + " + ".join(p[common:]) for p in parts]


def render_comparison(images, labels, runs, cases, output, style, crop=None, seed=0,
                      classes=None, class_names=None, title=None, channel_planes=None,
                      layout="pair", per_row=1):
    """Cases down the figure and models across it: the image, the ground truth, then each prediction.

    `per_row` puts several cases side by side on one row, each with its own full set of columns. With
    only a few models the single-case row is mostly whitespace, and two cases fill the same width.

    `layout` is read the way `qualitative.LAYOUTS` reads it, so the same setting shapes both kinds of
    figure: a `mask` layout paints on black instead of over the image, and `overlay` drops the raw
    image column.
    """
    on_black = layout in ("masks", "mask_pair")
    # `masks` paints the label maps flat, exactly as `qualitative.LAYOUTS` does for its mask slots --
    # no style and no alpha, so the mask is read as a shape rather than as a blend.
    flat = layout == "masks"
    show_image = layout != "overlay"
    rng = np.random.default_rng(seed)
    if classes is None:
        classes = _classes_in(labels, runs[0][2], cases[0][0])

    gt_colors, pred_colors = mask_colors(classes, style)

    # A dataset with no annotations has no ground-truth column and no Dice to print; the figure is
    # then the image beside one prediction per model.
    scored = labels is not None
    columns = ((["image"] if show_image else []) + (["ground truth"] if scored else [])
               + _distinct([label for label, *_ in runs]))
    per_row = max(1, per_row)
    grid_columns = len(columns) * per_row
    grid_rows = -(-len(cases) // per_row)
    aspect = panel_aspect(images, cases[0][0], crop, channel_planes)
    cell_width = 3.2
    width = cell_width * grid_columns
    height = cell_width * aspect * grid_rows
    scale = min(1.0, MAX_FIGURE_INCHES / max(width, height))
    width, height = width * scale, height * scale
    dpi = min(140, MAX_FIGURE_PIXELS / max(width, height))
    figure, axes = plt.subplots(grid_rows, grid_columns, figsize=(width, height), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    # What one panel is actually worth in pixels. Anything beyond this is drawn and then discarded.
    panel_width = int(width / grid_columns * dpi)

    # A row decodes one source image plus one prediction per column, and the lesion slides are tens of
    # megapixels each -- without this a figure looks like a hang for minutes at a time.
    progress = tqdm(list(enumerate(cases)), desc=Path(output).stem, unit="case", leave=False)
    for index, (case_id, _) in progress:
        # A case owns one block of `len(columns)` panels; `per_row` blocks share a row.
        row, block = divmod(index, per_row)
        first = block * len(columns)
        progress.set_postfix_str(_shorten(case_id, 24))
        image = (
            read_channel_planes(images, case_id, channel_planes)
            if channel_planes
            else read_image(_find(images, case_id))
        )
        label = read_image(_find(labels, case_id)) if scored else None
        reference = label if scored else read_image(_find(runs[0][2], case_id))
        window = (crop_window(np.asarray(reference), np.asarray(image).shape[:2], crop, rng)
                  if crop else None)
        if window:
            image = np.asarray(image)[window]
            label = None if label is None else np.asarray(label)[window]
        image = _fit(image, panel_width, nearest=False)
        label = None if label is None else _fit(label, panel_width, nearest=True)
        plain = _to_rgb(np.asarray(image))

        column = first
        if show_image:
            axes[row][column].imshow(plain.clip(0, 255).astype(np.uint8))
            column += 1
        if scored:
            truth = np.zeros_like(plain) if on_black else plain.copy()
            for value, color in gt_colors.items():
                if flat:
                    truth[label == value] = color
                else:
                    draw(truth, label == value, style["gt_style"], color,
                         style["gt_width"], style["alpha"])
            axes[row][column].imshow(truth.clip(0, 255).astype(np.uint8))
            column += 1
        axes[row][first].annotate(_shorten(case_id, 20), xy=(0, 0.5), xytext=(-6, 0),
                                  xycoords="axes fraction", textcoords="offset points",
                                  ha="right", va="center", fontsize=7, rotation=90)

        for _, _, prediction_dir, metrics_path in runs:
            prediction = np.asarray(read_image(_find(prediction_dir, case_id)))
            if window:
                prediction = prediction[window]
            prediction = _fit(prediction, panel_width, nearest=True)
            panel = np.zeros_like(plain) if on_black else plain.copy()
            for value, color in pred_colors.items():
                if flat:
                    panel[prediction == value] = color
                else:
                    draw(panel, prediction == value, style["pred_style"], color,
                         style["pred_width"], style["alpha"])
            if scored and not flat:
                # The ground truth goes over every prediction too, so a column is read against the
                # truth without looking back at the second column. A flat mask has no room for it.
                for value, color in gt_colors.items():
                    draw(panel, label == value, style["gt_style"], color,
                         style["gt_width"], style["alpha"])
            axes[row][column].imshow(panel.clip(0, 255).astype(np.uint8))
            dice = _case_dice(metrics_path, case_id)
            if dice is not None:
                # Inside the panel, not a title: a title reserves a strip of vertical space in every
                # row, which is most of the gap between rows once there are several of them.
                axes[row][column].text(
                    0.02, 0.98, f"{dice:.3f}", transform=axes[row][column].transAxes,
                    ha="left", va="top", fontsize=11, color="white",
                    bbox=dict(facecolor="black", alpha=0.55, pad=2.0, edgecolor="none"),
                )
            column += 1

    for column in range(grid_columns):
        # The column header names the model; the per-cell title is that case's Dice, so both fit.
        # Every block gets its own headers, since each is a complete set of columns.
        axes[0][column].annotate(_shorten(columns[column % len(columns)], 46), xy=(0.5, 1.0),
                                 xytext=(0, 18), xycoords="axes fraction",
                                 textcoords="offset points",
                                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    names = class_names or {}
    handles = [
        mpatches.Patch(color=np.array(color) / 255, label=names.get(value, f"class {value}"))
        for value, color in sorted(pred_colors.items())
    ]
    if scored:
        handles.append(mpatches.Patch(color=np.array(next(iter(gt_colors.values()))) / 255,
                                      label=f"ground truth ({style['gt_style']})"))
    figure.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
                  frameon=False, fontsize=9)
    if title:
        figure.suptitle(title)
    # Zero padding everywhere: the panels are the figure, and the row gaps are what the layout is for.
    inches = figure.get_size_inches()[1]
    figure.tight_layout(pad=0, h_pad=0, w_pad=0, rect=(0, 0.34 / inches, 1, 1))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="models")
    parser.add_argument("--experiments", nargs="*", default=[],
                        help="one column each, as `<model>/<trained-on>/<run>/fold_<n>` under "
                             "--results-dir. Globs allowed; the order given is the column order. "
                             "Empty takes every run.")
    parser.add_argument("--datasets", nargs="*", default=[],
                        help="evaluation sets to draw. Empty takes every set the chosen runs share.")
    parser.add_argument("--splits", nargs="*", default=["test"], choices=("validation", "test"))
    parser.add_argument("--output-dir", default="results/qualitative",
                        help="figures land in <output-dir>/<split>__<tested-on>.png")
    parser.add_argument("--crop", default="auto", help="auto | full | pixels")
    parser.add_argument("--per-row", type=int, default=1,
                        help="cases side by side on one row, each with its own set of columns")
    parser.add_argument("--seed-max", type=int, default=1000,
                        help="upper bound on the seed drawn when --seed is negative")
    parser.add_argument("--skip-unchanged", action="store_true",
                        help="leave figures newer than every metrics.csv they draw from. Detects new "
                             "results only, never a changed layout or style.")
    add_style_arguments(parser)
    args = parser.parse_args()
    style = style_from_arguments(args)

    drawn = skipped = 0
    run_dirs = resolve_runs(args.results_dir, args.experiments)
    for kind in args.splits:
        # Which evaluation sets exist is discovered rather than assumed, so a dataset added to
        # `test_datasets` later shows up without editing this.
        available = _evaluation_sets(run_dirs, kind)
        targets = [d for d in available if _matches(d, args.datasets)]
        for tested_on in targets:
            runs, missing = _columns(run_dirs, kind, tested_on)
            if runs is None:
                print(f"skipped {kind}/{tested_on}: {missing} has no predictions for it")
                continue
            cfg = ExperimentConfig.from_yaml(runs[0][1] / "config.yaml")
            source_path = runs[0][2].parent / "source.json"
            if source_path.exists():
                source = json.loads(source_path.read_text())
                source_dataset, split = source["dataset"], source["split"]
            else:
                source_dataset = tested_on
                split = cfg.test_split if tested_on != cfg.train_dataset else (
                    "Ts" if kind == "test" else "Tr"
                )
            dataset_dir = resolve_dataset_dir(cfg.raw_data_dir, source_dataset)
            images, labels = dataset_dir / f"images{split}", dataset_dir / f"labels{split}"
            # No annotations, no ground-truth column: the comparison is then between the models.
            if not (labels.is_dir() and any(labels.iterdir())):
                labels = None
            channel_planes = rgb_planes(load_dataset_json(dataset_dir)["channel_names"])
            # Restricted to the planes these runs were trained on, so the backdrop is their input.
            # It is the union across columns: a plane one column never saw is still part of what
            # another column was given.
            trained_on = {ExperimentConfig.from_yaml(run[1] / "config.yaml").train_dataset
                          for run in runs}
            keeps = [active_planes(load_dataset_json(resolve_dataset_dir(cfg.raw_data_dir, t))["channel_names"])
                     for t in sorted(trained_on)]
            keep = None if any(k is None for k in keeps) else frozenset().union(*keeps)
            if channel_planes and keep is not None:
                channel_planes = {stored: rgb for stored, rgb in channel_planes.items()
                                  if rgb in keep}
            crop = None if args.crop == "full" else (
                cfg.patching.patch_size if args.crop == "auto" and cfg.patching else
                None if args.crop == "auto" else int(args.crop)
            )
            # A negative seed asks for a fresh sample of cases every run, overwriting the
            # figure that was there. Pin a seed to keep drawing the same cases.
            seed = args.seed if args.seed >= 0 else int(
                np.random.default_rng().integers(args.seed_max)
            )
            rng = np.random.default_rng(seed)
            cases = _common_cases(runs, args.rows, runs[0], rng)
            if not cases:
                print(f"skipped {kind}/{tested_on} (no cases shared by all runs)")
                continue
            label_values = load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["labels"]
            names = {int(v): k for k, v in label_values.items() if int(v) != 0}
            # Flat: with a list of rows there is no one training set to file the figure under, so the
            # split and the evaluation set name it outright.
            output = Path(args.output_dir) / f"{kind}__{tested_on}.png"
            # An unscored dataset has no metrics.csv anywhere, so freshness comes from the
            # predictions themselves rather than from a file that will never exist.
            stamps = [m.stat().st_mtime for *_, m in runs if m.exists()]
            stamps += [p.stat().st_mtime for _, _, p, m in runs if not m.exists()]
            newest = max(stamps)
            if args.skip_unchanged and output.exists() and output.stat().st_mtime >= newest:
                skipped += 1
                continue
            path = render_comparison(
                images, labels, runs, cases, output, style,
                crop=crop, seed=seed, classes=sorted(names), class_names=names,
                channel_planes=channel_planes,
                layout=args.layout, per_row=args.per_row,
            )
            drawn += 1
            # How many cases each column holds, when they differ. A row that trained on part of this
            # set has only its `imagesTs`, a row that never touched it has the whole thing, and the
            # figure is drawn on the intersection -- 8 of 73 should not read the same as 8 of 8.
            held = [len([q for q in p.iterdir() if q.suffix.lower() == ".png"])
                    for _, _, p, _ in runs]
            note = "" if len(set(held)) == 1 else f", columns hold {'/'.join(map(str, held))}"
            print(f"wrote {path}  ({len(runs)} models x {len(cases)} cases, seed {seed}{note})")
    print(f"{drawn} comparison figure(s) drawn, {skipped} up to date")


if __name__ == "__main__":
    main()
