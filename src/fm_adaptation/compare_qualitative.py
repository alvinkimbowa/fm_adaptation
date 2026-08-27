"""One figure comparing several models on the same cases: a row per case, a column per model.

`qualitative.py` draws one run at a time, which answers "how does this model do?" but not "which model
does this case better?" -- reading that off separate figures fails because each run picks its own cases.
Here the cases are chosen once and every model is drawn on the same ones, so a row is a like-for-like
comparison across the columns.

    python -m fm_adaptation.compare_qualitative \
        --datasets 204 --splits test \
        --experiments upernet_inj_ft_ours upernet_ours --rows 5

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
from .report import ADAPTATIONS, _config_label, _dataset_label, _model_rank, _split_adaptation
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
from .selection import matches as _matches


# Where the across-training-sets figures go. Every other figure directory is named after the training
# set its columns share; these have no single training set to be named after, because that is the
# thing being compared.
ACROSS_DIR = "across_training_sets"


def _runs(results_dir, dataset, kind, tested_on, models, experiments, name_training_set=False):
    """Every run that has predictions for `tested_on`, as (label, run_dir, prediction_dir, metrics).

    `name_training_set` adds what the run was trained on to its label, which is the only thing that
    tells two columns apart when the same configuration is compared across training sets.
    """
    found = []
    pattern = f"*/{dataset}/*/fold_*/{kind}/{tested_on}/predictions"
    for prediction_dir in sorted(Path(results_dir).glob(pattern)):
        fold_dir = prediction_dir.parents[2]
        run_name = fold_dir.parent.name
        model = fold_dir.parents[2].name
        if models and not _matches(model, models):
            continue
        if experiments and not _matches(run_name, experiments):
            continue
        # The column carries the label the report gives this row, so a figure and the table name the
        # same thing the same way -- a column is one row of the results table, config and all.
        label = _config_label(model, run_name)
        if name_training_set:
            # Joined with the same separator the labels already use, so `_distinct` strips the shared
            # configuration and leaves the training set, which is what the comparison is about.
            label = f"{label} + trained on {_training_tag(dataset)}"
        found.append((label, fold_dir,
                      prediction_dir, prediction_dir.parent / "metrics.csv", model, run_name))
    # Columns in the order the report puts its rows, so reading across the figure and down the table
    # follow the same sequence.
    found.sort(key=lambda run: (_model_rank(_report_model(run[4])), run[4],
                                ADAPTATIONS.get(_split_adaptation(run[5])[0], ("", 99))[1],
                                _split_adaptation(run[5])[1]))
    return [run[:4] for run in found]


def _training_tag(dataset):
    """A short name for a training set, for a column header that has to fit.

    Every SCI set carries the same `_smi_gfap` stain suffix, which is exactly the part that does not
    distinguish one from another.
    """
    return re.sub(r"_(smi_)?gfap$", "", _dataset_label(dataset))


def _report_model(model):
    """`_model_rank` keys foundation models by their config name, which is what the paths carry."""
    return model


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
    parser.add_argument("--datasets", nargs="*", default=[], help="what the runs were trained on")
    parser.add_argument("--tested-on", nargs="*", default=[], help="which evaluation set to draw")
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--experiments", nargs="*", default=[])
    parser.add_argument("--splits", nargs="*", default=["test"], choices=("validation", "test"))
    parser.add_argument("--output-dir", default="results/qualitative",
                        help="figures land in <output-dir>/<trained-on>/")
    parser.add_argument("--crop", default="auto", help="auto | full | pixels")
    parser.add_argument("--across-training-sets", action="store_true",
                        help="one figure per evaluation set with every selected training set in it, "
                             "instead of one figure per training set")
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
    # Normally each training set gets its own figure. `--across-training-sets` puts them in one, so a
    # column is "the same configuration trained on a different dataset" rather than "another model".
    groups = [(d,) for d in args.datasets] if not args.across_training_sets else [tuple(args.datasets)]
    for group in groups:
        dataset = group[0] if len(group) == 1 else ACROSS_DIR
        for kind in args.splits:
            # Which evaluation sets exist is discovered rather than assumed, so a dataset added to
            # `test_datasets` later shows up without editing this.
            available = sorted({
                p.parent.name
                for trained_on in group
                for p in Path(args.results_dir).glob(f"*/{trained_on}/*/fold_*/{kind}/*/predictions")
            })
            targets = [d for d in available if not args.tested_on or _matches(d, args.tested_on)]
            for tested_on in targets:
                runs = [
                    run
                    for trained_on in group
                    for run in _runs(args.results_dir, trained_on, kind, tested_on,
                                     args.models, args.experiments,
                                     name_training_set=len(group) > 1)
                ]
                # Across training sets a single run has nothing to compare against, so the figure
                # would be a column on its own. Within one training set it is still worth drawing:
                # a training set with one adaptation is exactly the case where its predictions have
                # nowhere else to be seen.
                if len(runs) < (2 if len(group) > 1 else 1):
                    print(f"skipped {dataset}/{kind}/{tested_on} ({len(runs)} run(s) matched)")
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
                # Restricted to the planes these runs were trained on, so the backdrop is their
                # input. Across training sets it is the union: a plane one column never saw is still
                # part of what another column was given.
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
                    print(f"skipped {dataset}/{kind}/{tested_on} (no cases shared by all runs)")
                    continue
                label_values = load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["labels"]
                names = {int(v): k for k, v in label_values.items() if int(v) != 0}
                # Filed under what the models were trained on, which is what the figure compares;
                # the split and the evaluation set name the file within it.
                output = Path(args.output_dir) / dataset / f"{kind}__{tested_on}.png"
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
                print(f"wrote {path}  ({len(runs)} models x {len(cases)} cases, seed {seed})")
    print(f"{drawn} comparison figure(s) drawn, {skipped} up to date")


if __name__ == "__main__":
    main()
