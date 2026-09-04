"""One figure comparing several models on the same cases: a row per case, a column per model.

`qualitative.py` draws one run at a time, which answers "how does this model do?" but not "which model
does this case better?" -- reading that off separate figures fails because each run picks its own cases.
Here the cases are chosen once and every model is drawn on the same ones, so a row is a like-for-like
comparison across the columns.

Runs are selected the way `plot_qualitative` selects them -- a list per part of a run directory --
and each selected run becomes a column rather than a figure of its own.

    python -m fm_adaptation.compare_qualitative \
        --models dinov3 --train-datasets Dataset208_lesion_MYKE_smi_gfap Dataset219_lesion_MYK_smi_gfap \
        --configs upernet_inj_ft_balanced_aug_ours --folds 0 \
        --test-datasets Dataset211_lesion_paul_widefield_smi_gfap --splits test --rows 5

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

from .config import describe_run_dir
from .data import active_planes, load_dataset_json, rgb_planes
from .naming import MODEL_NAMES, describe_run
from .metrics import read_case_metrics
from .qualitative import (
    IMAGE_SUFFIXES,
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
from .datasets import dataset_dir as resolve_dataset_dir, dataset_id, family, split_cases
from .selection import matches as _matches, select_runs


def _columns(run_dirs, kind, tested_on):
    """(label, run_dir, prediction_dir, metrics) per run, or the first run that has no predictions.

    A run only predicts the sets its own config names, so a row's own training set is one the other
    rows never evaluated. That is the one case where a comparison cannot be drawn at all, and the
    caller reports which row was missing rather than silently dropping a column.
    """
    columns = []
    for run_dir in run_dirs:
        prediction_dir = next(
            (run_dir / kind / tested_on / name
             for name in ("predictions", "preds")
             if (run_dir / kind / tested_on / name).is_dir()),
            None,
        )
        if prediction_dir is None:
            return None, run_dir
        # Everything that tells this run from another: what it is, and what it was trained on. Two
        # columns can differ in either alone, so both are always written and `_distinct` drops back
        # whichever part they turn out to share.
        model = run_dir.parents[2].name
        label = " + ".join((
            MODEL_NAMES.get(model, model),
            describe_run(run_dir.parent.name),
            f"trained on {dataset_id(run_dir.parents[1].name)}",
        ))
        columns.append((label, run_dir, prediction_dir, prediction_dir.parent / "metrics.csv"))
    return columns, None


def _evaluation_sets(run_dirs, kind):
    """{evaluation set: the runs holding predictions for it}, in the order the runs were selected.

    A run only predicts the sets its own config names, so no set is held by every run of a wide
    selection -- a run is never evaluated on what it trained on, and a baseline from another project
    carries whatever sets that project asked for. Each set therefore takes the runs that have it and
    leaves out the ones that do not, rather than one missing column dropping the figure.
    """
    per_set = {}
    for run_dir in run_dirs:
        for path in (run_dir / kind).glob("*/pred*"):
            if path.is_dir():
                per_set.setdefault(path.parent.name, []).append(run_dir)
    return per_set


def _group_key(run_dir, group_by):
    """What run_dir is filed under, or None when every run belongs to one figure."""
    return run_dir.parents[1].name if group_by == "train-dataset" else None


def _repeat_cases(cases, count):
    """Cases repeated round-robin up to `count`, for a set too small to fill the figure.

    Patchwise rows show a window rather than a whole slide, so an evaluation set of two sections
    still has plenty left to show; the second visit to a case is cropped somewhere else, which
    `crop_window` arranges. Round-robin rather than blocked, so repeats of one case sit apart.
    """
    if not cases or len(cases) >= count:
        return cases
    return [cases[index % len(cases)] for index in range(count)]


def _common_cases(runs, count, reference, rng):
    """Cases every run predicted, sampled across the reference run's Dice range."""
    shared = None
    for _, _, prediction_dir, _ in runs:
        # Whatever the trainer chose to write: ours saves PNG, an external one may save TIFF, and a
        # column read as holding nothing empties the intersection and drops the whole figure.
        stems = {p.stem for p in prediction_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
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
    """Scale to roughly the pixels the panel will occupy.

    Only the image goes through this. A mask keeps its own resolution until its style has been
    applied, since a contour or a skeleton found on a shrunk mask describes a different shape; see
    `qualitative.coverage`, which resizes the drawn coverage instead.
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


# Height of one line of an 8pt header, in points, leading included.
HEADER_LINE_POINTS = 11.0
# Gap between one case's row of panels and the next, in points.
ROW_GAP_POINTS = 2.0


def _header(name):
    """A column header stacked one term to a line.

    A run's name is a list of terms and reads as well down as across, while set on one line it
    decides how wide the column has to be and squeezes the panels underneath it. Stacking spends
    height instead, which is the direction a comparison figure has to spare, and nothing has to be
    cut to fit.
    """
    return "\n".join(name.split(" + "))


def _distinct(labels):
    """Column headers cut down to the part that tells the runs apart.

    Every column here is the same architecture on the same dataset, so the labels share a long
    prefix and often a trailing `trained on ...` as well; a header truncated from the right reads
    identically in all of them, which is the one thing a comparison figure must not do. Drop the
    terms every column agrees on, at both ends, and keep whatever is left.
    """
    parts = [label.split(" + ") for label in labels]
    if len(parts) < 2:
        return list(labels)

    def shared(index):
        return len({p[index] for p in parts}) == 1

    lead, tail = 0, 0
    while all(len(p) > lead + tail + 1 for p in parts) and shared(lead):
        lead += 1
    while all(len(p) > lead + tail + 1 for p in parts) and shared(-1 - tail):
        tail += 1
    if not (lead or tail):
        return list(labels)
    kept = [p[lead : len(p) - tail] for p in parts]
    return [("… + " if lead else "") + " + ".join(k) + (" + …" if tail else "") for k in kept]


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
    # A layout says how the panels are arranged and what sits under a mask -- black or the image --
    # and nothing else. How a mask is painted is `gt_style`/`pred_style`, in every layout.
    on_black = layout in ("masks", "mask_pair")
    # `masks` keeps the truth in its own column, as `qualitative.LAYOUTS` does; every other layout
    # repeats it over each prediction so a column can be read without looking back at the second one.
    overlay_truth = layout != "masks"
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
    # Where each case has already been cropped, so a case drawn more than once shows a different
    # part of the slide each time -- see `_repeat_cases`.
    taken = {}
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
        window = (crop_window(np.asarray(reference), np.asarray(image).shape[:2], crop, rng,
                              avoid=taken.get(case_id, ()))
                  if crop else None)
        if window:
            taken.setdefault(case_id, []).append((window[0].start, window[1].start))
            image = np.asarray(image)[window]
            label = None if label is None else np.asarray(label)[window]
        # Only the image is reduced here. The masks keep the resolution they were drawn at until
        # their style has been applied -- a skeleton or a contour of a shrunk mask is the wrong one --
        # and it is their coverage that gets resized, inside `draw`.
        image = _fit(image, panel_width, nearest=False)
        plain = _to_rgb(np.asarray(image))
        panel_size = (plain.shape[1], plain.shape[0])

        column = first
        if show_image:
            axes[row][column].imshow(plain.clip(0, 255).astype(np.uint8))
            column += 1
        if scored:
            truth = np.zeros_like(plain) if on_black else plain.copy()
            for value, color in gt_colors.items():
                draw(truth, label == value, style["gt_style"], color,
                     style["gt_width"], style["alpha"], panel_size)
            axes[row][column].imshow(truth.clip(0, 255).astype(np.uint8))
            column += 1
        axes[row][first].annotate(_shorten(case_id, 20), xy=(0, 0.5), xytext=(-6, 0),
                                  xycoords="axes fraction", textcoords="offset points",
                                  ha="right", va="center", fontsize=7, rotation=90)

        for _, _, prediction_dir, metrics_path in runs:
            prediction = np.asarray(read_image(_find(prediction_dir, case_id)))
            if window:
                prediction = prediction[window]
            panel = np.zeros_like(plain) if on_black else plain.copy()
            for value, color in pred_colors.items():
                draw(panel, prediction == value, style["pred_style"], color,
                     style["pred_width"], style["alpha"], panel_size)
            if scored and overlay_truth:
                for value, color in gt_colors.items():
                    draw(panel, label == value, style["gt_style"], color,
                         style["gt_width"], style["alpha"], panel_size)
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

    # Every block gets its own headers, since each is a complete set of columns.
    headings = [_header(name) for name in columns]
    for column in range(grid_columns):
        axes[0][column].annotate(headings[column % len(columns)], xy=(0.5, 1.0),
                                 xytext=(0, 8), xycoords="axes fraction",
                                 textcoords="offset points", linespacing=1.35,
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
    # Columns stay flush -- the panels of one case belong together -- but the rows are separate cases
    # and need a gap, or a column of small panels reads as one continuous strip.
    # The top is reserved for the headers, from however many lines the tallest of them runs to.
    inches = figure.get_size_inches()[1]
    header = max((name.count("\n") + 1 for name in headings), default=1) * HEADER_LINE_POINTS / 72
    figure.tight_layout(pad=0, h_pad=ROW_GAP_POINTS, w_pad=0,
                        rect=(0, 0.34 / inches, 1, max(0.5, 1 - (header + 0.2) / inches)))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return Path(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", nargs="*", default=["models"],
                        help="result trees to draw from. An external trainer's tree is laid out the "
                             "same way, so naming it here is all it takes.")
    parser.add_argument("--raw-data-dir", default=None,
                        help="where the datasets live, for runs that ship no config.yaml of their own")
    # One list per part of a run directory, `<model>/<train dataset>/<config>/fold_<n>`, plus the
    # evaluation set. Each selected run is a column, ordered as the lists order it.
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--train-datasets", nargs="*", default=[],
                        help="what a run was trained on")
    parser.add_argument("--configs", nargs="*", default=[],
                        help="configurations to draw. A run with no config.yaml of its own is not "
                             "filtered by this; --models decides whether it is drawn.")
    parser.add_argument("--folds", nargs="*", default=[],
                        help="folds to draw, by number; empty draws every one")
    parser.add_argument("--test-datasets", nargs="*", default=[],
                        help="evaluation sets to draw. Empty takes every set the chosen runs share.")
    parser.add_argument("--splits", nargs="*", default=["test"], choices=("validation", "test"))
    parser.add_argument("--group-by", default="none", choices=("none", "train-dataset"),
                        help="none draws one figure per evaluation set, columns from every selected "
                             "run that has it. train-dataset draws one per training set as well, so "
                             "a figure compares configurations rather than training sets.")
    parser.add_argument("--output-dir", default="results/qualitative",
                        help="figures land in <output-dir>/<family>/<split>__<tested-on>.<format>, with a "
                             "trained_on_<set>__ prefix when grouped by training set")
    parser.add_argument("--format", default="png",
                        help="figure file format, as matplotlib names it: png, svg, pdf")
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
    run_dirs = select_runs(
        args.results_dir, args.models, args.train_datasets, args.configs, args.folds
    )
    groups = {}
    for run_dir in run_dirs:
        groups.setdefault(_group_key(run_dir, args.group_by), []).append(run_dir)
    for group, group_runs in groups.items():
        for kind in args.splits:
            # Which evaluation sets exist is discovered rather than assumed, so a dataset added to
            # `test_datasets` later shows up without editing this.
            per_set = _evaluation_sets(group_runs, kind)
            targets = [d for d in sorted(per_set) if _matches(d, args.test_datasets)]
            if not targets:
                available = ", ".join(sorted(per_set)) or "nothing"
                print(f"no {kind} set to draw for {group or 'the selection'}: runs hold {available}")
            for tested_on in targets:
                # A single column is a figure with nothing to compare, which `plot_qualitative`
                # already draws beside the run itself.
                if len(per_set[tested_on]) < 2:
                    continue
                runs, missing = _columns(per_set[tested_on], kind, tested_on)
                if runs is None:
                    print(f"skipped {kind}/{tested_on}: {missing} has no predictions for it")
                    continue
                cfg = describe_run_dir(runs[0][1], args.raw_data_dir)
                source_path = runs[0][2].parent / "source.json"
                if source_path.exists():
                    source = json.loads(source_path.read_text())
                    source_dataset, split = source["dataset"], source["split"]
                else:
                    source_dataset = tested_on
                    if not cfg.test_split:
                        # Nothing recorded the split, so the predictions say which one it was.
                        predicted = {path.stem for path in runs[0][2].glob("*.png")}
                        directory = resolve_dataset_dir(cfg.raw_data_dir, tested_on)
                        split = next(
                            (s for s in ("Ts", "Tr") if predicted & split_cases(directory, s)), "Ts"
                        )
                    elif tested_on != cfg.train_dataset:
                        split = cfg.test_split
                    else:
                        split = "Ts" if kind == "test" else "Tr"
                dataset_dir = resolve_dataset_dir(cfg.raw_data_dir, source_dataset)
                images, labels = dataset_dir / f"images{split}", dataset_dir / f"labels{split}"
                # No annotations, no ground-truth column: the comparison is then between the models.
                if not (labels.is_dir() and any(labels.iterdir())):
                    labels = None
                channel_planes = rgb_planes(load_dataset_json(dataset_dir)["channel_names"])
                # Restricted to the planes these runs were trained on, so the backdrop is their input.
                # It is the union across columns: a plane one column never saw is still part of what
                # another column was given.
                trained_on = {describe_run_dir(run[1], args.raw_data_dir).train_dataset
                              for run in runs}
                keeps = [active_planes(load_dataset_json(resolve_dataset_dir(cfg.raw_data_dir, t))["channel_names"])
                         for t in sorted(trained_on)]
                keep = None if any(k is None for k in keeps) else frozenset().union(*keeps)
                if channel_planes and keep is not None:
                    channel_planes = {stored: rgb for stored, rgb in channel_planes.items()
                                      if rgb in keep}
                # One window has to serve every column, so `auto` takes the smallest patch any column
                # was trained on: it fits inside the field of view of the wider ones, while the widest
                # would show the narrower columns more than they ever saw. A column with no patching --
                # a baseline that carries no config of its own -- has no say either way rather than
                # dropping the whole figure back to the slide.
                patch_sizes = [
                    run_cfg.patching.patch_size
                    for run_cfg in (describe_run_dir(run[1], args.raw_data_dir) for run in runs)
                    if run_cfg is not None and run_cfg.patching is not None
                ]
                crop = None if args.crop == "full" else (
                    (min(patch_sizes) if patch_sizes else None) if args.crop == "auto"
                    else int(args.crop)
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
                if crop:
                    cases = _repeat_cases(cases, args.rows)
                label_values = load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["labels"]
                names = {int(v): k for k, v in label_values.items() if int(v) != 0}
                # Ungrouped, a figure can hold columns from several training sets, so only the
                # split and the evaluation set name it. Grouped, the training set is what tells two
                # otherwise identical figures apart and has to be part of the name. Either way the
                # figure is filed under the task it is drawn on, so one directory holds one question.
                prefix = f"trained_on_{group}__" if group else ""
                folder = re.sub(r"[^a-z0-9]+", "_", family(tested_on).lower()).strip("_")
                output = (Path(args.output_dir) / folder
                          / f"{prefix}{kind}__{tested_on}.{args.format}")
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
                held = [len([q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES])
                        for _, _, p, _ in runs]
                note = "" if len(set(held)) == 1 else f", columns hold {'/'.join(map(str, held))}"
                print(f"wrote {path}  ({len(runs)} models x {len(cases)} cases, seed {seed}{note})")
    print(f"{drawn} comparison figure(s) drawn, {skipped} up to date")


if __name__ == "__main__":
    main()
