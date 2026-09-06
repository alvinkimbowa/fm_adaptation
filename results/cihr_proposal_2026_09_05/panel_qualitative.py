"""One composite figure of a single case: a 2x2 grid of bare panels, lettered a, b, c, d.

Where `plot_qualitative` and `compare_qualitative` draw grids of many cases carrying headers, Dice
and a legend, this draws one case as a publication plate -- four panels, thin white gutters, a letter
in each corner and no other text at all. A panel is the image, the annotator's traces, another
method's traces, or a run's prediction, in whatever order `--panels` gives:

    python results/cihr_proposal_2026_09_05/panel_qualitative.py \
        --models dinov3 --train-datasets Dataset302_neurite_yvonne_b2_smi_1px \
        --configs convnextt_upernet_ft_aug_p512_red_ours --folds 0 \
        --test-datasets Dataset303_neurite_yvonne_in_vitro_smi --size 2048 --rotate 90

Every panel shows the same centre crop of the same case, so the four line up pixel for pixel.
Drawing is `qualitative.py`'s, including the `centerline` style that reduces a mask to its one-pixel
skeleton before it is painted.
"""

import argparse
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fm_adaptation.config import describe_run_dir
from fm_adaptation.data import active_planes, load_dataset_json, rgb_planes
from fm_adaptation.datasets import dataset_dir as resolve_dataset_dir, family
from fm_adaptation.qualitative import (
    COLORS,
    IMAGE_SUFFIXES,
    MAX_FIGURE_INCHES,
    MAX_FIGURE_PIXELS,
    _find,
    _to_panel,
    _to_rgb,
    draw,
    read_channel_planes,
    read_image,
)
from fm_adaptation.selection import matches as _matches, select_runs, source_dirs

SLOTS = ("image", "manual", "other", "prediction")
# What a mask panel is reduced to before it is painted. `skeleton` is the one-pixel centreline, so
# annotator and model are compared at the same stroke width whatever each of them drew.
POSTPROCESS = {"skeleton": "centerline", "none": "overlay"}


def centre_window(shape, size):
    """A centred window of `size` = (height, width), clamped to an image of `shape`.

    A requested extent wider than the image is cut back to the image, and the other side is cut with
    it so the window keeps the aspect ratio that was asked for -- an image narrower than the request
    yields a smaller window of the same shape rather than a stretched one. `size` of None is the
    whole image.
    """
    height, width = shape[:2]
    if size is None:
        return slice(0, height), slice(0, width)
    scale = min(1.0, height / size[0], width / size[1])
    take = (max(1, int(size[0] * scale)), max(1, int(size[1] * scale)))
    top, left = (height - take[0]) // 2, (width - take[1]) // 2
    return slice(top, top + take[0]), slice(left, left + take[1])


def _oriented(array, rotate):
    """The array turned by `rotate` degrees anticlockwise.

    Masks are turned here, before they are drawn, rather than the finished panel being turned: a
    skeleton has to be found on the pixels the annotator drew, and a resampled panel is not those.
    """
    return array if not rotate else np.rot90(np.asarray(array), rotate // 90)


def _mask_panel(shape, mask, style, color, width, size):
    """One mask painted white on black, or a black panel when there is no mask to paint.

    `shape` is the panel's own (height, width) and `size` the (width, height) the mask's coverage is
    reduced to once its style has been applied -- `qualitative.coverage` does the reduction there,
    not on the mask, so a one-pixel line survives as a line rather than being dropped.
    """
    panel = np.zeros((*shape, 3), dtype=np.float32)
    if mask is not None:
        for value in (v for v in np.unique(mask) if v != 0):
            draw(panel, mask == value, style, color, width, 1.0, size)
    return panel


def _panel_size(figure, dpi, columns, rows):
    inches = figure.get_size_inches()
    return (max(1, int(inches[0] / columns * dpi)), max(1, int(inches[1] / rows * dpi)))


def render_panels(contents, shape, output, letters="abcd", letter_size=42, letter_color="white",
                  gutter=0.01, style="centerline", line_width=1.5, mask_color=COLORS["white"],
                  format_dpi=None):
    """Write the 2x2 plate. `contents` holds four `(kind, array)` panels, reading across then down.

    `kind` is "image" for the case's image and "mask" for a label map; every array is at `shape`,
    already cropped and oriented the same way, so the four panels line up pixel for pixel. A mask of
    None leaves its panel black, which is what an unfilled slot -- a method whose results have not
    been produced yet -- looks like.
    """
    height, width = shape[:2]
    cell = 4.0
    figure_width, figure_height = cell * 2, cell * 2 * height / width
    scale = min(1.0, MAX_FIGURE_INCHES / max(figure_width, figure_height))
    figure_width, figure_height = figure_width * scale, figure_height * scale
    dpi = format_dpi or min(200, MAX_FIGURE_PIXELS / max(figure_width, figure_height))

    figure, axes = plt.subplots(
        2, 2, figsize=(figure_width, figure_height), squeeze=False,
        gridspec_kw={"wspace": gutter, "hspace": gutter},
    )
    figure.patch.set_facecolor("white")
    # Everything is reduced to the pixels a panel actually has. Only images are resampled; a mask
    # keeps its own resolution until its style has been applied to it, and its coverage is what
    # `_mask_panel` reduces. Every panel shares `shape`, so one target serves all of them.
    panel_size = _panel_size(figure, dpi, 2, 2)
    target = _to_panel(np.zeros((height, width), dtype=np.uint8), panel_size)[1]
    panel_shape = (height, width) if target is None else (target[1], target[0])
    panels = [
        _to_rgb(_to_panel(np.asarray(array), panel_size)[0]) if kind == "image"
        else _mask_panel(panel_shape, array, style, mask_color, line_width, target)
        for kind, array in contents
    ]
    for index, (ax, panel) in enumerate(zip(axes.ravel(), panels)):
        ax.axis("off")
        ax.imshow(panel.clip(0, 255).astype(np.uint8))
        if index < len(letters):
            ax.text(0.02, 0.98, letters[index], transform=ax.transAxes, ha="left", va="top",
                    color=letter_color, fontsize=letter_size)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    return Path(output)


def _prediction_dir(run_dir, kind, tested_on):
    for name in ("predictions", "preds"):
        directory = run_dir / kind / tested_on / name
        if directory.is_dir():
            return directory
    return None


def _read_mask(directory, case_id):
    """A case's mask from a directory of them, or None when that directory does not hold it."""
    if directory is None:
        return None
    try:
        return np.asarray(read_image(_find(Path(directory), case_id)))
    except FileNotFoundError:
        return None


def _parse_size(values):
    if not values:
        return None
    numbers = [int(v) for v in values]
    return (numbers[0], numbers[0]) if len(numbers) == 1 else (numbers[0], numbers[1])


def _find_ok(directory, case_id):
    try:
        _find(Path(directory), case_id)
        return True
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", nargs="*", default=["models"],
                        help="result trees to draw from. An external trainer's tree is laid out the "
                             "same way, so naming it here is all it takes.")
    parser.add_argument("--raw-data-dir", default=None,
                        help="where the datasets live, for runs that ship no config.yaml of their own")
    # One list per part of a run directory, `<model>/<train dataset>/<config>/fold_<n>`, plus the
    # evaluation set. Each is matched exactly, as a glob, as a `_suffix` tag or, for a dataset, by
    # its number; empty keeps every value that part can take.
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--train-datasets", nargs="*", default=[])
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--folds", nargs="*", default=["0"])
    parser.add_argument("--test-datasets", nargs="*", default=[])
    parser.add_argument("--splits", nargs="*", default=["test"], choices=("validation", "test"))
    parser.add_argument("--case", default=None,
                        help="the case to draw. Empty draws one at random from those predicted.")
    parser.add_argument("--panels", nargs=4, default=list(SLOTS), choices=SLOTS,
                        help="what each of the four panels holds, reading left to right, top to "
                             "bottom. Each `prediction` takes the next selected run and each "
                             "`other` the next --other-predictions directory.")
    parser.add_argument("--other-predictions", nargs="*", default=[],
                        help="directories of another method's masks, named after the same cases. A "
                             "slot with no directory, or a directory without this case, is black.")
    parser.add_argument("--size", nargs="*", default=[],
                        help="centre crop applied to every panel, in source pixels: one number for "
                             "a square, two for height and width. Empty keeps the whole image.")
    parser.add_argument("--rotate", type=int, default=90, choices=(0, 90, 180, 270),
                        help="degrees anticlockwise. The slides are taller than they are wide, so "
                             "the default turns the plate landscape.")
    parser.add_argument("--postprocess", default="skeleton", choices=sorted(POSTPROCESS),
                        help="skeleton reduces every mask to its one-pixel centreline; none draws "
                             "each at the width it was stored at.")
    parser.add_argument("--line-width", type=float, default=1.5,
                        help="how many panel pixels wide a trace is drawn, after the postprocess")
    parser.add_argument("--mask-color", default="white", help="|".join(COLORS))
    parser.add_argument("--letters", default="abcd", help="panel letters; empty draws none")
    parser.add_argument("--letter-size", type=float, default=42)
    parser.add_argument("--letter-color", default="white")
    parser.add_argument("--gutter", type=float, default=0.01,
                        help="space between panels, as a fraction of a panel. The white figure "
                             "behind it is what makes the separating lines.")
    parser.add_argument("--output-dir", default="results/cihr_proposal_2026_09_05/figures/neurites")
    parser.add_argument("--format", default="png",
                        help="figure file format, as matplotlib names it: png, svg, pdf")
    parser.add_argument("--dpi", type=float, default=None)
    parser.add_argument("--seed", type=int, default=-1,
                        help="negative draws a fresh case every run")
    args = parser.parse_args()

    run_dirs = select_runs(
        args.results_dir, args.models, args.train_datasets, args.configs, args.folds
    )
    if not run_dirs:
        print("no run matched the selection")
        return
    wanted = args.panels.count("prediction")
    if len(run_dirs) < wanted:
        print(f"{args.panels.count('prediction')} prediction panel(s) asked for, "
              f"{len(run_dirs)} run(s) matched")
        return
    run_dirs = run_dirs[:wanted]
    others = [Path(d) for d in args.other_predictions]
    rng = np.random.default_rng(None if args.seed < 0 else args.seed)
    size = _parse_size(args.size)
    style = POSTPROCESS[args.postprocess]
    color = COLORS.get(args.mask_color, COLORS["white"])

    drawn = 0
    for kind in args.splits:
        # Which evaluation sets exist is discovered rather than assumed, so a dataset added later
        # shows up without editing this.
        available = sorted({d.name for run in run_dirs for d in (run / kind).glob("*")
                            if _prediction_dir(run, kind, d.name)})
        for tested_on in [d for d in available if _matches(d, args.test_datasets)]:
            prediction_dirs = [_prediction_dir(run, kind, tested_on) for run in run_dirs]
            if any(d is None for d in prediction_dirs):
                continue
            cfg = describe_run_dir(run_dirs[0], args.raw_data_dir)
            _, images, labels = source_dirs(
                cfg, tested_on, kind, prediction_dirs[0].parent, prediction_dirs[0]
            )
            cases = sorted(p.stem for p in prediction_dirs[0].iterdir()
                           if p.suffix.lower() in IMAGE_SUFFIXES)
            for directory in prediction_dirs[1:]:
                cases = [c for c in cases if _find_ok(directory, c)]
            if args.case:
                if args.case not in cases:
                    print(f"skipped {kind}/{tested_on}: no prediction for {args.case}")
                    continue
                case_id = args.case
            elif cases:
                case_id = cases[int(rng.integers(len(cases)))]
            else:
                print(f"skipped {kind}/{tested_on}: no case predicted by every run")
                continue

            # The planes these runs were trained on, so the backdrop is the input they were given --
            # an SMI-only run is drawn in red because that is the plane its stain sits in.
            dataset_dir = resolve_dataset_dir(cfg.raw_data_dir, tested_on)
            channel_planes = rgb_planes(load_dataset_json(dataset_dir)["channel_names"])
            keep = active_planes(
                load_dataset_json(resolve_dataset_dir(cfg.raw_data_dir, cfg.train_dataset))["channel_names"]
            )
            if channel_planes and keep is not None:
                channel_planes = {stored: rgb for stored, rgb in channel_planes.items() if rgb in keep}
            image = (read_channel_planes(images, case_id, channel_planes) if channel_planes
                     else read_image(_find(images, case_id)))
            image = np.asarray(image)

            window = centre_window(image.shape, size)
            image = _oriented(image[window], args.rotate)
            # Each slot draws on its own queue of directories, so two `prediction` panels are two
            # runs and two `other` panels two directories, in the order they were named.
            sources = {"manual": [labels], "other": others, "prediction": prediction_dirs}
            taken = {slot: 0 for slot in sources}
            contents = []
            for slot in args.panels:
                if slot == "image":
                    contents.append(("image", image))
                    continue
                pool = sources[slot]
                mask = _read_mask(pool[taken[slot]], case_id) if taken[slot] < len(pool) else None
                taken[slot] += 1
                contents.append(("mask", None if mask is None
                                 else _oriented(mask[window], args.rotate)))

            folder = re.sub(r"[^a-z0-9]+", "_", family(tested_on).lower()).strip("_")
            output = (Path(args.output_dir) / folder
                      / f"{kind}__{tested_on}__{case_id}.{args.format}")
            path = render_panels(
                contents, image.shape, output,
                letters=args.letters, letter_size=args.letter_size,
                letter_color=args.letter_color, gutter=args.gutter, style=style,
                line_width=args.line_width, mask_color=color, format_dpi=args.dpi,
            )
            drawn += 1
            print(f"wrote {path}")
    print(f"{drawn} panel figure(s) drawn")


if __name__ == "__main__":
    main()
