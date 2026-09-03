"""Render a qualitative figure for every run under the results directory.

All the drawing lives in `qualitative`, which knows nothing about this repo; this module only works out
which images, labels and predictions belong to each run and where the figure should go.
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import describe_run_dir
from .data import load_dataset_json, rgb_planes, trained_planes
from .qualitative import add_style_arguments, render, style_from_arguments
from .datasets import dataset_dir as resolve_dataset_dir, split_cases
from .selection import matches as _matches, select_runs


def _classes(cfg, dataset_name):
    """The foreground classes and their names, from the dataset the run was *trained* on.

    Taking them from the training dataset rather than the one being drawn keeps a class in the same
    colour across every figure a run produces, including a transfer set that happens to lack one.
    """
    labels = load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["labels"]
    names = {int(value): name for name, value in labels.items() if int(value) != 0}
    return sorted(names), names


def _source_dirs(cfg, dataset_name, kind, output_dir, prediction_dir):
    """The training dataset's `test/` results come from its held-out imagesTs, everything else from Tr."""
    source_path = output_dir / "source.json"
    if source_path.exists():
        source = json.loads(source_path.read_text())
        dataset_name, split = source["dataset"], source["split"]
    elif not cfg.test_split:
        # Nothing recorded the split, so the predictions say which one it was: the images that were
        # predicted are the images of exactly one of them.
        predicted = {path.stem for path in prediction_dir.glob("*.png")}
        directory = resolve_dataset_dir(cfg.raw_data_dir, dataset_name)
        split = next(
            (s for s in ("Ts", "Tr") if predicted & split_cases(directory, s)), "Ts"
        )
    elif dataset_name != cfg.train_dataset:
        split = cfg.test_split
    else:
        split = "Ts" if kind == "test" else "Tr"
    dataset_dir = resolve_dataset_dir(cfg.raw_data_dir, dataset_name)
    labels = dataset_dir / f"labels{split}"
    # A dataset can ship images with no annotations -- then the figure is image and prediction only.
    return (dataset_name, dataset_dir / f"images{split}",
            labels if labels.is_dir() and any(labels.iterdir()) else None)


def _crop_size(cfg, crop):
    if crop == "full":
        return None
    if crop == "auto":
        return cfg.patching.patch_size if cfg.patching is not None else None
    return int(crop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", nargs="*", default=["models"],
                        help="result trees to draw from. An external trainer's tree is laid out the "
                             "same way, so naming it here is all it takes.")
    parser.add_argument("--raw-data-dir", default=None,
                        help="where the datasets live, for runs that ship no config.yaml of their own")
    # One list per part of a run directory, `<model>/<train dataset>/<config>/fold_<n>`, plus the
    # evaluation set. Each is matched exactly, as a glob, as a `_suffix` tag or, for a dataset, by its
    # number; empty keeps every value that part can take.
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--train-datasets", nargs="*", default=[],
                        help="what a run was trained on")
    parser.add_argument("--configs", nargs="*", default=[],
                        help="configurations to draw. A run with no config.yaml of its own is not "
                             "filtered by this; --models decides whether it is drawn.")
    parser.add_argument("--folds", nargs="*", default=[],
                        help="folds to draw, by number; empty draws every one")
    parser.add_argument("--test-datasets", nargs="*", default=[],
                        help="evaluation sets to draw. Empty draws every one a run has.")
    parser.add_argument("--splits", nargs="*", default=["validation", "test"])
    parser.add_argument("--crop", default="auto", help="auto | full | pixels")
    parser.add_argument("--skip-unchanged", action="store_true",
                        help="leave figures that are newer than their metrics alone. Only detects new "
                             "results, never changed layout or style, so it is for the automatic "
                             "post-run refresh; a deliberate run should always redraw.")
    add_style_arguments(parser)
    args = parser.parse_args()
    style = style_from_arguments(args)
    drawn = skipped = 0

    # Selecting the work up front means the progress bar knows its total, and a run that matches nothing
    # says so immediately instead of after a silent walk of the results tree.
    selected = set(select_runs(
        args.results_dir, args.models, args.train_datasets, args.configs, args.folds
    ))
    jobs = []
    # Driven by the predictions rather than the metrics: a dataset with no labels is never scored, so
    # it has no metrics.csv, and keying off that would silently skip every figure it could draw.
    # `preds` as well as `predictions`: the same layout, a different word for the same directory.
    found = sorted(
        path
        for root in args.results_dir
        for name in ("predictions", "preds")
        for path in Path(root).glob(f"*/Dataset*/*/fold_*/*/*/{name}")
    )
    for prediction_dir in found:
        output_dir = prediction_dir.parent
        metrics_path = output_dir / "metrics.csv"
        fold_dir = output_dir.parents[1]
        dataset_name, kind = output_dir.name, output_dir.parent.name
        if not _matches(dataset_name, args.test_datasets):
            continue
        if fold_dir not in selected:
            continue
        if not _matches(kind, args.splits):
            continue
        cfg = describe_run_dir(fold_dir, args.raw_data_dir)
        if cfg is None:
            print(f"skipped {output_dir} (no config.yaml, and no --raw-data-dir to stand in for it)")
            continue
        figure_path = output_dir / "qualitative.png"
        # Redrawing every run costs minutes; only the runs whose metrics moved need a new figure.
        newest = metrics_path if metrics_path.exists() else prediction_dir
        if args.skip_unchanged and figure_path.exists() and figure_path.stat().st_mtime >= newest.stat().st_mtime:
            skipped += 1
            continue
        jobs.append((metrics_path, output_dir, fold_dir, dataset_name, kind, cfg, prediction_dir))

    if not jobs:
        print(f"nothing to draw ({skipped} up to date)")
        return
    print(f"drawing {len(jobs)} figure(s), {skipped} up to date")

    progress = tqdm(jobs, desc="qualitative", unit="fig")
    for metrics_path, output_dir, fold_dir, dataset_name, kind, cfg, prediction_dir in progress:
        # Each figure reads and crops rows x cols source images, so on the whole-slide datasets a single
        # one takes a while; name the run being drawn rather than leaving a bare counter.
        progress.set_postfix_str(f"{fold_dir.parents[1].name}/{fold_dir.parent.name} {kind}")
        source_dataset, images, labels = _source_dirs(cfg, dataset_name, kind, output_dir, prediction_dir)
        classes, class_names = _classes(cfg, dataset_name)
        # Draw the stains into the same planes the model was given them in, so a dataset shipping
        # its stains as separate greyscale files does not come out grey beside one storing the same
        # signal inside an RGB file.
        channel_planes = rgb_planes(
            load_dataset_json(resolve_dataset_dir(cfg.raw_data_dir, source_dataset))["channel_names"]
        )
        # Only the planes the model was trained on, so a czi_B run's figure on a two-stain dataset
        # shows the GFAP it was given and not the SMI it never saw.
        keep = trained_planes(
            load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["channel_names"], cfg.stains
        )
        if channel_planes and keep is not None:
            channel_planes = {stored: rgb for stored, rgb in channel_planes.items() if rgb in keep}
        path = render(
            images,
            labels,
            prediction_dir,
            output_dir / "qualitative.png",
            layout=args.layout,
            rows=args.rows,
            cols=args.cols,
            metrics=metrics_path if metrics_path.exists() else None,
            crop=_crop_size(cfg, args.crop),
            seed=args.seed,
            style=style,
            title=f"{dataset_name} — {fold_dir.parent.name}/{fold_dir.name} ({kind})",
            classes=classes,
            class_names=class_names,
            channel_planes=channel_planes,
        )
        drawn += 1
        progress.write(f"wrote {path}" if path else f"skipped {output_dir} (no metrics)")

    # A silent no-op is indistinguishable from a filter that matched nothing, so always say what happened.
    print(f"{drawn} figure(s) drawn, {skipped} up to date")


if __name__ == "__main__":
    main()
