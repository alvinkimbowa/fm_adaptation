"""Render a qualitative figure for every run under the results directory.

All the drawing lives in `qualitative`, which knows nothing about this repo; this module only works out
which images, labels and predictions belong to each run and where the figure should go.
"""

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .qualitative import add_style_arguments, render, style_from_arguments
from .selection import matches as _matches


def _source_dirs(cfg, dataset_name, kind):
    """The training dataset's `test/` results come from its held-out imagesTs, everything else from Tr."""
    if dataset_name != cfg.train_dataset:
        split = cfg.test_split
    else:
        split = "Ts" if kind == "test" else "Tr"
    dataset_dir = cfg.raw_data_dir / dataset_name
    return dataset_dir / f"images{split}", dataset_dir / f"labels{split}"


def _crop_size(cfg, crop):
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
    parser.add_argument("--crop", default="auto", help="auto | full | pixels")
    add_style_arguments(parser)
    args = parser.parse_args()
    style = style_from_arguments(args)

    for metrics_path in sorted(Path(args.results_dir).glob("*/Dataset*/*/fold_*/*/*/metrics.csv")):
        output_dir = metrics_path.parent
        fold_dir = output_dir.parents[1]
        dataset_name, kind = output_dir.name, output_dir.parent.name
        if not _matches(fold_dir.parents[1].name, args.datasets):
            continue
        if not _matches(fold_dir.parent.name, args.experiments):
            continue
        if not _matches(kind, args.splits):
            continue
        config_path = fold_dir / "config.yaml"
        if not config_path.exists():
            print(f"skipped {output_dir} (no config.yaml)")
            continue
        cfg = ExperimentConfig.from_yaml(config_path)
        images, labels = _source_dirs(cfg, dataset_name, kind)
        path = render(
            images,
            labels,
            output_dir / "predictions",
            output_dir / "qualitative.png",
            layout=args.layout,
            rows=args.rows,
            cols=args.cols,
            metrics=metrics_path,
            crop=_crop_size(cfg, args.crop),
            seed=args.seed,
            style=style,
            title=f"{dataset_name} — {fold_dir.parent.name}/{fold_dir.name} ({kind})",
        )
        print(f"wrote {path}" if path else f"skipped {output_dir} (no metrics)")


if __name__ == "__main__":
    main()
