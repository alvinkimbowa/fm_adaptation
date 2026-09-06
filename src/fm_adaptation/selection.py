import json
from fnmatch import fnmatch
from pathlib import Path

from .datasets import dataset_dir as resolve_dataset_dir, dataset_id, split_cases


def matches(name, patterns):
    """Empty means keep everything; entries match exactly, as a glob, as a `_suffix` tag, or, for a
    dataset, by its number alone."""
    if not patterns:
        return True
    return any(
        name == pattern
        or fnmatch(name, pattern)
        or name.endswith(f"_{pattern}")
        or (pattern.isdigit() and dataset_id(name) == dataset_id(pattern))
        for pattern in patterns
    )


def list_order(value, listed):
    """Sort key placing `value` where `listed` puts it, and after everything it names when absent.

    An empty list means "keep everything", so it has no opinion on order; the value's own name then
    decides, which at least keeps the result stable between runs.
    """
    listed = list(listed)
    return (listed.index(value), "") if value in listed else (len(listed), value)


def select_runs(results_dirs, models=(), train_datasets=(), configs=(), folds=()):
    """Run directories matching every named part, in the order the lists put them.

    A run is one directory, `<model>/<train dataset>/<configuration>/fold_<n>`, in any of the trees
    `results_dirs` names. A part with an empty list is not narrowed. Ordering follows the parts in
    the order the path writes them: model, then training set, then configuration.

    A run that ships no `config.yaml` names its directory in its trainer's vocabulary rather than
    this project's, so `configs` cannot name it and `models` alone decides whether it is kept.
    """
    roots = [Path(results_dirs)] if isinstance(results_dirs, (str, Path)) else [Path(r) for r in results_dirs]
    found = {}
    for root in roots:
        for run_dir in root.glob("*/Dataset*/*/fold_*"):
            if not any(run_dir.glob("*/*/pred*")):
                continue
            model, trained_on, config = (
                run_dir.parents[2].name, run_dir.parents[1].name, run_dir.parent.name
            )
            if not (matches(model, models) and matches(trained_on, train_datasets)):
                continue
            if (run_dir / "config.yaml").is_file() and not matches(config, configs):
                continue
            if not matches(run_dir.name.removeprefix("fold_"), folds):
                continue
            found[run_dir] = (
                list_order(model, models),
                list_order(trained_on, train_datasets),
                list_order(config, configs),
                run_dir.name,
            )
    return sorted(found, key=found.get)


def source_dirs(cfg, dataset_name, kind, output_dir, prediction_dir):
    """Where the images and labels a run was evaluated on live, as (dataset, images, labels).

    The training dataset's `test/` results come from its held-out imagesTs, everything else from Tr.
    `output_dir` is the directory the predictions sit in, `<run>/<split>/<evaluation set>/`.
    """
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
