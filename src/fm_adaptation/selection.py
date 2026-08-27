from fnmatch import fnmatch
from pathlib import Path

from .datasets import dataset_id


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
