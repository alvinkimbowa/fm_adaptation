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


def resolve_runs(results_dir, patterns):
    """Run directories, in the order asked for, from `<model>/<trained-on>/<run>/fold_<n>` patterns.

    A run is a model, a training set, a configuration and a fold, and on disk that is exactly one
    directory, in any of the result trees `results_dir` names. Naming runs by their path means any set of them can be selected -- they need not
    share a training set or a configuration, and the order asked for is the order returned.

    Globs are honoured, so `*/Dataset219*/*aug*/fold_0` picks a family. Empty takes every run.
    """
    roots = [Path(results_dir)] if isinstance(results_dir, (str, Path)) else [Path(r) for r in results_dir]
    resolved = []
    for pattern in patterns or ["*/*/*/fold_*"]:
        found = sorted(
            path
            for root in roots
            for path in root.glob(pattern)
            if path.is_dir() and any(path.glob("*/*/pred*"))
        )
        if not found:
            raise SystemExit(f"no run under {', '.join(map(str, roots))} matches {pattern!r}")
        for path in found:
            if path not in resolved:
                resolved.append(path)
    return resolved
