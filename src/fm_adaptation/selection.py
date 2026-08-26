from fnmatch import fnmatch

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
