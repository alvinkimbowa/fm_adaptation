"""Datasets that ship the same images.

One section can appear in several datasets under a different case id in each -- `10_018`,
`Mohammad__10` and `Yvonne__Cond-Lesion-...-Rat-10-slide1-section-8` are one slide -- so which
evaluation sets sit inside a training set is decided by what the images look like. That is a property
of the data rather than of any run, so it is measured once alongside everything else and the report
only reads it back.
"""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from .data import FINGERPRINT, SAME_IMAGE, fingerprints

FIELDS = ("dataset_a", "dataset_b", "shared")


def _images(dataset_dir):
    """Every image a dataset ships, across all of its splits, as fingerprints."""
    found = []
    for image_dir in sorted(Path(dataset_dir).glob("images*")):
        split = image_dir.name[len("images"):]
        found += [value for value in fingerprints(dataset_dir, split).values() if value is not None]
    return np.stack(found) if found else None


def measure(dataset_dirs):
    """How many images each pair of datasets has in common."""
    prints = {Path(d).name: _images(d) for d in dataset_dirs}
    rows = []
    for first, second in combinations(sorted(prints), 2):
        a, b = prints[first], prints[second]
        if a is None or b is None:
            rows.append((first, second, 0))
            continue
        similarity = a @ b.T / np.prod(FINGERPRINT)
        rows.append((first, second, int((similarity.max(axis=1) >= SAME_IMAGE).sum())))
    return rows


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        writer.writerows(sorted(rows))


def read(path):
    """The pairs of datasets that share at least one image, or an empty set where none is recorded."""
    if not Path(path).exists():
        return frozenset()
    with open(path, newline="") as f:
        return frozenset(
            frozenset((row["dataset_a"], row["dataset_b"]))
            for row in csv.DictReader(f)
            if int(row["shared"]) > 0
        )


def shares_images(pairs, first, second):
    return frozenset((str(first), str(second))) in pairs
