"""Cache human-to-human neurite agreement, without loading model predictions.

PYTHONPATH=src python -m fm_adaptation.neurite_agreement \
    --raw-data-dir /path/to/nnUNet_raw --results-dir models
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from .data import load_dataset_json, num_classes
from .metrics import CLDICE_FIELDS, cldice
from .neurite_annotations import SCALE_SUFFIX, identity, mapping, is_target_dataset
from .selection import matches


FIELDS = ("annotator_a", "annotator_b", "image", "case_a", "case_b", "dice", *CLDICE_FIELDS)


def path_for(results_dir, dataset):
    return Path(results_dir) / "agreement" / f"{dataset}_neurites.csv"


def annotations(dataset_dir):
    """One native mask per original annotation, across all label splits."""
    ending = load_dataset_json(dataset_dir)["file_ending"]
    found = {}
    for folder in sorted(dataset_dir.glob("labels*")):
        for path in sorted(folder.glob(f"*{ending}")):
            case = path.name[:-len(ending)]
            if SCALE_SUFFIX.search(case):
                continue
            who = identity(case)
            if who is None:
                continue
            key = who
            if key in found:
                # Duplicate split entries must refer to the same annotation.
                previous = found[key][1]
                if previous.resolve() != path.resolve() and previous.read_bytes() != path.read_bytes():
                    raise ValueError(f"Conflicting duplicate annotation: {previous} and {path}")
                continue
            found[key] = (case, path)
    return found


def signature(found):
    masks = [(key, str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
             for key, (_, path) in sorted(found.items())]
    return hashlib.sha256(json.dumps([mapping(), masks], sort_keys=True).encode()).hexdigest()


def measure(dataset_dir, found=None):
    found = annotations(dataset_dir) if found is None else found
    groups = defaultdict(list)
    for (source, person, image), (case, path) in found.items():
        groups[(source, image)].append((person, case, path))
    classes = num_classes(dataset_dir)
    rows = []
    for (source, image), entries in sorted(groups.items()):
        for (person_a, case_a, path_a), (person_b, case_b, path_b) in combinations(sorted(entries), 2):
            a = cv2.imread(str(path_a), cv2.IMREAD_UNCHANGED)
            b = cv2.imread(str(path_b), cv2.IMREAD_UNCHANGED)
            if a is None or b is None or a.ndim != 2 or b.ndim != 2 or a.shape != b.shape:
                raise ValueError(f"Paired masks must have matching native dimensions: {path_a}, {path_b}")
            dice = []
            for label in range(1, classes):
                x, y = a == label, b == label
                denominator = int(x.sum()) + int(y.sum())
                if denominator:
                    dice.append(2 * np.logical_and(x, y).sum() / denominator)
            rows.append(dict(zip(FIELDS, [person_a, person_b, f"{source}__{image}", case_a, case_b,
                                          float(np.mean(dice)) if dice else float("nan"),
                                          *cldice(a, b, classes)])))
    return rows


def read(path):
    if not Path(path).exists():
        return []
    with open(path, newline="") as stream:
        return [{**row, **{key: float(row[key]) if row[key] else float("nan")
                          for key in ("dice", *CLDICE_FIELDS) if key in row}}
                for row in csv.DictReader(stream)]


def cache(dataset_dir, results_dir, overwrite=False):
    found = annotations(dataset_dir)
    if not found:
        return None
    path = path_for(results_dir, dataset_dir.name)
    meta_path = path.with_suffix(".json")
    stamp = signature(found)
    if path.exists() and meta_path.exists() and not overwrite:
        with open(path, newline="") as stream:
            fields = csv.DictReader(stream).fieldnames or []
        if set(FIELDS) <= set(fields) and json.loads(meta_path.read_text()).get("signature") == stamp:
            return f"{dataset_dir.name}: agreement up to date"
    rows = measure(dataset_dir, found)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    meta_path.write_text(json.dumps({"signature": stamp, "annotations": len(found), "pairs": len(rows)}, indent=2) + "\n")
    return f"{dataset_dir.name}: measured {len(rows)} human pairs ({len(found)} native annotations)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("models"))
    parser.add_argument("--datasets", nargs="+", default=["203", "300", "301", "302", "304", "306"],
                        help="native neurite datasets and scale variants; defaults to 203,300,301,302,304,306")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    found = False
    for dataset in sorted(args.raw_data_dir.glob("Dataset*")):
        if not is_target_dataset(dataset.name) or not matches(dataset.name, args.datasets):
            continue
        result = cache(dataset, args.results_dir, args.overwrite)
        if result:
            found = True
            print(result, flush=True)
    if not found:
        parser.error("No mapped neurite annotations found for the selected datasets")


if __name__ == "__main__":
    main()
