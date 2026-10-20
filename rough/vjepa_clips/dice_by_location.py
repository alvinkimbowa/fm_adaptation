"""Per-image test Dice on 701 Ts, grouped by scan location, as the arrays the results page plots.

    PYTHONPATH=src python rough/vjepa_clips/dice_by_location.py KEY=RUN_DIR [KEY=RUN_DIR ...]

Each RUN_DIR is a fold directory; its test/Dataset701_nmus_median_rgb/metrics.csv is read. Prints
one JSON object, KEY -> five lists in LOCATIONS order, each in image-id order. An image with no
Dice (nothing predicted and nothing traced) is left out, as the page has always done.
"""

import csv
import json
import sys
from pathlib import Path

from fm_adaptation.datasets import metadata_values

RAW = Path("../nmus_segmentation/data/nnUNet_raw/Dataset701_nmus_median_rgb")
LOCATIONS = ["wrist", "elbow", "mid-arm", "mid-forearm", "distal forearm"]


def main():
    where = metadata_values(RAW, "location")
    out = {}
    for spec in sys.argv[1:]:
        key, run_dir = spec.split("=", 1)
        rows = csv.DictReader(open(Path(run_dir) / "test/Dataset701_nmus_median_rgb/metrics.csv"))
        groups = {location: [] for location in LOCATIONS}
        for row in sorted(rows, key=lambda r: r["image_id"]):
            location = (where.get(row["image_id"]) or ("",))[0]
            if location in groups and row["dice"]:
                groups[location].append(round(float(row["dice"]), 3))
        out[key] = [groups[location] for location in LOCATIONS]
    print(json.dumps(out))


if __name__ == "__main__":
    main()
