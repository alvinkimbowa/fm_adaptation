"""Agreement between two annotators who drew the same image.

The ceiling every model in the tables is read against. Measuring it means pairing the cases by what
they look like and then scoring one annotation against the other, so it is done where everything else
is measured and written down beside the results, and the report only reads it back.
"""

import csv
import re
from pathlib import Path

import cv2
import numpy as np

from .data import (
    FINGERPRINT,
    SAME_IMAGE,
    fingerprint,
    load_dataset_json,
    num_classes,
    stain_channel,
)
from .metrics import compute_metrics


def annotator(case_id):
    """Who drew this annotation, from the case ID.

    `Mohammad__1` and `Yvonne_b2__Cond-Lesion-...` name the annotator before the `__`; a `_b2` on the
    end of that is a second batch by the same person rather than a second person. A `_rater1` suffix
    on the case itself is the one place two annotators share a source name.
    """
    source = case_id.split("__", 1)[0]
    source = re.sub(r"_b\d+$", "", source)
    rater = re.search(r"_rater(\d+)$", case_id)
    return f"{source} rater{rater.group(1)}" if rater else source


def image_name(case_ids):
    """A name for the image the pair shares, taken from whichever ID describes it."""
    name = max(case_ids, key=len).split("__", 1)[-1]
    return re.sub(r"_rater\d+$", "", name)


def interrater_pairs(dataset_dir, split):
    """The same image annotated twice, as (image name, (case A, case B)), plus anything unpaired.

    The pairing cannot come from the case IDs -- `Mohammad__1` and `Yvonne__...Rat-1-slide11-section-1`
    are the same slide -- so it comes from the images. Two cases pair when each is the other's best
    match, the correlation clears `SAME_IMAGE`, and their labels have identical dimensions; without
    that last condition a near-miss would produce a Dice that means nothing.
    """
    info = load_dataset_json(dataset_dir)
    # The stain the models are actually shown, so agreement is read on the same picture they saw.
    channel = stain_channel(info)
    ending = info["file_ending"]
    image_dir, label_dir = dataset_dir / f"images{split}", dataset_dir / f"labels{split}"
    cases = sorted(p.name[: -len(f"_{channel:04d}{ending}")]
                   for p in image_dir.glob(f"*_{channel:04d}{ending}"))
    prints, shapes = [], []
    for case_id in cases:
        prints.append(fingerprint(image_dir / f"{case_id}_{channel:04d}{ending}"))
        label = cv2.imread(str(label_dir / f"{case_id}{ending}"), cv2.IMREAD_GRAYSCALE)
        shapes.append(None if label is None else label.shape)
    if not cases or any(f is None for f in prints):
        return [], cases
    similarity = np.stack(prints) @ np.stack(prints).T / np.prod(FINGERPRINT)
    np.fill_diagonal(similarity, -1.0)
    best = similarity.argmax(axis=1)
    pairs, paired = [], set()
    for i, j in enumerate(best):
        if i in paired or best[j] != i or similarity[i, j] < SAME_IMAGE:
            continue
        if shapes[i] is None or shapes[i] != shapes[j]:
            continue
        pairs.append((image_name((cases[i], cases[j])), (cases[i], cases[j])))
        paired |= {i, j}
    return pairs, [case for index, case in enumerate(cases) if index not in paired]


def measure(dataset_dir, split):
    """Dice and MASD between the two annotations of every paired image."""
    pairs, unpaired = interrater_pairs(dataset_dir, split)
    classes = num_classes(dataset_dir)
    ending = load_dataset_json(dataset_dir)["file_ending"]
    label_dir = dataset_dir / f"labels{split}"
    rows = []
    for name, (first, second) in pairs:
        a = cv2.imread(str(label_dir / f"{first}{ending}"), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(label_dir / f"{second}{ending}"), cv2.IMREAD_GRAYSCALE)
        # Dice and MASD are both symmetric, so which annotation is passed first does not matter.
        dice, masd = compute_metrics(a, b, classes)
        rows.append((sorted((annotator(first), annotator(second))), name, dice, masd))
    return rows, unpaired


FIELDS = ("annotator_a", "annotator_b", "image", "dice", "masd")


def write(rows, unpaired, path):
    """One row per paired image, with whatever was left unpaired recorded alongside."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)
        for annotators, image, dice, masd in rows:
            writer.writerow([*annotators, image, f"{dice:.6f}", f"{masd:.6f}"])
        for case in unpaired:
            writer.writerow(["", "", case, "", ""])


def read(path):
    """(rows, unpaired) as `measure` returned them, or ([], []) where nothing was measured."""
    if not Path(path).exists():
        return [], []
    rows, unpaired = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["dice"]:
                rows.append((
                    [row["annotator_a"], row["annotator_b"]],
                    row["image"],
                    float(row["dice"]),
                    float(row["masd"]),
                ))
            else:
                unpaired.append(row["image"])
    return rows, unpaired


def splits(dataset_dir):
    """The label splits of a dataset that hold the same image drawn twice."""
    found = sorted(p.name[len("labels"):] for p in dataset_dir.glob("labels*interrater*"))
    if not found and "interrater" in dataset_dir.name:
        found = sorted(p.name[len("labels"):] for p in dataset_dir.glob("labels*"))
    return found


def path_for(agreement_dir, dataset, split):
    return Path(agreement_dir) / f"{dataset}{split or '_Tr'}.csv"
