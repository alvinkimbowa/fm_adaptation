"""Dataset identity.

A dataset is identified by its number. The rest of the directory name is a label that describes what
is in it, and it is read from the raw data directory whenever one is needed, so selections and stored
paths stay valid however the label reads.

Anything that turns a dataset into a path goes through `dataset_dir`; anything that names one for a
person to read goes through `resolve`.
"""

import re
from functools import cache
from pathlib import Path

DATASET_DIR = re.compile(r"Dataset(\d+)(?:_(.*))?$")


def dataset_id(value):
    """The number identifying a dataset, from a bare id, a directory name, or a path.

    `217`, `Dataset217` and `Dataset217_lesion_MY_smi_gfap` all name the same dataset. Leading zeros
    are dropped so the three-digit directory form and the number a script carries agree.
    """
    text = Path(str(value)).name
    match = DATASET_DIR.match(text)
    if match:
        return str(int(match.group(1)))
    if text.isdigit():
        return str(int(text))
    return None


@cache
def _by_id(raw_data_dir):
    """Maps each id under a raw data directory to the directory holding it."""
    found = {}
    for path in sorted(Path(raw_data_dir).glob("Dataset*")):
        if not path.is_dir():
            continue
        identifier = dataset_id(path.name)
        if identifier is None:
            continue
        if identifier in found:
            raise RuntimeError(
                f"Dataset {identifier} is under {raw_data_dir} twice, as "
                f"{found[identifier]} and {path.name}"
            )
        found[identifier] = path.name
    return found


def resolve(raw_data_dir, value):
    """The directory name a dataset has under `raw_data_dir`.

    Falls back to whatever was asked for when the id is not there to look up, which is what lets a
    run whose data has moved on still be read, listed and tabulated from what it stored beside it.
    """
    identifier = dataset_id(value)
    if identifier is None or raw_data_dir is None:
        return str(value)
    return _by_id(Path(raw_data_dir)).get(identifier, str(value))


def dataset_dir(raw_data_dir, value):
    return Path(raw_data_dir) / resolve(raw_data_dir, value)


def split_cases(dataset_dir, split):
    """Case ids in one split of a dataset, read off the image filenames.

    An evaluation set need not ship a `splits_final.json`, so the files are the only thing that
    always answers which cases a split holds.
    """
    image_dir = Path(dataset_dir) / f"images{split}"
    if not image_dir.is_dir():
        return set()
    return {path.name.rsplit("_", 1)[0] for path in image_dir.glob("*_0000.*")}
