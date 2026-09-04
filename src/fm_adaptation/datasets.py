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
    if raw_data_dir is None:
        return str(value)
    # An exact directory name is already unambiguous. Accept it without indexing every sibling:
    # some raw-data roots legitimately carry two revisions under another dataset number, and that
    # must not prevent a fully named, unrelated dataset from being used.
    exact = Path(raw_data_dir) / str(value)
    if exact.is_dir():
        return exact.name
    identifier = dataset_id(value)
    if identifier is None:
        return str(value)
    return _by_id(Path(raw_data_dir)).get(identifier, str(value))


def dataset_dir(raw_data_dir, value):
    return Path(raw_data_dir) / resolve(raw_data_dir, value)


def dataset_root(raw_data_dirs, value):
    """Which of several raw data directories holds a dataset.

    A number identifies a dataset everywhere, so two roots naming it differently means two different
    datasets wearing one number, and every path built from either would be a guess -- that is an
    error, the same one `_by_id` raises for a single root holding a number twice. The same name under
    several roots is one dataset mirrored, which is how a project reads another's data, and the first
    root wins. A number no root knows falls back to the last, so what a caller reports still names a
    real place rather than nothing.
    """
    roots = [Path(directory) for directory in raw_data_dirs]
    if not roots:
        raise ValueError(f"no raw data directory to resolve {value} against")
    found = {}
    for root in roots:
        try:
            name = resolve(root, value)
        except RuntimeError:
            # A root carrying some other number twice cannot be searched by number at all. That says
            # nothing about the dataset being looked for, so the remaining roots still answer, and a
            # fully named dataset is found in it either way -- `resolve` takes an exact directory
            # name without indexing its siblings.
            continue
        if (root / name).is_dir():
            found.setdefault(name, root)
    if len(found) > 1:
        places = ", ".join(f"{name} under {root}" for name, root in found.items())
        raise RuntimeError(f"Dataset {dataset_id(value)} is two datasets: {places}")
    return next(iter(found.values()), roots[-1])


def split_cases(dataset_dir, split):
    """Case ids in one split of a dataset, read off the image filenames.

    An evaluation set need not ship a `splits_final.json`, so the files are the only thing that
    always answers which cases a split holds.
    """
    image_dir = Path(dataset_dir) / f"images{split}"
    if not image_dir.is_dir():
        return set()
    return {path.name.rsplit("_", 1)[0] for path in image_dir.glob("*_0000.*")}


# The family is what a dataset name's second token says the task is. Sets that name the same task
# differently are pulled together here, and a task the names do not spell out is written in words.
FAMILY_ALIASES = {
    "neurite": "neurites",
    "Clarius": "Knee Cartilage",
    "Sonix-Touch": "Knee Cartilage",
    "GE": "Knee Cartilage",
}


def family(dataset):
    """The task a dataset belongs to, which is what groups it with others in a table or a directory."""
    name = dataset.split("_", maxsplit=2)[1]
    return FAMILY_ALIASES.get(name, name)
