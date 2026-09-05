"""Workbook-backed annotator identities, including legacy and augmented case IDs."""

import json
import re
from functools import lru_cache
from pathlib import Path


ANNOTATORS = {"Yvonne": ("Coco", "Yvonne", "Tanya"), "Yvonne_b2": ("Queena", "Sarah")}
SCALE_SUFFIX = re.compile(r"_scale\d+$")
STAIN_PANEL = re.compile(r"[-\s]*GFAP\s+SERT\d*\s+SMI\d*\s+PDGFRa[-\s]*", re.I)


def is_target_dataset(dataset):
    name = dataset.lower()
    return "neurite" in name and "yvonne" in name and "in_vitro" not in name


def image_id(name):
    """The dataset preprocessor's stain removal and filename sanitization."""
    name = re.sub(r"_(?:SMI|GFAP)\.tif$", "", name, flags=re.I)
    name = STAIN_PANEL.sub("-", name)
    return re.sub(r"[\s_-]+", "-", name.replace("#", "")).strip("-")


@lru_cache(maxsize=1)
def mapping():
    return json.loads((Path(__file__).parent / "assets/neurite_annotators.json").read_text())


def canonical_annotator(name):
    return "Coco" if name.casefold() in ("coco", "cocco") else name


def identity(case_id):
    """Return (source, annotator, original image), or None for unrelated sources.

    Unprefixed historical cases are matched against both workbooks, never inferred from a
    dataset's name (Dataset203 actually contains the second batch).
    """
    base = SCALE_SUFFIX.sub("", str(case_id))
    source, sep, name = base.partition("__")
    if sep and source not in ANNOTATORS:
        return None
    name = name if sep else base
    rater = re.search(r"_rater\d+$", name)
    suffix = rater.group() if rater else ""
    stem = image_id(name[:rater.start()] if rater else name)
    key = stem + suffix
    sources = [source] if sep else list(ANNOTATORS)
    found = [(s, canonical_annotator(mapping()[s]["cases"][key]), stem)
             for s in sources if key in mapping()[s]["cases"]]
    if len(found) != 1:
        raise ValueError(f"Unmapped or ambiguous neurite annotation {case_id!r}; "
                         "update assets/neurite_annotators.json from the tracing workbooks")
    return found[0]
