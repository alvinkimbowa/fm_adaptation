"""ROI helpers copied from spinal_cord_injury/src/datasets/preprocess_yvonne_dataset.py.
Copyright (c) 2025 Alvin Kimbowa. MIT license; see YVONNE_LICENSE.
Copied 2026-09-05. No runtime dependency on the source project.
"""
import re
from pathlib import Path
import cv2
import numpy as np
from roifile import roiread

CASE_ID_PATTERNS = (
    re.compile(r"rat\s*#?\s*(\d+).*?slide\s*(\d+).*?sect(?:ion)?\s*(\d+)", re.I),
    re.compile(r"cervical\s*#\s*(\d+).*?slide\s*(\d+)\s*section\s*(\d+)", re.I),
    re.compile(r"animal\s*(\d+)\s*slide\s*(\d+)\s*sect\s*(\d+)", re.I),
)
RATER_PATTERN = re.compile(r"(?:_rater(\d)|(?:_lesion|_neurites)-(\d))")

# The stain panel is the one part of the name that says nothing about which section this
# is, and it is written three ways (SERT/SERT5, SMI/SMI312, with and without a leading
# dash), so it comes out of the emitted id.
STAIN_PANEL_PATTERN = re.compile(r"[-\s]*GFAP\s+SERT\d*\s+SMI\d*\s+PDGFRa[-\s]*", re.I)
CHANNEL_SUFFIX_PATTERN = re.compile(r"(?:_rater\d+)?_(?:SMI|GFAP)\.tif$", re.I)


def _sanitize(text: str) -> str:
    """Collapse a raw filename fragment to a path-safe, hyphen-joined token."""
    return re.sub(r"[\s_-]+", "-", text.replace("#", "")).strip("-")


def _display_id(tif_name: str) -> str:
    """The emitted case id: the source stem, sanitized.

    Nothing is normalised away beyond the stain panel and the channel suffix, so the id
    traces straight back to the file it came from. That matters here because the four
    cohorts in this source are not interchangeable -- an earlier scheme rewrote them all
    to `Rat-N_slide-S_Section-X`, which renamed `animal2` and `cervical #12` to "Rat",
    dropped the IKVAV condition, and let `Gelma Nov 24 animal2 slide11 sect2` collide
    with `Cond Lesion GelMa rods-Rat 2 slide11 section 2` in the 2026 root.

        Cond Lesion GelMa rods-Rat 10 slide1 section 8-GFAP SERT SMI PDGFRa_SMI.tif
            -> Cond-Lesion-GelMa-rods-Rat-10-slide1-section-8
        GelMa cervical #12   GFAP SERT5 SMI312 PDGFRa  slide 11 section 1_SMI.tif
            -> GelMa-cervical-12-slide-11-section-1
    """
    stem = CHANNEL_SUFFIX_PATTERN.sub("", tif_name)
    return _sanitize(STAIN_PANEL_PATTERN.sub("-", stem))


def _join_key(name: str) -> str:
    """Internal key matching one section across the three folders.

    Carries the cohort -- everything before the subject number -- so that two cohorts
    that happen to share a subject/slide/section triple stay distinct.
    """
    for pattern in CASE_ID_PATTERNS:
        match = pattern.search(name)
        if match is not None:
            subject, slide, section = match.groups()
            cohort = _sanitize(STAIN_PANEL_PATTERN.sub("-", name[: match.start()]))
            return f"{cohort}|{subject}|{slide}|{section}"
    raise ValueError(f"Could not parse a subject/slide/section id from {name}")


def _rater(name: str) -> str | None:
    """Rater suffix, or None for a singly-annotated case.

    `_lesion-1.roi` and `_neurites-1.zip` are the same person as `_rater1_SMI.tif`;
    verified at Dice 1.0000 against the existing preprocessed masks, and 0.86-0.96
    when crossed.
    """
    match = RATER_PATTERN.search(name)
    if match is None:
        return None
    return f"rater{match.group(1) or match.group(2)}"


def _annotation_key(path: Path) -> tuple[str, str | None]:
    return _join_key(path.name), _rater(path.name)


def _mask_stem(case_id: str, rater: str | None) -> str:
    return case_id if rater is None else f"{case_id}_{rater}"


def index_source(input_root: Path) -> tuple[dict, dict, dict]:
    """Map join key -> SMI path, and (join key, rater) -> lesion/neurite annotation."""
    images: dict[str, Path] = {}
    for path in sorted((input_root / "TIF Files").glob("*_SMI.tif")):
        # The rater TIFs are pixel-identical, so any one of them stands in for the case.
        images.setdefault(_join_key(path.name), path)

    lesions: dict[tuple[str, str | None], Path] = {}
    for path in sorted((input_root / "Lesion Masks").glob("*.roi")):
        lesions[_annotation_key(path)] = path

    neurites: dict[tuple[str, str | None], Path] = {}
    for path in sorted((input_root / "Neurites").glob("*.zip")):
        neurites[_annotation_key(path)] = path

    orphans = sorted(
        {key for key in lesions if key[0] not in images}
        | {key for key in neurites if key[0] not in images}
    )
    if orphans:
        raise ValueError(f"Annotations with no matching SMI image: {orphans}")
    return images, lesions, neurites


def load_neurite_coordinates(path: Path, coordinate_scale: float = 1.0) -> list[np.ndarray]:
    """Load curated ROI polylines in image coordinates, before pixel rounding."""
    rois = roiread(path)
    rois = rois if isinstance(rois, list) else [rois]
    return [roi.coordinates() * coordinate_scale for roi in rois]


def _neurite_mask(
    shape: tuple[int, int],
    path: Path,
    stroke_width: int,
    coordinate_scale: float = 1.0,
) -> np.ndarray:
    """Rasterize the open polyline traces at a fixed stroke width. Returns uint8 {0, 1}."""
    mask = np.zeros(shape, dtype=np.uint8)
    for points in load_neurite_coordinates(path, coordinate_scale):
        coordinates = np.round(points).astype(np.int32)
        if len(coordinates) >= 2:
            cv2.polylines(mask, [coordinates], False, 1, stroke_width)
        elif len(coordinates) == 1:
            x, y = coordinates[0]
            if 0 <= y < shape[0] and 0 <= x < shape[1]:
                mask[y, x] = 1
    if not mask.any():
        raise ValueError(f"Neurite ROIs in {path.name} rasterized to an empty mask")
    return mask

