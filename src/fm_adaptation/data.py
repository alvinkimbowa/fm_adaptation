import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


STAIN_PLANES = {"SMI": 0, "GFAP": 2}
# The older datasets name their channel by the plane it occupies rather than by the stain: the czi_B
# lesion sets declare `B` and store signal in blue alone, the neurite sets declare `R`.
PLANE_LETTERS = {"R": 0, "G": 1, "B": 2}


def load_dataset_json(dataset_dir: Path) -> dict:
    with open(dataset_dir / "dataset.json") as f:
        return json.load(f)


def stain_planes(channel_names: dict) -> dict[str, tuple[int, int]] | None:
    """Map known stains to ``(stored channel, RGB plane)``.

    Returning ``None`` for any ordinary/unknown channel declaration is deliberate: those datasets
    must keep using their existing colour-image path rather than having an RGB plane interpreted as
    a standalone greyscale file.
    """
    if not channel_names:
        return None
    names = {str(name).upper(): int(index) for index, name in channel_names.items()}
    if any(name not in STAIN_PLANES for name in names):
        return None
    return {name: (stored, STAIN_PLANES[name]) for name, stored in names.items()}


def active_planes(channel_names: dict) -> frozenset[int] | None:
    """RGB planes a dataset's images can carry signal in, or None when that cannot be known.

    This is what a model was trained to look at. A run trained on `czi_B` has only ever seen blue, so
    handing it a two-stain image would put signal in a plane it has never seen; `predict.py` uses this
    to keep such a model on the stains it knows. `None` -- an ultrasound set, say, whose single grey
    channel is replicated across all three -- means no restriction rather than no planes.
    """
    if not channel_names:
        return None
    planes = set()
    for name in channel_names.values():
        name = str(name).upper()
        if name in STAIN_PLANES:
            planes.add(STAIN_PLANES[name])
        elif name in PLANE_LETTERS:
            planes.add(PLANE_LETTERS[name])
        else:
            return None
    return frozenset(planes)


def rgb_planes(channel_names: dict) -> dict[int, int] | None:
    """`stain_planes` keyed the way a reader wants it: {stored channel: RGB plane}, or None.

    The figures need to know which file goes into which plane, not which stain is which; keeping the
    conversion here means the loader and the figures cannot drift into two different answers.
    """
    planes = stain_planes(channel_names)
    return None if planes is None else {stored: rgb for stored, rgb in planes.values()}


def num_classes(dataset_dir: Path) -> int:
    labels = load_dataset_json(dataset_dir)["labels"]
    values = labels.values()
    return max(int(v) for v in values) + 1


def _case_ids(dataset_dir: Path, split: str, fold: str, subset: str) -> list[str]:
    if split == "Tr" and subset in {"train", "val"}:
        with open(dataset_dir / "splits_final.json") as f:
            folds = json.load(f)
        if fold == "all":
            if subset == "val":
                raise ValueError("fold=all has no validation split")
            return sorted({case for item in folds for case in item["train"] + item["val"]})
        key = "train" if subset == "train" else "val"
        return list(folds[int(fold)][key])

    ending = load_dataset_json(dataset_dir)["file_ending"]
    image_dir = dataset_dir / f"images{split}"
    suffix = f"_0000{ending}"
    return sorted(p.name[: -len(suffix)] for p in image_dir.glob(f"*{suffix}"))


class NnUNet2DDataset(Dataset):
    def __init__(self, raw_dir, dataset_name, split, fold, subset, preprocess,
                 channel_dropout=(), channel_dropout_p=0.5, keep_planes=None,
                 require_labels=True):
        self.dataset_dir = Path(raw_dir) / dataset_name
        self.split = split
        self.subset = subset
        self.preprocess = preprocess
        info = load_dataset_json(self.dataset_dir)
        self.ending = info["file_ending"]
        if self.ending not in {".png", ".tif", ".tiff"}:
            raise ValueError(f"Only 2D PNG/TIFF datasets are supported, got {self.ending}")
        self.channel_names = info["channel_names"]
        self.stain_planes = stain_planes(self.channel_names)
        # Which RGB planes this dataset may fill. `None` fills every plane the dataset declares; a set
        # keeps a model to the planes it was trained on, so a czi_B model reads GFAP and not SMI.
        self.keep_planes = None if keep_planes is None else frozenset(keep_planes)
        if self.stain_planes and self.keep_planes is not None:
            self.stain_planes = {
                stain: mapping for stain, mapping in self.stain_planes.items()
                if mapping[1] in self.keep_planes
            }
            if not self.stain_planes:
                raise ValueError(
                    f"{dataset_name} has no stain in the planes this model was trained on"
                )
        # A dataset that ships images without labels can still be predicted; it just cannot be scored.
        self.require_labels = require_labels
        self.ids = _case_ids(self.dataset_dir, split, fold, subset)
        self.channel_dropout = tuple(str(name).upper() for name in channel_dropout)
        self.channel_dropout_p = float(channel_dropout_p)
        if not 0.0 <= self.channel_dropout_p <= 1.0:
            raise ValueError("channel_dropout_p must be between 0 and 1")
        if self.channel_dropout:
            declared = set(self.stain_planes or {})
            unknown = sorted(set(self.channel_dropout) - declared)
            if unknown:
                raise ValueError(
                    f"channel_dropout contains stains not declared by {dataset_name}: {unknown}"
                )
        self.present_stains = {}
        if self.channel_dropout and subset == "train":
            image_dir = self.dataset_dir / f"images{self.split}"
            for case_id in self.ids:
                present = set()
                for stain, (stored, _) in self.stain_planes.items():
                    path = image_dir / f"{case_id}_{stored:04d}{self.ending}"
                    probe = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_8)
                    if probe is None:
                        raise FileNotFoundError(path)
                    if probe.any():
                        present.add(stain)
                self.present_stains[case_id] = present

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        case_id = self.ids[index]
        image_dir = self.dataset_dir / f"images{self.split}"
        label_path = self.dataset_dir / f"labels{self.split}" / f"{case_id}{self.ending}"
        if self.stain_planes is None:
            image_path = image_dir / f"{case_id}_0000{self.ending}"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        else:
            present = self.present_stains.get(case_id)
            dropped = None
            candidates = [] if present is None else [
                stain for stain in self.channel_dropout if stain in present
            ]
            if (
                present is not None
                and len(present) >= 2
                and candidates
                and torch.rand(()) < self.channel_dropout_p
            ):
                dropped = candidates[int(torch.randint(len(candidates), ()).item())]
            image = None
            for stain, (stored, rgb_plane) in self.stain_planes.items():
                if stain == dropped or (present is not None and stain not in present):
                    continue
                image_path = image_dir / f"{case_id}_{stored:04d}{self.ending}"
                plane = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if plane is None:
                    raise FileNotFoundError(image_path)
                if image is None:
                    image = np.zeros((*plane.shape, 3), dtype=plane.dtype)
                elif plane.shape != image.shape[:2]:
                    raise ValueError(f"stain planes have different shapes for {case_id}")
                image[..., rgb_plane] = plane
            if image is None:
                # All declared stains were known absent. Use one file only to recover shape/dtype.
                stored = next(iter(self.stain_planes.values()))[0]
                image_path = image_dir / f"{case_id}_{stored:04d}{self.ending}"
                plane = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if plane is None:
                    raise FileNotFoundError(image_path)
                image = np.zeros((*plane.shape, 3), dtype=plane.dtype)
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        has_label = mask is not None
        if not has_label:
            if self.require_labels:
                raise FileNotFoundError(label_path)
            # Geometry is derived from the pair, so the placeholder has to be the image's own size.
            mask = np.zeros(np.asarray(image).shape[:2], dtype=np.uint8)
        if self.stain_planes is None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_t, mask_t, geometry = self.preprocess(image, mask)
        return image_t, mask_t, {"case_id": case_id, "has_label": has_label, **geometry}


class CachedFeatureDataset(Dataset):
    def __init__(self, cache_dir, case_ids):
        self.cache_dir = Path(cache_dir)
        self.case_ids = list(case_ids)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        item = torch.load(
            self.cache_dir / f"{self.case_ids[index]}.pt",
            map_location="cpu",
            weights_only=True,
        )
        return item["feature"], item["mask"].long(), {"case_id": self.case_ids[index]}


def collate_cases(batch):
    images, masks, metadata = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(metadata)
