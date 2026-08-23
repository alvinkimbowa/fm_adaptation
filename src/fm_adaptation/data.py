import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


def load_dataset_json(dataset_dir: Path) -> dict:
    with open(dataset_dir / "dataset.json") as f:
        return json.load(f)


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
    def __init__(self, raw_dir, dataset_name, split, fold, subset, preprocess):
        self.dataset_dir = Path(raw_dir) / dataset_name
        self.split = split
        self.preprocess = preprocess
        info = load_dataset_json(self.dataset_dir)
        self.ending = info["file_ending"]
        if self.ending not in {".png", ".tif", ".tiff"}:
            raise ValueError(f"Only 2D PNG/TIFF datasets are supported, got {self.ending}")
        self.ids = _case_ids(self.dataset_dir, split, fold, subset)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        case_id = self.ids[index]
        image_path = self.dataset_dir / f"images{self.split}" / f"{case_id}_0000{self.ending}"
        label_path = self.dataset_dir / f"labels{self.split}" / f"{case_id}{self.ending}"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        if mask is None:
            raise FileNotFoundError(label_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_t, mask_t, geometry = self.preprocess(image, mask)
        return image_t, mask_t, {"case_id": case_id, **geometry}


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


class TeacherLogitDataset(Dataset):
    """A dataset's cases with the teacher's cached logits attached.

    `NnUNet2DDataset` applies no augmentation, so the teacher's output for a case is the same in every
    epoch and is read from disk rather than recomputed. That is what keeps the teacher out of GPU memory
    for the whole of student training.
    """

    def __init__(self, base: Dataset, cache_dir):
        self.base = base
        self.cache_dir = Path(cache_dir)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, mask, metadata = self.base[index]
        cached = torch.load(
            self.cache_dir / f"{metadata['case_id']}.pt", map_location="cpu", weights_only=True
        )
        return image, mask, cached["logits"], metadata


def _warp(tensor, theta, mode, fill=0.0):
    """Resample `tensor` through `theta`, filling anything pulled in from outside with `fill`.

    `grid_sample` can only pad with zeros, so the fill value is subtracted first and added back after.
    `theta` is in normalised coordinates, which is what lets the same transform apply unchanged to the
    896-pixel image and the teacher's 224-pixel logits.
    """
    grid = F.affine_grid(theta[None], (1, *tensor.shape), align_corners=False)
    warped = F.grid_sample(
        (tensor[None] - fill), grid, mode=mode, padding_mode="zeros", align_corners=False
    )
    return warped[0] + fill


class SpatialAugmentDataset(Dataset):
    """Random flip, rotation and scale, applied identically to the image, the label and the teacher.

    Spatial only, and deliberately so: the teacher's cached logits are a spatial map, so the same
    transform that moves the image moves them, and one cached pass stays valid for every epoch.
    Intensity augmentation would not -- it changes what the teacher would have predicted, so it would
    mean either rerunning the teacher live or accepting a target that no longer matches its input.

    Anything rotated in from outside the canvas is labelled -1, the same ignore value `_resize_and_pad`
    already uses for its padding, so neither loss term sees invented pixels.
    """

    def __init__(self, base: Dataset, rotation_degrees: float = 180.0,
                 scale_range: tuple[float, float] = (0.7, 1.4), flip: bool = True):
        self.base = base
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.flip = flip

    def __len__(self):
        return len(self.base)

    def _theta(self):
        angle = math.radians(float(torch.empty(()).uniform_(-self.rotation_degrees, self.rotation_degrees)))
        scale = float(torch.empty(()).uniform_(*self.scale_range))
        cos, sin = math.cos(angle), math.sin(angle)
        # `affine_grid` maps output coordinates back into the input, so this is the inverse transform:
        # dividing by `scale` makes the object larger, and negating the first column mirrors it.
        theta = torch.tensor(
            [[cos / scale, -sin / scale, 0.0], [sin / scale, cos / scale, 0.0]], dtype=torch.float32
        )
        if self.flip and torch.rand(()) < 0.5:
            theta[:, 0] *= -1
        return theta

    def __getitem__(self, index):
        item = self.base[index]
        image, mask = item[0], item[1]
        theta = self._theta()
        image = _warp(image, theta, "bilinear")
        # The label is warped as a float so `grid_sample` accepts it, with nearest sampling so no class
        # index is ever interpolated into existence.
        mask = _warp(mask[None].float(), theta, "nearest", fill=-1.0)[0].round().long()
        if len(item) == 4:
            return image, mask, _warp(item[2].float(), theta, "bilinear").half(), item[3]
        return image, mask, item[2]


def collate_cases(batch):
    images, masks, metadata = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(metadata)


def collate_teacher_cases(batch):
    images, masks, teacher, metadata = zip(*batch)
    return torch.stack(images), torch.stack(masks), torch.stack(teacher), list(metadata)
