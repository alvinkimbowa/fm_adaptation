import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_dataset_json(dataset_dir: Path) -> dict:
    with open(dataset_dir / "dataset.json") as f:
        return json.load(f)


def _blue_channel(channel_names: dict) -> int | None:
    """Which stored channel to read on its own, or None to read the case the way every other one is.

    The czi_B lesion datasets declare a single channel `B` and store it as an RGB image whose red and
    green planes are zero, so a model trained on them has only ever seen signal in blue. A dataset that
    ships its stains as separate greyscale files instead -- GFAP alongside SMI -- has to be assembled
    the same way, or `cv2.IMREAD_COLOR` replicates one plane across all three and the model is handed
    something it was never trained on.
    """
    for index, name in channel_names.items():
        if name.upper() == "GFAP":
            return int(index)
    return None


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
        self.blue_channel = _blue_channel(info["channel_names"])
        self.ids = _case_ids(self.dataset_dir, split, fold, subset)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        case_id = self.ids[index]
        channel = 0 if self.blue_channel is None else self.blue_channel
        image_path = self.dataset_dir / f"images{self.split}" / f"{case_id}_{channel:04d}{self.ending}"
        label_path = self.dataset_dir / f"labels{self.split}" / f"{case_id}{self.ending}"
        if self.blue_channel is None:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        if mask is None:
            raise FileNotFoundError(label_path)
        if self.blue_channel is None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Blue last, matching the RGB order the conversion above produces.
            planes = np.zeros((*image.shape, 3), dtype=image.dtype)
            planes[..., 2] = image
            image = planes
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


def collate_cases(batch):
    images, masks, metadata = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(metadata)
