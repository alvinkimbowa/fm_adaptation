import json
import math
from pathlib import Path

from .datasets import dataset_dir

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


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


def trained_planes(channel_names: dict, stains=()) -> frozenset[int] | None:
    """The planes a model actually saw, which is not always what its training set declares.

    `Dataset218_lesion_eric_smi_gfap` declares SMI and GFAP but ships a blank SMI file for every
    case, so a model trained on it has only ever seen GFAP. Naming the stains in the config keeps
    such a run from being handed real SMI by an evaluation set that has it.
    """
    if not stains:
        return active_planes(channel_names)
    planes = set()
    for stain in (str(name).upper() for name in stains):
        if stain in STAIN_PLANES:
            planes.add(STAIN_PLANES[stain])
        elif stain in PLANE_LETTERS:
            planes.add(PLANE_LETTERS[stain])
        else:
            raise ValueError(f"unknown stain: {stain}")
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


def _inscribed(width, height, degrees):
    """Sides of the largest axis-aligned rectangle inside a `width` x `height` rectangle turned by
    `degrees` -- the region that is still real image in every row and every column."""
    angle = math.radians(abs(degrees) % 180)
    if angle > math.pi / 2:
        angle = math.pi - angle
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    long_side, short_side = max(width, height), min(width, height)
    if short_side <= 2 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
        half = 0.5 * short_side
        w, h = (half / sin_a, half / cos_a) if width >= height else (half / cos_a, half / sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        w = (width * cos_a - height * sin_a) / cos_2a
        h = (height * cos_a - width * sin_a) / cos_2a
    return max(int(w), 1), max(int(h), 1)


def _label_box(mask):
    """(top, bottom, left, right) of the annotation, or None where there is none to keep in frame."""
    rows = torch.any(mask > 0, dim=1).nonzero()
    if not len(rows):
        return None
    columns = torch.any(mask > 0, dim=0).nonzero()
    return int(rows[0]), int(rows[-1]), int(columns[0]), int(columns[-1])


def _centred(size, keep):
    start = (size - keep) // 2
    return start, start + keep


def _holds(box, top, bottom, left, right):
    return box is None or (top <= box[0] and box[1] < bottom and left <= box[2] and box[3] < right)


def _augment(image, mask, geometry, cfg, fill):
    """Flip, rotate and zoom one preprocessed case, introducing nothing that was never imaged.

    This runs on the square canvas the encoder sees rather than on the image on disk: these sections
    reach 4598x15031, and warping one at full resolution once per epoch would cost more than the
    training step it feeds.

    A rotation pulls area from outside the image into the frame. Rather than fill it, the turned case
    is cropped back to the rows and columns that are real image all the way across -- the largest
    rectangle inside the turned section -- and then refitted to the canvas the way `_resize_and_pad`
    fits an image in the first place. The letterbox therefore stays upright and no invented detail
    ever reaches the model.

    That crop costs real tissue on a section four times taller than it is wide, which is why the
    configs draw from a narrow angle: at 10 degrees such a section keeps 37% of itself, at 25 it
    keeps 16% and the lesion no longer fits in what is left.

    The annotation must survive both the crop and any zoom past the canvas. A candidate that would
    cut it is halved back toward the identity and retried; four failures leave the case with its
    flips alone.
    """
    if cfg.hflip and torch.rand(()) < cfg.flip_p:
        image, mask = torch.flip(image, [-1]), torch.flip(mask, [-1])
    if cfg.vflip and torch.rand(()) < cfg.flip_p:
        image, mask = torch.flip(image, [-2]), torch.flip(mask, [-2])

    angle = float(torch.empty(()).uniform_(-cfg.rotation, cfg.rotation)) if cfg.rotation else 0.0
    scale = (
        float(torch.empty(()).uniform_(cfg.zoom_min, cfg.zoom_max))
        if (cfg.zoom_min, cfg.zoom_max) != (1.0, 1.0)
        else 1.0
    )
    for _ in range(4):
        if (angle, scale) == (0.0, 1.0):
            return image, mask
        turned = _turn(image, mask, geometry, angle, scale, fill)
        if turned is not None:
            return turned
        angle, scale = angle / 2, 1.0 + (scale - 1.0) / 2
    return image, mask


def _turn(image, mask, geometry, angle, scale, fill):
    """One candidate rotation and zoom, or None where it would cut the annotation."""
    size = image.shape[-1]
    turned_image = TF.affine(
        image, angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0],
        interpolation=TF.InterpolationMode.BILINEAR, fill=list(fill),
    )
    turned_mask = TF.affine(
        mask.unsqueeze(0), angle=angle, translate=[0, 0], scale=1.0, shear=[0.0, 0.0],
        interpolation=TF.InterpolationMode.NEAREST, fill=[-1.0],
    ).squeeze(0)

    # Back to what is real image in every row and column. The section sits centred on the canvas, so
    # the rectangle to keep is centred too.
    keep_w, keep_h = _inscribed(geometry["resized_width"], geometry["resized_height"], angle)
    # A few pixels off each side. The section is centred to within half a pixel and the rotation
    # turns about the canvas centre, so without this slack the odd corner keeps a sliver of padding.
    keep_w, keep_h = max(keep_w - 8, 1), max(keep_h - 8, 1)
    top, bottom = _centred(size, min(keep_h, size))
    left, right = _centred(size, min(keep_w, size))
    if not _holds(_label_box(turned_mask), top, bottom, left, right):
        return None
    turned_image = turned_image[:, top:bottom, left:right]
    turned_mask = turned_mask[top:bottom, left:right]

    # Refit, aspect preserved, at the drawn zoom: the same fit `_resize_and_pad` performs, on what is
    # left of the section rather than on the whole of it.
    height, width = turned_mask.shape
    factor = scale * size / max(height, width)
    new_h, new_w = max(round(height * factor), 1), max(round(width * factor), 1)
    turned_image = F.interpolate(
        turned_image.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False,
    ).squeeze(0)
    turned_mask = F.interpolate(
        turned_mask[None, None].float(), size=(new_h, new_w), mode="nearest",
    ).squeeze(0).squeeze(0).long()

    # A zoom past the canvas is a centre crop, which the annotation also has to survive.
    if new_h > size or new_w > size:
        top, bottom = _centred(new_h, min(new_h, size))
        left, right = _centred(new_w, min(new_w, size))
        if not _holds(_label_box(turned_mask), top, bottom, left, right):
            return None
        turned_image = turned_image[:, top:bottom, left:right]
        turned_mask = turned_mask[top:bottom, left:right]
        new_h, new_w = turned_mask.shape

    pad_top, pad_left = (size - new_h) // 2, (size - new_w) // 2
    canvas = torch.empty((image.shape[0], size, size), dtype=turned_image.dtype)
    for channel, value in enumerate(fill):
        canvas[channel] = value
    canvas[:, pad_top:pad_top + new_h, pad_left:pad_left + new_w] = turned_image
    label = torch.full((size, size), -1, dtype=torch.long)
    label[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = turned_mask
    return canvas, label


class NnUNet2DDataset(Dataset):
    def __init__(self, raw_dir, dataset_name, split, fold, subset, preprocess,
                 channel_dropout=(), channel_dropout_p=0.5, keep_planes=None,
                 require_labels=True, augment=None):
        self.dataset_dir = dataset_dir(raw_dir, dataset_name)
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
        # Geometric augmentation, applied to the training subset alone; `None` is every run that
        # existed before it was an option.
        self.augment = augment
        # What black becomes once this encoder has normalised it, so a rotation fills its corners with
        # the value the resize already pads with rather than one no image ever contains.
        self.fill = (
            preprocess(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8))[0][:, 0, 0].tolist()
            if augment is not None
            else None
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
        if self.augment is not None and self.subset == "train":
            # `geometry` says where the section sits on the canvas, which is what the crop needs.
            image_t, mask_t = _augment(image_t, mask_t, geometry, self.augment, self.fill)
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
