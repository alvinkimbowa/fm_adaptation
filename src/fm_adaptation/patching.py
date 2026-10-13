"""Patchwise training and inference for images too large to resize to the encoder input.

Whole-slide images are read lazily through memory maps, so a patch never costs more than the pixels it
covers. Which patches are worth taking is decided once per case from a coarse block-occupancy map of the
region of interest (everything above ``roi_threshold``); the masked-out surround is never sampled.

Training patches are augmented by cutting a larger patch than the one asked for, turning that, and
taking the requested size out of its middle -- see ``_source_size``. A patch already cut to its final
size cannot be turned without fill arriving in its corners, and the margin a slide offers around a
patch is free, so this is the one place where a rotation costs no tissue at all.
"""

import math
from collections import OrderedDict
from pathlib import Path

from .datasets import dataset_dir as resolve_dataset_dir, split_cases
from .slabs import slab_members

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import _case_ids, load_dataset_json, stain_planes, trained_planes

BLOCKS_PER_PATCH = 8
# Decoded pixels one process may keep between patches; see `_PlaneCache`.
DECODED_PLANE_BUDGET = 2 * 1024**3


def open_image(path: Path) -> np.ndarray:
    """Memory-mapped when the format allows it, eagerly decoded otherwise."""
    if path.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        try:
            return tifffile.memmap(path, mode="r")
        except (ValueError, MemoryError):
            return tifffile.imread(path)
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


class _PlaneCache:
    """Recently used planes of one process, decoded ones held to a byte budget.

    A format that cannot be memory-mapped is decoded whole, and every dataloader worker ends up
    touching every case, so planes kept without limit cost the decoded size of the dataset once per
    worker. Past the budget the least recently used plane is dropped and decoded again when next
    asked for. Memory maps hold no pixels and are never dropped.
    """

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.held_bytes = 0
        self.planes: OrderedDict = OrderedDict()

    def get(self, path: Path, gray: bool = False) -> np.ndarray:
        key = (path, gray)
        plane = self.planes.get(key)
        if plane is not None:
            self.planes.move_to_end(key)
            return plane
        plane = open_image(path)
        if gray and plane.ndim != 2:
            plane = cv2.cvtColor(np.asarray(plane), cv2.COLOR_RGB2GRAY)
        self.planes[key] = plane
        self.held_bytes += self._cost(plane)
        while self.held_bytes > self.budget_bytes and len(self.planes) > 1:
            _, dropped = self.planes.popitem(last=False)
            self.held_bytes -= self._cost(dropped)
        return plane

    @staticmethod
    def _cost(plane: np.ndarray) -> int:
        return 0 if isinstance(plane, np.memmap) else plane.nbytes


_planes = _PlaneCache(DECODED_PLANE_BUDGET)


def foreground_statistics(plane) -> tuple[float, float] | None:
    """Mean and standard deviation of a plane's non-zero pixels; None where it has none to vary."""
    values = np.asarray(plane, dtype=np.float32)
    foreground = values[values != 0]
    if foreground.size == 0:
        return None
    std = float(foreground.std())
    return (float(foreground.mean()), std) if std > 0 else None


def zscore(plane, statistics) -> np.ndarray:
    """A plane standardised by `foreground_statistics`, zeros included; all zero without any."""
    values = np.asarray(plane, dtype=np.float32)
    if statistics is None:
        return np.zeros_like(values)
    mean, std = statistics
    return (values - mean) / std


def bell_weights(annotated: np.ndarray, width: float, floor: float) -> np.ndarray:
    """Loss weights of a boolean annotation, in the 8 bits a stored weight map has.

    255 on the annotation. Beside it a bell curve of the distance to the nearest annotated pixel,
    `width` pixels wide, that starts at 255 on the pixels touching the annotation and never drops
    below `floor`.
    """
    if not annotated.any():
        return np.full(annotated.shape, round(255 * floor), dtype=np.uint8)
    distance = cv2.distanceTransform(
        (~annotated).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    halo = np.exp(-((np.maximum(distance, 1) - 1) ** 2) / (2 * width**2))
    weights = np.where(annotated, 1.0, np.maximum(halo, floor))
    return np.round(255 * weights).astype(np.uint8)


def bell_reach(width: float, floor: float) -> int:
    """Distance in pixels at which `bell_weights` has come down to its floor."""
    return math.ceil(1 + width * math.sqrt(2 * math.log(1 / floor)))


class CaseIndex:
    """Paths, shape and per-block ROI occupancy for one case.

    `slab` names the cases read together with this one, itself included: every crop is then the
    per-pixel maximum over them. `weight_bell`, a `(width, floor)` pair, computes the loss weights
    from the label with `bell_weights` instead of reading a stored map.
    """

    def __init__(self, dataset_dir: Path, split: str, case_id: str, ending: str, block: int,
                 channel_layout=None, weight_maps: bool = False, zscore: bool = False,
                 slab=None, weight_bell=None):
        self.case_id = case_id
        self.slab = tuple(slab) if slab else (case_id,)
        self.weight_bell = weight_bell
        # Under z-scoring the layout's second entry is a channel's place in the patch, not an RGB
        # plane, and the patch has as many channels as the layout names.
        self.zscore = zscore
        self._statistics = None
        image_dir = dataset_dir / f"images{split}"
        self.channel_layout = None if channel_layout is None else tuple(sorted(channel_layout))
        self.layout_key = (
            "grayscale" if self.channel_layout is None
            else "channels:" + ",".join(f"{stored}>{rgb}" for stored, rgb in self.channel_layout)
        )
        self.image_paths = (
            ((None, image_dir / f"{case_id}_0000{ending}"),)
            if self.channel_layout is None
            else tuple((rgb, image_dir / f"{case_id}_{stored:04d}{ending}")
                       for stored, rgb in self.channel_layout)
        )
        self.label_path = dataset_dir / f"labels{split}" / f"{case_id}{ending}"
        # One folder holds the maps of every split, named as the labels are.
        self.weight_path = dataset_dir / "weight_maps" / f"{case_id}{ending}" if weight_maps else None
        self.block = block
        self.shape: tuple[int, int] = (0, 0)
        self.occupancy: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    @property
    def images(self):
        loaded = tuple((rgb, _planes.get(path, gray=rgb is not None)) for rgb, path in self.image_paths)
        shapes = {tuple(image.shape[:2]) for _, image in loaded}
        if len(shapes) != 1:
            raise ValueError(f"stain planes have different shapes for {self.case_id}")
        return loaded

    def build(self, roi_threshold: int) -> None:
        image = self.images[0][1]
        self.shape = (int(image.shape[0]), int(image.shape[1]))
        rows = self.shape[0] // self.block
        cols = self.shape[1] // self.block
        roi = np.zeros((rows * self.block, cols * self.block), dtype=bool)
        for _, plane in self.images:
            occupied = np.asarray(plane[: rows * self.block, : cols * self.block]) > roi_threshold
            if occupied.ndim == 3:
                occupied = occupied.any(axis=2)
            roi |= occupied
        self.occupancy = roi.reshape(rows, self.block, cols, self.block).mean(axis=(1, 3)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, occupancy=self.occupancy, shape=np.array(self.shape), block=self.block,
                            layout=np.array(self.layout_key))

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path)
        if int(data["block"]) != self.block:
            return False
        cached_layout = str(data["layout"]) if "layout" in data.files else "grayscale"
        if cached_layout != self.layout_key:
            return False
        self.occupancy = data["occupancy"]
        self.shape = tuple(int(v) for v in data["shape"])
        return True

    @property
    def statistics(self):
        """`foreground_statistics` of each whole image plane, in the order of `image_paths`."""
        if self._statistics is None:
            self._statistics = tuple(foreground_statistics(plane) for _, plane in self.images)
        return self._statistics

    def standardise(self, patch: np.ndarray) -> np.ndarray:
        """A raw patch from `crop` with each channel z-scored by its whole image's statistics."""
        if patch.ndim == 2:
            patch = patch[..., None]
        return np.stack(
            [zscore(patch[..., slot], stats)
             for (slot, _), stats in zip(self.image_paths, self.statistics)],
            axis=-1,
        )

    @property
    def label(self) -> np.ndarray:
        return _planes.get(self.label_path)

    def _projected(self, path: Path, window, gray: bool = False) -> np.ndarray:
        """A window of one of this case's files, as the maximum over the same file of its slab."""
        crop = None
        for member in self.slab:
            member_path = path.with_name(path.name.replace(self.case_id, member, 1))
            part = np.asarray(_planes.get(member_path, gray=gray)[window])
            crop = part if crop is None else np.maximum(crop, part)
        return crop

    def crop_weight(self, y: int, x: int, patch_size: int) -> np.ndarray:
        """The loss weights over the window `crop` cuts: 8 bits spanning 0 to 1."""
        if self.weight_bell is None:
            return self._projected(self.weight_path, (slice(y, y + patch_size), slice(x, x + patch_size)))
        # The label is read wider than the window by the reach of the bell curve, so an annotation
        # just outside the window still weights the pixels inside it.
        reach = bell_reach(*self.weight_bell)
        top, left = max(y - reach, 0), max(x - reach, 0)
        label = self._projected(
            self.label_path,
            (slice(top, y + patch_size + reach), slice(left, x + patch_size + reach)),
        )
        weights = bell_weights(label > 0, *self.weight_bell)
        return weights[y - top : y - top + patch_size, x - left : x - left + patch_size]

    def crop(self, y: int, x: int, patch_size: int, require_label: bool = True):
        window = (slice(y, y + patch_size), slice(x, x + patch_size))
        if self.channel_layout is None:
            image = _to_rgb(self._projected(self.image_paths[0][1], window))
        else:
            first = self.images[0][1]
            channels = len(self.image_paths) if self.zscore else 3
            image = np.zeros((min(patch_size, first.shape[0] - y),
                              min(patch_size, first.shape[1] - x), channels), dtype=first.dtype)
            for rgb, path in self.image_paths:
                image[..., rgb] = self._projected(path, window, gray=True)
        label = self._projected(self.label_path, window) if require_label else None
        return image, label

    def roi_fractions(self, patch_size: int) -> np.ndarray:
        """ROI fraction of the patch anchored at each block position, as a summed-area lookup."""
        span = -(-patch_size // self.block)
        rows = self.shape[0] // self.block - span + 1
        cols = self.shape[1] // self.block - span + 1
        if rows <= 0 or cols <= 0:
            return np.zeros((0, 0), dtype=np.float32)
        table = np.pad(self.occupancy.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        window = (
            table[span:, span:] - table[:-span, span:] - table[span:, :-span] + table[:-span, :-span]
        ) / (span * span)
        return window[:rows, :cols].astype(np.float32)

    def anchors(self, patch_size: int, min_roi_fraction: float) -> np.ndarray:
        """Block-aligned (y, x) pixel anchors whose patch clears the ROI threshold."""
        fractions = self.roi_fractions(patch_size)
        rows, cols = np.nonzero(fractions >= min_roi_fraction)
        anchors = np.stack([rows, cols], axis=1) * self.block
        limit = np.array([self.shape[0] - patch_size, self.shape[1] - patch_size])
        return np.clip(anchors, 0, np.maximum(limit, 0))

    def roi_bounds(self, patch_size: int) -> tuple[int, int, int, int]:
        """Pixel bounding box of the occupied blocks, widened so a full patch always fits inside it."""
        rows, cols = np.nonzero(self.occupancy > 0)
        if len(rows) == 0:
            return 0, min(patch_size, self.shape[0]), 0, min(patch_size, self.shape[1])
        y0, y1 = int(rows.min()) * self.block, (int(rows.max()) + 1) * self.block
        x0, x1 = int(cols.min()) * self.block, (int(cols.max()) + 1) * self.block
        if y1 - y0 < patch_size:
            y0 = max(0, min(y0, self.shape[0] - patch_size))
            y1 = min(self.shape[0], y0 + patch_size)
        if x1 - x0 < patch_size:
            x0 = max(0, min(x0, self.shape[1] - patch_size))
            x1 = min(self.shape[1], x0 + patch_size)
        return y0, min(y1, self.shape[0]), x0, min(x1, self.shape[1])


def build_index(raw_dir, dataset_name, split, fold, subset, patch_cfg, cache_dir,
                keep_planes=None, weight_maps=False, weight_bell=None) -> list[CaseIndex]:
    """One `CaseIndex` per case, with the ROI occupancy map cached to disk on first use.

    With `patch_cfg.slab_slices` above 1 each case reads its neighbouring slices too. The ROI
    occupancy stays that of the case's own slice.
    """
    dataset_dir = resolve_dataset_dir(raw_dir, dataset_name)
    info = load_dataset_json(dataset_dir)
    ending = info["file_ending"]
    layout = None
    standardised = patch_cfg.normalization == "zscore"
    if standardised:
        layout = tuple(
            (stored, slot) for slot, stored in enumerate(sorted(int(key) for key in info["channel_names"]))
        )
    elif patch_cfg.respect_channels:
        declared = stain_planes(info["channel_names"])
        if declared is not None:
            layout = tuple(
                mapping for mapping in declared.values()
                if keep_planes is None or mapping[1] in keep_planes
            )
            if not layout:
                raise ValueError(f"{dataset_name} has no stain in the planes this model was trained on")
    block = max(1, patch_cfg.patch_size // BLOCKS_PER_PATCH)
    cache_dir = Path(cache_dir)
    case_ids = _case_ids(dataset_dir, split, fold, subset)
    slabs = {}
    if patch_cfg.slab_slices > 1:
        if standardised:
            raise ValueError("patching.slab_slices is not supported with zscore normalization")
        # Neighbours are looked up among every case the split stores, not only this subset's.
        slabs = slab_members(sorted(split_cases(dataset_dir, split) | set(case_ids)),
                             patch_cfg.slab_slices)
    cases = [
        CaseIndex(dataset_dir, split, case_id, ending, block, layout, weight_maps, standardised,
                  slabs.get(case_id), weight_bell)
        for case_id in case_ids
    ]
    if weight_maps:
        missing = [case.case_id for case in cases if not case.weight_path.exists()]
        if missing:
            raise FileNotFoundError(
                f"{dataset_name} has no weight map for {len(missing)} of {len(cases)} cases, "
                f"first {missing[0]}"
            )
    pending = [case for case in cases if not case.load(cache_dir / f"{case.case_id}.npz")]
    for case in tqdm(pending, desc=f"indexing {dataset_name} ROI", leave=False):
        case.build(patch_cfg.roi_threshold)
        case.save(cache_dir / f"{case.case_id}.npz")
    return cases


def patch_loader(cfg, preprocess, subset, shuffle=False):
    """Random patches for training, a deterministic overlapping grid for validation.

    Only the training subset is augmented: validation has to measure the same patches every epoch."""
    from torch.utils.data import DataLoader

    from .data import collate_cases

    keep_planes = None
    if cfg.patching.respect_channels and cfg.stains:
        keep_planes = trained_planes(
            load_dataset_json(resolve_dataset_dir(cfg.raw_data_dir, cfg.train_dataset))["channel_names"],
            cfg.stains,
        )
    cases = build_index(
        cfg.raw_data_dir,
        cfg.train_dataset,
        "Tr",
        cfg.fold,
        subset,
        cfg.patching,
        cfg.patch_cache_dir(cfg.train_dataset),
        keep_planes,
        cfg.weight_maps and not cfg.weight_maps_from_labels,
        (cfg.weight_map_width, cfg.weight_map_floor)
        if cfg.weight_maps and cfg.weight_maps_from_labels else None,
    )
    if cfg.patching.slab_slices > 1 and all(len(case.slab) == 1 for case in cases):
        raise ValueError(
            f"patching.slab_slices is {cfg.patching.slab_slices} but no case of {cfg.train_dataset} "
            "is a slice of a stack"
        )
    if subset == "train":
        dataset = RandomPatchDataset(cases, cfg.patching, preprocess, cfg.augment)
    else:
        dataset = GridPatchDataset(cases, cfg.patching, preprocess)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_cases,
        pin_memory=True,
    )


def _to_rgb(patch: np.ndarray) -> np.ndarray:
    if patch.ndim == 2:
        return cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)


def _source_size(patch_size, augment):
    """Side to cut before augmenting, so the patch that survives the turn is all real image.

    A square of side `S` turned by `a` holds a centred, axis-aligned square of side
    `S / (|cos a| + |sin a|)`, worst at 45 degrees where the factor is sqrt(2). Zooming out past 1.0
    pulls in more still, by the same ratio.
    """
    if augment is None:
        return patch_size
    turning = augment.rotation or augment.zoom_min < 1.0
    if not turning:
        return patch_size
    angle = math.radians(min(abs(augment.rotation), 45.0))
    margin = (math.cos(angle) + math.sin(angle)) / min(1.0, augment.zoom_min)
    return int(math.ceil(patch_size * margin))


def _centre_crop(array, size):
    top = (array.shape[0] - size) // 2
    left = (array.shape[1] - size) // 2
    return array[top : top + size, left : left + size]


def _augment_patch(image, label, patch_size, augment, rng, weight=None):
    """Flip and turn an oversized patch, then take `patch_size` out of its middle.

    A `weight` map goes through the same flips and turn and comes back as a third value.

    The centre square is inside the turned source by construction, so `warpAffine`'s border value
    never reaches the output and the model is never shown a corner that was not imaged. Where the
    case was too small to widen the crop, the angle is drawn from what the margin actually supports
    rather than from the configured range.
    """
    if augment.hflip and rng.random() < augment.flip_p:
        image, label = image[:, ::-1], label[:, ::-1]
        weight = None if weight is None else weight[:, ::-1]
    if augment.vflip and rng.random() < augment.flip_p:
        image, label = image[::-1], label[::-1]
        weight = None if weight is None else weight[::-1]
    limit = _supported_rotation(min(image.shape[:2]), patch_size, augment)
    zooming = (augment.zoom_min, augment.zoom_max) != (1.0, 1.0)
    if limit > 0 or zooming:
        angle = rng.uniform(-limit, limit)
        scale = rng.uniform(augment.zoom_min, augment.zoom_max) if zooming else 1.0
        centre = ((image.shape[1] - 1) / 2, (image.shape[0] - 1) / 2)
        matrix = cv2.getRotationMatrix2D(centre, angle, scale)
        size = (image.shape[1], image.shape[0])
        image = cv2.warpAffine(np.ascontiguousarray(image), matrix, size, flags=cv2.INTER_LINEAR)
        label = cv2.warpAffine(np.ascontiguousarray(label), matrix, size, flags=cv2.INTER_NEAREST)
        if weight is not None:
            weight = cv2.warpAffine(np.ascontiguousarray(weight), matrix, size, flags=cv2.INTER_LINEAR)
    image, label = _centre_crop(image, patch_size), _centre_crop(label, patch_size)
    if weight is None:
        return image, label
    return image, label, _centre_crop(weight, patch_size)


def _fit_weight(weight, geometry, shape):
    """A cropped weight map laid over the label tensor `preprocess` made of the same crop."""
    size = (geometry["resized_width"], geometry["resized_height"])
    weight = cv2.resize(np.ascontiguousarray(weight), size, interpolation=cv2.INTER_LINEAR)
    fitted = np.zeros(tuple(shape), dtype=np.float32)
    top, left = geometry["pad_top"], geometry["pad_left"]
    fitted[top : top + size[1], left : left + size[0]] = weight.astype(np.float32) / 255.0
    return torch.from_numpy(fitted)


def _supported_rotation(source, patch_size, augment):
    """The widest angle the margin actually available allows, capped at what was configured."""
    if not augment.rotation or source <= patch_size:
        return 0.0
    # Invert `_source_size` for the angle: cos a + sin a = sqrt(2) sin(a + 45).
    ratio = min(source / patch_size, math.sqrt(2.0))
    return min(augment.rotation, math.degrees(math.asin(ratio / math.sqrt(2.0))) - 45.0)


def _prepare(case, y, x, patch_cfg, preprocess, roi_threshold, augment=None, rng=None):
    size = patch_cfg.patch_size
    if augment is not None:
        # Widen the crop around the anchor's centre, clamped into the case; the anchor itself is
        # still drawn against `patch_size`, so `min_roi_fraction` keeps its meaning.
        source = min(_source_size(size, augment), *case.shape[:2])
        y = int(np.clip(y - (source - size) // 2, 0, max(case.shape[0] - source, 0)))
        x = int(np.clip(x - (source - size) // 2, 0, max(case.shape[1] - source, 0)))
        size = source
    image, label = case.crop(y, x, size)
    weighted = case.weight_path is not None or getattr(case, "weight_bell", None) is not None
    weight = case.crop_weight(y, x, size) if weighted else None
    if augment is not None:
        image, label, *turned = _augment_patch(image, label, patch_cfg.patch_size, augment, rng, weight)
        weight = turned[0] if turned else None
    if patch_cfg.ignore_masked_out:
        occupied = image > roi_threshold
        if occupied.ndim == 3:
            occupied = occupied.any(axis=2)
        label = np.where(occupied, label, -1).astype(np.int16)
    if getattr(case, "zscore", False):
        image = case.standardise(image)
    elif image.ndim == 2:
        image = _to_rgb(image)
    image_t, label_t, geometry = preprocess(image, label)
    meta = {"case_id": case.case_id, "y": y, "x": x, **geometry}
    if weight is not None:
        meta["weight"] = _fit_weight(weight, geometry, label_t.shape)
    return image_t, label_t, meta


class RandomPatchDataset(Dataset):
    """`patches_per_case` random ROI patches per case per epoch, uniform over valid anchors."""

    def __init__(self, cases, patch_cfg, preprocess, augment=None):
        self.cases = cases
        self.patch_cfg = patch_cfg
        self.preprocess = preprocess
        self.augment = augment
        self.anchors = [case.anchors(patch_cfg.patch_size, patch_cfg.min_roi_fraction) for case in cases]
        self.usable = [index for index, anchors in enumerate(self.anchors) if len(anchors)]
        if not self.usable:
            raise ValueError("No patch anchors cleared min_roi_fraction; lower it or check roi_threshold")

    def __len__(self):
        return len(self.usable) * self.patch_cfg.patches_per_case

    def __getitem__(self, index):
        case = self.cases[self.usable[index // self.patch_cfg.patches_per_case]]
        anchors = self.anchors[self.usable[index // self.patch_cfg.patches_per_case]]
        # Drawn from torch's RNG so patches differ every epoch yet stay reproducible under manual_seed,
        # with or without dataloader workers.
        rng = np.random.default_rng(int(torch.randint(0, 2**62, (1,)).item()))
        y, x = anchors[rng.integers(len(anchors))]
        # Jitter within the block so sampling is not locked to the occupancy lattice.
        y = int(np.clip(y + rng.integers(case.block), 0, case.shape[0] - self.patch_cfg.patch_size))
        x = int(np.clip(x + rng.integers(case.block), 0, case.shape[1] - self.patch_cfg.patch_size))
        return _prepare(case, y, x, self.patch_cfg, self.preprocess, self.patch_cfg.roi_threshold,
                        self.augment, rng)


class GridPatchDataset(Dataset):
    """Deterministic overlapping grid over the ROI, used for validation."""

    def __init__(self, cases, patch_cfg, preprocess):
        self.cases = cases
        self.patch_cfg = patch_cfg
        self.preprocess = preprocess
        self.items = [
            (case, int(y), int(x))
            for case in cases
            for y, x in grid_anchors(case, patch_cfg)
        ]
        if not self.items:
            raise ValueError("No grid patches cleared min_roi_fraction; lower it or check roi_threshold")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        case, y, x = self.items[index]
        return _prepare(case, y, x, self.patch_cfg, self.preprocess, self.patch_cfg.roi_threshold)


def grid_anchors(case: CaseIndex, patch_cfg) -> np.ndarray:
    """Anchors on a `stride` lattice covering the ROI, with the last row/column clamped inside."""
    stride = max(case.block, round(patch_cfg.stride / case.block) * case.block)
    valid = set(map(tuple, case.anchors(patch_cfg.patch_size, patch_cfg.min_roi_fraction).tolist()))
    if not valid:
        return np.zeros((0, 2), dtype=int)
    y0, y1, x0, x1 = case.roi_bounds(patch_cfg.patch_size)
    anchors = []
    for y in _lattice(y0, y1, stride, patch_cfg.patch_size, case.shape[0]):
        for x in _lattice(x0, x1, stride, patch_cfg.patch_size, case.shape[1]):
            snapped = (y - y % case.block, x - x % case.block)
            if snapped in valid:
                anchors.append((y, x))
    return np.array(anchors, dtype=int) if anchors else np.zeros((0, 2), dtype=int)


def _lattice(start, stop, stride, patch_size, limit):
    positions = list(range(start, max(start + 1, stop - patch_size + 1), stride))
    last = min(stop, limit) - patch_size
    if last > positions[-1]:
        positions.append(last)
    return [max(0, min(p, limit - patch_size)) for p in positions]


def _importance_map(patch_size: int) -> np.ndarray:
    """Gaussian weighting so patch centres dominate the seams, as nnU-Net does."""
    axis = cv2.getGaussianKernel(patch_size, patch_size / 8.0).astype(np.float32)
    weights = axis @ axis.T
    return weights / weights.max()


def _restore_probabilities(probabilities: np.ndarray, geometry: dict, patch_size: int) -> np.ndarray:
    """Undo the encoder's resize-and-pad so probabilities line up with the raw patch."""
    top, left = geometry["pad_top"], geometry["pad_left"]
    height, width = geometry["resized_height"], geometry["resized_width"]
    probabilities = probabilities[:, top : top + height, left : left + width]
    if (height, width) == (patch_size, patch_size):
        return probabilities
    return np.stack(
        [cv2.resize(channel, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR) for channel in probabilities]
    )


def predict_case(model, case, patch_cfg, classes, device, amp, batch_size, preprocess):
    """Overlapping-patch inference aggregated back to the full image; outside the ROI stays background."""
    anchors = grid_anchors(case, patch_cfg)
    prediction = np.zeros(case.shape, dtype=np.uint8)
    if len(anchors) == 0:
        return prediction
    y0, y1, x0, x1 = case.roi_bounds(patch_cfg.patch_size)
    accumulator = np.zeros((classes, y1 - y0, x1 - x0), dtype=np.float32)
    weights = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    importance = _importance_map(patch_cfg.patch_size)
    empty = np.zeros((patch_cfg.patch_size, patch_cfg.patch_size), dtype=np.uint8)

    for start in range(0, len(anchors), batch_size):
        batch = anchors[start : start + batch_size]
        tensors, geometries = [], []
        for y, x in batch:
            image, _ = case.crop(int(y), int(x), patch_cfg.patch_size, require_label=False)
            if case.zscore:
                image = case.standardise(image)
            image_t, _, geometry = preprocess(image, empty)
            tensors.append(image_t)
            geometries.append(geometry)
        with torch.no_grad(), amp:
            logits = model(torch.stack(tensors).to(device))
        probabilities = logits.float().softmax(1).cpu().numpy()
        for (y, x), geometry, probability in zip(batch, geometries, probabilities):
            probability = _restore_probabilities(probability, geometry, patch_cfg.patch_size)
            ys, xs = int(y) - y0, int(x) - x0
            window = (slice(ys, ys + patch_cfg.patch_size), slice(xs, xs + patch_cfg.patch_size))
            accumulator[(slice(None), *window)] += probability * importance
            weights[window] += importance

    covered = weights > 0
    accumulator[:, ~covered] = 0
    prediction[y0:y1, x0:x1] = np.where(covered, accumulator.argmax(0), 0).astype(np.uint8)
    return prediction
