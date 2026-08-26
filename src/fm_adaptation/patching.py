"""Patchwise training and inference for images too large to resize to the encoder input.

Whole-slide images are read lazily through memory maps, so a patch never costs more than the pixels it
covers. Which patches are worth taking is decided once per case from a coarse block-occupancy map of the
region of interest (everything above ``roi_threshold``); the masked-out surround is never sampled.
"""

from pathlib import Path

from .datasets import dataset_dir as resolve_dataset_dir

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data import _case_ids, load_dataset_json

BLOCKS_PER_PATCH = 8


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


class CaseIndex:
    """Paths, shape and per-block ROI occupancy for one case."""

    def __init__(self, dataset_dir: Path, split: str, case_id: str, ending: str, block: int):
        self.case_id = case_id
        self.image_path = dataset_dir / f"images{split}" / f"{case_id}_0000{ending}"
        self.label_path = dataset_dir / f"labels{split}" / f"{case_id}{ending}"
        self.block = block
        self.shape: tuple[int, int] = (0, 0)
        self.occupancy: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self._image = None
        self._label = None

    def build(self, roi_threshold: int) -> None:
        image = open_image(self.image_path)
        self.shape = (int(image.shape[0]), int(image.shape[1]))
        rows = self.shape[0] // self.block
        cols = self.shape[1] // self.block
        roi = np.asarray(image[: rows * self.block, : cols * self.block]) > roi_threshold
        self.occupancy = roi.reshape(rows, self.block, cols, self.block).mean(axis=(1, 3)).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, occupancy=self.occupancy, shape=np.array(self.shape), block=self.block)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        data = np.load(path)
        if int(data["block"]) != self.block:
            return False
        self.occupancy = data["occupancy"]
        self.shape = tuple(int(v) for v in data["shape"])
        return True

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = open_image(self.image_path)
        return self._image

    @property
    def label(self) -> np.ndarray:
        if self._label is None:
            self._label = open_image(self.label_path)
        return self._label

    def crop(self, y: int, x: int, patch_size: int):
        image = np.asarray(self.image[y : y + patch_size, x : x + patch_size])
        label = np.asarray(self.label[y : y + patch_size, x : x + patch_size])
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


def build_index(raw_dir, dataset_name, split, fold, subset, patch_cfg, cache_dir) -> list[CaseIndex]:
    """One `CaseIndex` per case, with the ROI occupancy map cached to disk on first use."""
    dataset_dir = resolve_dataset_dir(raw_dir, dataset_name)
    ending = load_dataset_json(dataset_dir)["file_ending"]
    block = max(1, patch_cfg.patch_size // BLOCKS_PER_PATCH)
    cache_dir = Path(cache_dir)
    cases = [
        CaseIndex(dataset_dir, split, case_id, ending, block)
        for case_id in _case_ids(dataset_dir, split, fold, subset)
    ]
    pending = [case for case in cases if not case.load(cache_dir / f"{case.case_id}.npz")]
    for case in tqdm(pending, desc=f"indexing {dataset_name} ROI", leave=False):
        case.build(patch_cfg.roi_threshold)
        case.save(cache_dir / f"{case.case_id}.npz")
    return cases


def patch_loader(cfg, preprocess, subset, shuffle=False):
    """Random patches for training, a deterministic overlapping grid for validation."""
    from torch.utils.data import DataLoader

    from .data import collate_cases

    cases = build_index(
        cfg.raw_data_dir,
        cfg.train_dataset,
        "Tr",
        cfg.fold,
        subset,
        cfg.patching,
        cfg.patch_cache_dir(cfg.train_dataset),
    )
    if subset == "train":
        dataset = RandomPatchDataset(cases, cfg.patching, preprocess)
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


def _prepare(case, y, x, patch_cfg, preprocess, roi_threshold):
    image, label = case.crop(y, x, patch_cfg.patch_size)
    if patch_cfg.ignore_masked_out:
        label = np.where(image > roi_threshold, label, -1).astype(np.int16)
    image_t, label_t, geometry = preprocess(_to_rgb(image), label)
    return image_t, label_t, {"case_id": case.case_id, "y": y, "x": x, **geometry}


class RandomPatchDataset(Dataset):
    """`patches_per_case` random ROI patches per case per epoch, uniform over valid anchors."""

    def __init__(self, cases, patch_cfg, preprocess):
        self.cases = cases
        self.patch_cfg = patch_cfg
        self.preprocess = preprocess
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
        return _prepare(case, y, x, self.patch_cfg, self.preprocess, self.patch_cfg.roi_threshold)


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
            image, _ = case.crop(int(y), int(x), patch_cfg.patch_size)
            image_t, _, geometry = preprocess(_to_rgb(image), empty)
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
