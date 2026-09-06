"""Native-resolution, overlapping polyline targets; no on-disk patch cache."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils


def rle_encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def rle_decode(rle):
    return mask_utils.decode(rle).astype(bool)


def rasterize(points, shape):
    mask = np.zeros(shape, np.uint8)
    p = np.round(points).astype(np.int32)
    if len(p) >= 2:
        cv2.polylines(mask, [p], False, 1, 2)
    elif len(p) and 0 <= p[0, 0] < shape[1] and 0 <= p[0, 1] < shape[0]:
        mask[p[0, 1], p[0, 0]] = 1
    return mask.astype(bool)


def length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def clip_runs(points, size):
    """Liang–Barsky segment clipping; never join runs across an outside excursion."""
    p = np.asarray(points, float)
    if len(p) == 1:
        return [p] if ((p >= 0) & (p <= size - 1)).all() else []
    runs, run = [], []
    for a, b in zip(p[:-1], p[1:]):
        d = b - a
        lo, hi = 0., 1.
        for axis in range(2):
            if d[axis] == 0:
                if not 0 <= a[axis] <= size - 1:
                    hi = -1
            else:
                t = sorted(((0 - a[axis]) / d[axis], (size - 1 - a[axis]) / d[axis]))
                lo, hi = max(lo, t[0]), min(hi, t[1])
        if lo <= hi:
            first, last = a + lo * d, a + hi * d
            if run and not np.allclose(run[-1], first, atol=1e-7):
                runs.append(np.array(run)); run = []
            if not run:
                run.append(first)
            run.append(last)
            if hi < 1:
                runs.append(np.array(run)); run = []
        elif run:
            runs.append(np.array(run)); run = []
    if run:
        runs.append(np.array(run))
    return runs


def affine(left, top, size, angle=0., hflip=False, vflip=False):
    matrix = np.eye(3)
    matrix[:2, 2] = [-left, -top]
    rot = np.eye(3)
    rot[:2] = cv2.getRotationMatrix2D(((size-1)/2, (size-1)/2), angle, 1.)
    flip = np.diag([-1. if hflip else 1., -1. if vflip else 1., 1.])
    flip[:2, 2] = [size-1 if hflip else 0, size-1 if vflip else 0]
    return (flip @ rot @ matrix)[:2]


def patch_targets(fibers, matrix, size=256, threshold=16, queries=128):
    masks, ids, lines = [], [], []
    ignored = np.zeros((size, size), bool)
    for fiber in fibers:
        p = np.asarray(fiber['points'], float) @ matrix[:, :2].T + matrix[:, 2]
        if (p.max(axis=0) < -2).any() or (p.min(axis=0) > size+1).any():
            continue
        contained = bool(((p >= 0) & (p <= size-1)).all())
        runs = clip_runs(p, size)
        # Preserve stroke pixels from a centerline just outside the crop as ignored.
        complete = rasterize(p, (size, size))
        covered = np.zeros_like(complete)
        for run in runs:
            mask = rasterize(run, (size, size))
            covered |= mask
            if not contained and length(run) < threshold:
                ignored |= mask
            elif mask.any():
                masks.append(mask); ids.append(fiber['id']); lines.append(run)
        ignored |= complete & ~covered
    if len(masks) > queries:
        raise ValueError(f'Query budget overflow: {len(masks)} local instances > {queries}; increase model.num_queries')
    masks = np.stack(masks) if masks else np.zeros((0, size, size), bool)
    ignored &= ~masks.any(axis=0)
    return dict(masks=masks, ignore=ignored, source_ids=ids, lines=lines)


def red_image(gray):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError('Expected Dataset301 uint8 single-channel SMI')
    rgb = np.zeros((*gray.shape, 3), np.uint8)
    rgb[..., 0] = gray
    return rgb


def normalize(rgb):
    from fm_adaptation.models import DINOv3Encoder
    x = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float() / 255
    return (x - torch.tensor(DINOv3Encoder.mean)[:, None, None]) / torch.tensor(DINOv3Encoder.std)[:, None, None]


def positions(dimension, size=256, stride=128):
    return sorted(set(range(0, max(1, dimension-size+1), stride)) | {max(0, dimension-size)})


class InstancePatches(Dataset):
    def __init__(self, cfg, subset='train'):
        self.cfg, self.subset, self.epoch = cfg, subset, 0
        self.root = cfg.instance_data_dir
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        folds = json.loads((self.root / 'splits_final.json').read_text())
        self.cases = folds[int(cfg.fold)][subset]
        self.records = {c: json.loads((self.root / 'annotations' / (c+'.json')).read_text()) for c in self.cases}
        self.boxes = {}
        for case, rec in self.records.items():
            for f in rec['fibers']:
                f['points'] = np.asarray(f['points'], float)
            self.boxes[case] = np.array([np.r_[f['points'].min(0), f['points'].max(0)] for f in rec['fibers']])
        self.grid = [(c, x, y) for c in self.cases
                     for y in positions(self.records[c]['height'], cfg.input_size)
                     for x in positions(self.records[c]['width'], cfg.input_size)] if subset != 'train' else None
        self._cached_case = None
        self._fallback_positions = {}

    def __len__(self):
        return len(self.grid) if self.grid is not None else len(self.cases)*self.cfg.patching.patches_per_case

    def __getitem__(self, index):
        cfg = self.cfg
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, self.epoch, index]))
        if self.grid is None:
            case = self.cases[index // cfg.patching.patches_per_case]
            rec = self.records[case]
            left = int(rng.integers(max(1, rec['width']-cfg.input_size+1)))
            top = int(rng.integers(max(1, rec['height']-cfg.input_size+1)))
        else:
            case, left, top = self.grid[index]
            rec = self.records[case]
        if case != self._cached_case:
            self._cached_image = cv2.imread(str(self.root / rec['image']), cv2.IMREAD_UNCHANGED)
            self._cached_case = case
        if self.grid is None and cfg.patching.min_roi_fraction > 0:
            for attempt in range(64):
                crop = self._cached_image[top:top+cfg.input_size, left:left+cfg.input_size]
                if np.mean(crop > cfg.patching.roi_threshold) >= cfg.patching.min_roi_fraction:
                    break
                left = int(rng.integers(max(1, rec['width']-cfg.input_size+1)))
                top = int(rng.integers(max(1, rec['height']-cfg.input_size+1)))
            else:
                if case not in self._fallback_positions:
                    integral = cv2.integral((self._cached_image > cfg.patching.roi_threshold).astype(np.uint8))
                    candidates = []
                    for y in positions(rec['height'], cfg.input_size):
                        for x in positions(rec['width'], cfg.input_size):
                            bottom, right = min(y+cfg.input_size, rec['height']), min(x+cfg.input_size, rec['width'])
                            count = integral[bottom,right]-integral[y,right]-integral[bottom,x]+integral[y,x]
                            if count / ((bottom-y)*(right-x)) >= cfg.patching.min_roi_fraction:
                                candidates.append((x,y))
                    if not candidates:
                        raise ValueError(f'No imaged patch meets min_roi_fraction for {case}')
                    self._fallback_positions[case] = candidates
                candidates = self._fallback_positions[case]
                left, top = candidates[int(rng.integers(len(candidates)))]
        aug = cfg.augment if self.subset == 'train' else None
        matrix = affine(left, top, cfg.input_size,
                        rng.uniform(-aug.rotation, aug.rotation) if aug else 0,
                        bool(aug and aug.hflip and rng.random() < aug.flip_p),
                        bool(aug and aug.vflip and rng.random() < aug.flip_p))
        gray = cv2.warpAffine(self._cached_image, matrix, (cfg.input_size, cfg.input_size))
        inverse = cv2.invertAffineTransform(matrix)
        corners = np.array([[-2,-2],[-2,cfg.input_size+2],[cfg.input_size+2,-2],[cfg.input_size+2,cfg.input_size+2]])
        original = corners @ inverse[:, :2].T + inverse[:, 2]
        boxes = self.boxes[case]
        hits = np.flatnonzero(((boxes[:, 2:] >= original.min(0)) & (boxes[:, :2] <= original.max(0))).all(1))
        target = patch_targets([rec['fibers'][i] for i in hits], matrix, cfg.input_size, self.manifest['fragment_threshold'], cfg.num_queries)
        # Padding introduced by rotation or slide boundaries carries no supervision.
        valid = cv2.warpAffine(np.ones(self._cached_image.shape, np.uint8), matrix,
                               (cfg.input_size, cfg.input_size), flags=cv2.INTER_NEAREST).astype(bool)
        target['ignore'] |= ~valid & ~target['masks'].any(axis=0)
        target.update(case=case, left=left, top=top)
        return normalize(red_image(gray)), target


def collate(batch):
    images, targets = zip(*batch)
    return torch.stack(images), list(targets)


def prepare(source, raw, output):
    from .vendor.yvonne_rois import index_source, _display_id, _mask_stem, load_neurite_coordinates
    images, _, archives = index_source(raw)
    lookup = {'Yvonne_b2__'+_mask_stem(_display_id(images[key].name), rater): path
              for (key, rater), path in archives.items()}
    output.mkdir(parents=True, exist_ok=True)
    (output/'annotations').mkdir(exist_ok=True)
    folds = json.loads((source/'splits_final.json').read_text())
    train_lengths, all_records, audit = [], [], []
    for split in ('Tr', 'Ts'):
        (output/f'images{split}').mkdir(exist_ok=True)
        for image in sorted((source/f'images{split}').glob('*_0000.png')):
            case = image.name[:-9]
            gray = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
            label = cv2.imread(str(source/f'labels{split}'/(case+'.png')), 0) > 0
            union = np.zeros(gray.shape, bool)
            fibers = []
            for i, points in enumerate(load_neurite_coordinates(lookup[case])):
                if not len(points) or not np.isfinite(points).all():
                    raise ValueError(f'Empty/invalid source fiber {case}:{i}')
                lo = np.maximum(0, np.floor(points.min(axis=0)).astype(int)-2)
                hi = np.minimum(np.array(gray.shape[::-1]), np.ceil(points.max(axis=0)).astype(int)+3)
                x, y = lo; w, h = hi-lo
                local = rasterize(np.round(points)-lo, (h, w))
                union[y:y+h, x:x+w] |= local
                fibers.append(dict(id=f'{case}:{i:05d}', points=points.tolist(), bbox=[int(x),int(y),int(w),int(h)],
                                   offset=[int(x),int(y)], rle=rle_encode(local), length=length(points)))
            mismatch = int(np.count_nonzero(union != label))
            if mismatch:
                raise ValueError(f'{case}: instance union differs from Dataset301 at {mismatch} pixels')
            dst = output/f'images{split}'/image.name
            if not dst.is_symlink():
                dst.symlink_to(os.path.relpath(image, dst.parent))
            if not dst.exists() or not np.array_equal(cv2.imread(str(dst), -1), gray):
                raise ValueError(f'Image symlink verification failed: {dst}')
            rec = dict(case=case, image=str(dst.relative_to(output)), height=gray.shape[0], width=gray.shape[1], fibers=fibers)
            (output/'annotations'/(case+'.json')).write_text(json.dumps(rec))
            all_records.append(rec)
            if case in folds[0]['train']:
                train_lengths.extend(f['length'] for f in fibers)
            audit.append(dict(case=case, fibers=len(fibers), union_mismatch=mismatch))
    threshold = math.ceil(float(np.percentile(train_lengths, 5)))
    if threshold != 16:
        raise ValueError(f'Unexpected training p5 threshold {threshold}, expected 16')
    (output/'splits_final.json').write_bytes((source/'splits_final.json').read_bytes())
    manifest = dict(fragment_threshold=threshold, categories=[dict(id=1, name='neurite')],
                    split_sha256=hashlib.sha256((source/'splits_final.json').read_bytes()).hexdigest(), audit=audit)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    from fm_adaptation.instance_metrics import global_rle
    from pycocotools import mask as mu
    coco_images, annotations = [], []
    for image_id, rec in enumerate(all_records, 1):
        shape = (rec['height'], rec['width'])
        coco_images.append(dict(id=image_id, file_name=rec['image'], height=shape[0], width=shape[1], case=rec['case']))
        for fiber in rec['fibers']:
            rle = global_rle(fiber['rle'], fiber['offset'], shape)
            annotations.append(dict(id=len(annotations)+1, image_id=image_id, category_id=1, iscrowd=0,
                                    segmentation=rle, area=float(mu.area(rle)), bbox=mu.toBbox(rle).tolist(), source_fiber_id=fiber['id']))
    (output/'instances.coco.json').write_text(json.dumps(dict(images=coco_images, annotations=annotations, categories=manifest['categories'])))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('data/instances_yvonne_b2'))
    a = p.parse_args()
    prepare(a.source, a.raw, a.output)
