"""Synthetic nnU-Net datasets, so the tests never touch the real data directories."""

import json
from pathlib import Path

import cv2
import numpy as np
import torch


def preprocess(image, mask):
    """The identity preprocessing: tensors straight through, no resize or padding.

    The geometry a real encoder reports is still filled in -- it says the whole canvas is image --
    because the augmentation reads it to know where the section ends and the letterbox begins.
    """
    height, width = mask.shape
    return (
        torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1),
        torch.from_numpy(mask.copy()),
        {
            "original_height": height,
            "original_width": width,
            "resized_height": height,
            "resized_width": width,
            "pad_top": 0,
            "pad_left": 0,
        },
    )


class DatasetFixture:
    def __init__(self, root, name, channels):
        self.root = Path(root)
        self.name = name
        self.path = self.root / name
        for split in ("Tr", "Ts", "Ts_external", "Ts_interrater"):
            (self.path / f"images{split}").mkdir(parents=True)
            (self.path / f"labels{split}").mkdir()
        (self.path / "dataset.json").write_text(json.dumps({
            "channel_names": channels,
            "labels": {"background": 0, "lesion": 1},
            "file_ending": ".png",
        }))

    def add(self, case_id, split="Tr", planes=None, color=None, label=None):
        planes = planes or {}
        if color is not None:
            cv2.imwrite(str(self.path / f"images{split}" / f"{case_id}_0000.png"), color)
        else:
            for channel, value in planes.items():
                # A plane may be given as a constant or as an array, which the pairing tests need:
                # a flat image has no texture to match another image on.
                image = (value if isinstance(value, np.ndarray)
                         else np.full((16, 12), value, dtype=np.uint8))
                cv2.imwrite(
                    str(self.path / f"images{split}" / f"{case_id}_{channel:04d}.png"), image
                )
        shape = next(
            (v.shape for v in planes.values() if isinstance(v, np.ndarray)),
            None if color is None else color.shape[:2],
        ) or (16, 12)
        cv2.imwrite(
            str(self.path / f"labels{split}" / f"{case_id}.png"),
            np.zeros(shape, dtype=np.uint8) if label is None else label,
        )

    def split(self, train, val=()):
        (self.path / "splits_final.json").write_text(json.dumps([
            {"train": list(train), "val": list(val)}
        ]))
