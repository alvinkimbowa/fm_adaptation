import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from fm_adaptation.config import AugmentConfig, ClipConfig
from fm_adaptation.data import ClipDataset, _augment, _label_box, clip_starts
from fm_adaptation.naming import describe_run
from fixtures import DatasetFixture, preprocess


class ClipStartTests(unittest.TestCase):
    """Where the windows over a video start."""

    def test_windows_cover_the_video_and_the_last_ends_on_its_last_frame(self):
        self.assertEqual(clip_starts(40, 16, 16), [0, 16, 24])
        self.assertEqual(clip_starts(40, 16, 8), [0, 8, 16, 24])
        self.assertEqual(clip_starts(16, 16, 8), [0])

    def test_a_video_shorter_than_a_clip_is_refused(self):
        with self.assertRaises(ValueError):
            clip_starts(10, 16, 8)


class ClipDatasetTests(unittest.TestCase):
    """Clips cut from an unbroken frames/ folder, labelled where another dataset traced a frame."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        torch.manual_seed(0)
        frames = DatasetFixture(self.root, "Dataset001_frames", {"0": "R", "1": "G", "2": "B"})
        (frames.path / "frames").mkdir()
        for frame in range(8):
            image = np.full((16, 12, 3), frame * 10, dtype=np.uint8)
            cv2.imwrite(str(frames.path / "frames" / f"V_{frame:03d}.png"), image)
        labelled = DatasetFixture(self.root, "Dataset002_labels", {"0": "R", "1": "G", "2": "B"})
        # Frames 1, 2 and 6 traced, with a gap between them.
        for frame in (1, 2, 6):
            label = np.zeros((16, 12), dtype=np.uint8)
            label[4:8, 4:8] = 1
            labelled.add(f"V_{frame:03d}", color=np.zeros((16, 12, 3), np.uint8), label=label)
        # One video serves as both subsets, so each can be read on its own.
        labelled.split(["V_001", "V_002", "V_006"], ["V_001", "V_002", "V_006"])
        self.clip = ClipConfig(frames=4, frames_from="Dataset001_frames")

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, subset, **kwargs):
        return ClipDataset(
            self.root, "Dataset002_labels", self.clip, "Tr", "0", subset, preprocess, **kwargs
        )

    def test_a_clip_is_unbroken_and_unlabelled_frames_are_ignored(self):
        images, masks, metadata = self._dataset("val", stride=4)[0]
        self.assertEqual(tuple(images.shape), (4, 3, 16, 12))
        self.assertEqual(metadata["frame_ids"], ["V_000", "V_001", "V_002", "V_003"])
        self.assertEqual(metadata["labelled"], [False, True, True, False])
        # Each frame is the one on disk: the frames folder fills the gap the labels leave.
        self.assertEqual([int(images[t, 0, 0, 0]) for t in range(4)], [0, 10, 20, 30])
        self.assertTrue((masks[0] == -1).all() and (masks[3] == -1).all())
        self.assertEqual(int(masks[1].sum()), 16)

    def test_every_training_clip_holds_a_label(self):
        dataset = self._dataset("train")
        # Three labelled frames fit one clip of four.
        self.assertEqual(len(dataset), 1)
        for _ in range(20):
            _, _, metadata = dataset[0]
            self.assertTrue(any(metadata["labelled"]))

    def test_tiling_keeps_only_windows_with_a_label(self):
        dataset = self._dataset("val", stride=2)
        self.assertEqual([start for _, start in dataset.windows], [0, 2, 4])


class ClipAugmentTests(unittest.TestCase):
    """A clip's frames are flipped, turned and zoomed as one."""

    def test_a_stack_of_masks_is_boxed_as_one(self):
        masks = torch.zeros((2, 10, 10), dtype=torch.long)
        masks[0, 1, 2] = 1
        masks[1, 7, 8] = 1
        self.assertEqual(_label_box(masks), (1, 7, 2, 8))

    def test_every_frame_moves_with_its_mask(self):
        torch.manual_seed(1)
        canvas, frames = 64, 3
        image = torch.zeros((3 * frames, canvas, canvas))
        masks = torch.zeros((frames, canvas, canvas), dtype=torch.long)
        for t in range(frames):
            image[3 * t:3 * t + 3, 20 + t:28 + t, 24:32] = 1.0
            masks[t, 20 + t:28 + t, 24:32] = 1
        geometry = {"resized_width": canvas, "resized_height": canvas}
        cfg = AugmentConfig(hflip=True, vflip=False, rotation=20.0, zoom_min=0.8, zoom_max=1.2)
        for _ in range(10):
            turned, turned_masks = _augment(image, masks, geometry, cfg, [0.0] * (3 * frames))
            self.assertEqual(tuple(turned_masks.shape), (frames, canvas, canvas))
            for t in range(frames):
                bright = turned[3 * t] > 0.5
                labelled = turned_masks[t] == 1
                overlap = (bright & labelled).sum() / labelled.sum()
                self.assertGreater(float(overlap), 0.8)


class ClipNamingTests(unittest.TestCase):
    def test_a_clip_run_reads_as_its_parts(self):
        self.assertEqual(
            describe_run("upernet_fullres_ln_ft_aug_clip16_split_ours"),
            "UperNet + full-res + LN head + FT + aug + 16-frame clips + tubelet split + ours",
        )
        self.assertEqual(
            describe_run("upernet_ln_ft_aug_clip16_repeat_ours"),
            "UperNet + LN head + FT + aug + 16-frame clips + frame repeat + ours",
        )


if __name__ == "__main__":
    unittest.main()
