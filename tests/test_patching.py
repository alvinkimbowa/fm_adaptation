import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from fm_adaptation.config import AugmentConfig, PatchConfig
from fixtures import DatasetFixture, preprocess
from fm_adaptation.patching import (
    _PlaneCache, _augment_patch, _prepare, _source_size, build_index, foreground_statistics, zscore,
)


def _augment(**overrides):
    return AugmentConfig(**{"hflip": False, "vflip": False, "flip_p": 0.0, **overrides})


class SourceSizeTests(unittest.TestCase):
    """How much wider than the patch the crop has to be for the turn to cost nothing."""

    def test_no_rotation_needs_no_margin(self):
        self.assertEqual(_source_size(512, _augment(rotation=0.0)), 512)
        self.assertEqual(_source_size(512, None), 512)

    def test_any_angle_needs_the_diagonal(self):
        self.assertEqual(_source_size(512, _augment(rotation=180.0)), math.ceil(512 * math.sqrt(2)))
        self.assertEqual(_source_size(512, _augment(rotation=45.0)), math.ceil(512 * math.sqrt(2)))

    def test_a_narrow_angle_needs_less(self):
        margin = _source_size(512, _augment(rotation=10.0))
        self.assertGreater(margin, 512)
        self.assertLess(margin, math.ceil(512 * math.sqrt(2)))

    def test_zooming_out_widens_it_further(self):
        self.assertGreater(
            _source_size(512, _augment(rotation=10.0, zoom_min=0.5)),
            _source_size(512, _augment(rotation=10.0)),
        )


class AugmentPatchTests(unittest.TestCase):
    """What reaches the model after the turn."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.patch = 64
        self.source = _source_size(self.patch, _augment(rotation=180.0))

    def test_the_turned_patch_is_all_real_image(self):
        """The point of the oversized crop: at any angle every pixel of the result was imaged, so a
        rotation never teaches the model to expect fill in the corners."""
        image = np.full((self.source, self.source), 255, dtype=np.uint8)
        label = np.ones_like(image)
        for _ in range(20):
            turned, _ = _augment_patch(image, label, self.patch, _augment(rotation=180.0), self.rng)
            self.assertEqual(turned.shape, (self.patch, self.patch))
            self.assertTrue((turned > 0).all(), "fill reached the patch")

    def test_without_margin_the_patch_is_left_unturned(self):
        """The failure this design avoids. Turning a patch already cut to its final size has to fill
        the corners, so where the case affords no margin the rotation is dropped rather than the
        model being shown invented pixels."""
        image = np.full((self.patch, self.patch), 255, dtype=np.uint8)
        label = np.ones_like(image)
        centre = ((self.patch - 1) / 2, (self.patch - 1) / 2)
        naive = cv2.warpAffine(image, cv2.getRotationMatrix2D(centre, 30.0, 1.0),
                               (self.patch, self.patch), flags=cv2.INTER_LINEAR)
        self.assertTrue((naive == 0).any(), "the naive turn should show fill")
        kept, _ = _augment_patch(image, label, self.patch, _augment(rotation=180.0), self.rng)
        np.testing.assert_array_equal(kept, image)

    def test_flips_alone_leave_the_pixels_untouched(self):
        image = np.arange(self.patch * self.patch, dtype=np.uint16).reshape(self.patch, self.patch)
        label = (image % 2).astype(np.uint8)
        flipped, flipped_label = _augment_patch(
            image, label, self.patch, _augment(hflip=True, flip_p=1.0), self.rng
        )
        np.testing.assert_array_equal(flipped, image[:, ::-1])
        np.testing.assert_array_equal(flipped_label, label[:, ::-1])

    def test_the_label_turns_with_the_image(self):
        image = np.zeros((self.source, self.source), dtype=np.uint8)
        image[self.source // 2 - 20 : self.source // 2 + 20, :] = 255
        label = (image > 0).astype(np.uint8)
        turned, turned_label = _augment_patch(
            image, label, self.patch, _augment(rotation=180.0), self.rng
        )
        np.testing.assert_array_equal(turned_label > 0, turned > 127)


class PrepareTests(unittest.TestCase):
    """The crop `_prepare` takes, which is what widens around the anchor."""

    def case(self, side=2000):
        image = np.random.default_rng(0).integers(1, 255, (side, side), dtype=np.uint8)
        return SimpleNamespace(
            case_id="c", shape=image.shape, weight_path=None,
            crop=lambda y, x, size: (image[y : y + size, x : x + size],
                                     np.ones((size, size), dtype=np.uint8)),
        )

    def prepare(self, case, augment, patch=512):
        cfg = PatchConfig(patch_size=patch)
        sizes = {}

        def preprocess(image, label):
            sizes["shape"] = np.asarray(image).shape[:2]
            return image, label, {}

        _prepare(case, 100, 100, cfg, preprocess, 0, augment, np.random.default_rng(0))
        return sizes["shape"]

    def test_without_augmentation_the_crop_is_the_patch(self):
        self.assertEqual(self.prepare(self.case(), None), (512, 512))

    def test_with_augmentation_the_patch_still_comes_out_at_its_size(self):
        self.assertEqual(self.prepare(self.case(), _augment(rotation=180.0)), (512, 512))

    def test_a_case_too_small_to_widen_still_yields_a_patch(self):
        """No margin to be had, so the angle is cut back to what the case supports rather than the
        crop being padded out to the size the rotation would like."""
        self.assertEqual(self.prepare(self.case(side=520), _augment(rotation=180.0)), (512, 512))


class WeightMapTests(unittest.TestCase):
    """The loss weight map, which has to stay registered with the label it weights."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = DatasetFixture(self.root, "Dataset001_weighted", {"0": "SMI"})
        self.label = np.zeros((16, 12), dtype=np.uint8)
        self.label[3:6, 2:9] = 1
        self.data.add("case", planes={0: 31}, label=self.label)
        self.data.split(["case"])

    def tearDown(self):
        self.tmp.cleanup()

    def index(self):
        return build_index(self.root, self.data.name, "Tr", "0", "train", PatchConfig(patch_size=8),
                           self.root / "cache", weight_maps=True)

    def write_map(self):
        """255 where the label is set and 51 elsewhere."""
        (self.data.path / "weight_maps").mkdir()
        cv2.imwrite(str(self.data.path / "weight_maps" / "case.png"),
                    np.where(self.label > 0, 255, 51).astype(np.uint8))

    def test_a_case_without_a_map_stops_the_index(self):
        with self.assertRaises(FileNotFoundError):
            self.index()

    def test_the_patch_carries_its_weights_scaled_to_one(self):
        self.write_map()
        case = self.index()[0]
        _, label, meta = _prepare(case, 0, 0, PatchConfig(patch_size=8), preprocess, 0)
        weight = meta["weight"]
        self.assertEqual(weight.shape, label.shape)
        self.assertTrue(torch.allclose(weight[label > 0], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(weight[label == 0], torch.tensor(0.2)))

    def test_the_weights_flip_with_the_label(self):
        label = np.zeros((16, 16), dtype=np.uint8)
        label[2:5, 1:4] = 1
        weight = np.where(label > 0, 255, 51).astype(np.uint8)
        image = np.dstack([label * 200] * 3)
        augment = AugmentConfig(hflip=True, vflip=True, flip_p=1.0)
        _, flipped, weights = _augment_patch(image, label, 16, augment, np.random.default_rng(0), weight)
        self.assertTrue((flipped != label).any())
        self.assertTrue(((weights == 255) == (flipped > 0)).all())


class ChannelLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = DatasetFixture(self.root, "Dataset001_stains", {"0": "SMI", "1": "GFAP"})
        self.data.add("case", planes={0: 31, 1: 47})
        self.data.split(["case"])

    def tearDown(self):
        self.tmp.cleanup()

    def crop(self, respect_channels):
        cfg = PatchConfig(patch_size=8, respect_channels=respect_channels)
        case = build_index(
            self.root, self.data.name, "Tr", "0", "train", cfg,
            self.root / f"cache_{respect_channels}",
        )[0]
        return case.crop(0, 0, 8)[0]

    def test_patch_channels_are_opt_in(self):
        grayscale = self.crop(False)
        self.assertTrue(np.all(grayscale == 31))

        channels = self.crop(True)
        self.assertTrue(np.all(channels[..., 0] == 31))
        self.assertFalse(channels[..., 1].any())
        self.assertTrue(np.all(channels[..., 2] == 47))


class ZScoreTests(unittest.TestCase):
    """Patches standardised by their whole image, as the tracer app standardises what it sends."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.image = rng.integers(0, 255, (64, 48), dtype=np.uint8)
        self.image[:, :16] = 0
        data = DatasetFixture(self.root, "Dataset311_z", {"0": "SMI"})
        data.add("case", planes={0: self.image})
        data.split(["case"])
        self.cfg = PatchConfig(patch_size=16, respect_channels=True, normalization="zscore")
        self.case = build_index(self.root, data.name, "Tr", "0", "train", self.cfg, self.root / "cache")[0]

    def test_statistics_leave_the_zeros_out(self):
        foreground = self.image[self.image != 0].astype(np.float32)
        mean, std = foreground_statistics(self.image)
        self.assertAlmostEqual(mean, float(foreground.mean()), places=4)
        self.assertAlmostEqual(std, float(foreground.std()), places=4)
        self.assertAlmostEqual(
            float(zscore(np.zeros((2, 2), np.uint8), (mean, std))[0, 0]), -mean / std, places=5
        )

    def test_an_image_with_nothing_to_vary_becomes_zeros(self):
        for flat in (np.zeros((4, 4), np.uint8), np.full((4, 4), 9, np.uint8)):
            self.assertIsNone(foreground_statistics(flat))
            self.assertFalse(zscore(flat, None).any())

    def test_a_patch_is_the_same_window_of_the_standardised_image(self):
        whole = zscore(self.image, foreground_statistics(self.image))
        image, _, meta = _prepare(self.case, 8, 24, self.cfg, preprocess, 0)
        self.assertEqual(tuple(image.shape), (1, 16, 16))
        self.assertTrue(np.allclose(image[0].numpy(), whole[8:24, 24:40]))

    def test_the_raw_crop_keeps_one_channel_per_stored_plane(self):
        raw, _ = self.case.crop(8, 24, 16)
        self.assertEqual(raw.shape, (16, 16, 1))
        self.assertTrue(np.array_equal(raw[..., 0], self.image[8:24, 24:40]))

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(ValueError):
            PatchConfig(normalization="minmax")


class PlaneCacheTests(unittest.TestCase):
    """Decoded planes are kept only up to a byte budget; one that was dropped is read again."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.paths = []
        for value in (10, 20, 30):
            path = Path(self.directory.name) / f"{value}.png"
            cv2.imwrite(str(path), np.full((8, 8), value, dtype=np.uint8))
            self.paths.append(path)

    def test_a_plane_within_budget_is_decoded_once(self):
        cache = _PlaneCache(budget_bytes=64 * 3)
        self.assertIs(cache.get(self.paths[0]), cache.get(self.paths[0]))

    def test_the_least_recently_used_plane_is_dropped_past_the_budget(self):
        cache = _PlaneCache(budget_bytes=64 * 2)
        first = cache.get(self.paths[0])
        second = cache.get(self.paths[1])
        cache.get(self.paths[0])
        cache.get(self.paths[2])
        self.assertIs(cache.get(self.paths[0]), first)
        self.assertIsNot(cache.get(self.paths[1]), second)
        self.assertLessEqual(cache.held_bytes, 64 * 2)

    def test_a_dropped_plane_reads_back_the_same(self):
        cache = _PlaneCache(budget_bytes=64)
        cache.get(self.paths[0])
        cache.get(self.paths[1])
        self.assertTrue((cache.get(self.paths[0]) == 10).all())


if __name__ == "__main__":
    unittest.main()
