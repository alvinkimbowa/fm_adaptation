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
    _PlaneCache, _augment_patch, _prepare, _source_size, bell_reach, bell_weights, build_index,
    foreground_statistics, zscore,
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


class SlabTests(unittest.TestCase):
    """Patches projected from neighbouring slices of a stack, and the weights computed for them."""

    SHAPE = (48, 40)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = DatasetFixture(self.root, "Dataset001_slices", {"0": "SMI"})
        rng = np.random.default_rng(0)
        self.images, self.labels = {}, {}
        for z in range(1, 6):
            case_id = f"Src__a_z{z:03d}"
            self.images[case_id] = rng.integers(1, 255, self.SHAPE, dtype=np.uint8)
            label = np.zeros(self.SHAPE, dtype=np.uint8)
            # One line per slice, each in its own row, so a projection is told from a single slice.
            label[4 + 8 * z, 2:30] = 1
            self.labels[case_id] = label
            self.data.add(case_id, planes={0: self.images[case_id]}, label=label)
        self.data.split(list(self.images))

    def tearDown(self):
        self.tmp.cleanup()

    def index(self, slices, **kwargs):
        cfg = PatchConfig(patch_size=16, respect_channels=True, slab_slices=slices)
        return {case.case_id: case
                for case in build_index(self.root, self.data.name, "Tr", "0", "train", cfg,
                                        self.root / "cache", **kwargs)}

    def projected(self, arrays, centre):
        members = [f"Src__a_z{z:03d}" for z in (centre - 1, centre, centre + 1) if 1 <= z <= 5]
        return np.max([arrays[member] for member in members], axis=0)

    def test_one_slice_reads_the_case_as_it_is_stored(self):
        case = self.index(1)["Src__a_z003"]
        image, label = case.crop(8, 4, 16)
        np.testing.assert_array_equal(image[..., 0], self.images["Src__a_z003"][8:24, 4:20])
        self.assertFalse(image[..., 1:].any())
        np.testing.assert_array_equal(label, self.labels["Src__a_z003"][8:24, 4:20])

    def test_a_slab_is_the_maximum_over_its_slices(self):
        cases = self.index(3)
        for centre in (1, 3, 5):
            image, label = cases[f"Src__a_z{centre:03d}"].crop(8, 4, 32)
            np.testing.assert_array_equal(image[..., 0], self.projected(self.images, centre)[8:40, 4:36])
            np.testing.assert_array_equal(label, self.projected(self.labels, centre)[8:40, 4:36])

    def test_computed_weights_are_those_of_the_whole_projected_label(self):
        """The window is cut from the map of the full label, so a line just outside it still
        weights the pixels inside."""
        case = self.index(3, weight_bell=(5.0, 0.1))["Src__a_z003"]
        whole = bell_weights(self.projected(self.labels, 3) > 0, 5.0, 0.1)
        # Rows 21 to 27 hold no line; the lines of slices 2 and 3 sit just above and below.
        window = case.crop_weight(21, 4, 7)
        np.testing.assert_array_equal(window, whole[21:28, 4:11])
        self.assertGreater(int(window.min()), round(255 * 0.1))
        _, label, meta = _prepare(case, 8, 4, PatchConfig(patch_size=16), preprocess, 0)
        np.testing.assert_allclose(meta["weight"].numpy() * 255, whole[8:24, 4:20], atol=1e-3)
        self.assertTrue(torch.allclose(meta["weight"][label > 0], torch.tensor(1.0)))

    def test_the_bell_curve_has_the_shape_the_stored_maps_have(self):
        annotated = np.zeros((40, 40), dtype=bool)
        annotated[20, 5:35] = True
        weights = bell_weights(annotated, 5.0, 0.1)
        self.assertEqual(weights[20, 20], 255)
        self.assertEqual(weights[21, 20], 255)
        self.assertEqual(weights[25, 20], round(255 * math.exp(-(4**2) / 50)))
        self.assertEqual(weights[20 + bell_reach(5.0, 0.1), 20], round(255 * 0.1))
        self.assertEqual(weights[39, 20], round(255 * 0.1))
        self.assertTrue((bell_weights(np.zeros((4, 4), dtype=bool), 5.0, 0.1) == 26).all())

    def test_stored_maps_of_a_slab_are_the_map_of_its_projected_label(self):
        (self.data.path / "weight_maps").mkdir()
        for case_id, label in self.labels.items():
            cv2.imwrite(str(self.data.path / "weight_maps" / f"{case_id}.png"),
                        bell_weights(label > 0, 5.0, 0.1))
        case = self.index(3, weight_maps=True)["Src__a_z003"]
        whole = bell_weights(self.projected(self.labels, 3) > 0, 5.0, 0.1)
        np.testing.assert_array_equal(case.crop_weight(0, 0, 48), whole)

    def test_a_slab_is_refused_under_zscoring(self):
        cfg = PatchConfig(patch_size=16, normalization="zscore", slab_slices=3)
        with self.assertRaisesRegex(ValueError, "slab_slices"):
            build_index(self.root, self.data.name, "Tr", "0", "train", cfg, self.root / "cache")


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
