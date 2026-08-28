import math
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from fm_adaptation.config import AugmentConfig, PatchConfig
from fm_adaptation.patching import _augment_patch, _prepare, _source_size


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
            case_id="c", shape=image.shape,
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


if __name__ == "__main__":
    unittest.main()
