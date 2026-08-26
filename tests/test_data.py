import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from fm_adaptation.config import AugmentConfig
from fm_adaptation.data import (
    NnUNet2DDataset,
    _augment,
    _inscribed,
    _turn,
    active_planes,
    stain_planes,
    trained_planes,
)
from fixtures import DatasetFixture, preprocess


class LoaderTests(unittest.TestCase):
        """How a case on disk becomes the tensor the model is handed."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_legacy_rgb_and_gfap_layout(self):
            rgb = DatasetFixture(self.root, "Dataset001_rgb", {"0": "RGB"})
            bgr = np.zeros((16, 12, 3), dtype=np.uint8)
            bgr[..., 0], bgr[..., 1], bgr[..., 2] = 3, 7, 11
            rgb.add("case", color=bgr)
            rgb.split(["case"])
            image, _, _ = NnUNet2DDataset(
                self.root, rgb.name, "Tr", "0", "train", preprocess
            )[0]
            expected = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()).permute(2, 0, 1)
            self.assertTrue(torch.equal(image, expected))

            gfap = DatasetFixture(self.root, "Dataset207_gfap", {"0": "GFAP"})
            gfap.add("case", planes={0: 19})
            gfap.split(["case"])
            image, _, _ = NnUNet2DDataset(
                self.root, gfap.name, "Tr", "0", "train", preprocess
            )[0]
            self.assertFalse(image[0].any())
            self.assertFalse(image[1].any())
            self.assertTrue(torch.all(image[2] == 19))

        def test_two_stains_compose_red_and_blue(self):
            data = DatasetFixture(self.root, "Dataset208_two", {"0": "SMI", "1": "GFAP"})
            data.add("Katie__case", planes={0: 31, 1: 47})
            data.split(["Katie__case"])
            image, _, _ = NnUNet2DDataset(
                self.root, data.name, "Tr", "0", "train", preprocess
            )[0]
            self.assertTrue(torch.all(image[0] == 31))
            self.assertFalse(image[1].any())
            self.assertTrue(torch.all(image[2] == 47))

        def test_dropout_distribution_and_single_stain_protection(self):
            data = DatasetFixture(self.root, "Dataset208_drop", {"0": "SMI", "1": "GFAP"})
            data.add("Katie__dual", planes={0: 31, 1: 47})
            data.add("Eric__single", planes={0: 0, 1: 47})
            data.split(["Katie__dual", "Eric__single"])
            dataset = NnUNet2DDataset(
                self.root, data.name, "Tr", "0", "train", preprocess,
                channel_dropout=("SMI", "GFAP"), channel_dropout_p=0.5,
            )
            torch.manual_seed(3)
            states = {"both": 0, "smi": 0, "gfap": 0}
            for _ in range(1200):
                image, _, _ = dataset[0]
                has_smi, has_gfap = bool(image[0].any()), bool(image[2].any())
                states["both" if has_smi and has_gfap else "smi" if has_smi else "gfap"] += 1
            self.assertTrue(520 <= states["both"] <= 680, states)
            self.assertTrue(240 <= states["smi"] <= 360, states)
            self.assertTrue(240 <= states["gfap"] <= 360, states)
            for _ in range(20):
                image, _, _ = dataset[1]
                self.assertTrue(image.any())
                self.assertTrue(image[2].any())

            validation = NnUNet2DDataset(
                self.root, data.name, "Tr", "0", "val", preprocess,
                channel_dropout=("SMI", "GFAP"), channel_dropout_p=1.0,
            )
            validation.ids = ["Katie__dual"]
            image, _, _ = validation[0]
            self.assertTrue(image[0].any() and image[2].any())

        def test_planes_a_model_was_trained_on(self):
            """A czi_B model reads GFAP out of a two-stain set and never meets SMI."""
            self.assertEqual(active_planes({"0": "B"}), frozenset({2}))
            self.assertEqual(active_planes({"0": "R"}), frozenset({0}))
            self.assertEqual(active_planes({"0": "SMI", "1": "GFAP"}), frozenset({0, 2}))
            # An ultrasound set replicates one grey channel across all three, so it restricts nothing.
            self.assertIsNone(active_planes({"0": "US"}))

            data = DatasetFixture(self.root, "Dataset207_two", {"0": "SMI", "1": "GFAP"})
            data.add("Katie__case", planes={0: 31, 1: 47})
            data.split(["Katie__case"])
            both, _, _ = NnUNet2DDataset(self.root, data.name, "Tr", "0", "eval", preprocess)[0]
            self.assertTrue(torch.all(both[0] == 31) and torch.all(both[2] == 47))
            blue, _, _ = NnUNet2DDataset(
                self.root, data.name, "Tr", "0", "eval", preprocess,
                keep_planes=active_planes({"0": "B"}),
            )[0]
            self.assertFalse(blue[0].any())
            self.assertFalse(blue[1].any())
            self.assertTrue(torch.all(blue[2] == 47))

        def test_a_run_can_narrow_the_planes_its_training_set_declares(self):
            """Dataset218 declares SMI and GFAP but ships a blank SMI, so its runs are GFAP-only."""
            declared = {"0": "SMI", "1": "GFAP"}
            self.assertEqual(trained_planes(declared), frozenset({0, 2}))
            self.assertEqual(trained_planes(declared, ["GFAP"]), frozenset({2}))
            # Plane letters name the same thing the older datasets do.
            self.assertEqual(trained_planes(declared, ["B"]), frozenset({2}))
            with self.assertRaises(ValueError):
                trained_planes(declared, ["DAPI"])

        def test_images_without_labels(self):
            """Dataset212 ships images to predict and nothing to score them against."""
            data = DatasetFixture(self.root, "Dataset212_none", {"0": "SMI", "1": "GFAP"})
            data.add("Katie__case", split="Ts", planes={0: 31, 1: 47})
            (data.path / "labelsTs" / "Katie__case.png").unlink()
            with self.assertRaises(FileNotFoundError):
                NnUNet2DDataset(self.root, data.name, "Ts", "0", "eval", preprocess)[0]
            image, mask, meta = NnUNet2DDataset(
                self.root, data.name, "Ts", "0", "eval", preprocess, require_labels=False
            )[0]
            self.assertFalse(meta["has_label"])
            self.assertFalse(mask.any())
            self.assertEqual(mask.shape, image.shape[1:])

        def test_stain_mapping_requires_all_known_channels(self):
            self.assertEqual(stain_planes({"0": "SMI", "1": "GFAP"}), {
                "SMI": (0, 0), "GFAP": (1, 2),
            })
            self.assertIsNone(stain_planes({"0": "B"}))
            self.assertIsNone(stain_planes({"0": "GFAP", "1": "unknown"}))


class AugmentTests(unittest.TestCase):
        """Flips, rotation and zoom, and the two things they must never do: invent image, cut label."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)
            torch.manual_seed(0)

        def tearDown(self):
            self.tmp.cleanup()

        def _case(self, top, left, size=8, canvas=64, content=(32, 64)):
            """A block on a letterboxed canvas: `content` is the section, the rest is padding."""
            width, height = content
            image = torch.full((3, canvas, canvas), -2.0)
            pad_top, pad_left = (canvas - height) // 2, (canvas - width) // 2
            image[:, pad_top:pad_top + height, pad_left:pad_left + width] = 0.2
            image[:, top : top + size, left : left + size] = 1.0
            mask = torch.full((canvas, canvas), -1, dtype=torch.long)
            mask[pad_top:pad_top + height, pad_left:pad_left + width] = 0
            mask[top : top + size, left : left + size] = 1
            geometry = {"resized_width": width, "resized_height": height}
            return image, mask, geometry

        def test_a_dataset_does_not_augment_unless_asked(self):
            data = DatasetFixture(self.root, "Dataset999_plain", {"0": "SMI", "1": "GFAP"})
            data.add("Mohammad__case", planes={0: 31, 1: 47})
            data.split(["Mohammad__case"])
            dataset = NnUNet2DDataset(self.root, data.name, "Tr", "0", "train", preprocess)
            self.assertIsNone(dataset.augment)
            self.assertIsNone(dataset.fill)

        def test_augmentation_reaches_training_and_nothing_else(self):
            """A validation or prediction case must be the same image for every run that sees it."""
            data = DatasetFixture(self.root, "Dataset999_aug", {"0": "SMI", "1": "GFAP"})
            for case in ("Mohammad__a", "Mohammad__b"):
                data.add(case, planes={0: 31, 1: 47})
            data.split(["Mohammad__a"], ["Mohammad__b"])
            augment = AugmentConfig(rotation=10.0, zoom_min=0.5, zoom_max=1.5)
            for subset, expected in (("train", True), ("val", False), ("eval", False)):
                dataset = NnUNet2DDataset(
                    self.root, data.name, "Tr", "0", subset, preprocess, augment=augment,
                )
                plain = NnUNet2DDataset(self.root, data.name, "Tr", "0", subset, preprocess)
                changed = not torch.equal(dataset[0][0], plain[0][0])
                self.assertEqual(changed, expected, subset)

        def test_the_crop_keeps_only_what_is_real_image_in_every_row_and_column(self):
            """The point of the crop: a rotation must not leave a wedge of invented canvas."""
            image, mask, geometry = self._case(top=28, left=28)
            turned = _turn(image, mask, geometry, angle=10.0, scale=1.0, fill=[-2.0] * 3)
            self.assertIsNotNone(turned)
            _, label = turned
            # What is left of the section is a solid upright block: every row is either all section
            # or all padding, and the same for every column. A tilted edge would break that.
            for line in (label, label.T):
                inside = (line >= 0).sum(dim=1)
                self.assertEqual(set(int(v) for v in inside) - {0}, {int(inside.max())})

        def test_the_letterbox_that_is_left_is_the_padding_the_pipeline_already_writes(self):
            image, mask, geometry = self._case(top=28, left=28)
            canvas, label = _turn(image, mask, geometry, angle=10.0, scale=1.0, fill=[-2.0] * 3)
            self.assertEqual(int(label[0, 0]), -1)
            self.assertAlmostEqual(float(canvas[0, 0, 0]), -2.0, places=5)

        def test_the_inscribed_rectangle_shrinks_the_way_the_geometry_says(self):
            self.assertEqual(_inscribed(100, 100, 0), (100, 100))
            # A section four times taller than wide keeps a sixth of itself at 25 degrees, which is
            # why the configs draw from ten.
            narrow_10 = _inscribed(227, 896, 10)
            narrow_25 = _inscribed(227, 896, 25)
            self.assertLess(narrow_25[0] * narrow_25[1], narrow_10[0] * narrow_10[1])
            self.assertLess(narrow_25[0] * narrow_25[1], 0.2 * 227 * 896)

        def test_a_transform_that_would_cut_the_annotation_is_refused(self):
            """A lesion at the end of the section cannot survive the crop, so the candidate falls."""
            image, mask, geometry = self._case(top=17, left=17, size=6, content=(30, 30))
            self.assertIsNone(_turn(image, mask, geometry, angle=45.0, scale=1.4, fill=[-2.0] * 3))

        def test_no_draw_ever_loses_the_annotation(self):
            augment = AugmentConfig(rotation=10.0, zoom_min=0.5, zoom_max=1.5)
            for top, left in ((18, 18), (28, 28), (40, 30)):
                image, mask, geometry = self._case(top, left)
                for _ in range(50):
                    _, out = _augment(image.clone(), mask.clone(), geometry, augment, [-2.0] * 3)
                    self.assertTrue((out == 1).any(), (top, left))

        def test_an_empty_annotation_has_nothing_to_keep_in_frame(self):
            image, mask, geometry = self._case(top=28, left=28)
            mask[mask == 1] = 0
            _, out = _augment(
                image, mask, geometry, AugmentConfig(rotation=10.0, zoom_min=1.5, zoom_max=1.5),
                [-2.0] * 3,
            )
            self.assertEqual(out.shape, mask.shape)

if __name__ == "__main__":
    unittest.main()
