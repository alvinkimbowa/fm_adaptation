import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from fm_adaptation.data import NnUNet2DDataset, active_planes, stain_planes, trained_planes
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

if __name__ == "__main__":
    unittest.main()
