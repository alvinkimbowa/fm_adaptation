import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_adaptation.report import _annotator, _interrater_pairs, _interrater_rows
from fixtures import DatasetFixture


class InterraterTests(unittest.TestCase):
        """Finding the same image annotated twice, and naming who drew each one."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_interrater_pairing_and_annotator_names(self):
            """The same image annotated twice is found by content, not by case ID."""
            from fm_adaptation.report import _annotator, _interrater_pairs, _interrater_rows

            rng = np.random.default_rng(0)
            data = DatasetFixture(self.root, "Dataset209_combined_two_raters", {"0": "SMI", "1": "GFAP"})
            slide = rng.integers(0, 255, (96, 40), dtype=np.uint8)
            other = rng.integers(0, 255, (96, 40), dtype=np.uint8)
            # The same slide as two annotators received it: one re-exported, so it is not byte-identical.
            nudged = np.clip(slide.astype(np.int16) + rng.integers(-6, 6, slide.shape), 0, 255)
            label = np.zeros((96, 40), dtype=np.uint8)
            label[20:60, 10:30] = 1
            shifted = np.zeros_like(label)
            shifted[24:60, 10:28] = 1
            for case_id, plane, mask in (
                ("Ann__7", slide, label),
                ("Bea_b2__Rat-7-slide1", nudged.astype(np.uint8), shifted),
                ("Bea__solo-section", other, label),
            ):
                data.add(case_id, split="Ts_interrater", planes={0: plane, 1: plane}, label=mask)

            pairs, unpaired = _interrater_pairs(data.path, "Ts_interrater")
            self.assertEqual([case for _, case in pairs], [("Ann__7", "Bea_b2__Rat-7-slide1")])
            self.assertEqual(unpaired, ["Bea__solo-section"])

            rows, unpaired = _interrater_rows(data.path, "Ts_interrater")
            (annotators, name, dice, _), = rows
            # `_b2` is the same person's second batch; a `_raterN` suffix is a different person.
            self.assertEqual(annotators, ["Ann", "Bea"])
            self.assertEqual(name, "Rat-7-slide1")
            self.assertGreater(dice, 0.8)
            self.assertEqual(unpaired, ["Bea__solo-section"])
            self.assertEqual(_annotator("Yvonne__x_rater2"), "Yvonne rater2")
            self.assertEqual(_annotator("Yvonne_b2__x"), "Yvonne")

if __name__ == "__main__":
    unittest.main()
