import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_adaptation.agreement import annotator, interrater_pairs, measure
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
            from fm_adaptation.agreement import annotator, interrater_pairs, measure

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

            pairs, unpaired = interrater_pairs(data.path, "Ts_interrater")
            self.assertEqual([case for _, case in pairs], [("Ann__7", "Bea_b2__Rat-7-slide1")])
            self.assertEqual(unpaired, ["Bea__solo-section"])

            rows, unpaired = measure(data.path, "Ts_interrater")
            (annotators, name, dice, _), = rows
            # `_b2` is the same person's second batch; a `_raterN` suffix is a different person.
            self.assertEqual(annotators, ["Ann", "Bea"])
            self.assertEqual(name, "Rat-7-slide1")
            self.assertGreater(dice, 0.8)
            self.assertEqual(unpaired, ["Bea__solo-section"])
            self.assertEqual(annotator("Yvonne__x_rater2"), "Yvonne rater2")
            self.assertEqual(annotator("Yvonne_b2__x"), "Yvonne")

if __name__ == "__main__":
    unittest.main()


class StoredAgreementTests(unittest.TestCase):
        """What compute_metrics writes down is what the report reads back."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_rows_and_unpaired_survive_a_round_trip(self):
            from fm_adaptation.agreement import path_for, read, write

            rows = [(["Mohammad", "Yvonne"], "slide-1", 0.7561, 155.29)]
            path = path_for(self.root, "Dataset210_lesion_interrater_MY_smi_gfap", "Ts")
            write(rows, ["Yvonne__lonely"], path)
            self.assertEqual(read(path), (rows, ["Yvonne__lonely"]))

        def test_nothing_measured_reads_as_nothing(self):
            from fm_adaptation.agreement import read

            self.assertEqual(read(self.root / "absent.csv"), ([], []))
