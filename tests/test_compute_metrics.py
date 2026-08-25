import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_adaptation.metrics import read_case_metrics
from fixtures import DatasetFixture


class MetricsFileTests(unittest.TestCase):
        """Reading a metrics CSV written by either tool, and what counts as a case."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def _write(self, name, text):
            path = self.root / name
            path.write_text(text)
            return path

        def test_aggregate_row_is_not_a_case(self):
            """The MEAN row the scorer appends must never be counted as another image."""
            path = self._write("metrics.csv", (
                "image_id,dice,hd95,masd\n"
                "one,0.800000,10.000000,2.000000\n"
                "two,0.600000,20.000000,4.000000\n"
                "MEAN,0.700000,15.000000,3.000000\n"
            ))
            rows = read_case_metrics(path)
            self.assertEqual([row["case_id"] for row in rows], ["one", "two"])
            self.assertAlmostEqual(float(np.mean([row["dice"] for row in rows])), 0.7)

        def test_the_older_schema_still_reads(self):
            """Runs this project scored itself wrote `case_id,dice,masd` and no aggregate."""
            path = self._write("old.csv", (
                "case_id,dice,masd\n"
                "one,0.8,2.0\n"
            ))
            rows = read_case_metrics(path)
            self.assertEqual([row["case_id"] for row in rows], ["one"])
            self.assertAlmostEqual(rows[0]["masd"], 2.0)

        def test_a_case_with_no_ground_truth_is_undefined_not_infinite(self):
            """Two widefield slides have empty labels; a distance to nothing is not a huge distance."""
            path = self._write("empty.csv", (
                "image_id,dice,hd95,masd\n"
                "empty,,,inf\n"
                "real,0.500000,10.000000,2.000000\n"
            ))
            rows = read_case_metrics(path)
            self.assertTrue(np.isnan(rows[0]["dice"]))
            self.assertTrue(np.isnan(rows[0]["masd"]), "inf would drag the column mean to infinity")
            self.assertAlmostEqual(float(np.nanmean([row["masd"] for row in rows])), 2.0)


class LabelResolutionTests(unittest.TestCase):
        """Pairing a prediction directory with the labels it should be scored against."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_labels_are_found_across_every_split(self):
            """A column can hold cases from both splits, so resolution is per case, not per column."""
            from fm_adaptation.compute_metrics import _label_index

            data = DatasetFixture(self.root, "Dataset215_both_splits", {"0": "SMI", "1": "GFAP"})
            data.add("Yvonne__held", split="Ts", planes={0: 1, 1: 2})
            data.add("Yvonne__train", split="Tr", planes={0: 1, 1: 2})
            index = _label_index(data.path)
            self.assertEqual(sorted(index), ["Yvonne__held", "Yvonne__train"])
            self.assertEqual(index["Yvonne__held"].parent.name, "labelsTs")
            self.assertEqual(index["Yvonne__train"].parent.name, "labelsTr")

        def test_a_dataset_with_no_labels_is_skipped(self):
            """Dataset212 ships images only, so there is nothing to score it against."""
            from fm_adaptation.compute_metrics import _label_index

            data = DatasetFixture(self.root, "Dataset212_unlabelled", {"0": "SMI", "1": "GFAP"})
            data.add("Case__one", split="Ts", planes={0: 1, 1: 2})
            (data.path / "labelsTs" / "Case__one.png").unlink()
            self.assertEqual(_label_index(data.path), {})


if __name__ == "__main__":
    unittest.main()
