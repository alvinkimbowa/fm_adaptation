import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_adaptation.metrics import read_case_metrics
from fixtures import DatasetFixture

try:
    import fm_adaptation.compute_metrics  # noqa: F401
    SCORING = True
except ImportError:  # pragma: no cover - environment, not behaviour
    # Scoring needs monai, which only .venv-mm carries. The tests below that never open a metric
    # still run everywhere; only the ones that reach into the scoring module stand down.
    SCORING = False

needs_scoring = unittest.skipUnless(SCORING, "monai is not installed in this environment")



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


@needs_scoring
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

        def test_derived_label_renderings_are_not_ground_truth(self):
            from fm_adaptation.compute_metrics import _label_index

            data = DatasetFixture(self.root, "Dataset070_renderings", {"0": "US"})
            data.add("Case__one", split="Tr", planes={0: 1})
            rendered = data.path / "labelsVal_fold0_alvin_visualized"
            rendered.mkdir()
            (rendered / "Case__one.png").write_bytes(b"rendered")
            index = _label_index(data.path)
            self.assertEqual(index["Case__one"].parent.name, "labelsTr")


@needs_scoring
class NnunetColumnTests(unittest.TestCase):
        """Selecting the baseline columns out of an nnU-Net results tree, which carries no config."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)
            self.results = self.root / "nnUNet_results"
            for trained_on, tested_on in (
                ("Dataset105_lesion_eric_gfap_resized", "Dataset105_lesion_eric_gfap_resized"),
                ("Dataset105_lesion_eric_gfap_resized", "Dataset207_lesion_katie_contusion_smi_gfap"),
                ("Dataset116_neurites_mohammad_smi", "Dataset121_neurites_yvonne_smi"),
            ):
                (
                    self.results / "nnunet" / trained_on
                    / "nnUNetTrainer__nnUNetResEncUNetMPlans__2d" / "fold_0" / "test" / tested_on
                    / "preds"
                ).mkdir(parents=True)

        def tearDown(self):
            self.tmp.cleanup()

        def _select(self, **kwargs):
            from argparse import Namespace
            from fm_adaptation.compute_metrics import _nnunet_columns

            args = Namespace(**{"datasets": [], "folds": [], "tested_on": [], **kwargs})
            return [
                (path.parents[4].name, path.parent.name)
                for path, _ in _nnunet_columns(self.results, self.root, args)
            ]

        def test_every_column_is_found_without_a_selection(self):
            self.assertEqual(len(self._select()), 3)

        def test_a_number_selects_a_column(self):
            self.assertEqual(
                self._select(datasets=["105"], tested_on=["207"]),
                [("Dataset105_lesion_eric_gfap_resized",
                  "Dataset207_lesion_katie_contusion_smi_gfap")],
            )

        def test_the_selection_narrows_by_training_and_evaluation_set(self):
            """The other runs in a shared results tree belong to work this table never shows."""
            self.assertEqual(
                self._select(
                    datasets=["Dataset105_lesion_eric_gfap_resized"],
                    tested_on=["Dataset207_lesion_katie_contusion_smi_gfap"],
                ),
                [("Dataset105_lesion_eric_gfap_resized",
                  "Dataset207_lesion_katie_contusion_smi_gfap")],
            )

        def test_a_tif_only_column_counts_as_current(self):
            """The czi predictions are TIFFs, so a PNG-only freshness check never sees them."""
            from fm_adaptation.compute_metrics import _is_current

            predictions = self.root / "preds"
            predictions.mkdir()
            (predictions / "case.tif").write_bytes(b"")
            metrics = self.root / "metrics.csv"
            metrics.write_text("image_id,dice,hd95,masd\n")
            self.assertTrue(_is_current(metrics, predictions))


if __name__ == "__main__":
    unittest.main()
