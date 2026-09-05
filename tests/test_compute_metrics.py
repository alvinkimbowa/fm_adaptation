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


def _trained_on(column):
    """The training set a selected nnU-Net column belongs to, whichever split it came from."""
    return (
        column.predictions.parents[2].name if column.dataset
        else column.predictions.parents[4].name
    )


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
            self.validation = (
                self.results / "nnunet" / "Dataset116_neurites_mohammad_smi"
                / "nnUNetTrainer__nnUNetResEncUNetMPlans__2d" / "fold_0" / "validation"
            )
            self.validation.mkdir(parents=True)

        def tearDown(self):
            self.tmp.cleanup()

        def _select(self, **kwargs):
            from argparse import Namespace
            from fm_adaptation.compute_metrics import _nnunet_columns

            args = Namespace(
                **{"datasets": [], "folds": [], "splits": [], "tested_on": [], **kwargs}
            )
            return [
                (_trained_on(column), column.dataset or column.predictions.parent.name)
                for column in _nnunet_columns(self.results, [self.root], args)
            ]

        def test_every_column_is_found_without_a_selection(self):
            self.assertEqual(len(self._select()), 4)

        def test_a_held_out_fold_is_a_column_for_the_training_set(self):
            """nnU-Net names no set on that path, so the column has to carry what it was measured on."""
            self.assertIn(
                ("Dataset116_neurites_mohammad_smi", "Dataset116_neurites_mohammad_smi"),
                self._select(datasets=["116"]),
            )

        def test_a_split_selection_leaves_out_the_held_out_fold(self):
            self.assertEqual(self._select(datasets=["116"], splits=["test"]),
                             [("Dataset116_neurites_mohammad_smi", "Dataset121_neurites_yvonne_smi")])

        def test_the_held_out_fold_is_measured_where_its_predictions_are(self):
            """There is no directory above them belonging to that column alone."""
            from fm_adaptation.compute_metrics import _nnunet_columns
            from argparse import Namespace

            args = Namespace(datasets=["116"], folds=[], splits=["validation"], tested_on=[])
            [column] = _nnunet_columns(self.results, [self.root], args)
            self.assertEqual(column.predictions, self.validation)

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
            from fm_adaptation.metrics import METRIC_FIELDS
            metrics.write_text(",".join(("image_id", *METRIC_FIELDS)) + "\n")
            self.assertTrue(_is_current(metrics, predictions))


@needs_scoring
class MonounetColumnTests(unittest.TestCase):
        """A MonoUNet architecture directory, which carries no config and no configuration level."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)
            self.results = self.root / "MonoUNetE123V2GatedDA"
            for trained_on, tested_on in (
                ("Dataset070_Clarius_L15", "Dataset070_Clarius_L15"),
                ("Dataset070_Clarius_L15", "Dataset072_GE_LQP9"),
                ("Dataset073_GE_LE", "Dataset072_GE_LQP9"),
            ):
                (self.results / trained_on / "fold_0" / "test" / tested_on / "preds").mkdir(
                    parents=True
                )

        def tearDown(self):
            self.tmp.cleanup()

        def _select(self, **kwargs):
            from argparse import Namespace
            from fm_adaptation.compute_metrics import _monounet_columns

            args = Namespace(**{"datasets": [], "folds": [], "tested_on": [], **kwargs})
            return [
                (column.predictions.parents[3].name, column.predictions.parent.name)
                for column in _monounet_columns(self.results, [self.root], args)
            ]

        def test_every_column_is_found_without_a_selection(self):
            self.assertEqual(len(self._select()), 3)

        def test_the_selection_narrows_by_training_and_evaluation_set(self):
            self.assertEqual(
                self._select(datasets=["070"], tested_on=["072"]),
                [("Dataset070_Clarius_L15", "Dataset072_GE_LQP9")],
            )

        def test_a_column_is_read_into_the_label_space(self):
            """Every MonoUNet column carries the reader; ours and nnU-Net's carry none."""
            from fm_adaptation.compute_metrics import binary_mask_in_label_space

            readers = {column.prepare for column in self._columns()}
            self.assertEqual(readers, {binary_mask_in_label_space})

        def _columns(self):
            from argparse import Namespace
            from fm_adaptation.compute_metrics import _monounet_columns

            args = Namespace(datasets=[], folds=[], tested_on=[])
            return _monounet_columns(self.results, [self.root], args)


@needs_scoring
class MaskNormalisationTests(unittest.TestCase):
        """A mask saved on the network's own canvas, as 0/255, measured against a native label."""

        def test_a_canvas_sized_mask_is_thresholded_and_resampled(self):
            from fm_adaptation.compute_metrics import binary_mask_in_label_space

            prediction = np.zeros((4, 4), dtype=np.uint8)
            prediction[1:3, 1:3] = 255
            label = np.zeros((8, 8), dtype=np.uint8)
            label[2:6, 2:6] = 1
            mask = binary_mask_in_label_space(prediction, label)
            self.assertEqual(mask.shape, label.shape)
            self.assertEqual(sorted(np.unique(mask)), [0, 1])
            np.testing.assert_array_equal(mask, label)

        def test_a_mask_already_in_the_label_space_is_only_thresholded(self):
            from fm_adaptation.compute_metrics import binary_mask_in_label_space

            prediction = np.array([[0, 255], [255, 0]], dtype=np.uint8)
            label = np.array([[0, 1], [1, 0]], dtype=np.uint8)
            np.testing.assert_array_equal(binary_mask_in_label_space(prediction, label), label)


if __name__ == "__main__":
    unittest.main()


@needs_scoring
class CldiceToleranceTests(unittest.TestCase):
    def test_offsets_and_euclidean_diagonal(self):
        from fm_adaptation.compute_metrics import _cldice
        truth = np.zeros((20, 20), dtype=np.uint8)
        truth[5, 5] = 1
        for offset in range(6):
            pred = np.zeros_like(truth)
            pred[5, 5 + offset] = 1
            expected = tuple(float(r >= offset) for r in range(5))
            self.assertEqual(_cldice(pred, truth, 2), expected)
            self.assertEqual(_cldice(truth, pred, 2), expected)
        pred = np.zeros_like(truth)
        pred[6, 6] = 1
        self.assertEqual(_cldice(pred, truth, 2), (0, 0, 1, 1, 1))

    def test_zero_matches_original_and_work_is_reused(self):
        from unittest.mock import patch
        import fm_adaptation.metrics as scoring
        rng = np.random.default_rng(42)
        pred, truth = rng.random((2, 30, 30)) > 0.7
        ps, ts = scoring.skeletonize(pred), scoring.skeletonize(truth)
        precision = (ps & truth).sum() / ps.sum()
        sensitivity = (ts & pred).sum() / ts.sum()
        expected = 2 * precision * sensitivity / (precision + sensitivity)
        with patch.object(scoring, "skeletonize", wraps=scoring.skeletonize) as skeleton, \
             patch.object(scoring, "distance_transform_edt", wraps=scoring.distance_transform_edt) as distance:
            scores = scoring.cldice(pred, truth, 2)
        self.assertEqual(scores[0], expected)
        self.assertTrue(np.all(np.diff(scores) >= 0))
        self.assertEqual(skeleton.call_count, 2)
        self.assertEqual(distance.call_count, 2)

    def test_multiclass_and_empty_policy(self):
        from fm_adaptation.compute_metrics import _cldice
        truth = np.zeros((20, 20), dtype=np.uint8)
        truth[3, 3], truth[12, 12] = 1, 2
        pred = np.zeros_like(truth)
        pred[3, 3], pred[12, 14] = 1, 2
        self.assertEqual(_cldice(pred, truth, 4), (0.5, 0.5, 1, 1, 1))
        self.assertTrue(np.isnan(_cldice(np.zeros_like(truth), truth, 3)).all())

    def test_csv_roundtrip_and_schema_upgrade(self):
        import csv
        from fm_adaptation.compute_metrics import CaseMetrics, compute_case_metrics, write_csv, _is_current
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 5] = 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case.png").touch()
            path = root / "metrics.csv"
            path.write_text("image_id,dice,cldice,hd95,masd\ncase,1,1,0,0\n")
            self.assertFalse(_is_current(path, root))
            row = CaseMetrics("case", *compute_case_metrics(mask, mask, 2))
            write_csv([row], path)
            self.assertTrue(_is_current(path, root))
            values = read_case_metrics(path)[0]
            for key in ("cldice", "cldice_1px", "cldice_2px", "cldice_3px", "cldice_4px"):
                self.assertEqual(values[key], 1.0)
            with path.open() as stream:
                aggregate = list(csv.DictReader(stream))[-1]
            self.assertEqual(aggregate["image_id"], "MEAN")
            self.assertEqual(float(aggregate["cldice_4px"]), 1.0)
