import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_adaptation.report import _best_values, _in_domain, _model_matches, _pool_folds
from fixtures import DatasetFixture


def column(cases, dice):
    dice = np.asarray(dice, dtype=float)
    return {"dice": dice, "masd": dice.copy(), "cases": np.array(cases)}


class ModelMatchTests(unittest.TestCase):
    """`--models`, matched the way someone would type it."""

    def test_case_and_hyphens_are_ignored(self):
        self.assertTrue(_model_matches("nnU-Net", ["nnunet"]))
        self.assertTrue(_model_matches("nnU-Net", ["nnU_net"]))

    def test_a_plans_variant_answers_to_its_base_name(self):
        self.assertTrue(_model_matches("nnU-Net (Res Enc M)", ["nnunet"]))
        self.assertFalse(_model_matches("nnU-Net (Res Enc M)", ["dinov3"]))

    def test_a_glob_can_narrow_to_one_variant(self):
        self.assertTrue(_model_matches("nnU-Net (xtiny32)", ["*xtiny*"]))
        self.assertFalse(_model_matches("nnU-Net (Res Enc M)", ["*xtiny*"]))

    def test_an_empty_list_keeps_everything(self):
        self.assertTrue(_model_matches("dinov3", []))


class PoolFoldTests(unittest.TestCase):
    """Several folds compiled into one row."""

    def records(self, folds):
        return {
            ("dinov3", "run", "Dataset208_MYKE", fold): {"Dataset211_paul": column([f"p{fold}"], [0.5])}
            for fold in folds
        }

    def test_the_requested_folds_cases_end_up_in_one_row(self):
        pooled = _pool_folds(self.records(["0", "1", "2"]), ["0", "1", "2"])
        self.assertEqual(len(pooled), 1)
        (key,) = pooled
        self.assertEqual(sorted(pooled[key]["Dataset211_paul"]["cases"]), ["p0", "p1", "p2"])

    def test_a_fold_that_was_not_asked_for_is_dropped(self):
        pooled = _pool_folds(self.records(["0", "1"]), ["0"])
        (key,) = pooled
        self.assertEqual(list(pooled[key]["Dataset211_paul"]["cases"]), ["p0"])

    def test_a_row_is_labelled_with_the_folds_it_holds(self):
        """A run that has only fold 0 must not read as an average over three."""
        pooled = _pool_folds(self.records(["0"]), ["0", "1", "2"])
        self.assertEqual([key[3] for key in pooled], ["0"])
        pooled = _pool_folds(self.records(["0", "1", "2"]), ["0", "1", "2"])
        self.assertEqual([key[3] for key in pooled], ["0,1,2"])


class InDomainTests(unittest.TestCase):
    """Which cells are scored on images the row's training set also holds."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        combined = DatasetFixture(self.root, "Dataset208_combined", {"0": "SMI", "1": "GFAP"})
        for case_id in ("Katie__one", "Katie__two"):
            combined.add(case_id, planes={0: 1, 1: 2})
        katie = DatasetFixture(self.root, "Dataset207_katie", {"0": "SMI", "1": "GFAP"})
        katie.add("Katie__one", planes={0: 1, 1: 2})
        katie.add("Katie__held", split="Ts", planes={0: 1, 1: 2})
        paul = DatasetFixture(self.root, "Dataset211_paul", {"0": "SMI", "1": "GFAP"})
        paul.add("Paul__a1", split="Ts", planes={0: 1, 1: 2})
        self.names = [combined.name, katie.name, paul.name]
        self.raw_dirs = {name: self.root for name in self.names}
        self.records = {("dinov3", "run", combined.name, "0"): {}}

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_set_sharing_a_case_with_the_training_set_is_in_domain(self):
        pairs = _in_domain(self.records, self.names, self.raw_dirs)
        self.assertIn(("Dataset208_combined", "Dataset207_katie"), pairs)

    def test_a_set_sharing_nothing_is_not(self):
        pairs = _in_domain(self.records, self.names, self.raw_dirs)
        self.assertNotIn(("Dataset208_combined", "Dataset211_paul"), pairs)

    def test_a_training_set_is_in_domain_against_itself(self):
        """The rule is the same for every column: a set the run trained on is in-domain, so a run
        shown against its own training set reads greyed there as well."""
        pairs = _in_domain(self.records, self.names, self.raw_dirs)
        self.assertIn(("Dataset208_combined", "Dataset208_combined"), pairs)


class BestValueTests(unittest.TestCase):
    """Which cell in a column is marked."""

    def records(self):
        return {
            ("dinov3", "a", "Dataset208_MYKE", "0"): {"Dataset211_paul": column(["p"], [0.4])},
            ("dinov3", "b", "Dataset219_MYK", "0"): {"Dataset211_paul": column(["p"], [0.9])},
            ("dinov3", "c", "Dataset207_katie", "0"): {"Dataset211_paul": column(["p"], [0.6])},
        }

    def test_best_and_second_span_rows_from_different_training_sets(self):
        best = _best_values(self.records(), ["Dataset211_paul"], np.mean)
        self.assertEqual(best[("Dataset211_paul", "dice")], (0.9, 0.6))

    def test_an_in_domain_cell_cannot_be_marked(self):
        best = _best_values(
            self.records(),
            ["Dataset211_paul"],
            np.mean,
            in_domain={("Dataset219_MYK", "Dataset211_paul")},
        )
        self.assertEqual(best[("Dataset211_paul", "dice")], (0.6, 0.4))

    def test_masd_is_ranked_the_other_way_up(self):
        best = _best_values(self.records(), ["Dataset211_paul"], np.mean)
        self.assertEqual(best[("Dataset211_paul", "masd")], (0.4, 0.6))

    def test_a_single_row_has_nothing_to_win_against(self):
        one = dict(list(self.records().items())[:1])
        self.assertEqual(_best_values(one, ["Dataset211_paul"], np.mean), {})


if __name__ == "__main__":
    unittest.main()
