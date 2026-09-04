import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_adaptation.report import (PARAMETER_COUNTS, _best_values, _dataset_family,
                                  _experiment_order, _in_domain, _model_matches, _pool_folds,
                                  nnunet_label)
from fixtures import DatasetFixture


def column(cases, dice):
    dice = np.asarray(dice, dtype=float)
    return {"dice": dice, "masd": dice.copy(), "cases": np.array(cases)}


class DatasetFamilyTests(unittest.TestCase):
    """Which table a run lands in, and which metrics that table carries."""

    def test_the_second_token_names_the_family(self):
        self.assertEqual(_dataset_family("Dataset207_lesion_katie_contusion_smi_gfap"), "lesion")

    def test_the_neurite_sets_answer_to_one_family_however_they_are_named(self):
        self.assertEqual(_dataset_family("Dataset203_neurites_yvonne_smi_2px_scaleaug"), "neurites")
        self.assertEqual(_dataset_family("Dataset300_neurite_yvonne_smi"), "neurites")


class NnunetLabelTests(unittest.TestCase):
    """What tells one nnU-Net directory from another ends up in its row's name."""

    def test_the_stock_trainer_plans_and_configuration_read_plainly(self):
        self.assertEqual(nnunet_label("nnUNetTrainer__nnUNetPlans__2d"), "nnU-Net")

    def test_the_plans_names_the_model(self):
        """A plans is a different network, not a way of training one, so it belongs to the name."""
        self.assertEqual(nnunet_label("nnUNetTrainer__nnUNetResEncUNetMPlans__2d"),
                         "nnU-Net Res Enc M")

    def test_a_non_stock_trainer_is_named_too(self):
        """Neurites train for 100 epochs, lesions for the default 1000: same plans, different runs,
        so the label has to keep them apart."""
        self.assertEqual(nnunet_label("nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__2d"),
                         "nnU-Net Res Enc M (100 epochs)")
        self.assertEqual(
            nnunet_label("nnUNetTrainerSkeletonRecall_100epochs__nnUNetResEncUNetMPlans__2d"),
            "nnU-Net Res Enc M (Skeleton Recall 100 epochs)",
        )

    def test_a_configuration_that_is_its_own_network_keeps_its_own_name(self):
        self.assertEqual(nnunet_label("nnUNetTrainer__nnUNetPlans__2d_xtiny8"), "XTinyUNet")


class ModelMatchTests(unittest.TestCase):
    """`--models`, matched the way someone would type it."""

    def test_case_and_hyphens_are_ignored(self):
        self.assertTrue(_model_matches("nnU-Net", ["nnunet"]))
        self.assertTrue(_model_matches("nnU-Net", ["nnU_net"]))

    def test_each_plans_is_selected_on_its_own(self):
        """The stock plans and the residual-encoder preset are separate rows to ask for."""
        self.assertTrue(_model_matches("nnU-Net Res Enc M", ["nnUNetResEncM"]))
        self.assertFalse(_model_matches("nnU-Net Res Enc M", ["nnunet"]))
        self.assertFalse(_model_matches("nnU-Net", ["nnUNetResEncM"]))
        self.assertFalse(_model_matches("nnU-Net Res Enc M", ["dinov3"]))

    def test_a_glob_takes_every_plans_at_once(self):
        self.assertTrue(_model_matches("nnU-Net Res Enc M", ["nnUNet*"]))
        self.assertTrue(_model_matches("nnU-Net", ["nnUNet*"]))

    def test_how_a_network_was_trained_does_not_change_what_it_answers_to(self):
        self.assertTrue(_model_matches("nnU-Net Res Enc M (100 epochs)", ["nnUNetResEncM"]))

    def test_a_glob_can_narrow_to_one_variant(self):
        self.assertTrue(_model_matches("nnU-Net (xtiny32)", ["*xtiny*"]))
        self.assertFalse(_model_matches("nnU-Net (Res Enc M)", ["*xtiny*"]))

    def test_an_empty_list_keeps_everything(self):
        self.assertTrue(_model_matches("dinov3", []))


class RowOrderTests(unittest.TestCase):
    """Rows come out where the selection lists put them."""

    def rows(self, models=(), train_datasets=(), configs=(), group=False, sort_by="",
             descending=False):
        records = {
            ("dinov3", "balanced_aug", "Dataset208_MYKE", "0"): {},
            ("dinov3", "balanced", "Dataset219_MYK", "0"): {},
            ("nnunet", "", "Dataset105_eric", "0"): {},
        }
        key = _experiment_order(list(models), list(train_datasets), list(configs), group, sort_by,
                                descending)
        return [(k[0], k[2], k[1]) for k, _ in sorted(records.items(), key=key)]

    def sized(self):
        """Counts for every row, so only the sort column decides the order."""
        self.addCleanup(PARAMETER_COUNTS.clear)
        PARAMETER_COUNTS.update({
            "dinov3|balanced_aug|": {"total": 300, "trainable": 3},
            "dinov3|balanced|": {"total": 100, "trainable": 1},
            "nnunet||Dataset105_eric": {"total": 200, "trainable": 2},
        })

    def test_model_is_weighed_first_then_training_set_then_configuration(self):
        self.assertEqual(
            self.rows(models=["nnunet", "dinov3"],
                      train_datasets=["Dataset219_MYK", "Dataset208_MYKE", "Dataset105_eric"]),
            [("nnunet", "Dataset105_eric", ""),
             ("dinov3", "Dataset219_MYK", "balanced"),
             ("dinov3", "Dataset208_MYKE", "balanced_aug")],
        )

    def test_reordering_a_list_reorders_the_rows(self):
        first = self.rows(models=["dinov3", "nnunet"])
        second = self.rows(models=["nnunet", "dinov3"])
        self.assertEqual(first[0][0], "dinov3")
        self.assertEqual(second[0][0], "nnunet")

    def test_a_value_no_list_names_sorts_after_the_ones_named(self):
        rows = self.rows(models=["nnunet"])
        self.assertEqual(rows[0][0], "nnunet")

    def test_a_sort_column_outweighs_the_lists(self):
        self.sized()
        self.assertEqual(
            [row[0:2] for row in self.rows(models=["nnunet", "dinov3"], sort_by="params")],
            [("dinov3", "Dataset219_MYK"),
             ("nnunet", "Dataset105_eric"),
             ("dinov3", "Dataset208_MYKE")],
        )

    def test_naming_no_column_leaves_the_lists_in_charge(self):
        self.sized()
        self.assertEqual(self.rows(models=["nnunet", "dinov3"])[0][0], "nnunet")

    def test_descending_puts_the_largest_first(self):
        self.sized()
        self.assertEqual(
            [row[0:2] for row in self.rows(sort_by="trainable", descending=True)],
            [("dinov3", "Dataset208_MYKE"),
             ("nnunet", "Dataset105_eric"),
             ("dinov3", "Dataset219_MYK")],
        )

    def test_a_run_of_unknown_size_sorts_last_whichever_way_round(self):
        self.addCleanup(PARAMETER_COUNTS.clear)
        PARAMETER_COUNTS.update({"dinov3|balanced_aug|": {"total": 300, "trainable": 3}})
        for descending in (False, True):
            with self.subTest(descending=descending):
                rows = self.rows(models=["nnunet", "dinov3"], sort_by="params",
                                 descending=descending)
                self.assertEqual(rows[0], ("dinov3", "Dataset208_MYKE", "balanced_aug"))


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

    def test_sets_declared_to_hold_the_same_images_are_in_domain_without_sharing_a_case_id(self):
        """Two builds of one collection name their cases differently, so ids cannot answer this."""
        from fm_adaptation import report

        original = report.SHARED_IMAGES
        report.SHARED_IMAGES = (frozenset({"Dataset208_combined", "Dataset211_paul"}),)
        try:
            pairs = _in_domain(self.records, self.names, self.raw_dirs)
        finally:
            report.SHARED_IMAGES = original
        self.assertIn(("Dataset208_combined", "Dataset211_paul"), pairs)

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


class NnunetRecordTests(unittest.TestCase):
    """Which columns an nnU-Net results tree contributes, given how it lays its splits out."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.results = Path(self.tmp.name)
        self.trainer = (
            self.results / "nnunet" / "Dataset070_Clarius_L15"
            / "nnUNetTrainer__nnUNetResEncUNetMPlans__2d" / "fold_0"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, *parts):
        path = self.trainer.joinpath(*parts, "metrics.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("image_id,dice,cldice,hd95,masd\ncase,0.9,0.9,3,1\n")

    def _records(self):
        from collections import defaultdict
        from fm_adaptation.report import _add_nnunet_records

        records = defaultdict(dict)
        _add_nnunet_records(records, self.results)
        return {key: sorted(results) for key, results in records.items()}

    def test_a_held_out_fold_is_the_training_set_s_own_column(self):
        self._write("validation")
        self._write("test", "Dataset072_GE_LQP9")
        self.assertEqual(
            self._records(),
            {("nnU-Net Res Enc M", "", "Dataset070_Clarius_L15", "0"):
             ["Dataset070_Clarius_L15", "Dataset072_GE_LQP9"]},
        )

    def test_an_own_test_set_is_preferred_over_the_held_out_fold(self):
        """The training dataset is reported once, and imagesTs is the better of the two."""
        self._write("validation")
        self._write("test", "Dataset070_Clarius_L15")
        key = ("nnU-Net Res Enc M", "", "Dataset070_Clarius_L15", "0")
        self.assertEqual(self._records(), {key: ["Dataset070_Clarius_L15"]})

    def test_a_tree_holding_neither_split_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self._records()
