import unittest

import numpy as np

from fm_adaptation.report import (
    ANNOTATOR,
    _annotator,
    _annotator_columns,
    _column_annotator,
)


def column(cases, dice=None):
    dice = np.arange(len(cases), dtype=float) if dice is None else np.asarray(dice, dtype=float)
    return {"dice": dice, "masd": dice.copy(), "cases": np.array(cases)}


class AnnotatorTests(unittest.TestCase):
    """Who drew a case, read off its id."""

    def test_reads_the_source_the_id_names(self):
        self.assertEqual(_annotator("Mohammad__10"), "Mohammad")
        self.assertEqual(_annotator("Eric__Rat-12_slide-5_Section-6"), "Eric")

    def test_a_batch_or_an_injury_model_is_not_a_second_annotator(self):
        self.assertEqual(_annotator("Yvonne_b2__Cond-Lesion-GelMa-rods-Rat-10"), "Yvonne")
        self.assertEqual(_annotator("Katie_contusion__CLE2-rat1_13-3"), "Katie")

    def test_an_id_that_names_nobody_names_nobody(self):
        self.assertIsNone(_annotator("Drug-study_3_GFAP_Rat-12_slide-5_analyzed_043"))


class ColumnAnnotatorTests(unittest.TestCase):
    def test_a_column_of_one_annotator_is_that_annotator(self):
        self.assertEqual(_column_annotator(column(["Mohammad__1", "Mohammad__2"])), "Mohammad")

    def test_a_column_of_two_annotators_belongs_to_neither(self):
        self.assertIsNone(_column_annotator(column(["Mohammad__1", "Yvonne__2"])))

    def test_a_column_whose_ids_name_nobody_belongs_to_nobody(self):
        self.assertIsNone(_column_annotator(column(["case_1", "case_2"])))


class AnnotatorColumnTests(unittest.TestCase):
    """The two places a per-annotator column can come from, and which one wins."""

    def setUp(self):
        self.trained_on = "Dataset217_lesion_MY_smi_gfap"
        self.key = ("dinov3", "upernet_inj_ft_ours", self.trained_on, "0")

    def test_the_training_sets_own_split_fills_the_annotators_it_holds(self):
        records = {self.key: {self.trained_on: column(["Mohammad__1", "Yvonne__2", "Yvonne_b2__3"])}}
        rewritten, columns, in_domain = _annotator_columns(records, [self.trained_on])
        self.assertIn(ANNOTATOR + "Mohammad", columns)
        self.assertIn(ANNOTATOR + "Yvonne", columns)
        self.assertEqual(len(rewritten[self.key][ANNOTATOR + "Yvonne"]["cases"]), 2)
        self.assertIn((self.trained_on, ANNOTATOR + "Mohammad"), in_domain)

    def test_an_annotator_the_run_never_trained_on_keeps_its_own_dataset(self):
        records = {
            self.key: {
                self.trained_on: column(["Mohammad__1"]),
                "Dataset218_lesion_eric_smi_gfap": column(["Eric__a", "Eric__b"]),
            }
        }
        rewritten, columns, in_domain = _annotator_columns(
            records, [self.trained_on, "Dataset218_lesion_eric_smi_gfap"]
        )
        self.assertIn(ANNOTATOR + "Eric", columns)
        self.assertNotIn((self.trained_on, ANNOTATOR + "Eric"), in_domain)
        self.assertEqual(len(rewritten[self.key][ANNOTATOR + "Eric"]["cases"]), 2)

    def test_the_run_s_own_split_wins_where_both_hold_an_annotator(self):
        records = {
            self.key: {
                self.trained_on: column(["Mohammad__1"], dice=[0.9]),
                "Dataset214_lesion_mohammad_smi_gfap": column(["Mohammad__7", "Mohammad__8"], dice=[0.1, 0.2]),
            }
        }
        rewritten, _, in_domain = _annotator_columns(
            records, [self.trained_on, "Dataset214_lesion_mohammad_smi_gfap"]
        )
        self.assertEqual(list(rewritten[self.key][ANNOTATOR + "Mohammad"]["dice"]), [0.9])
        self.assertIn((self.trained_on, ANNOTATOR + "Mohammad"), in_domain)

    def test_a_set_holding_two_annotators_stays_a_column_of_its_own(self):
        interrater = "Dataset210_lesion_interrater_MY_smi_gfap"
        records = {self.key: {interrater: column(["Mohammad__1", "Yvonne__2"])}}
        _, columns, _ = _annotator_columns(records, [interrater])
        self.assertEqual(columns, [interrater])


if __name__ == "__main__":
    unittest.main()
