import tempfile
import unittest
from pathlib import Path

from fm_adaptation.datasets import dataset_dir, dataset_id, dataset_root, resolve
from fm_adaptation.selection import matches


class DatasetIdTests(unittest.TestCase):
    """What counts as naming a dataset."""

    def test_reads_a_number_from_every_form_a_dataset_is_written_in(self):
        for value in ("217", "Dataset217", "Dataset217_lesion_MY_smi_gfap"):
            self.assertEqual(dataset_id(value), "217", value)

    def test_reads_a_number_from_a_path(self):
        self.assertEqual(dataset_id(Path("/data/raw/Dataset217_lesion_MY_smi_gfap")), "217")

    def test_ignores_the_leading_zeros_the_directory_form_carries(self):
        self.assertEqual(dataset_id("Dataset080_BUSBRA_GE_Logiq_5"), dataset_id("80"))

    def test_returns_none_for_something_that_is_not_a_dataset(self):
        self.assertIsNone(dataset_id("upernet_inj_ft_ours"))


class ResolveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "Dataset217_lesion_MY_smi_gfap").mkdir()
        (self.root / "Dataset080_BUSBRA_GE_Logiq_5").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_names_a_dataset_the_way_the_raw_data_directory_does(self):
        self.assertEqual(resolve(self.root, "217"), "Dataset217_lesion_MY_smi_gfap")
        self.assertEqual(resolve(self.root, "Dataset217"), "Dataset217_lesion_MY_smi_gfap")

    def test_renames_a_dataset_stored_under_an_earlier_name(self):
        self.assertEqual(resolve(self.root, "Dataset217_something_else"), "Dataset217_lesion_MY_smi_gfap")

    def test_keeps_what_it_was_given_when_the_number_is_not_there_to_look_up(self):
        self.assertEqual(resolve(self.root, "Dataset999_gone"), "Dataset999_gone")

    def test_builds_the_directory_a_dataset_lives_in(self):
        self.assertEqual(
            dataset_dir(self.root, "217"), self.root / "Dataset217_lesion_MY_smi_gfap"
        )

    def test_refuses_a_number_that_names_two_directories(self):
        (self.root / "Dataset217_lesion_MY_smi_gfap_v2").mkdir()
        with self.assertRaises(RuntimeError):
            resolve(self.root, "217")

    def test_exact_name_is_usable_when_an_unrelated_number_is_ambiguous(self):
        (self.root / "Dataset075_first").mkdir()
        (self.root / "Dataset075_second").mkdir()
        self.assertEqual(
            resolve(self.root, "Dataset217_lesion_MY_smi_gfap"),
            "Dataset217_lesion_MY_smi_gfap",
        )


class DatasetRootTests(unittest.TestCase):
    """Which of several raw data directories a dataset is read from."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.first, self.second = self.root / "first", self.root / "second"
        (self.first / "Dataset217_lesion_MY_smi_gfap").mkdir(parents=True)
        (self.second / "Dataset217_lesion_MY_smi_gfap").mkdir(parents=True)
        (self.second / "Dataset080_BUSBRA_GE_Logiq_5").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_takes_the_root_that_holds_the_dataset(self):
        self.assertEqual(dataset_root([self.first, self.second], "080"), self.second)

    def test_a_dataset_mirrored_under_both_roots_comes_from_the_first(self):
        self.assertEqual(dataset_root([self.first, self.second], "217"), self.first)

    def test_refuses_a_number_naming_two_different_datasets(self):
        (self.second / "Dataset300_one").mkdir()
        (self.first / "Dataset300_another").mkdir()
        with self.assertRaises(RuntimeError):
            dataset_root([self.first, self.second], "300")

    def test_a_number_no_root_knows_still_names_a_root(self):
        self.assertEqual(dataset_root([self.first, self.second], "999"), self.second)

    def test_an_ambiguous_root_does_not_stop_the_others_answering(self):
        """A root carrying some other number twice says nothing about the dataset being looked for."""
        (self.first / "Dataset075_first").mkdir()
        (self.first / "Dataset075_second").mkdir()
        self.assertEqual(dataset_root([self.first, self.second], "080"), self.second)


class SelectionTests(unittest.TestCase):
    """Datasets select by number; everything else selects as it did."""

    def test_a_number_selects_the_dataset_it_identifies(self):
        self.assertTrue(matches("Dataset217_lesion_MY_smi_gfap", ["217"]))
        self.assertTrue(matches("Dataset080_BUSBRA_GE_Logiq_5", ["80"]))
        self.assertFalse(matches("Dataset218_lesion_eric_smi_gfap", ["217"]))

    def test_a_number_selects_whatever_the_dataset_is_called(self):
        self.assertTrue(matches("Dataset217_under_any_name", ["217"]))

    def test_names_globs_and_suffix_tags_still_select(self):
        self.assertTrue(matches("Dataset217_lesion_MY_smi_gfap", ["Dataset217_lesion_MY_smi_gfap"]))
        self.assertTrue(matches("Dataset217_lesion_MY_smi_gfap", ["Dataset21*"]))
        self.assertTrue(matches("upernet_inj_ft_ours", ["ours"]))

    def test_an_empty_selection_keeps_everything(self):
        self.assertTrue(matches("Dataset217_lesion_MY_smi_gfap", []))


if __name__ == "__main__":
    unittest.main()
