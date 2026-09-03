import tempfile
import unittest
from pathlib import Path

from fm_adaptation.selection import list_order, select_runs


def _run(root, model, trained_on, config, config_yaml=True):
    """A run directory with one column of predictions, as `<split>/<dataset>/predictions`."""
    run_dir = root / model / trained_on / config / "fold_0"
    (run_dir / "test" / "Dataset211_paul" / "predictions").mkdir(parents=True)
    if config_yaml:
        (run_dir / "config.yaml").write_text("")
    return run_dir


class ListOrderTests(unittest.TestCase):
    def test_a_listed_value_sorts_where_the_list_puts_it(self):
        listed = ["b", "a", "c"]
        self.assertEqual(sorted(["a", "b", "c"], key=lambda v: list_order(v, listed)), listed)

    def test_an_unlisted_value_sorts_after_every_listed_one(self):
        self.assertGreater(list_order("z", ["b", "a"]), list_order("a", ["b", "a"]))

    def test_an_empty_list_falls_back_to_the_value_itself(self):
        self.assertEqual(sorted(["b", "a"], key=lambda v: list_order(v, [])), ["a", "b"])


class SelectRunsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.aug = _run(self.root, "dinov3", "Dataset208_MYKE", "balanced_aug")
        self.plain = _run(self.root, "dinov3", "Dataset219_MYK", "balanced")
        self.baseline = _run(self.root, "nnunet", "Dataset105_eric", "someTrainer__somePlans__2d",
                             config_yaml=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_run_is_kept_only_when_every_named_part_matches(self):
        self.assertEqual(
            select_runs(self.root, models=["dinov3"], train_datasets=["Dataset208_MYKE"],
                        configs=["balanced_aug"], folds=["0"]),
            [self.aug],
        )
        self.assertEqual(
            select_runs(self.root, models=["dinov3"], train_datasets=["Dataset208_MYKE"],
                        configs=["balanced_aug"], folds=["1"]),
            [],
        )

    def test_an_empty_list_narrows_nothing(self):
        self.assertEqual(len(select_runs(self.root)), 3)

    def test_a_run_without_a_config_of_its_own_is_not_filtered_by_configs(self):
        """Its directory is named in its trainer's vocabulary, so listing our configurations cannot
        be expected to name it."""
        self.assertIn(self.baseline, select_runs(self.root, configs=["balanced_aug"]))

    def test_but_models_still_decides_whether_it_is_kept(self):
        self.assertNotIn(self.baseline, select_runs(self.root, models=["dinov3"]))

    def test_order_is_model_then_training_set_then_configuration(self):
        runs = select_runs(
            self.root,
            models=["nnunet", "dinov3"],
            train_datasets=["Dataset219_MYK", "Dataset208_MYKE", "Dataset105_eric"],
        )
        self.assertEqual(runs, [self.baseline, self.plain, self.aug])

    def test_a_directory_with_no_predictions_is_not_a_run(self):
        (self.root / "dinov3" / "Dataset300_empty" / "config" / "fold_0").mkdir(parents=True)
        self.assertEqual(len(select_runs(self.root)), 3)


if __name__ == "__main__":
    unittest.main()
