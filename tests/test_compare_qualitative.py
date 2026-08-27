import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from fm_adaptation.compare_qualitative import _columns, _common_cases, _evaluation_sets, resolve_runs
except ImportError as error:  # pragma: no cover - environment, not behaviour
    # Drawing needs scikit-image, which only the SAM3 environment carries; the rest of the suite runs
    # under .venv-mm. Skipping keeps one interpreter from failing to collect what the other checks.
    raise unittest.SkipTest(f"compare_qualitative unavailable here: {error}") from error


def _run(root, name, dices):
    """One run's prediction directory and metrics CSV, scored as `dices` says."""
    predictions = root / name / "predictions"
    predictions.mkdir(parents=True)
    for case in dices:
        (predictions / f"{case}.png").write_bytes(b"")
    metrics = root / name / "metrics.csv"
    rows = "".join(f"{case},{dice:.3f},1.0,1.0\n" for case, dice in dices.items())
    mean = sum(dices.values()) / len(dices)
    metrics.write_text(f"image_id,dice,hd95,masd\n{rows}MEAN,{mean:.3f},1.0,1.0\n")
    return (None, None, predictions, metrics)


class CommonCaseTests(unittest.TestCase):
    """Which cases a comparison figure draws, and whether the seed can move them."""

    def _runs(self, root, cases):
        dices = {f"case_{i:02d}": 1.0 - i / cases for i in range(cases)}
        return [_run(root, "a", dices), _run(root, "b", dices)]

    def test_the_seed_moves_the_sample_on_a_set_barely_larger_than_the_figure(self):
        """The regression this guards: 22 shared cases used to give one fixed sample forever.

        `select_cases` is asked for three times the cases the figure holds, so any dataset smaller
        than that returned its whole ranking, and thinning it by even steps then picked the same
        rows whatever the seed. Only Eric's 73 cases were large enough to vary.
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = self._runs(root, 22)
            samples = {
                tuple(case for case, _ in _common_cases(runs, 8, runs[0], np.random.default_rng(seed)))
                for seed in range(6)
            }
            self.assertGreater(len(samples), 1)

    def test_every_sample_still_spans_the_quality_range(self):
        """Varying the sample must not cost the spread -- a figure of near-identical rows says
        nothing about where a model struggles."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = self._runs(root, 40)
            for seed in range(6):
                picked = _common_cases(runs, 8, runs[0], np.random.default_rng(seed))
                scores = [dice for _, dice in picked if dice is not None]
                self.assertEqual(len(picked), 8)
                self.assertGreater(max(scores) - min(scores), 0.7)

    def test_a_set_smaller_than_the_figure_is_drawn_whole_and_stays_put(self):
        """Six cases cannot fill eight rows, so there is nothing to choose and nothing to vary."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = self._runs(root, 6)
            samples = {
                tuple(sorted(case for case, _ in _common_cases(runs, 8, runs[0], np.random.default_rng(seed))))
                for seed in range(4)
            }
            self.assertEqual(len(samples), 1)
            self.assertEqual(len(next(iter(samples))), 6)

    def test_only_cases_every_run_predicted_are_offered(self):
        """A column is a run, so a case one run is missing cannot be a row."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = {f"case_{i:02d}": 1.0 - i / 20 for i in range(20)}
            partial = {case: dice for case, dice in full.items() if case != "case_00"}
            runs = [_run(root, "a", full), _run(root, "b", partial)]
            picked = {case for case, _ in _common_cases(runs, 8, runs[0], np.random.default_rng(0))}
            self.assertNotIn("case_00", picked)


def _fold(root, model, trained_on, run_name, columns):
    """A run directory shaped like models/, with `columns` as {evaluation set: [case ids]}."""
    fold = root / model / trained_on / run_name / "fold_0"
    fold.mkdir(parents=True)
    (fold / "config.yaml").write_text("")
    for tested_on, case_ids in columns.items():
        predictions = fold / "test" / tested_on / "predictions"
        predictions.mkdir(parents=True)
        for case_id in case_ids:
            (predictions / f"{case_id}.png").write_bytes(b"")
    return fold


class RunSelectionTests(unittest.TestCase):
    """Naming a column by its path, which is how a row of the results table is identified."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.myke = _fold(self.root, "dinov3", "Dataset208_MYKE", "balanced_aug",
                          {"Dataset211_paul": ["p1", "p2"], "Dataset218_eric": ["e_held"]})
        self.myk = _fold(self.root, "dinov3", "Dataset219_MYK", "balanced_aug",
                         {"Dataset211_paul": ["p1", "p2"],
                          "Dataset218_eric": ["e_held", "e1", "e2"],
                          "Dataset219_MYK": ["own"]})

    def tearDown(self):
        self.tmp.cleanup()

    def test_runs_resolve_in_the_order_given(self):
        """The list is the column order. Sorting by anything else would mean the script cannot say
        which column goes where, which is the whole point of naming them one by one."""
        for order in ([self.myk, self.myke], [self.myke, self.myk]):
            asked = [str(path.relative_to(self.root)) for path in order]
            self.assertEqual(resolve_runs(self.root, asked), list(order))

    def test_a_glob_expands_to_the_runs_it_matches(self):
        self.assertEqual(
            resolve_runs(self.root, ["*/Dataset2*/balanced_aug/fold_0"]),
            sorted([self.myke, self.myk]),
        )

    def test_no_match_is_an_error_rather_than_a_missing_column(self):
        with self.assertRaises(SystemExit):
            resolve_runs(self.root, ["dinov3/Dataset999_nothing/*/fold_0"])

    def test_only_sets_every_run_evaluated_are_offered(self):
        """A run's own training set is one the other rows never predicted, so it cannot be a figure."""
        self.assertEqual(_evaluation_sets([self.myke, self.myk], "test"),
                         ["Dataset211_paul", "Dataset218_eric"])

    def test_a_run_without_predictions_for_a_set_is_named_rather_than_dropped(self):
        columns, missing = _columns([self.myke, self.myk], "test", "Dataset219_MYK")
        self.assertIsNone(columns)
        self.assertEqual(missing, self.myke)

    def test_columns_holding_different_splits_of_one_set_intersect(self):
        """MYKE trained on Eric, so it holds Eric's imagesTs; MYK never did, so it holds all of
        Eric. The shared cases are the held-out ones, and the figure is drawn on those rather than
        the set being skipped."""
        columns, missing = _columns([self.myke, self.myk], "test", "Dataset218_eric")
        self.assertIsNone(missing)
        cases = _common_cases(columns, 8, columns[0], np.random.default_rng(0))
        self.assertEqual([case for case, _ in cases], ["e_held"])

    def test_a_column_is_labelled_by_its_configuration_and_its_training_set(self):
        """Read off the path, so a run the figure has never seen before still labels itself."""
        columns, _ = _columns([self.myke, self.myk], "test", "Dataset211_paul")
        self.assertEqual([label for label, *_ in columns],
                         ["DINOv3 + balanced + aug + trained on MYKE",
                          "DINOv3 + balanced + aug + trained on MYK"])


if __name__ == "__main__":
    unittest.main()
