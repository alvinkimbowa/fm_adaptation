import tempfile
import unittest
from pathlib import Path

import numpy as np

from fm_adaptation.selection import select_runs

try:
    from fm_adaptation.qualitative import crop_window, _window_overlap
    from fm_adaptation.compare_qualitative import (_columns, _common_cases, _evaluation_sets,
                                                   _repeat_cases)
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
    """Which runs become columns, and in what order."""

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

    def test_columns_follow_the_order_the_lists_put_them_in(self):
        for order in (["Dataset219_MYK", "Dataset208_MYKE"], ["Dataset208_MYKE", "Dataset219_MYK"]):
            runs = select_runs(self.root, train_datasets=order)
            self.assertEqual([run.parents[1].name for run in runs], order)

    def test_every_part_narrows_and_an_empty_list_does_not(self):
        self.assertEqual(select_runs(self.root, train_datasets=["Dataset208_MYKE"]), [self.myke])
        self.assertEqual(len(select_runs(self.root)), 2)
        self.assertEqual(select_runs(self.root, configs=["nothing_like_this"]), [])

    def test_each_set_is_offered_with_the_runs_that_hold_it(self):
        """A run predicts only what its own config names, so no set is held by every run."""
        self.assertEqual(
            _evaluation_sets([self.myke, self.myk], "test"),
            {"Dataset211_paul": [self.myke, self.myk],
             "Dataset218_eric": [self.myke, self.myk],
             "Dataset219_MYK": [self.myk]},
        )

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
        """Read off the path, so a run the figure has never seen before still labels itself. The
        training set is named by its id: the rest of a directory name is too wide for a header."""
        columns, _ = _columns([self.myke, self.myk], "test", "Dataset211_paul")
        self.assertEqual([label for label, *_ in columns],
                         ["DINOv3 + balanced + aug + trained on 208",
                          "DINOv3 + balanced + aug + trained on 219"])


class RepeatCasesTests(unittest.TestCase):
    """Filling a patchwise figure from an evaluation set with only a case or two in it."""

    def test_a_set_that_already_fills_the_figure_is_left_alone(self):
        cases = [("a", 0.1), ("b", 0.2), ("c", 0.3)]
        self.assertEqual(_repeat_cases(cases, 3), cases)
        self.assertEqual(_repeat_cases(cases, 2), cases)

    def test_too_few_cases_are_cycled_so_repeats_sit_apart(self):
        cases = [("a", 0.1), ("b", 0.2)]
        self.assertEqual([c for c, _ in _repeat_cases(cases, 5)], ["a", "b", "a", "b", "a"])

    def test_nothing_to_repeat_stays_empty(self):
        self.assertEqual(_repeat_cases([], 4), [])


class CropWindowTests(unittest.TestCase):
    """A case drawn more than once has to show somewhere else each time."""

    def setUp(self):
        self.label = np.zeros((2000, 2000), dtype=np.uint8)
        self.label[::40, :] = 1          # annotation spread over the whole image
        self.rng = np.random.default_rng(0)

    def test_a_window_avoids_the_ones_already_taken(self):
        first = crop_window(self.label, self.label.shape, 256, self.rng)
        taken = [(first[0].start, first[1].start)]
        second = crop_window(self.label, self.label.shape, 256, self.rng, avoid=taken)
        overlap = _window_overlap((second[0].start, second[1].start), taken[0], 256)
        self.assertLess(overlap, 0.25)

    def test_an_annotation_confined_to_one_spot_still_yields_a_window(self):
        """Nothing else to show, so the least overlapping try is taken rather than raising."""
        label = np.zeros((600, 600), dtype=np.uint8)
        label[300, 300] = 1
        taken = [(172, 172)]
        window = crop_window(label, label.shape, 256, self.rng, avoid=taken)
        self.assertEqual((window[0].start, window[1].start), (172, 172))


if __name__ == "__main__":
    unittest.main()
