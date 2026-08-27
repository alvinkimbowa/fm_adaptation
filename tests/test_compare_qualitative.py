import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    from fm_adaptation.compare_qualitative import _common_cases
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


if __name__ == "__main__":
    unittest.main()
