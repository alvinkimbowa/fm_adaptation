import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

from fm_adaptation.predict import _jobs, _seen_in_training
from fixtures import DatasetFixture


class JobTests(unittest.TestCase):
        """Which evaluations a run produces, and what each one's column is called."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_held_out_job_discovery(self):
            data = DatasetFixture(self.root, "Dataset208_jobs", {"0": "SMI", "1": "GFAP"})
            data.add("Train__one", planes={0: 1, 1: 2})
            data.split(["Train__one"])
            for split in ("Ts", "Ts_external", "Ts_interrater"):
                data.add(f"Held__{split}", split=split, planes={0: 1, 1: 2})
            cfg = SimpleNamespace(
                raw_data_dir=self.root,
                train_dataset=data.name,
                test_split="Tr",
                fold="0",
            )
            jobs = _jobs(cfg, [data.name])
            self.assertEqual([job[4] for job in jobs[1:]], [
                data.name,
                f"{data.name}_external",
                f"{data.name}_interrater",
            ])

        def test_training_cases_are_excluded_from_other_datasets(self):
            """A run is scored on an evaluation set only over the cases it did not fit.

            Dataset207 is wholly contained in Dataset208, so a Dataset208 run must be scored on the
            Katie slides it held out and on nothing else.
            """
            data = DatasetFixture(self.root, "Dataset208_seen", {"0": "SMI", "1": "GFAP"})
            for case_id in ("Katie__fitted", "Katie__validated", "Katie__held_out"):
                data.add(case_id, planes={0: 1, 1: 2})
            data.split(["Katie__fitted"], val=["Katie__validated"])
            cfg = SimpleNamespace(raw_data_dir=self.root, train_dataset=data.name, fold="0")
            self.assertEqual(
                _seen_in_training(cfg), {"Katie__fitted", "Katie__validated"}
            )

        def test_nothing_is_excluded_without_a_split_file(self):
            """An evaluation-only dataset ships no splits_final.json and must not raise."""
            data = DatasetFixture(self.root, "Dataset211_nosplit", {"0": "SMI", "1": "GFAP"})
            cfg = SimpleNamespace(raw_data_dir=self.root, train_dataset=data.name, fold="0")
            self.assertEqual(_seen_in_training(cfg), set())


if __name__ == "__main__":
    unittest.main()
