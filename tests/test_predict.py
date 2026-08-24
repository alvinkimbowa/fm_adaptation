import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

from fm_adaptation.predict import _jobs
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

if __name__ == "__main__":
    unittest.main()
