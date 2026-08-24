import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

import torch

from fm_adaptation.training import _balanced_sampler, _validate_data_mode
from fixtures import DatasetFixture


class SamplingTests(unittest.TestCase):
        """Source-balanced sampling, and the run kinds it cannot be combined with."""

        def setUp(self):
            self.tmp = tempfile.TemporaryDirectory()
            self.root = Path(self.tmp.name)

        def tearDown(self):
            self.tmp.cleanup()

        def test_balanced_sampler_and_guards(self):
            ids = [f"Eric__{i}" for i in range(8)] + [f"Yvonne__{i}" for i in range(2)]
            sampler = _balanced_sampler(ids)
            self.assertEqual(set(sampler.weights[:8].tolist()), {0.125})
            self.assertEqual(set(sampler.weights[8:].tolist()), {0.5})
            torch.manual_seed(5)
            counts = {"Eric": 0, "Yvonne": 0}
            for _ in range(300):
                for index in sampler:
                    counts[ids[index].split("__", 1)[0]] += 1
            self.assertLess(abs(counts["Eric"] - counts["Yvonne"]), 250)

            data = DatasetFixture(self.root, "Dataset208_guard", {"0": "SMI", "1": "GFAP"})
            data.add("Eric__case", planes={0: 0, 1: 1})
            data.split(["Eric__case"])
            cfg = SimpleNamespace(
                raw_data_dir=self.root,
                train_dataset=data.name,
                patching=object(),
                channel_dropout=("SMI",),
            )
            with self.assertRaisesRegex(ValueError, "patchwise"):
                _validate_data_mode(cfg)
            cfg.patching = None
            with self.assertRaisesRegex(ValueError, "uncached"):
                _validate_data_mode(cfg, encoder_trains=False)

if __name__ == "__main__":
    unittest.main()
