import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

import torch

from fm_adaptation.progress_samples import pick_samples, render_samples
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

            # A frozen trunk on the patchwise path caches nothing -- every patch is cut and perturbed
            # fresh -- so perturbing it is not the mistake the guard is there to catch.
            single = DatasetFixture(self.root, "Dataset300_single", {"0": "R"})
            single.add("Eric__case", planes={0: 0})
            single.split(["Eric__case"])
            patched = SimpleNamespace(
                raw_data_dir=self.root,
                train_dataset=single.name,
                patching=object(),
                channel_dropout=(),
                augment=object(),
            )
            _validate_data_mode(patched, encoder_trains=False)
            patched.patching = None
            with self.assertRaisesRegex(ValueError, "uncached"):
                _validate_data_mode(patched, encoder_trains=False)

class ProgressSampleTests(unittest.TestCase):
        """The figure of validation samples written while a run trains."""

        def test_background_only_items_are_skipped_and_a_figure_is_written(self):
            items = []
            for index in range(12):
                mask = torch.zeros(16, 16, dtype=torch.long)
                if index % 3 == 0:
                    mask[4:8, 4:8] = 1
                items.append((torch.rand(3, 16, 16), mask, {"case_id": f"case{index}", "y": 0, "x": 0}))
            samples = pick_samples(items, count=3)
            self.assertEqual(len(samples), 3)
            self.assertTrue(all((mask > 0).any() for _, mask, _ in samples))
            self.assertEqual([t for _, _, t in samples], [t for _, _, t in pick_samples(items, count=3)])

            module = torch.nn.Conv2d(3, 2, 1)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "samples" / "epoch_001.png"
                render_samples(module, samples, torch.device("cpu"), path,
                               forward=lambda m, images, masks: m(images))
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
