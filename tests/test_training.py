import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

import numpy as np
import torch

from fm_adaptation.models import RefinementModel, SegmentationModel, build_model
from fm_adaptation.progress_samples import pick_samples, render_samples
from fm_adaptation.training import (
    _balanced_sampler,
    _train_steps,
    _training_batches,
    _truncate_history,
    _validate_data_mode,
)
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
                refines=None,
                weight_maps=False,
                weight_maps_dice=False,
                distance_weight_tau=0.0,
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
                refines=None,
                weight_maps=False,
                weight_maps_dice=False,
                distance_weight_tau=0.0,
            )
            _validate_data_mode(patched, encoder_trains=False)
            patched.patching = None
            with self.assertRaisesRegex(ValueError, "uncached"):
                _validate_data_mode(patched, encoder_trains=False)

        def test_a_refining_run_needs_an_empty_plane_for_the_refined_output(self):
            data = DatasetFixture(self.root, "Dataset307_refined", {"0": "SMI"})
            data.add("Eric__case", planes={0: 0})
            data.split(["Eric__case"])
            cfg = SimpleNamespace(
                raw_data_dir=self.root,
                train_dataset=data.name,
                patching=SimpleNamespace(respect_channels=True),
                channel_dropout=(),
                augment=None,
                refines="first",
                refines_plane=1,
                weight_maps=False,
                weight_maps_dice=False,
                distance_weight_tau=0.0,
            )
            _validate_data_mode(cfg)
            cfg.refines_plane = 0
            with self.assertRaisesRegex(ValueError, "empty RGB plane"):
                _validate_data_mode(cfg)
            cfg.refines_plane = 1
            cfg.patching = SimpleNamespace(respect_channels=False)
            with self.assertRaisesRegex(ValueError, "respect_channels"):
                _validate_data_mode(cfg)

        def test_weight_maps_need_patches_and_no_other_weighting(self):
            data = DatasetFixture(self.root, "Dataset311_weighted", {"0": "SMI"})
            data.add("Eric__case", planes={0: 0})
            data.split(["Eric__case"])
            cfg = SimpleNamespace(
                raw_data_dir=self.root,
                train_dataset=data.name,
                patching=SimpleNamespace(respect_channels=True),
                channel_dropout=(),
                augment=None,
                refines=None,
                weight_maps=True,
                weight_maps_dice=True,
                distance_weight_tau=0.0,
            )
            _validate_data_mode(cfg)
            cfg.distance_weight_tau = 5.0
            with self.assertRaisesRegex(ValueError, "distance_weight_tau"):
                _validate_data_mode(cfg)
            cfg.distance_weight_tau = 0.0
            cfg.weight_maps = False
            with self.assertRaisesRegex(ValueError, "weight_maps_dice requires"):
                _validate_data_mode(cfg)
            cfg.weight_maps, cfg.patching = True, None
            with self.assertRaisesRegex(ValueError, "patchwise"):
                _validate_data_mode(cfg)


class _StubEncoder(torch.nn.Module):
    mean = (0.5, 0.25, 0.5)
    std = (0.5, 0.5, 0.5)
    trainable = True

    def forward(self, images):
        self.seen = images
        return images


class _StubProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Conv2d(3, 2, 1)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, features, output_size):
        return self.classifier(features)


class StepwiseTrainingTests(unittest.TestCase):
        """Training counted in optimiser steps, validated every so many of them."""

        def test_batches_run_on_across_epochs_and_a_resume_finishes_its_epoch(self):
            loader = list(range(5))
            batches = _training_batches(loader, 0, 1)
            self.assertEqual([next(batches) for _ in range(7)],
                             [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (1, 0), (2, 1)])
            # Two batches to a step leaves an epoch of 5 three steps long, its last batch alone. A
            # resume after two steps draws a fresh pass and takes the three batches left.
            resumed = _training_batches(loader, 4, 2)
            self.assertEqual([next(resumed) for _ in range(3)], [(3, 0), (4, 1), (5, 2)])
            self.assertEqual(next(resumed), (1, 0))

        def test_a_stretch_takes_its_steps_and_the_next_carries_on(self):
            torch.manual_seed(0)
            model = SegmentationModel(_StubEncoder(), _StubProbe())
            item = (torch.rand(1, 3, 4, 4), torch.zeros(1, 4, 4, dtype=torch.long), [{}])
            loader = [item] * 5
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            calls = []
            original = optimizer.step
            optimizer.step = lambda *a, **k: (calls.append(1), original(*a, **k))[1]
            loss_fn = lambda logits, masks: torch.nn.functional.cross_entropy(logits, masks)  # noqa: E731
            batches = _training_batches(loader, 0, 2)
            device = torch.device("cpu")
            forward = lambda m, images, masks: m(images)  # noqa: E731
            run = lambda: _train_steps(  # noqa: E731
                model, batches, 2, len(loader), loss_fn, device, optimizer, forward=forward,
                accumulation_steps=2,
            )
            run()
            self.assertEqual(len(calls), 2)
            # Steps three and four: the epoch's lone fifth batch, then two from the next epoch.
            run()
            self.assertEqual(len(calls), 4)
            self.assertEqual(next(batches)[0], 3)

        def test_history_is_cut_back_to_the_resumed_step(self):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "history.csv"
                path.write_text("epoch,train_loss,lr,step\n0.5,1,1,10\n1.0,1,1,20\n1.5,1,1,30\n")
                header = _truncate_history(path, 20, 20)
                self.assertEqual(header, ["epoch", "train_loss", "lr", "step"])
                self.assertEqual(path.read_text().splitlines()[-1], "1.0,1,1,20")
                # Written once an epoch, before the step column: placed by its epoch.
                path.write_text("epoch,train_loss\n1,1\n2,1\n3,1\n")
                _truncate_history(path, 40, 20)
                self.assertEqual(path.read_text().splitlines()[-1], "2,1")


class RefinementModelTests(unittest.TestCase):
        """A model trained to correct the output of another, frozen one."""

        def test_the_first_models_probability_fills_one_plane_and_it_stays_frozen(self):
            first = SegmentationModel(_StubEncoder(), _StubProbe())
            encoder = _StubEncoder()
            model = RefinementModel(first, encoder, _StubProbe(), plane=1).train()
            self.assertFalse(first.training)
            self.assertTrue(model.probe.training)
            self.assertFalse(any(p.requires_grad for p in first.parameters()))

            images = torch.rand(2, 3, 8, 8)
            logits = model(images)
            self.assertEqual(tuple(logits.shape), (2, 2, 8, 8))
            foreground = first(images).softmax(1)[:, 1]
            self.assertTrue(torch.allclose(encoder.seen[:, 1], (foreground - 0.25) / 0.5, atol=1e-6))
            self.assertTrue(torch.equal(encoder.seen[:, 0], images[:, 0]))
            self.assertTrue(torch.equal(encoder.seen[:, 2], images[:, 2]))

            logits.sum().backward()
            self.assertIsNotNone(model.probe.classifier.weight.grad)
            self.assertIsNone(first.probe.classifier.weight.grad)

class FullResolutionUperNetTests(unittest.TestCase):
        """UperNet extended from its stride-4 logits to stride 1 by stages that read the image."""

        def setUp(self):
            try:
                import mmseg.models  # noqa: F401
            except ImportError:
                self.skipTest("needs mmsegmentation")
            from fm_adaptation.models import UperNetDecoder

            self.widths = [96, 192, 384, 768]
            self.decoder = UperNetDecoder(self.widths, 2, full_resolution=True)

        def _features(self, size):
            return tuple(
                torch.rand(2, width, size // stride, size // stride)
                for width, stride in zip(self.widths, (4, 8, 16, 32))
            )

        def test_logits_come_out_at_the_image_size_and_the_skips_train(self):
            stages = self.decoder.full_resolution
            self.assertEqual(sum(p.numel() for p in stages.parameters()), 271_954)
            self.assertTrue(self.decoder.reads_image)

            logits = self.decoder(self._features(64), (64, 64), torch.rand(2, 3, 64, 64))
            self.assertEqual(tuple(logits.shape), (2, 2, 64, 64))
            logits.sum().backward()
            self.assertIsNotNone(stages.skip_full[0][0].weight.grad)
            self.assertIsNotNone(stages.up_half[0].weight.grad)

        def test_an_input_off_a_multiple_of_four_still_meets_its_skips(self):
            logits = self.decoder(self._features(64), (66, 66), torch.rand(2, 3, 66, 66))
            self.assertEqual(tuple(logits.shape), (2, 2, 66, 66))

        def test_it_needs_the_image_and_only_extends_upernet(self):
            with self.assertRaises(ValueError):
                self.decoder(self._features(64), (64, 64))
            with self.assertRaises(ValueError):
                build_model("dinov3", "linear", 2, None, full_resolution=True)

        def test_the_plain_head_is_unchanged(self):
            from fm_adaptation.models import UperNetDecoder

            plain = UperNetDecoder(self.widths, 2)
            self.assertFalse(plain.reads_image)
            self.assertFalse(any(key.startswith("full_resolution") for key in plain.state_dict()))
            self.assertEqual(tuple(plain(self._features(64), (64, 64)).shape), (2, 2, 64, 64))

class UNetTests(unittest.TestCase):
        """The UNet trained from scratch, which predicts at the resolution of its input."""

        def setUp(self):
            try:
                import dynamic_network_architectures  # noqa: F401
            except ImportError:
                self.skipTest("needs nnU-Net's dynamic_network_architectures")

        def test_logits_come_out_at_the_input_size_and_every_part_trains(self):
            model = build_model("unet", "unet", 2, None, train_encoder=True)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 25_948_488)
            self.assertFalse(any(key.startswith("decoder.encoder.") for key in model.probe.state_dict()))

            logits = model(torch.rand(1, 3, 64, 64))
            self.assertEqual(tuple(logits.shape), (1, 2, 64, 64))
            logits.sum().backward()
            self.assertTrue(all(p.grad is not None for p in model.encoder.trunk.parameters()))

        def test_standardised_patches_come_in_as_stored_and_leave_unscaled(self):
            model = build_model("unet", "unet", 2, None, train_encoder=True, input_channels=1, zscore=True)
            patch = np.random.default_rng(0).normal(size=(40, 64, 1)).astype(np.float32)
            image, mask, geometry = model.encoder.preprocess(patch, np.ones((40, 64), np.uint8))
            self.assertEqual(tuple(image.shape), (1, 512, 512))
            top, left = geometry["pad_top"], geometry["pad_left"]
            self.assertTrue(np.array_equal(image[0, top : top + 40, left : left + 64].numpy(), patch[..., 0]))
            self.assertEqual(int(mask[0, 0]), -1)
            self.assertEqual(tuple(model(torch.rand(1, 1, 64, 64)).shape), (1, 2, 64, 64))
            with self.assertRaises(ValueError):
                build_model("dinov3", "linear", 2, None, zscore=True)

        def test_it_takes_no_foundation_model_settings(self):
            for kwargs in (
                {"train_encoder": False},
                {"train_encoder": True, "injector": True},
                {"train_encoder": True, "variant": "convnextt"},
            ):
                with self.assertRaises(ValueError):
                    build_model("unet", "unet", 2, None, **kwargs)
            with self.assertRaises(ValueError):
                build_model("unet", "linear", 2, None, train_encoder=True)
            with self.assertRaises(ValueError):
                build_model("unet", "unet", 2, "weights.pth", train_encoder=True)


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
