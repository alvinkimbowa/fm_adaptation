import tempfile
import unittest

import cv2
import numpy as np
import torch

from fm_adaptation.config import AugmentConfig, ContextConfig
from fm_adaptation.data import NnUNet2DDataset, collate_cases, context_batch, context_frames
from fm_adaptation.models import ContextAttention, ContextConv, ContextFusion, SegmentationModel
from fixtures import DatasetFixture, preprocess


class WindowTests(unittest.TestCase):
    """Which frames of a video a case is shown with."""

    def test_a_case_in_the_middle_is_centred(self):
        self.assertEqual(context_frames(50, 60, 3, 2), [44, 46, 48, 52, 54, 56])

    def test_the_first_frame_takes_all_of_them_from_after(self):
        self.assertEqual(context_frames(0, 60, 3, 2), [2, 4, 6, 8, 10, 12])

    def test_the_last_frame_takes_all_of_them_from_before(self):
        self.assertEqual(context_frames(59, 60, 3, 2), [47, 49, 51, 53, 55, 57])

    def test_a_short_side_is_made_up_from_the_other(self):
        self.assertEqual(context_frames(3, 60, 3, 2), [1, 5, 7, 9, 11, 13])
        self.assertEqual(context_frames(57, 60, 3, 2), [47, 49, 51, 53, 55, 59])

    def test_every_frame_of_the_shortest_video_gets_a_full_window(self):
        for frame in range(42):
            frames = context_frames(frame, 42, 3, 2)
            self.assertEqual(len(set(frames)), 6)
            self.assertNotIn(frame, frames)
            self.assertTrue(all(0 <= index < 42 for index in frames))
            self.assertTrue(all((index - frame) % 2 == 0 for index in frames))

    def test_a_video_too_short_for_the_window_is_refused(self):
        with self.assertRaises(ValueError):
            context_frames(2, 8, 3, 2)


class DatasetTests(unittest.TestCase):
    """A case arriving with its context frames."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = DatasetFixture(self.tmp.name, "Dataset001_video", {"0": "R", "1": "G", "2": "B"})
        (self.data.path / "frames").mkdir()
        # Every frame is flat at its own index, with one bright corner that a flip would move.
        for index in range(20):
            frame = np.full((16, 16, 3), index, dtype=np.uint8)
            frame[:4, :4] = 200
            cv2.imwrite(str(self.data.path / "frames" / f"clip_{index:03d}.png"), frame)
        label = np.zeros((16, 16), dtype=np.uint8)
        label[6:10, 6:10] = 1
        for index in (0, 10):
            frame = cv2.imread(str(self.data.path / "frames" / f"clip_{index:03d}.png"))
            self.data.add(f"clip_{index:03d}", color=frame, label=label)
        self.data.split(["clip_000", "clip_010"])

    def dataset(self, context, augment=None):
        return NnUNet2DDataset(
            self.tmp.name, self.data.name, "Tr", "0", "train", preprocess,
            augment=augment, context=context,
        )

    def test_the_case_comes_first_and_its_context_in_time_order(self):
        image, _, metadata = self.dataset(ContextConfig(per_side=2, stride=3))[1]
        self.assertEqual(tuple(image.shape), (5, 3, 16, 16))
        self.assertEqual(image[:, 0, 8, 8].tolist(), [10, 4, 7, 13, 16])
        self.assertEqual(metadata["context_steps"], (-2, -1, 1, 2))

    def test_a_case_at_the_start_reports_the_steps_it_was_given(self):
        image, _, metadata = self.dataset(ContextConfig(per_side=2, stride=3))[0]
        self.assertEqual(image[:, 0, 8, 8].tolist(), [0, 3, 6, 9, 12])
        self.assertEqual(metadata["context_steps"], (1, 2, 3, 4))

    def test_copies_show_the_case_in_place_of_every_context_frame(self):
        image, _, metadata = self.dataset(ContextConfig(per_side=2, stride=3, copies=True))[1]
        self.assertEqual(image[:, 0, 8, 8].tolist(), [10] * 5)
        self.assertEqual(metadata["context_steps"], (-2, -1, 1, 2))

    def test_one_augmentation_is_drawn_for_all_the_frames(self):
        augment = AugmentConfig(hflip=True, vflip=True, flip_p=0.5)
        dataset = self.dataset(ContextConfig(per_side=2, stride=3), augment=augment)
        torch.manual_seed(0)
        for _ in range(20):
            image, _, _ = dataset[1]
            corners = (image[:, 0] == 200).flatten(1)
            self.assertTrue((corners == corners[0]).all())

    def test_a_batch_carries_its_steps(self):
        dataset = self.dataset(ContextConfig(per_side=2, stride=3))
        images, _, metadata = collate_cases([dataset[0], dataset[1]])
        self.assertEqual(tuple(images.shape), (2, 5, 3, 16, 16))
        self.assertEqual(context_batch(metadata).tolist(), [[1, 2, 3, 4], [-2, -1, 1, 2]])

    def test_a_dataset_without_frames_says_so(self):
        for path in (self.data.path / "frames").iterdir():
            path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset(ContextConfig())


class FusionTests(unittest.TestCase):
    """The attention between a case's features and its context's."""

    def test_a_fresh_block_is_the_identity(self):
        block = ContextAttention(16, frames=4, heads=4)
        target = torch.randn(2, 16, 6, 5)
        fused = block(target, torch.randn(2, 4, 16, 3, 3), torch.tensor([[-2, -1, 1, 2]] * 2))
        self.assertTrue(torch.equal(fused, target))

    def test_a_trained_block_reads_the_context_and_its_order(self):
        block = ContextAttention(16, frames=4, heads=4)
        torch.nn.init.normal_(block.out.weight)
        target, context = torch.randn(1, 16, 6, 5), torch.randn(1, 4, 16, 3, 3)
        steps = torch.tensor([[-2, -1, 1, 2]])
        fused = block(target, context, steps)
        self.assertFalse(torch.allclose(fused, block(target, torch.randn_like(context), steps)))
        self.assertFalse(torch.allclose(fused, block(target, context, steps + 1)))

    def test_a_fresh_convolution_block_is_the_identity(self):
        block = ContextConv(16, frames=4)
        target = torch.randn(2, 16, 6, 5)
        fused = block(target, torch.randn(2, 4, 16, 6, 5), torch.tensor([[-2, -1, 1, 2]] * 2))
        self.assertTrue(torch.equal(fused, target))

    def test_a_trained_convolution_block_reads_the_context_and_its_steps(self):
        block = ContextConv(16, frames=4)
        torch.nn.init.normal_(block.out.weight)
        target, context = torch.randn(1, 16, 6, 5), torch.randn(1, 4, 16, 6, 5)
        steps = torch.tensor([[-2, -1, 1, 2]])
        fused = block(target, context, steps)
        self.assertFalse(torch.allclose(fused, block(target, torch.randn_like(context), steps)))
        self.assertFalse(torch.allclose(fused, block(target, context, steps + 1)))

    def test_late_fuses_the_coarsest_level_and_intermediate_all_of_them(self):
        channels = [8, 16, 32, 64]
        features = [torch.randn(2, width, 32 >> level, 32 >> level)
                    for level, width in enumerate(channels)]
        context = [torch.randn(2 * 4, *feature.shape[1:]) for feature in features]
        steps = torch.tensor([[-2, -1, 1, 2]] * 2)
        for operator in ("attention", "conv"):
            for fusion, levels in (("late", {3}), ("intermediate", {0, 1, 2, 3})):
                module = ContextFusion(
                    channels, ContextConfig(per_side=2, fusion=fusion, operator=operator)
                )
                for block in module.blocks.values():
                    torch.nn.init.normal_(block.out.weight)
                fused = module(features, module.keep(context), steps)
                changed = {
                    level for level in range(4) if not torch.equal(fused[level], features[level])
                }
                self.assertEqual(changed, levels)
                self.assertEqual([f.shape for f in fused], [f.shape for f in features])


class _Pyramid(torch.nn.Module):
    """A four-level encoder small enough to run in a test."""

    def __init__(self):
        super().__init__()
        self.levels = torch.nn.ModuleList(
            torch.nn.Conv2d(3, 8, 2 ** (level + 1), 2 ** (level + 1)) for level in range(4)
        )

    def forward(self, images):
        return tuple(level(images) for level in self.levels)


class _Head(torch.nn.Module):
    def forward(self, features, output_size, prompt=None):
        return sum(feature.mean((1, 2, 3)) for feature in features)


class GradientTests(unittest.TestCase):
    """Whether the encoder is trained through the context frames."""

    def model(self, gradient):
        context = ContextConfig(per_side=1, fusion="intermediate", operator="conv", gradient=gradient)
        fusion = ContextFusion([8] * 4, context)
        for block in fusion.blocks.values():
            torch.nn.init.normal_(block.out.weight)
        return SegmentationModel(_Pyramid(), _Head(), context=fusion, context_gradient=gradient)

    def frame_gradients(self, model):
        images = torch.randn(2, 3, 3, 32, 32, requires_grad=True)
        model(images, None, torch.tensor([[-1, 1]] * 2)).sum().backward()
        return images.grad.abs().sum((0, 2, 3, 4))

    def test_without_it_only_the_case_reaches_the_encoder(self):
        case, before, after = self.frame_gradients(self.model(gradient=False))
        self.assertGreater(case, 0)
        self.assertEqual((before, after), (0, 0))

    def test_with_it_every_frame_does(self):
        self.assertTrue((self.frame_gradients(self.model(gradient=True)) > 0).all())

    def test_it_changes_nothing_about_what_is_computed(self):
        frozen, trained = self.model(gradient=False), self.model(gradient=True)
        trained.load_state_dict(frozen.state_dict())
        images, steps = torch.randn(2, 3, 3, 32, 32), torch.tensor([[-1, 1]] * 2)
        self.assertTrue(torch.allclose(frozen(images, None, steps), trained(images, None, steps)))


if __name__ == "__main__":
    unittest.main()
