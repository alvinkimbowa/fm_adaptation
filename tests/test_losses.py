import unittest

import torch

from fm_adaptation.losses import (DiceCrossEntropyLoss, DiceCrossEntropySkeletonRecallLoss,
                                  SkeletonRecallLoss, distance_weights, tubed_skeleton)


def _disc(size=32, radius=9):
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    centre = (size - 1) / 2
    return (((ys - centre) ** 2 + (xs - centre) ** 2) <= radius ** 2).long()[None]


def _logits_for(target, classes=2, confidence=10.0):
    one_hot = torch.nn.functional.one_hot(target.clamp(min=0), classes)
    return (one_hot.permute(0, 3, 1, 2).float() * 2 - 1) * confidence


class SkeletonTests(unittest.TestCase):
        """The tubed skeletonisation the recall term is taken over."""

        def test_skeleton_is_thinner_than_the_shape_and_stays_inside_it(self):
            target = _disc()
            skeleton = tubed_skeleton(target)
            self.assertLess(skeleton.sum().item(), target.sum().item())
            self.assertTrue(bool(((skeleton > 0) <= (target > 0)).all()))

        def test_tube_widens_but_a_thin_label_survives_unchanged(self):
            target = _disc()
            self.assertGreater(tubed_skeleton(target).sum().item(),
                               tubed_skeleton(target, tube=False).sum().item())
            line = torch.zeros(1, 32, 32, dtype=torch.long)
            line[0, 16, 4:28] = 1
            self.assertTrue(torch.equal(tubed_skeleton(line), line))

        def test_empty_and_ignored_targets_give_an_empty_skeleton(self):
            self.assertEqual(tubed_skeleton(torch.zeros(1, 16, 16, dtype=torch.long)).sum().item(), 0)
            self.assertEqual(tubed_skeleton(torch.full((1, 16, 16), -1)).sum().item(), 0)

        def test_classes_are_kept(self):
            target = _disc() * 2
            self.assertEqual(set(tubed_skeleton(target).unique().tolist()), {0, 2})


class SkeletonRecallTests(unittest.TestCase):
        """The loss term itself, and the compound loss it is added to."""

        def test_a_perfect_prediction_reaches_full_recall(self):
            target = _disc()
            loss = SkeletonRecallLoss()(_logits_for(target), target)
            self.assertAlmostEqual(loss.item(), -1.0, places=3)

        def test_a_break_in_the_structure_costs_recall(self):
            target = torch.zeros(1, 32, 32, dtype=torch.long)
            target[0, 14:18, 4:28] = 1
            broken = target.clone()
            broken[0, :, 15:17] = 0
            whole = SkeletonRecallLoss()(_logits_for(target), target)
            self.assertGreater(SkeletonRecallLoss()(_logits_for(broken), target).item(), whole.item())

        def test_zero_weight_leaves_the_generic_loss_untouched(self):
            target = _disc()
            logits = _logits_for(target, confidence=1.0)
            self.assertAlmostEqual(DiceCrossEntropySkeletonRecallLoss(0.0)(logits, target).item(),
                                   DiceCrossEntropyLoss()(logits, target).item(), places=6)

        def test_the_weighted_term_is_added_at_its_weight(self):
            target = _disc()
            logits = _logits_for(target, confidence=1.0)
            expected = (DiceCrossEntropyLoss()(logits, target)
                        + 0.5 * SkeletonRecallLoss()(logits, target))
            self.assertAlmostEqual(DiceCrossEntropySkeletonRecallLoss(0.5)(logits, target).item(),
                                   expected.item(), places=6)


class DistanceWeightTests(unittest.TestCase):
        """The per-pixel weight map the cross entropy is taken over."""

        def setUp(self):
            self.target = torch.zeros(1, 64, 64, dtype=torch.long)
            self.target[0, 32, 8:56] = 1

        def test_weights_fall_with_distance_and_stop_at_the_floor(self):
            weights = distance_weights(self.target, tau=5.0, floor=0.1)[0]
            self.assertGreater(weights[32, 32].item(), weights[36, 32].item())
            self.assertGreater(weights[36, 32].item(), weights[45, 32].item())
            self.assertAlmostEqual(weights[0, 0].item() / weights[32, 32].item(), 0.1, places=2)

        def test_the_mean_over_scored_pixels_is_one(self):
            weights = distance_weights(self.target, tau=5.0, floor=0.1)
            self.assertAlmostEqual(weights.mean().item(), 1.0, places=5)

        def test_ignored_pixels_are_left_out_of_the_normalisation(self):
            target = self.target.clone()
            target[0, 48:, :] = -1
            weights = distance_weights(target, tau=5.0, floor=0.1)
            self.assertAlmostEqual(weights[target != -1].mean().item(), 1.0, places=5)

        def test_a_target_without_annotation_weights_every_pixel_alike(self):
            weights = distance_weights(torch.zeros(1, 16, 16, dtype=torch.long), tau=5.0, floor=0.1)
            self.assertTrue(torch.allclose(weights, torch.ones_like(weights)))

        def test_tau_of_zero_leaves_the_loss_unweighted(self):
            logits = _logits_for(self.target, confidence=1.0)
            self.assertAlmostEqual(DiceCrossEntropyLoss(0.0)(logits, self.target).item(),
                                   DiceCrossEntropyLoss()(logits, self.target).item(), places=6)

        def test_an_error_beside_the_structure_costs_more_than_one_far_away(self):
            # A false positive on background, once two pixels from the tracing and once far from it.
            near, far = _logits_for(self.target), _logits_for(self.target)
            near[0, :, 34, 32] = torch.tensor([-10.0, 10.0])
            far[0, :, 60, 32] = torch.tensor([-10.0, 10.0])
            loss = DiceCrossEntropyLoss(distance_tau=5.0)
            self.assertGreater(loss(near, self.target).item(), loss(far, self.target).item())
            unweighted = DiceCrossEntropyLoss()
            self.assertAlmostEqual(unweighted(near, self.target).item(),
                                   unweighted(far, self.target).item(), places=6)

        def test_the_skeleton_loss_carries_the_weighting_through(self):
            logits = _logits_for(self.target)
            logits[0, :, 34, 32] = torch.tensor([-10.0, 10.0])
            weighted = DiceCrossEntropySkeletonRecallLoss(1.0, distance_tau=5.0)(logits, self.target)
            plain = DiceCrossEntropySkeletonRecallLoss(1.0)(logits, self.target)
            self.assertGreater(weighted.item(), plain.item())


if __name__ == "__main__":
    unittest.main()
