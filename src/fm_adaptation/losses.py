import cv2
import numpy as np
import torch
from skimage.morphology import dilation, skeletonize
from torch import nn
from torch.nn import functional as F


def distance_weights(target: torch.Tensor, tau: float, floor: float) -> torch.Tensor:
    """Per-pixel weights that fall off with distance to the nearest annotated pixel.

    `floor + (1 - floor) * exp(-d / tau)`, normalised so the mean over the pixels the loss scores is
    1, which keeps the weighted loss on the same scale as the unweighted one. A centreline annotation
    says nothing about how wide the structure is, so the pixels beside it are where the decision is
    genuinely uncertain; `tau` is how many pixels wide that band is.
    """
    valid = target != -1
    background = (target <= 0).to(torch.uint8).cpu().numpy() * 255
    weights = np.empty(background.shape, dtype=np.float32)
    for index, plane in enumerate(background):
        distance = cv2.distanceTransform(plane, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        weights[index] = floor + (1.0 - floor) * np.exp(-distance / tau)
    weights = torch.from_numpy(weights).to(device=target.device)
    mean = weights[valid].mean() if valid.any() else weights.new_tensor(1.0)
    return weights / mean


class DiceCrossEntropyLoss(nn.Module):
    """Equal-weight cross entropy and foreground soft Dice loss.

    `distance_tau` above zero weights the cross entropy by `distance_weights`; the Dice term stays a
    plain overlap either way, so it remains comparable between a weighted run and an unweighted one.
    """

    def __init__(self, distance_tau: float = 0.0, distance_floor: float = 0.1):
        super().__init__()
        self.distance_tau = distance_tau
        self.distance_floor = distance_floor

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != -1
        if self.distance_tau:
            weights = distance_weights(target, self.distance_tau, self.distance_floor)
            per_pixel = F.cross_entropy(logits, target, ignore_index=-1, reduction="none")
            ce = (per_pixel * weights * valid).sum() / valid.sum().clamp(min=1)
        else:
            ce = F.cross_entropy(logits, target, ignore_index=-1)
        probabilities = logits.softmax(dim=1)
        dice_target = torch.where(valid, target, 0)
        one_hot = F.one_hot(dice_target, logits.shape[1]).permute(0, 3, 1, 2).float()
        valid = valid[:, None]
        probabilities = probabilities[:, 1:] * valid
        one_hot = one_hot[:, 1:]
        reduce_dims = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(reduce_dims)
        denominator = probabilities.sum(reduce_dims) + one_hot.sum(reduce_dims)
        dice = (2 * intersection + 1e-5) / (denominator + 1e-5)
        return ce - dice.mean()


def tubed_skeleton(target: torch.Tensor, tube: bool = True) -> torch.Tensor:
    """The class-labelled skeleton of `target`, optionally widened into a two-pixel tube.

    Skeleton Recall Loss (Kirchhoff et al., ECCV 2024) takes the skeleton of the annotation rather
    than of the prediction, so it needs no differentiable skeletonisation: binarise, thin, widen, then
    multiply by the labels again to recover the classes. The multiplication also clips the tube back
    inside the annotation, so a tracing narrower than the tube comes back unchanged -- on a
    centreline-width label the result is the label itself.
    """
    labels = torch.where(target > 0, target, torch.zeros_like(target)).to(torch.int16).cpu().numpy()
    skeletons = np.zeros_like(labels)
    for index, label in enumerate(labels):
        if not label.any():
            continue
        skeleton = skeletonize(label > 0).astype(np.int16)
        if tube:
            skeleton = dilation(dilation(skeleton))
        skeletons[index] = skeleton * label
    return torch.from_numpy(skeletons).to(device=target.device, dtype=target.dtype)


class SkeletonRecallLoss(nn.Module):
    """Soft recall of the prediction over the annotation's skeleton, foreground classes only.

    Recall alone is what makes this a connectivity term: a break in a thin structure costs the whole
    skeleton that runs through it, while the Dice and cross-entropy terms it is added to are what
    still hold the prediction back from covering everything.
    """

    def __init__(self, tube: bool = True):
        super().__init__()
        self.tube = tube

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != -1
        skeleton = tubed_skeleton(target, self.tube)
        one_hot = F.one_hot(skeleton, logits.shape[1]).permute(0, 3, 1, 2).float()[:, 1:]
        probabilities = logits.softmax(dim=1)[:, 1:] * valid[:, None]
        reduce_dims = (0, 2, 3)
        recall = ((probabilities * one_hot).sum(reduce_dims) + 1e-5) / (one_hot.sum(reduce_dims) + 1e-5)
        return -recall.mean()


class DiceCrossEntropySkeletonRecallLoss(nn.Module):
    """`DiceCrossEntropyLoss` with the skeleton recall term added at `weight`.

    The paper's own combination, and the weights the reference implementation ships with: cross
    entropy and Dice at 1 each, the recall term at `w`, which it only ever evaluates at 0.1 and 1.0.
    """

    def __init__(self, weight: float, tube: bool = True, distance_tau: float = 0.0,
                 distance_floor: float = 0.1):
        super().__init__()
        self.weight = weight
        self.generic = DiceCrossEntropyLoss(distance_tau, distance_floor)
        self.skeleton = SkeletonRecallLoss(tube)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.generic(logits, target) + self.weight * self.skeleton(logits, target)


def build_loss(cfg) -> nn.Module:
    """The loss a run's config asks for: Dice and cross entropy, plus whatever is switched on."""
    if cfg.skeleton_recall_weight:
        return DiceCrossEntropySkeletonRecallLoss(
            cfg.skeleton_recall_weight, cfg.skeleton_tube,
            cfg.distance_weight_tau, cfg.distance_weight_floor,
        )
    return DiceCrossEntropyLoss(cfg.distance_weight_tau, cfg.distance_weight_floor)


def mean_foreground_dice(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = logits.argmax(dim=1)
    valid = target != -1
    scores = []
    for label in range(1, logits.shape[1]):
        pred = (prediction == label) & valid
        truth = target == label
        denominator = pred.sum() + truth.sum()
        if denominator > 0:
            scores.append((2 * (pred & truth).sum() / denominator).item())
    return float(sum(scores) / len(scores)) if scores else float("nan")
