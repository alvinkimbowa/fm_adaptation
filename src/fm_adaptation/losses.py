import torch
from torch import nn
from torch.nn import functional as F


class DiceCrossEntropyLoss(nn.Module):
    """Equal-weight cross entropy and foreground soft Dice loss."""

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != -1
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
