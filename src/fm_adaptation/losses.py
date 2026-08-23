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


class DistillationLoss(nn.Module):
    """The teacher's dense predictions alongside the ground truth.

    The teacher's logits are cached at its head's own stride-4 resolution, because that is all it ever
    computed -- `UperNetDecoder` bilinearly upsamples that to the image size, so re-doing the same
    interpolation here reproduces the teacher's full-resolution output exactly while storing 16x less.

    `T**2` is Hinton's scaling: softening the distributions shrinks the KD gradient by roughly `1/T**2`,
    so without it the balance against the supervised term would move every time `temperature` did.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.supervised = DiceCrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor, teacher_logits: torch.Tensor):
        supervised = self.supervised(logits, target)
        teacher = F.interpolate(
            teacher_logits.float(), size=logits.shape[-2:], mode="bilinear", align_corners=False
        )
        temperature = self.temperature
        student_log = F.log_softmax(logits / temperature, dim=1)
        teacher_probabilities = F.softmax(teacher / temperature, dim=1)
        divergence = F.kl_div(student_log, teacher_probabilities, reduction="none").sum(dim=1, keepdim=True)
        # Padding carries nothing worth matching, and the supervised term already ignores it.
        valid = (target != -1).unsqueeze(1)
        distillation = (divergence * valid).sum() / valid.sum().clamp(min=1) * temperature**2
        total = (1.0 - self.alpha) * supervised + self.alpha * distillation
        return total, supervised, distillation


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
