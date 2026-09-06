"""MMDetection Mask2Former with a thin-fiber, ignore-aware set criterion."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


def sample_indices(targets, valid, logits, num_points=4096):
    """Every fiber contributes positive support; remaining points are random/uncertain.

    Integer pixel centers avoid blending ignored pixels into supervised samples.
    The same coordinates are used for every pair in Hungarian matching.
    """
    available = valid.flatten().nonzero().flatten()
    if not len(available):
        return available
    positives = []
    for mask in targets:
        indices = (mask & valid).flatten().nonzero().flatten()
        if len(indices):
            positives.append(indices[torch.randperm(len(indices), device=indices.device)[:16]])
    support = torch.cat(positives) if positives else available[:0]
    n = max(0, num_points-len(support))
    random = available[torch.randint(len(available), (max(1, n//4),), device=available.device)]
    candidates = available[torch.randint(len(available), (max(1, n*3),), device=available.device)]
    uncertainty = logits.detach().flatten(1)[:, candidates].abs().amin(0)
    uncertain = candidates[uncertainty.topk(min(len(candidates), n-len(random)), largest=False).indices] if n > len(random) else candidates[:0]
    return torch.unique(torch.cat((support, random, uncertain)))


def layer_loss(classes, logits, target, num_points=4096):
    device = logits.device
    truth = torch.as_tensor(target['masks'], device=device, dtype=torch.bool)
    ignore = torch.as_tensor(target['ignore'], device=device, dtype=torch.bool)
    if len(truth) > len(logits):
        raise ValueError(f'Query budget overflow: {len(truth)} targets, {len(logits)} queries')
    full = F.interpolate(logits[:, None], size=ignore.shape, mode='bilinear', align_corners=False)[:, 0]
    indices = sample_indices(truth, ~ignore, full, num_points)
    if not len(indices):
        return (classes.sum()+logits.sum())*0
    sampled = full.flatten(1)[:, indices]
    gt = truth.flatten(1)[:, indices].float()
    prob = sampled.sigmoid()
    with torch.no_grad():
        cls_cost = -classes.softmax(-1)[:, :1].expand(-1, len(truth))
        bce = (F.softplus(sampled).sum(1)[:, None] - sampled @ gt.T) / len(indices)
        dice = 1-(2*prob @ gt.T+1)/(prob.sum(1)[:, None]+gt.sum(1)[None]+1)
        cost = 2*cls_cost+5*bce+5*dice
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    labels = torch.ones(len(classes), device=device, dtype=torch.long)
    labels[rows] = 0
    weights = torch.ones(len(classes), device=device)
    # Empty queries are supervised as no-object. Only nonempty predictions wholly
    # confined to ignored fragments lose no-object supervision.
    with torch.no_grad():
        binary = full.sigmoid() > .5
        confined = binary.flatten(1).any(1) & ~(binary & ~ignore).flatten(1).any(1)
        weights[confined] = 0
        weights[rows] = 1
    ce_weights = torch.where(labels == 0, 1., .1)*weights
    cls_loss = (F.cross_entropy(classes, labels, reduction='none')*ce_weights).sum()/ce_weights.sum().clamp_min(1)
    if len(rows):
        pred, matched = sampled[rows], gt[cols]
        mask_loss = F.binary_cross_entropy_with_logits(pred, matched)
        p = pred.sigmoid()
        dice_loss = (1-(2*(p*matched).sum(1)+1)/(p.sum(1)+matched.sum(1)+1)).mean()
    else:
        mask_loss = dice_loss = sampled.sum()*0
    return 2*cls_loss+5*mask_loss+5*dice_loss


class InstanceModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from mmengine.config import ConfigDict
        from mmdet.utils import register_all_modules
        register_all_modules(init_default_scope=True)
        from mmdet.models.dense_heads import Mask2FormerHead
        from .models import DINOv3AdapterEncoder, DINOv3ConvNeXtEncoder
        from .dinov3_mmseg import mask2former_cfg
        if cfg.variant.startswith('convnext'):
            self.encoder = DINOv3ConvNeXtEncoder(cfg.checkpoint, cfg.train_encoder, cfg.variant)
        else:
            self.encoder = DINOv3AdapterEncoder(cfg.checkpoint, cfg.train_encoder, cfg.injector, cfg.variant)
        self.encoder.input_size = cfg.input_size
        head = mask2former_cfg(1, (cfg.input_size, cfg.input_size))['decode_head']
        for key in ('type', 'num_classes', 'align_corners', 'strides'):
            head.pop(key)
        channels = self.encoder.feature_channels
        head.update(in_channels=channels if isinstance(channels, list) else [channels]*4,
                    num_things_classes=1,num_stuff_classes=0,num_queries=cfg.num_queries,train_cfg=None)
        self.head = Mask2FormerHead(**ConfigDict(head))
        self.head.init_weights()
        self.input_size = cfg.input_size

    def forward(self, images):
        if images.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError(f'Expected native {self.input_size}x{self.input_size}, got {images.shape}')
        return self.head(self.encoder(images.float()), [])

    def loss(self, outputs, targets):
        classes, masks = outputs
        return sum(layer_loss(c[i], m[i], t) for c, m in zip(classes, masks)
                   for i, t in enumerate(targets))/len(targets)

    @torch.no_grad()
    def predict(self, images, score_threshold=.05):
        classes, masks = self(images)
        probs = F.interpolate(masks[-1], size=images.shape[-2:], mode='bilinear', align_corners=False).sigmoid()
        output = []
        for cls, prob in zip(classes[-1], probs):
            binary = prob > .5
            scores = cls.softmax(-1)[:, 0]*(prob*binary).flatten(1).sum(1)/binary.flatten(1).sum(1).clamp_min(1)
            keep = (scores >= score_threshold) & binary.flatten(1).any(1)
            output.append(dict(probabilities=prob[keep].cpu().numpy(), scores=scores[keep].cpu().numpy(),
                               query_ids=keep.nonzero().flatten().cpu().numpy(),
                               query_saturation=int(keep.sum()) == len(keep), num_queries=len(keep)))
        return output


def validate_config(cfg):
    if cfg.task != 'instance' or cfg.probe_name != 'mask2former':
        raise ValueError('Instance training requires model.task: instance and probe: mask2former')
    if cfg.input_size != 256 or cfg.patching is None or cfg.patching.patch_size != cfg.input_size:
        raise ValueError('This instance experiment requires native 256x256 patches')
    if cfg.stains != ('SMI',) or not cfg.patching.respect_channels or cfg.balance_sources:
        raise ValueError('Require stains: [SMI], respect_channels: true, balance_sources: false')
    if cfg.skeleton_recall_weight or cfg.distance_weight_tau:
        raise ValueError('Instance losses cannot include skeleton or distance weights')
    if cfg.augment and (cfg.augment.zoom_min != 1 or cfg.augment.zoom_max != 1):
        raise ValueError('Instance augmentation does not support zoom')
