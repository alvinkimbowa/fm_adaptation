"""Register DINOv3's ViT-Adapter as an mmsegmentation backbone.

Nothing here reimplements a head: the adapter comes from DINOv3 unmodified and the decode heads
(`UPerHead`, `Mask2FormerHead`) come from mmsegmentation, so their losses and training loop are the
upstream ones. This module only bridges the two and adapts our nnU-Net-layout datasets.
"""

import sys
from pathlib import Path

import torch
from mmengine.model import BaseModule
from mmseg.registry import DATASETS, MODELS

REPO = Path(__file__).resolve().parents[2]
DINOV3_ROOT = REPO / "foundational_models" / "dinov3"
DEFAULT_CHECKPOINT = DINOV3_ROOT / "ckpts" / "dinov3_vitl16_pretrain_lvd1689m.pth"
# The four blocks DINOv3 taps for ViT-L/16, per its own segmentation config.
INTERACTION_INDEXES = [4, 11, 17, 23]
EMBED_DIM = 1024
# Cross-entropy alone collapses to all-background on targets as sparse as the neurites (~0.7% of
# pixels), and mmseg's DiceLoss flattens every class into one global Dice, which at 99.3% background
# reads ~0.997 however bad the foreground is. `ignore_index=0` drops the background channel, leaving a
# foreground-only Dice -- the same quantity the SAM3 arms optimise, so the comparison stays fair.
DECODE_LOSSES = [
    {"type": "CrossEntropyLoss", "use_sigmoid": False, "loss_name": "loss_ce", "loss_weight": 1.0},
    {"type": "DiceLoss", "loss_name": "loss_dice", "loss_weight": 3.0,
     "use_sigmoid": False, "ignore_index": 0},
]


@MODELS.register_module()
class DINOv3Adapter(BaseModule):
    """Frozen DINOv3 ViT-L/16 plus DINOv3's trainable adapter, emitting strides 4/8/16/32."""

    def __init__(self, checkpoint=None, interaction_indexes=None, injector=False, init_cfg=None):
        super().__init__(init_cfg)
        sys.path.insert(0, str(DINOV3_ROOT))
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
        from dinov3.hub.backbones import dinov3_vitl16

        backbone = dinov3_vitl16(pretrained=False)
        state = torch.load(Path(checkpoint or DEFAULT_CHECKPOINT), map_location="cpu", weights_only=True)
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"DINOv3 checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
        backbone.requires_grad_(False)
        self.injector = injector
        if injector:
            from .dinov3_injector import DINOv3AdapterWithInjector as adapter_cls
        else:
            adapter_cls = DINOv3_Adapter
        self.adapter = adapter_cls(
            backbone, interaction_indexes=interaction_indexes or INTERACTION_INDEXES
        )

    def train(self, mode=True):
        super().train(mode)
        self.adapter.backbone.eval()  # the ViT stays frozen in eval, as DINOv3 intends
        return self

    def forward(self, x):
        features = self.adapter(x)
        return tuple(features[key] for key in ("1", "2", "3", "4"))


def _dataset_class():
    from mmseg.datasets import BaseSegDataset

    @DATASETS.register_module()
    class NnUNetSegDataset(BaseSegDataset):
        """nnU-Net layout: images<split>/<case>_0000<ext> and labels<split>/<case><ext>."""

        METAINFO = {"classes": ("background", "foreground"), "palette": [[0, 0, 0], [255, 0, 0]]}

        def __init__(self, case_ids=None, ext=".png", classes=None, **kwargs):
            self._case_ids = list(case_ids or [])
            if classes:
                self.METAINFO = {
                    "classes": tuple(classes),
                    "palette": [[i * 40 % 256, (i * 90) % 256, (i * 150) % 256] for i in range(len(classes))],
                }
            super().__init__(img_suffix=f"_0000{ext}", seg_map_suffix=ext, **kwargs)

        def load_data_list(self):
            items = []
            for case_id in self._case_ids:
                items.append(
                    {
                        "img_path": str(Path(self.data_prefix["img_path"]) / f"{case_id}{self.img_suffix}"),
                        "seg_map_path": str(Path(self.data_prefix["seg_map_path"]) / f"{case_id}{self.seg_map_suffix}"),
                        "label_map": self.label_map,
                        "reduce_zero_label": self.reduce_zero_label,
                        "seg_fields": [],
                    }
                )
            return items

    return NnUNetSegDataset


def backbone_cfg(checkpoint=None, injector=False):
    return {
        "type": "DINOv3Adapter",
        "checkpoint": str(checkpoint) if checkpoint else None,
        "injector": injector,
    }


def upernet_cfg(num_classes, crop_size, checkpoint=None, injector=False):
    """Stock mmseg UPerHead on the adapter's four scales."""
    return {
        "type": "EncoderDecoder",
        "data_preprocessor": _preprocessor(crop_size),
        "backbone": backbone_cfg(checkpoint, injector),
        "decode_head": {
            "type": "UPerHead",
            "in_channels": [EMBED_DIM] * 4,
            "in_index": [0, 1, 2, 3],
            "pool_scales": (1, 2, 3, 6),
            "channels": 512,
            "dropout_ratio": 0.1,
            "num_classes": num_classes,
            "norm_cfg": {"type": "BN", "requires_grad": True},
            "align_corners": False,
            "loss_decode": DECODE_LOSSES,
        },
        "auxiliary_head": {
            "type": "FCNHead",
            "in_channels": EMBED_DIM,
            "in_index": 2,
            "channels": 256,
            "num_convs": 1,
            "concat_input": False,
            "dropout_ratio": 0.1,
            "num_classes": num_classes,
            "norm_cfg": {"type": "BN", "requires_grad": True},
            "align_corners": False,
            "loss_decode": [dict(loss, loss_weight=loss["loss_weight"] * 0.4) for loss in DECODE_LOSSES],
        },
        "train_cfg": {},
        "test_cfg": {"mode": "whole"},
    }


def mask2former_cfg(num_classes, crop_size, checkpoint=None, injector=False):
    """Stock mmseg Mask2FormerHead, which brings its own matcher and set criterion."""
    return {
        "type": "EncoderDecoder",
        "data_preprocessor": _preprocessor(crop_size),
        "backbone": backbone_cfg(checkpoint, injector),
        "decode_head": {
            "type": "Mask2FormerHead",
            "in_channels": [EMBED_DIM] * 4,
            "strides": [4, 8, 16, 32],
            "feat_channels": 256,
            "out_channels": 256,
            "num_classes": num_classes,
            "num_queries": 100,
            "num_transformer_feat_level": 3,
            "align_corners": False,
            "pixel_decoder": {
                "type": "mmdet.MSDeformAttnPixelDecoder",
                "num_outs": 3,
                "norm_cfg": {"type": "GN", "num_groups": 32},
                "act_cfg": {"type": "ReLU"},
                "encoder": {
                    "num_layers": 6,
                    "layer_cfg": {
                        "self_attn_cfg": {
                            "embed_dims": 256,
                            "num_heads": 8,
                            "num_levels": 3,
                            "num_points": 4,
                            "dropout": 0.0,
                            "batch_first": True,
                        },
                        "ffn_cfg": {
                            "embed_dims": 256,
                            "feedforward_channels": 1024,
                            "num_fcs": 2,
                            "ffn_drop": 0.0,
                            "act_cfg": {"type": "ReLU", "inplace": True},
                        },
                    },
                },
                "positional_encoding": {"num_feats": 128, "normalize": True},
            },
            "positional_encoding": {"num_feats": 128, "normalize": True},
            "transformer_decoder": {
                "return_intermediate": True,
                "num_layers": 9,
                "layer_cfg": {
                    "self_attn_cfg": {
                        "embed_dims": 256,
                        "num_heads": 8,
                        "dropout": 0.0,
                        "batch_first": True,
                    },
                    "cross_attn_cfg": {
                        "embed_dims": 256,
                        "num_heads": 8,
                        "dropout": 0.0,
                        "batch_first": True,
                    },
                    "ffn_cfg": {
                        "embed_dims": 256,
                        "feedforward_channels": 2048,
                        "num_fcs": 2,
                        "ffn_drop": 0.0,
                        "act_cfg": {"type": "ReLU", "inplace": True},
                    },
                },
                "init_cfg": None,
            },
            "loss_cls": {
                "type": "mmdet.CrossEntropyLoss",
                "use_sigmoid": False,
                "loss_weight": 2.0,
                "reduction": "mean",
                "class_weight": [1.0] * num_classes + [0.1],
            },
            "loss_mask": {
                "type": "mmdet.CrossEntropyLoss",
                "use_sigmoid": True,
                "reduction": "mean",
                "loss_weight": 5.0,
            },
            "loss_dice": {
                "type": "mmdet.DiceLoss",
                "use_sigmoid": True,
                "activate": True,
                "reduction": "mean",
                "naive_dice": True,
                "eps": 1.0,
                "loss_weight": 5.0,
            },
            "train_cfg": {
                "num_points": 12544,
                "oversample_ratio": 3.0,
                "importance_sample_ratio": 0.75,
                "assigner": {
                    "type": "mmdet.HungarianAssigner",
                    "match_costs": [
                        {"type": "mmdet.ClassificationCost", "weight": 2.0},
                        {"type": "mmdet.CrossEntropyLossCost", "weight": 5.0, "use_sigmoid": True},
                        {"type": "mmdet.DiceCost", "weight": 5.0, "pred_act": True, "eps": 1.0},
                    ],
                },
                "sampler": {"type": "mmdet.MaskPseudoSampler"},
            },
        },
        "train_cfg": {},
        "test_cfg": {"mode": "whole"},
    }


def _preprocessor(crop_size):
    return {
        "type": "SegDataPreProcessor",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "bgr_to_rgb": True,
        "pad_val": 0,
        "seg_pad_val": 255,
        "size": (crop_size, crop_size),
    }


def head_cfg(name, num_classes, crop_size, checkpoint=None, injector=False):
    """mmdet reaches into the decoder config by attribute, so every nested dict must be a ConfigDict."""
    from mmengine.config import ConfigDict

    builders = {"upernet": upernet_cfg, "m2f": mask2former_cfg}
    if name not in builders:
        raise ValueError(f"Unknown head: {name} (expected one of {sorted(builders)})")
    return ConfigDict(builders[name](num_classes, crop_size, checkpoint, injector))


HEADS = {"upernet": upernet_cfg, "m2f": mask2former_cfg}
