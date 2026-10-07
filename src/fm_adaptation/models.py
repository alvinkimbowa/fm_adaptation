import math
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from .prompting import PromptConditioner, PromptGate


def decoder_widths(in_channels):
    """One width per pyramid level. The adapters emit the trunk's token width four times over; a
    ConvNeXt widens as it goes, so it declares the four its stages actually have."""
    return list(in_channels) if isinstance(in_channels, (list, tuple)) else [in_channels] * 4


class LinearProbe(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, features, output_size, prompt=None):
        return F.interpolate(
            self.classifier(features), size=output_size, mode="bilinear", align_corners=False
        )


class NonlinearProbe(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        channels = (256, 128, 64, 32)
        layers = []
        current = in_channels
        for channel in channels:
            layers.extend(
                [nn.Conv2d(current, channel, 3, padding=1, bias=False), nn.BatchNorm2d(channel), nn.ReLU()]
            )
            current = channel
        self.decoder = nn.Sequential(*layers)
        self.classifier = nn.Conv2d(current, num_classes, 1)

    def forward(self, features, output_size, prompt=None):
        x = features
        for start in range(0, len(self.decoder), 3):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.decoder[start : start + 3](x)
        return F.interpolate(self.classifier(x), size=output_size, mode="bilinear", align_corners=False)


def _resize_and_pad(image: np.ndarray, mask: np.ndarray, input_size: int):
    """Aspect-preserving resize onto a square canvas; the label pad is -1 so the loss ignores it."""
    height, width = mask.shape
    scale = input_size / max(height, width)
    resized_h, resized_w = round(height * scale), round(width * scale)
    image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
    pad_top = (input_size - resized_h) // 2
    pad_left = (input_size - resized_w) // 2
    pad_bottom = input_size - resized_h - pad_top
    pad_right = input_size - resized_w - pad_left
    image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
    mask = np.pad(mask, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=-1)
    geometry = {
        "original_height": height,
        "original_width": width,
        "resized_height": resized_h,
        "resized_width": resized_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return image, torch.from_numpy(mask.copy()).long(), geometry


class PEEncoder(nn.Module):
    name = "sam3"
    feature_channels = 1024
    input_size = 1008

    def __init__(self, checkpoint: str | None, trainable: bool = False):
        super().__init__()
        sam3_root = Path(__file__).resolve().parents[2] / "foundational_models" / "sam3"
        sys.path.insert(0, str(sam3_root))
        from sam3.model_builder import build_sam3_image_model

        model = build_sam3_image_model(
            device="cpu",
            eval_mode=True,
            checkpoint_path=checkpoint,
            load_from_HF=checkpoint is None,
            enable_segmentation=False,
        )
        self.trunk = model.backbone.vision_backbone.trunk
        self.trainable = trainable
        if trainable:
            for block in self.trunk.blocks:
                block.mlp.forward = types.MethodType(_trainable_sam3_mlp_forward, block.mlp)
        self.trunk.requires_grad_(trainable)

    def preprocess(self, image: np.ndarray, mask: np.ndarray):
        image, mask_t, geometry = _resize_and_pad(image, mask, self.input_size)
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(127.5).sub_(1.0)
        return image_t, mask_t, geometry

    def forward(self, images):
        if self.trainable:
            features = self.trunk(images)[-1]
        else:
            with torch.no_grad():
                features = self.trunk(images)[-1]
        return features


def _dinov3_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "foundational_models" / "dinov3"
    sys.path.insert(0, str(root))
    return root


# The DINOv3 sizes the study can run, and everything about them that is not readable off the trunk.
# `interaction_indexes` names the last block of each quarter of the trunk, as DINOv3's own segmentation
# config does for ViT-L. `deform_num_heads` has to keep the deformable attention's channels-per-head a
# power of two -- at 384 dim with `deform_ratio=0.5`, 16 heads gives 24 and takes `MSDeformAttn`'s slow
# path, while 6 gives 32.
DINOV3_VARIANTS = {
    "vitl16": {
        "hub": "dinov3_vitl16",
        "checkpoint": "dinov3_vitl16_pretrain_lvd1689m.pth",
        "feature_channels": 1024,
        "last_layer": 23,
        "interaction_indexes": (4, 11, 17, 23),
        "deform_num_heads": 16,
    },
    "vitb16": {
        "hub": "dinov3_vitb16",
        "checkpoint": "dinov3_vitb16_pretrain_lvd1689m.pth",
        "feature_channels": 768,
        "last_layer": 11,
        "interaction_indexes": (2, 5, 8, 11),
        "deform_num_heads": 12,
    },
    "vits16": {
        "hub": "dinov3_vits16",
        "checkpoint": "dinov3_vits16_pretrain_lvd1689m.pth",
        "feature_channels": 384,
        "last_layer": 11,
        "interaction_indexes": (2, 5, 8, 11),
        "deform_num_heads": 6,
    },
    # ConvNeXt emits a stride-4/8/16/32 pyramid of its own, so the four widths take the place of the
    # single token width a ViT reports. Sizes are DINOv3's own, from `dinov3.models.convnext`.
    "convnextl": {
        "kind": "convnext",
        "checkpoint": "dinov3_convnextl_pretrain_lvd1689m.pth",
        "feature_channels": (192, 384, 768, 1536),
        "depths": (3, 3, 27, 3),
    },
    "convnextb": {
        "kind": "convnext",
        "checkpoint": "dinov3_convnextb_pretrain_lvd1689m.pth",
        "feature_channels": (128, 256, 512, 1024),
        "depths": (3, 3, 27, 3),
    },
    "convnexts": {
        "kind": "convnext",
        "checkpoint": "dinov3_convnexts_pretrain_lvd1689m.pth",
        "feature_channels": (96, 192, 384, 768),
        "depths": (3, 3, 27, 3),
    },
    "convnextt": {
        "kind": "convnext",
        "checkpoint": "dinov3_convnextt_pretrain_lvd1689m.pth",
        "feature_channels": (96, 192, 384, 768),
        "depths": (3, 3, 9, 3),
    },
}
DEFAULT_DINOV3_VARIANT = "vitl16"


def _dinov3_variant(variant: str) -> dict:
    if variant not in DINOV3_VARIANTS:
        raise ValueError(f"Unknown DINOv3 variant: {variant} (expected one of {sorted(DINOV3_VARIANTS)})")
    return DINOV3_VARIANTS[variant]


def _load_dinov3_backbone(checkpoint: str | None, variant: str = DEFAULT_DINOV3_VARIANT):
    """A DINOv3 trunk with the local weights loaded; the architecture comes from `variant`."""
    root = _dinov3_root()
    import dinov3.hub.backbones as backbones

    spec = _dinov3_variant(variant)
    if checkpoint is None:
        checkpoint = root / "ckpts" / spec["checkpoint"]
    if spec.get("kind") == "convnext":
        from dinov3.models.convnext import ConvNeXt

        trunk = ConvNeXt(depths=list(spec["depths"]), dims=list(spec["feature_channels"]))
    else:
        # The hub loader derives a config flag from an 8-char hash in the filename, which local
        # checkpoints do not carry; the LVD-1689M defaults are what `pretrained=False` already builds.
        trunk = getattr(backbones, spec["hub"])(pretrained=False)
    state = torch.load(Path(checkpoint).resolve(), map_location="cpu", weights_only=True)
    missing, unexpected = trunk.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DINOv3 checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return trunk


class DINOv3Encoder(nn.Module):
    """DINOv3 patch tokens, at the 896 resolution DINOv3 uses for dense adaptation.

    The class attributes describe the default ViT-L/16; a different `variant` overrides the ones that
    depend on the trunk's size, per instance.
    """

    name = "dinov3"
    feature_channels = 1024
    input_size = 896
    last_layer = 23
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        checkpoint: str | None,
        trainable: bool = False,
        variant: str = DEFAULT_DINOV3_VARIANT,
    ):
        super().__init__()
        spec = _dinov3_variant(variant)
        self.variant = variant
        self.feature_channels = spec["feature_channels"]
        self.last_layer = spec["last_layer"]
        self.trunk = _load_dinov3_backbone(checkpoint, variant)
        self.trainable = trainable
        self.trunk.requires_grad_(trainable)

    def preprocess(self, image: np.ndarray, mask: np.ndarray):
        image, mask_t, geometry = _resize_and_pad(image, mask, self.input_size)
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        image_t = (image_t - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]
        return image_t, mask_t, geometry

    def forward(self, images):
        with torch.no_grad() if not self.trainable else nullcontext():
            return self.trunk.get_intermediate_layers(images, n=[self.last_layer], reshape=True)[0]


class DINOv3AdapterEncoder(DINOv3Encoder):
    """Frozen DINOv3 trunk behind DINOv3's ViT-Adapter, emitting strides 4/8/16/32.

    The adapter trains while the trunk stays frozen, so unlike the plain encoder this one has parameters
    of its own and its features cannot be cached between epochs.
    """

    def __init__(
        self,
        checkpoint: str | None,
        trainable: bool = False,
        injector: bool = False,
        variant: str = DEFAULT_DINOV3_VARIANT,
    ):
        nn.Module.__init__(self)
        _dinov3_root()
        from mmengine.model import revert_sync_batchnorm

        spec = _dinov3_variant(variant)
        self.variant = variant
        self.feature_channels = spec["feature_channels"]
        self.last_layer = spec["last_layer"]
        self.interaction_indexes = spec["interaction_indexes"]
        backbone = _load_dinov3_backbone(checkpoint, variant)
        if injector:
            from .dinov3_injector import DINOv3AdapterWithInjector as adapter_cls
        else:
            from dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter as adapter_cls
        adapter = adapter_cls(
            backbone,
            interaction_indexes=list(self.interaction_indexes),
            deform_num_heads=spec["deform_num_heads"],
        )
        # The adapter is built with SyncBatchNorm, which needs a process group we do not have outside
        # mmseg's distributed Runner; mmengine's helper swaps it for plain BatchNorm in place.
        self.adapter = revert_sync_batchnorm(adapter)
        # `DINOv3_Adapter.__init__` freezes the trunk itself, so the say over it comes after, not before.
        self.adapter.backbone.requires_grad_(trainable)
        # `trainable` on the encoder means "do not force this module into eval" -- the adapter trains
        # whether or not the trunk does, so this is always True; `trunk_trains` is the trunk's own say.
        self.trainable = True
        self.trunk_trains = trainable

    @property
    def trunk(self):
        return self.adapter.backbone

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trunk_trains:
            self.adapter.backbone.eval()  # a frozen trunk carries stochastic depth and stays in eval
        return self

    def forward(self, images):
        features = self.adapter(images)
        return tuple(features[key] for key in ("1", "2", "3", "4"))


class DINOv3ConvNeXtEncoder(nn.Module):
    """DINOv3's ConvNeXt trunk, emitting the strides 4/8/16/32 its four stages already produce.

    Where a ViT needs the adapter to synthesise a feature pyramid out of one token grid, a ConvNeXt is
    a pyramid, so every stage feeds the decoder directly and there is no adapter and no injector.
    Runs at 512, the resolution ConvNeXt segments at.
    """

    name = "dinov3"
    input_size = 512
    mean = DINOv3Encoder.mean
    std = DINOv3Encoder.std

    def __init__(
        self,
        checkpoint: str | None,
        trainable: bool = False,
        variant: str = "convnextl",
        frames: int = 1,
    ):
        super().__init__()
        _dinov3_root()
        self.variant = variant
        self.feature_channels = list(_dinov3_variant(variant)["feature_channels"])
        self.trunk = _load_dinov3_backbone(checkpoint, variant)
        if frames > 1:
            self._widen_stem(frames)
        self.trainable = trainable
        self.trunk.requires_grad_(trainable)

    def _widen_stem(self, frames: int):
        """Have the stem read `frames` RGB images stacked as channels, the first being the case.

        The case keeps the pretrained filters and every other frame starts at zero, so the widened
        trunk computes exactly what the pretrained one does until training moves those weights.
        """
        stem = self.trunk.downsample_layers[0][0]
        wide = nn.Conv2d(3 * frames, stem.out_channels, stem.kernel_size, stem.stride)
        with torch.no_grad():
            wide.weight.zero_()
            wide.weight[:, :3] = stem.weight
            wide.bias.copy_(stem.bias)
        self.trunk.downsample_layers[0][0] = wide

    preprocess = DINOv3Encoder.preprocess

    def forward(self, images):
        with torch.no_grad() if not self.trainable else nullcontext():
            return self.trunk.get_intermediate_layers(images, n=[0, 1, 2, 3], reshape=True)


class SAM3AdapterEncoder(PEEncoder):
    """Frozen SAM3 PE trunk behind the same ViT-Adapter, emitting strides 4/8/16/32.

    Runs at 896 rather than SAM3's native 1008: the adapter's coarsest level needs the input to divide by
    32, which 1008 does not, while 896 divides by both 32 and SAM3's patch size of 14.
    """

    input_size = 896

    def __init__(self, checkpoint: str | None, trainable: bool = False, injector: bool = False):
        nn.Module.__init__(self)
        from mmengine.model import revert_sync_batchnorm

        from .sam3_adapter import SAM3Adapter, load_sam3_trunk

        backbone = load_sam3_trunk(checkpoint, image_size=self.input_size)
        # SAM3's fused PE MLP is inference-only and casts to bfloat16 internally. When the trunk is
        # frozen its weights never update, but the injector's gradients still flow back through these
        # blocks, so the
        # fused path is swapped for its differentiable equivalent exactly as finetuning does.
        for block in backbone.blocks:
            block.mlp.forward = types.MethodType(_trainable_sam3_mlp_forward, block.mlp)
        adapter = SAM3Adapter(backbone, use_injector=injector)
        # Built with SyncBatchNorm, which needs a process group we do not have outside mmseg's Runner.
        self.adapter = revert_sync_batchnorm(adapter)
        # The adapter freezes the trunk on construction, so the say over it comes after, not before.
        self.adapter.backbone.requires_grad_(trainable)
        self.trainable = True
        self.trunk_trains = trainable

    @property
    def trunk(self):
        return self.adapter.backbone

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trunk_trains:
            self.adapter.backbone.eval()  # a frozen trunk carries stochastic depth and stays in eval
        return self

    def forward(self, images):
        features = self.adapter(images)
        return tuple(features[key] for key in ("1", "2", "3", "4"))


class UperNetDecoder(nn.Module):
    """mmsegmentation's UPerHead over the adapter's four scales, driven directly as an nn.Module.

    Only the head is used -- none of mmseg's loss, auxiliary head or `EncoderDecoder` machinery -- so the
    run is identical to a probe run apart from the decoder itself.
    """

    def __init__(self, in_channels, num_classes: int, prompt_width: int | None = None):
        super().__init__()
        from mmseg.models.decode_heads import UPerHead

        widths = decoder_widths(in_channels)
        self.head = UPerHead(
            in_channels=widths,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg={"type": "BN", "requires_grad": True},
            align_corners=False,
            # Required by the constructor, never called: we compute the loss ourselves.
            loss_decode={"type": "CrossEntropyLoss"},
        )
        # Gates the features the FPN has already fused, so one gate reaches every scale at once.
        self.gate = None if prompt_width is None else PromptGate(self.head.channels, prompt_width)

    def forward(self, features, output_size, prompt=None):
        if self.gate is None:
            logits = self.head(features)
        else:
            fused = self.head._forward_feature(features)
            logits = self.head.cls_seg(self.gate(fused, prompt))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


def _sine_positions(height: int, width: int, channels: int, device) -> torch.Tensor:
    """A fixed position code per cell of a `height` x `width` grid, as `(height * width, channels)`.

    Cells are placed by where they fall in the image rather than by their index, so grids of
    different resolution over the same image agree on where a cell is.
    """
    if channels % 4:
        raise ValueError(f"a 2D sine position code needs a width divisible by 4, got {channels}")
    quarter = channels // 4
    frequency = torch.exp(
        torch.arange(quarter, device=device, dtype=torch.float32) * (-math.log(10000.0) / quarter)
    )
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / height * 2 * math.pi
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / width * 2 * math.pi
    y, x = y[:, None] / frequency, x[:, None] / frequency
    rows = torch.cat([y.sin(), y.cos()], dim=1)[:, None].expand(height, width, 2 * quarter)
    columns = torch.cat([x.sin(), x.cos()], dim=1)[None].expand(height, width, 2 * quarter)
    return torch.cat([rows, columns], dim=2).flatten(0, 1)


class ContextAttention(nn.Module):
    """One feature map of a case attending to the same map of its context frames.

    Every cell of the case's map is a query over every cell of every context frame. Both sides carry
    a position code, and each context frame an embedding of how many strides it lies from the case,
    before or after. `out` is zero-initialised, so a freshly built block is the identity.
    """

    def __init__(self, channels: int, frames: int, heads: int = 8):
        super().__init__()
        self.frames = frames
        self.query_norm = nn.LayerNorm(channels)
        self.context_norm = nn.LayerNorm(channels)
        # A context frame lies between `frames` strides before the case and `frames` after.
        self.step = nn.Embedding(2 * frames + 1, channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.out = nn.Linear(channels, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, target, context, steps):
        """`target` is `(B, C, H, W)`, `context` `(B, T, C, h, w)` and `steps` `(B, T)`."""
        batch, channels, height, width = target.shape
        query = self.query_norm(target.flatten(2).transpose(1, 2))
        query = query + _sine_positions(height, width, channels, target.device).to(query.dtype)
        keys = self.context_norm(context.flatten(3).transpose(2, 3))
        keys = keys + _sine_positions(*context.shape[-2:], channels, target.device).to(keys.dtype)
        keys = (keys + self.step(steps + self.frames)[:, :, None].to(keys.dtype)).flatten(1, 2)
        attended = self.attention(query, keys, keys, need_weights=False)[0]
        update = self.out(attended).transpose(1, 2).reshape(batch, channels, height, width)
        return target + update


class ContextConv(nn.Module):
    """One feature map of a case concatenated with the same map of its context frames, and convolved.

    The context frames are projected by one shared convolution, each given an embedding of how many
    strides it lies from the case, and averaged, so the block does not depend on how many there are
    or on which side of the case they fall. The average is concatenated with the case's map cell for
    cell, which assumes a structure sits in about the same place across the frames. `out` is
    zero-initialised, so a freshly built block is the identity.
    """

    def __init__(self, channels: int, frames: int):
        super().__init__()
        self.frames = frames
        self.project = nn.Conv2d(channels, channels, 1)
        self.step = nn.Embedding(2 * frames + 1, channels)
        self.mix = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, target, context, steps):
        """`target` is `(B, C, H, W)`, `context` `(B, T, C, H, W)` and `steps` `(B, T)`."""
        projected = self.project(context.flatten(0, 1)).unflatten(0, steps.shape)
        projected = projected + self.step(steps + self.frames)[..., None, None].to(projected.dtype)
        return target + self.out(self.mix(torch.cat([target, projected.mean(1)], dim=1)))


class ContextFusion(nn.Module):
    """Where a case's feature pyramid meets the pyramids of its context frames.

    `late` fuses the coarsest level alone and `intermediate` all four, by attention or by
    convolution. A context map that is attended to is first pooled to the coarsest level's grid,
    whichever level it belongs to, so the cost of a fine level is its own cells against a short list
    of keys rather than against as many again per context frame.
    """

    def __init__(self, channels, context):
        super().__init__()
        levels = range(len(channels)) if context.fusion == "intermediate" else [len(channels) - 1]
        self.attends = context.operator == "attention"
        block = ContextAttention if self.attends else ContextConv
        self.blocks = nn.ModuleDict(
            {str(level): block(channels[level], context.frames) for level in levels}
        )

    def keep(self, pyramid):
        """What of a context frame's pyramid the blocks read: one map per fused level."""
        grid = pyramid[-1].shape[-2:]
        maps = [pyramid[int(key)] for key in self.blocks]
        return [F.adaptive_avg_pool2d(kept, grid) for kept in maps] if self.attends else maps

    def forward(self, features, context_maps, steps):
        """`context_maps` is `keep` of every context frame, each map as `(B * T, C, h, w)`."""
        fused = list(features)
        for (key, block), frames in zip(self.blocks.items(), context_maps):
            level = int(key)
            fused[level] = block(fused[level], frames.unflatten(0, steps.shape), steps)
        return fused


def _trainable_sam3_mlp_forward(mlp, x):
    """Differentiable equivalent of SAM3's inference-only fused PE MLP."""
    x = mlp.fc1(x)
    x = mlp.act(x)
    x = mlp.drop1(x)
    x = mlp.norm(x)
    x = mlp.fc2(x)
    return mlp.drop2(x)


# Context frames encoded per forward pass of the encoder.
CONTEXT_ENCODE_BATCH = 2


class SegmentationModel(nn.Module):
    """An encoder, a decoder, and whatever conditions the features on their way between the two.

    A case arriving with context frames is `(B, 1 + T, 3, H, W)`, the case first. With `context` set
    each frame goes through the encoder on its own and the case's features are fused with those of
    its context; without it the frames are stacked as channels for an encoder whose stem takes them.
    `context_gradient` says whether the encoder is trained through the context frames too.
    """

    def __init__(self, encoder, probe, prompt=None, context=None, context_gradient=False):
        super().__init__()
        self.encoder = encoder
        self.probe = probe
        self.prompt = prompt
        self.context = context
        self.context_gradient = context_gradient

    def _context_maps(self, group):
        """The feature maps the fusion reads from a few context frames."""
        return tuple(self.context.keep(self.encoder(group)))

    def _encode_context(self, frames):
        """The feature maps the fusion reads from every context frame, each as `(B * T, C, h, w)`.

        The frames go through the encoder a few at a time, each group cut down to the maps the
        fusion reads before the next is encoded, so that a step never holds the encoder's
        activations for all of them at once.

        With `context_gradient` the encoder is trained through them: each group is checkpointed, its
        activations dropped after the forward pass and recomputed, one group at a time, on the
        backward pass. Normalisation layers that keep running statistics therefore see each group
        twice per step.

        Without it no gradient is kept and the encoder is in eval mode, so that what a small group
        is normalised by, and whether a block is dropped, does not depend on how the frames were
        grouped.
        """
        groups = frames.flatten(0, 1).split(CONTEXT_ENCODE_BATCH)
        if self.context_gradient:
            kept = [
                checkpoint.checkpoint(self._context_maps, group, use_reentrant=False)
                if torch.is_grad_enabled()
                else self._context_maps(group)
                for group in groups
            ]
            return [torch.cat(maps) for maps in zip(*kept)]
        training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            kept = [self._context_maps(group) for group in groups]
        self.encoder.train(training)
        # Autocast keeps the low-precision copy it makes of each weight for as long as its context
        # is open, and the copies made above carry no gradient. Left in place, the case's own pass
        # would reuse them and its weights would receive none either.
        torch.clear_autocast_cache()
        return [torch.cat(maps) for maps in zip(*kept)]

    def forward(self, images, prompt=None, steps=None):
        frames = None
        if images.dim() == 5:
            if self.context is None:
                images = images.flatten(1, 2)
            else:
                images, frames = images[:, 0], images[:, 1:]
        # Before the case, whose activations are held for the backward pass: the context's are
        # released as they are encoded, so the two never peak together.
        context_maps = None if frames is None else self._encode_context(frames)
        features = self.encoder(images)
        if context_maps is not None:
            features = self.context(features, context_maps, steps)
        embedding = None
        if self.prompt is not None:
            if prompt is None:
                raise ValueError("this model takes a prompt, and the batch carried none")
            embedding = self.prompt.encode(prompt)
            features = self.prompt.condition(features, embedding)
        return self.probe(features, images.shape[-2:], embedding)


def build_model(
    model_name: str,
    probe_name: str,
    classes: int,
    checkpoint: str | None,
    train_encoder: bool = False,
    injector: bool = False,
    variant: str = DEFAULT_DINOV3_VARIANT,
    prompt=None,
    context=None,
):
    encoders = {"sam3": PEEncoder, "dinov3": DINOv3Encoder}
    probes = {"linear": LinearProbe, "nonlinear": NonlinearProbe, "upernet": UperNetDecoder}
    if model_name not in encoders:
        raise ValueError(f"Unknown foundation model: {model_name} (expected one of {sorted(encoders)})")
    if probe_name not in probes:
        raise ValueError(f"Unknown probe: {probe_name} (expected one of {sorted(probes)})")
    # `variant` names a DINOv3 trunk size; SAM3 has only the one trunk, so asking for a size there is a
    # mistake in the config rather than something to ignore.
    if model_name != "dinov3" and variant != DEFAULT_DINOV3_VARIANT:
        raise ValueError(f"model.variant is only meaningful for dinov3, not {model_name}")
    extra = {"variant": variant} if model_name == "dinov3" else {}
    if context is not None and probe_name != "upernet":
        raise ValueError("data.context is fused over a feature pyramid, which only upernet decodes")
    is_convnext = model_name == "dinov3" and _dinov3_variant(variant).get("kind") == "convnext"
    stacks_frames = context is not None and context.fusion == "early"
    if stacks_frames and not (is_convnext and train_encoder):
        # The stem's filters for the context frames start at zero and have to be learned.
        raise ValueError("data.context.fusion early needs a convnext trunk that trains")
    trains_context = context is not None and context.gradient and not stacks_frames
    if probe_name == "upernet":
        # UperNet consumes a feature pyramid. A ConvNeXt trunk is one already; a ViT needs the adapter
        # to build one.
        if is_convnext:
            if injector:
                raise ValueError("model.injector belongs to the ViT-Adapter, which convnext does not use")
            encoder = DINOv3ConvNeXtEncoder(
                checkpoint, trainable=train_encoder, variant=variant,
                frames=1 + context.frames if stacks_frames else 1,
            )
        else:
            adapters = {"dinov3": DINOv3AdapterEncoder, "sam3": SAM3AdapterEncoder}
            encoder = adapters[model_name](checkpoint, trainable=train_encoder, injector=injector, **extra)
    else:
        encoder = encoders[model_name](checkpoint, trainable=train_encoder, **extra)
    if prompt is not None and probe_name != "upernet" and prompt.gates_decoder:
        raise ValueError("data.prompt.sites reaches the decoder, which only upernet has")
    probe_extra = (
        {"prompt_width": prompt.width}
        if prompt is not None and probe_name == "upernet" and prompt.gates_decoder
        else {}
    )
    probe = probes[probe_name](encoder.feature_channels, classes, **probe_extra)
    conditioner = None
    if prompt is not None:
        # A pyramid decoder is handed four maps; the probes are handed the trunk's one.
        channels = (
            decoder_widths(encoder.feature_channels)
            if probe_name == "upernet"
            else [encoder.feature_channels]
        )
        conditioner = PromptConditioner(
            prompt, prompt.width, channels, gate_features=prompt.gates_encoder,
        )
    fusion = (
        ContextFusion(decoder_widths(encoder.feature_channels), context)
        if context is not None and not stacks_frames
        else None
    )
    return SegmentationModel(encoder, probe, conditioner, fusion, context_gradient=trains_context)


def load_trained_model(cfg, checkpoint: str, device, classes: int):
    """The model a finished run left behind, built from its config and loaded from its checkpoint."""
    model = build_model(
        cfg.model_name, cfg.probe_name, classes, cfg.checkpoint,
        train_encoder=cfg.train_encoder, injector=cfg.injector, variant=cfg.variant,
        prompt=getattr(cfg, "prompt", None), context=getattr(cfg, "context", None),
    )
    name = "final" if cfg.fold == "all" else checkpoint
    path = cfg.run_dir / f"{name}.pt"
    # `last.pt` also carries optimiser and RNG state, which the safe loader cannot unpickle.
    state = torch.load(path, map_location="cpu", weights_only=name != "last")
    model.probe.load_state_dict(state["probe"])
    if "prompt" in state:
        model.prompt.load_state_dict(state["prompt"])
    if "context" in state:
        model.context.load_state_dict(state["context"])
    if "encoder" in state:
        model.encoder.trunk.load_state_dict(state["encoder"])
    if "adapter" in state:
        # Trained adapter weights only; the frozen trunk came from the DINOv3 checkpoint at build time.
        missing, unexpected = model.encoder.adapter.load_state_dict(state["adapter"], strict=False)
        missing = [key for key in missing if not key.startswith("backbone.")]
        if missing or unexpected:
            raise RuntimeError(
                f"adapter checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
            )
    return model.to(device).eval()


def restore_prediction(prediction: torch.Tensor, geometry: dict) -> np.ndarray:
    top, left = geometry["pad_top"], geometry["pad_left"]
    height, width = geometry["resized_height"], geometry["resized_width"]
    prediction = prediction[top : top + height, left : left + width]
    return cv2.resize(
        prediction.cpu().numpy().astype(np.uint8),
        (geometry["original_width"], geometry["original_height"]),
        interpolation=cv2.INTER_NEAREST,
    )
