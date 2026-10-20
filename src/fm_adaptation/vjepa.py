"""V-JEPA 2.1, a video ViT, as an encoder that turns a clip into a feature pyramid per frame.

The trunk reads a clip as one sequence of space-time tokens and attends across all of it, every
frame to every other, in both directions. Its patch embedding pairs consecutive frames into one
token step (a tubelet of 2), so a clip of T frames comes out as T/2 token steps; `temporal` says
how each frame gets features of its own again. The four taps the trunk was distilled at are each
turned into one level of a strides 4/8/16/32 pyramid, frame by frame, for UperNet to decode.
"""

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models import LayerNorm2d, _resize_and_pad

# What the study can build, and what is not readable off the trunk. `checkpoint_key` is the
# encoder copy the release ships its weights under, which differs between sizes.
VJEPA_VARIANTS = {
    "vitb16": {
        "arch": "vit_base",
        "checkpoint": "vjepa2_1_vitb_dist_vitG_384.pt",
        "checkpoint_key": "ema_encoder",
        "feature_channels": 768,
        "taps": (2, 5, 8, 11),
    },
    "vitl16": {
        "arch": "vit_large",
        "checkpoint": "vjepa2_1_vitl_dist_vitG_384.pt",
        "checkpoint_key": "ema_encoder",
        "feature_channels": 1024,
        "taps": (5, 11, 17, 23),
    },
}


def _vjepa2_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "foundational_models" / "vjepa2"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _load_trunk(checkpoint: str | None, variant: str, frames: int):
    root = _vjepa2_root()
    from app.vjepa_2_1.models import vision_transformer

    spec = VJEPA_VARIANTS[variant]
    # The settings V-JEPA 2.1 was trained with, as its hub builder passes them.
    trunk = vision_transformer.__dict__[spec["arch"]](
        img_size=(VJEPAEncoder.input_size, VJEPAEncoder.input_size),
        patch_size=16,
        num_frames=frames,
        tubelet_size=2,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
        out_layers=list(spec["taps"]),
    )
    path = Path(checkpoint) if checkpoint else root / "ckpts" / spec["checkpoint"]
    state = torch.load(path, map_location="cpu", weights_only=True)[spec["checkpoint_key"]]
    state = {key.removeprefix("module.").removeprefix("backbone."): value for key, value in state.items()}
    trunk.load_state_dict(state, strict=True)
    return trunk


class TemporalSplit(nn.Module):
    """Decodes each token step of a `(B, C, T/2, h, w)` map into the two frames it was read from.

    A transposed convolution in time alone, starting as a copy of the step into both frames, so a
    fresh split hands the pyramid the trunk's own features and learns from there how the two differ.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.up = nn.ConvTranspose3d(channels, channels, (2, 1, 1), stride=(2, 1, 1))
        with torch.no_grad():
            self.up.weight.zero_()
            eye = torch.eye(channels)
            self.up.weight[:, :, 0, 0, 0] = eye
            self.up.weight[:, :, 1, 0, 0] = eye
            self.up.bias.zero_()

    def forward(self, x):
        return self.up(x)


class VideoPyramid(nn.Module):
    """The four taps of the trunk, frame by frame, as one pyramid level each.

    The levels follow mmseg's `Feature2Pyramid` over a stride-16 ViT -- two transposed convolutions
    up to stride 4, one up to stride 8, the tap itself at 16, a max-pool down to 32 -- with layer
    norm where it has batch norm.
    """

    def __init__(self, channels: int, temporal: str):
        super().__init__()
        self.split = (
            nn.ModuleList(TemporalSplit(channels) for _ in range(4)) if temporal == "split" else None
        )
        self.levels = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
                LayerNorm2d(channels),
                nn.GELU(),
                nn.ConvTranspose2d(channels, channels, 2, stride=2),
            ),
            nn.ConvTranspose2d(channels, channels, 2, stride=2),
            nn.Identity(),
            nn.MaxPool2d(2),
        ])

    def forward(self, maps):
        """`maps` are four `(B, C, steps, h, w)` taps; returns four `(B * T, C, H_s, W_s)` levels."""
        levels = []
        for index, x in enumerate(maps):
            if self.split is not None:
                x = self.split[index](x)
            levels.append(self.levels[index](x.transpose(1, 2).flatten(0, 1)))
        return tuple(levels)


class VJEPAEncoder(nn.Module):
    """V-JEPA 2.1 over a `(B, T, 3, H, W)` clip, returning a strides 4/8/16/32 pyramid per frame.

    `split` reads the clip as it is, T/2 token steps, and decodes each step into its two frames.
    `repeat` feeds every frame twice, so each token step is one frame, at twice the tokens.
    """

    name = "vjepa21"
    input_size = 384
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    takes_clips = True

    def __init__(
        self,
        checkpoint: str | None,
        trainable: bool = False,
        variant: str = "vitb16",
        frames: int = 16,
        temporal: str = "split",
    ):
        super().__init__()
        if variant not in VJEPA_VARIANTS:
            raise ValueError(f"unknown vjepa21 variant {variant} (expected one of {sorted(VJEPA_VARIANTS)})")
        if frames % 2:
            raise ValueError(f"a clip is read two frames per token step, so {frames} frames will not do")
        self.variant = variant
        self.temporal = temporal
        self.feature_channels = VJEPA_VARIANTS[variant]["feature_channels"]
        self.trunk = _load_trunk(checkpoint, variant, frames * 2 if temporal == "repeat" else frames)
        self.trunk.use_activation_checkpointing = trainable
        self.trainable = trainable
        self.trunk.requires_grad_(trainable)
        self.adapter = VideoPyramid(self.feature_channels, temporal)

    def preprocess(self, image: np.ndarray, mask: np.ndarray):
        image, mask_t, geometry = _resize_and_pad(image, mask, self.input_size)
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(255.0)
        image_t = (image_t - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]
        return image_t, mask_t, geometry

    def forward(self, clips):
        batch, _, _, height, width = clips.shape
        video = clips.transpose(1, 2)
        if self.temporal == "repeat":
            video = video.repeat_interleave(2, dim=2)
        with torch.no_grad() if not self.trainable else nullcontext():
            taps = self.trunk(video)
        rows, columns = height // self.trunk.patch_size, width // self.trunk.patch_size
        maps = [
            tap.transpose(1, 2).reshape(batch, self.feature_channels, -1, rows, columns) for tap in taps
        ]
        return self.adapter(maps)
