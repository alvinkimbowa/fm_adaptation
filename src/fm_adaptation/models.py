import sys
import types
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LinearProbe(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, 1)

    def forward(self, features, output_size):
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

    def forward(self, features, output_size):
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


class DINOv3Encoder(nn.Module):
    """DINOv3 ViT-L/16 patch tokens, at the 896 resolution DINOv3 uses for dense adaptation."""

    name = "dinov3"
    feature_channels = 1024
    input_size = 896
    last_layer = 23
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, checkpoint: str | None, trainable: bool = False):
        super().__init__()
        root = Path(__file__).resolve().parents[2] / "foundational_models" / "dinov3"
        sys.path.insert(0, str(root))
        from dinov3.hub.backbones import dinov3_vitl16

        if checkpoint is None:
            checkpoint = root / "ckpts" / "dinov3_vitl16_pretrain_lvd1689m.pth"
        # The hub loader derives a config flag from an 8-char hash in the filename, which local
        # checkpoints do not carry; the LVD-1689M defaults are what `pretrained=False` already builds.
        self.trunk = dinov3_vitl16(pretrained=False)
        state = torch.load(Path(checkpoint).resolve(), map_location="cpu", weights_only=True)
        missing, unexpected = self.trunk.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"DINOv3 checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
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


def _trainable_sam3_mlp_forward(mlp, x):
    """Differentiable equivalent of SAM3's inference-only fused PE MLP."""
    x = mlp.fc1(x)
    x = mlp.act(x)
    x = mlp.drop1(x)
    x = mlp.norm(x)
    x = mlp.fc2(x)
    return mlp.drop2(x)


class SegmentationModel(nn.Module):
    def __init__(self, encoder, probe):
        super().__init__()
        self.encoder = encoder
        self.probe = probe

    def forward(self, images):
        return self.probe(self.encoder(images), images.shape[-2:])


def build_model(
    model_name: str,
    probe_name: str,
    classes: int,
    checkpoint: str | None,
    train_encoder: bool = False,
):
    encoders = {"sam3": PEEncoder, "dinov3": DINOv3Encoder}
    if model_name not in encoders:
        raise ValueError(f"Unknown foundation model: {model_name} (expected one of {sorted(encoders)})")
    encoder = encoders[model_name](checkpoint, trainable=train_encoder)
    probes = {"linear": LinearProbe, "nonlinear": NonlinearProbe}
    if probe_name not in probes:
        raise ValueError(f"Unknown probe: {probe_name}")
    probe = probes[probe_name](encoder.feature_channels, classes)
    return SegmentationModel(encoder, probe)


def restore_prediction(prediction: torch.Tensor, geometry: dict) -> np.ndarray:
    top, left = geometry["pad_top"], geometry["pad_left"]
    height, width = geometry["resized_height"], geometry["resized_width"]
    prediction = prediction[top : top + height, left : left + width]
    return cv2.resize(
        prediction.cpu().numpy().astype(np.uint8),
        (geometry["original_width"], geometry["original_height"]),
        interpolation=cv2.INTER_NEAREST,
    )
