import sys
import types
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


class PEEncoder(nn.Module):
    name = "pe"
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
        height, width = mask.shape
        scale = self.input_size / max(height, width)
        resized_h, resized_w = round(height * scale), round(width * scale)
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        pad_top = (self.input_size - resized_h) // 2
        pad_left = (self.input_size - resized_w) // 2
        pad_bottom = self.input_size - resized_h - pad_top
        pad_right = self.input_size - resized_w - pad_left
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)))
        mask = np.pad(
            mask,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            constant_values=-1,
        )
        image_t = torch.from_numpy(image.transpose(2, 0, 1).copy()).float().div_(127.5).sub_(1.0)
        mask_t = torch.from_numpy(mask.copy()).long()
        geometry = {
            "original_height": height,
            "original_width": width,
            "resized_height": resized_h,
            "resized_width": resized_w,
            "pad_top": pad_top,
            "pad_left": pad_left,
        }
        return image_t, mask_t, geometry

    def forward(self, images):
        if self.trainable:
            features = self.trunk(images)[-1]
        else:
            with torch.no_grad():
                features = self.trunk(images)[-1]
        return features


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
    if model_name != "pe":
        raise ValueError(f"Unknown foundation model: {model_name}")
    encoder = PEEncoder(checkpoint, trainable=train_encoder)
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
