"""Export a trained run to ONNX for deployment.

The exported graph takes one channel per stored stain plane with intensities scaled to 0-1, cut to
the run's patch size, and returns per-class logits at the same size. Placing each plane in the RGB
channel the model was trained to find it in, and the normalisation, are part of the graph. The file
says so in its metadata (`input_normalization: unit_range`), which is how the tracer app knows not to
standardise the image before handing it over.

A run trained on z-scored patches (`patching.normalization: zscore`) is exported as the bare
network instead and marked `input_normalization: zscore`: it takes one channel per stored plane,
each standardised by the mean and deviation of its whole image's non-zero pixels.

Quantising and checking the export is `quantize_onnx.py`, which needs ONNX Runtime and no torch; this
module needs torch and the model's own dependencies and no ONNX Runtime. `--reference` writes the
inputs and logits the second stage compares the exported file against.
"""

import argparse
import math
import re
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import onnx
import torch
from torch import nn

from .config import ExperimentConfig
from .data import load_dataset_json, num_classes, stain_planes
from .models import load_trained_model
from .patching import foreground_statistics, zscore

IMAGE_SUFFIXES = {".png", ".tif", ".tiff"}


def _pooling_matrix(input_size, output_size):
    """Row `i` averages the inputs adaptive pooling assigns to output `i`."""
    matrix = torch.zeros(output_size, input_size)
    for index in range(output_size):
        start = (index * input_size) // output_size
        end = math.ceil((index + 1) * input_size / output_size)
        matrix[index, start:end] = 1.0 / (end - start)
    return matrix


class FixedAdaptiveAvgPool2d(nn.Module):
    """`AdaptiveAvgPool2d` for one known input size, as two matrix products.

    ONNX has no adaptive pooling, and the exporter only rewrites it as an ordinary pooling when the
    output size divides the input size. UPerNet pools a 16x16 map down to 3x3 and 6x6, which does not.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.register_buffer("rows", _pooling_matrix(input_size[0], output_size[0]))
        self.register_buffer("columns", _pooling_matrix(input_size[1], output_size[1]).T)

    def forward(self, x):
        return self.rows @ (x @ self.columns)


def _fix_adaptive_pools(model, example):
    """Swap every adaptive pooling in `model` for its fixed equivalent at `example`'s size."""
    seen = {}
    hooks = [
        module.register_forward_hook(
            lambda module, inputs, _output: seen.__setitem__(module, tuple(inputs[0].shape[-2:]))
        )
        for module in model.modules() if isinstance(module, nn.AdaptiveAvgPool2d)
    ]
    with torch.no_grad():
        model(example)
    for hook in hooks:
        hook.remove()
    for parent in model.modules():
        for name, child in parent.named_children():
            if child in seen:
                size = child.output_size
                size = (size, size) if isinstance(size, int) else tuple(size)
                setattr(parent, name, FixedAdaptiveAvgPool2d(seen[child], size))


def channel_mixing(cfg):
    """A 3 x C matrix placing each stored channel in the RGB planes the model reads it from.

    Mirrors how `patching.CaseIndex.crop` builds a patch: a dataset declaring its stains puts each
    one in that stain's plane and leaves the others black, and any other dataset is one grey channel
    repeated across all three.
    """
    declared = None
    if cfg.patching is not None and cfg.patching.respect_channels:
        declared = stain_planes(
            load_dataset_json(cfg.raw_data_dir / cfg.train_dataset)["channel_names"]
        )
    if declared is None:
        return torch.ones(3, 1)
    layout = sorted(declared.values())
    mixing = torch.zeros(3, len(layout))
    for column, (_stored, plane) in enumerate(layout):
        mixing[plane, column] = 1.0
    return mixing


class DeployedModel(nn.Module):
    """A trained segmentation model behind the preprocessing it was trained with."""

    def __init__(self, model, mixing):
        super().__init__()
        self.model = model
        self.register_buffer("mixing", mixing[None, :, :, None, None])
        self.register_buffer("mean", torch.tensor(model.encoder.mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(model.encoder.std)[None, :, None, None])

    def forward(self, images):
        rgb = (images[:, None] * self.mixing).sum(2)
        return self.model((rgb - self.mean) / self.std)


def centre_patch(path, size, standardised=False):
    """The middle `size` x `size` of one stored 8-bit plane, zero-padded where it is smaller.

    In 0-1, or with `standardised` z-scored by the whole plane's non-zero pixels.
    """
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[..., 0]
    image = zscore(image, foreground_statistics(image)) if standardised else image / 255.0
    patch = np.zeros((size, size), dtype=np.float32)
    height, width = min(size, image.shape[0]), min(size, image.shape[1])
    top, left = (image.shape[0] - height) // 2, (image.shape[1] - width) // 2
    out_top, out_left = (size - height) // 2, (size - width) // 2
    patch[out_top : out_top + height, out_left : out_left + width] = (
        image[top : top + height, left : left + width]
    )
    return patch


def sample_inputs(image_dir, size, channels, limit, standardised=False):
    """Up to `limit` model inputs cut from the images in `image_dir`, with the files they came from."""
    first_planes = sorted(
        p for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and p.stem.endswith("_0000")
    )[:limit]
    inputs = [
        np.stack([
            centre_patch(
                path.with_name(f"{path.stem[:-5]}_{channel:04d}{path.suffix}"), size, standardised
            )
            for channel in range(channels)
        ])
        for path in first_planes
    ]
    return np.stack(inputs).astype(np.float32), [p.name for p in first_planes]


def default_name(cfg):
    """The file stem a run is deployed under: `dinov3-convnext-t-307` for ConvNeXt-T trained on 307.

    A run refining another carries both models in its one file and is named apart from the run it
    refines: `dinov3-convnext-t-307-refine`. The UNet comes in one size and is named without one:
    `unet-311`.
    """
    number = re.search(r"\d+", cfg.train_dataset).group(0)
    parts = [cfg.model_name, number]
    if cfg.model_name != "unet":
        parts.insert(1, re.sub(r"^convnext(?=.)", "convnext-", cfg.variant))
    return "-".join(parts) + ("-refine" if cfg.refines else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--fold", help="cross-validation fold to export, overriding the config's")
    parser.add_argument("--output", required=True, help="path of the .onnx file to write")
    parser.add_argument("--reference", help="write sample inputs and this model's logits for them here")
    parser.add_argument("--reference-dir", help="images the reference inputs are cut from")
    parser.add_argument("--reference-samples", type=int, default=3)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if args.fold is not None:
        cfg = replace(cfg, fold=str(args.fold))
    if cfg.patching is None:
        raise SystemExit("only patchwise runs have a fixed input size to export at")
    device = torch.device("cpu")
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    size = cfg.patching.patch_size
    model = load_trained_model(cfg, args.checkpoint, device, classes)
    if cfg.zscore:
        channels = cfg.input_channels
        deployed = model.eval()
    else:
        mixing = channel_mixing(cfg)
        channels = mixing.shape[1]
        deployed = DeployedModel(model, mixing).eval()
    example = torch.rand(1, channels, size, size)
    _fix_adaptive_pools(deployed, example)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        deployed,
        example,
        str(output),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        dynamo=False,
    )
    exported = onnx.load(str(output))
    entry = exported.metadata_props.add()
    entry.key, entry.value = "input_normalization", "zscore" if cfg.zscore else "unit_range"
    # Slices of a z-stack the model expects projected into each image it is given.
    entry = exported.metadata_props.add()
    entry.key, entry.value = "slab_slices", str(cfg.patching.slab_slices if cfg.patching else 1)
    onnx.save(exported, str(output))
    print(f"Saved ONNX model to {output}")

    if args.reference:
        inputs, names = sample_inputs(
            args.reference_dir, size, channels, args.reference_samples, cfg.zscore
        )
        with torch.no_grad():
            logits = deployed(torch.from_numpy(inputs)).numpy()
        np.savez_compressed(args.reference, inputs=inputs, logits=logits, names=np.array(names))
        print(f"Saved {len(names)} reference sample(s) to {args.reference}")


if __name__ == "__main__":
    main()
