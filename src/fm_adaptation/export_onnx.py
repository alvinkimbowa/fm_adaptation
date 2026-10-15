"""Write a finished run's model as ONNX, for apps that run it with ONNX Runtime.

The graph takes a batch of images already resized and normalised the way the run's encoder
preprocesses them, at its fixed square input size, and returns per-class logits at that size.
"""

import argparse
from pathlib import Path

import torch
from torch import nn

from .config import ExperimentConfig
from .data import num_classes
from .models import load_trained_model


class StaticAdaptiveAvgPool2d(nn.Module):
    """`AdaptiveAvgPool2d` spelt as slices and means, for an input whose size is fixed.

    ONNX has no adaptive pooling, and torch only exports it when the output divides the input. The
    bins here are torch's own -- bin `i` of `n` over `size` spans floor(i*size/n) to
    ceil((i+1)*size/n) -- so the result is identical; the slices are fixed when the graph is traced.
    """

    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)

    @staticmethod
    def _bins(size: int, count: int):
        return [((i * size) // count, -(-((i + 1) * size) // count)) for i in range(count)]

    def forward(self, x):
        rows = self._bins(x.shape[-2], self.output_size[0])
        cols = self._bins(x.shape[-1], self.output_size[1])
        pooled = [
            torch.cat([x[..., r0:r1, c0:c1].mean(dim=(-2, -1), keepdim=True) for c0, c1 in cols], dim=-1)
            for r0, r1 in rows
        ]
        return torch.cat(pooled, dim=-2)


def _replace_adaptive_pools(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.AdaptiveAvgPool2d):
            setattr(module, name, StaticAdaptiveAvgPool2d(child.output_size))
        else:
            _replace_adaptive_pools(child)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--output", required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    if cfg.prompt is not None or cfg.context is not None:
        raise SystemExit("only single-image runs export: this one takes a prompt or context frames")
    classes = num_classes(cfg.raw_data_dir / cfg.train_dataset)
    model = load_trained_model(cfg, args.checkpoint, torch.device("cpu"), classes)
    size = model.encoder.input_size
    example = torch.zeros(1, 3, size, size)
    with torch.no_grad():
        reference = model(example)
        _replace_adaptive_pools(model)
        if not torch.allclose(model(example), reference, atol=1e-4):
            raise RuntimeError("the static pooling does not reproduce the model's own")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (example,),
            str(output),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )
    print(f"Wrote {output} ({classes} classes, input 3x{size}x{size})")


if __name__ == "__main__":
    main()
