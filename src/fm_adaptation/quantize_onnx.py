"""Quantise an exported ONNX model to INT8 and check both files.

Runs on ONNX Runtime alone, so it imports nothing from the rest of the package: the interpreter that
has ONNX Runtime is not the one that has torch. The FP32 file is compared with the logits the
original model produced for the same inputs (`export_onnx.py --reference`), and the INT8 file with
the FP32 one.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

IMAGE_SUFFIXES = {".png", ".tif", ".tiff"}


def centre_patch(path, size):
    """The middle `size` x `size` of one stored 8-bit plane in 0-1, zero-padded where it is smaller."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    image = image / 255.0
    if image.ndim == 3:
        image = image[..., 0]
    patch = np.zeros((size, size), dtype=np.float32)
    height, width = min(size, image.shape[0]), min(size, image.shape[1])
    top, left = (image.shape[0] - height) // 2, (image.shape[1] - width) // 2
    out_top, out_left = (size - height) // 2, (size - width) // 2
    patch[out_top : out_top + height, out_left : out_left + width] = (
        image[top : top + height, left : left + width]
    )
    return patch


def model_input(path, size, channels):
    planes = [
        centre_patch(path.with_name(f"{path.stem[:-5]}_{channel:04d}{path.suffix}"), size)
        for channel in range(channels)
    ]
    return np.stack(planes)[None].astype(np.float32)


class PatchReader(CalibrationDataReader):
    def __init__(self, paths, input_name, size, channels):
        self._paths = iter(paths)
        self._input_name = input_name
        self._size = size
        self._channels = channels

    def get_next(self):
        path = next(self._paths, None)
        if path is None:
            return None
        return {self._input_name: model_input(path, self._size, self._channels)}


def dice(a, b):
    a, b = a > 0, b > 0
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2.0 * np.logical_and(a, b).sum() / total)


def run(session, inputs):
    return session.run(None, {session.get_inputs()[0].name: inputs})[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fp32", required=True, help="the exported model")
    parser.add_argument("--int8", required=True, help="where to write the quantised model")
    parser.add_argument("--calib-dir", required=True, help="images to calibrate on")
    parser.add_argument("--calib-limit", type=int, default=50)
    parser.add_argument("--per-channel", action="store_true",
                        help="quantise weights per channel instead of per tensor")
    parser.add_argument("--reference", help="inputs and logits written by export_onnx.py")
    parser.add_argument("--report", required=True, help="where to write the JSON report")
    args = parser.parse_args()

    graph_input = onnx.load(args.fp32).graph.input[0]
    _, channels, size, _ = (d.dim_value for d in graph_input.type.tensor_type.shape.dim)
    paths = sorted(
        p for p in Path(args.calib_dir).iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES and p.stem.endswith("_0000")
    )[: max(1, args.calib_limit)]
    if not paths:
        raise SystemExit(f"no calibration images in {args.calib_dir}")

    # The calibrater otherwise keeps every image's activations until the end, over a gigabyte each
    # for a network this size, so it is told to reduce them every few images. ONNX Runtime fails with
    # "No data is collected" when the last reduction leaves nothing behind, which is what happens
    # when the interval divides the image count.
    flush_every = next(n for n in (4, 3, 5) if len(paths) % n)
    quantize_static(
        args.fp32,
        args.int8,
        PatchReader(paths, graph_input.name, size, channels),
        per_channel=args.per_channel,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        quant_format=QuantFormat.QDQ,
        extra_options={"CalibMaxIntermediateOutputs": flush_every},
    )
    print(f"Saved INT8 model to {args.int8} (calibrated on {len(paths)} image(s))")

    providers = ["CPUExecutionProvider"]
    fp32 = ort.InferenceSession(args.fp32, providers=providers)
    int8 = ort.InferenceSession(args.int8, providers=providers)
    report = {"fp32_model": args.fp32, "int8_model": args.int8, "calibration_images": len(paths)}

    if args.reference:
        reference = np.load(args.reference)
        exported = run(fp32, reference["inputs"])
        report["fp32_vs_original"] = {
            "samples": [str(name) for name in reference["names"]],
            "max_abs_logit_difference": float(np.abs(exported - reference["logits"]).max()),
            "mask_dice": dice(exported.argmax(1), reference["logits"].argmax(1)),
        }
        print(f"FP32 vs original model: {report['fp32_vs_original']}")

    samples = []
    for path in paths:
        inputs = model_input(path, size, channels)
        fp32_mask, int8_mask = run(fp32, inputs).argmax(1), run(int8, inputs).argmax(1)
        samples.append({
            "image": path.name,
            "fp32_foreground": int(fp32_mask.sum()),
            "int8_foreground": int(int8_mask.sum()),
            "dice": dice(fp32_mask, int8_mask),
        })
    report["int8_vs_fp32"] = {
        "dice_mean": float(np.mean([s["dice"] for s in samples])),
        "samples": samples,
    }
    print(f"INT8 vs FP32 mask dice, mean over {len(samples)}: {report['int8_vs_fp32']['dice_mean']:.4f}")

    Path(args.report).write_text(json.dumps(report, indent=2))
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()
