#!/usr/bin/env bash
# Export a trained patchwise run to ONNX: an FP32 file, an INT8 file quantised from it, and a report
# checking the first against the original model and the second against the first. The models go to
# the tracer app's assets, where it discovers them; the report and the reference samples to its debug
# directory.
set -euo pipefail

config="${1:?usage: export_onnx.sh <config.yaml>}"
checkpoint=${checkpoint:-best}   # best | final
# Empty keeps the fold the config names.
fold=${fold:-}
# Images the INT8 calibration and both checks are cut from; empty takes the training dataset's own
# imagesTs. Point it at the images the model will meet once deployed.
calib_dir=${calib_dir:-}
calib_limit=${calib_limit:-50}
quantize=${quantize:-1}
app_dir=${app_dir:-~/GAA/MTGNeuriteTracer}
# The model needs torch and mmseg; quantising needs ONNX Runtime, which that environment lacks.
venv=${venv:-$PWD/.venv-mm}
onnx_venv=${onnx_venv:-~/GAA/MTGNeuriteTracer/.venv}

export PYTHONPATH="${PYTHONPATH:-}:src"
# The cards are usually training; export and quantisation are light enough for the CPU.
export CUDA_VISIBLE_DEVICES=""

describe=(--config "$config" --checkpoint "$checkpoint")
[[ -n "$fold" ]] && describe+=(--fold "$fold")
read -r name default_calib < <(
    "${venv/#\~/$HOME}/bin/python" - "${describe[@]}" <<'EOF'
import argparse
from dataclasses import replace

from fm_adaptation.config import ExperimentConfig
from fm_adaptation.export_onnx import default_name

parser = argparse.ArgumentParser()
parser.add_argument("--config")
parser.add_argument("--checkpoint")
parser.add_argument("--fold")
args = parser.parse_args()
cfg = ExperimentConfig.from_yaml(args.config)
if args.fold is not None:
    cfg = replace(cfg, fold=str(args.fold))
print(default_name(cfg), cfg.raw_data_dir / cfg.train_dataset / "imagesTs")
EOF
)
[[ "$checkpoint" != best ]] && name+="-$checkpoint"
calib_dir="${calib_dir:-$default_calib}"
out_dir="${app_dir/#\~/$HOME}/assets"
debug_dir="${app_dir/#\~/$HOME}/debug"
mkdir -p "$out_dir" "$debug_dir"

"${venv/#\~/$HOME}/bin/python" -m fm_adaptation.export_onnx \
    "${describe[@]}" \
    --output "$out_dir/${name}_fp32.onnx" \
    --reference "$debug_dir/${name}_reference.npz" \
    --reference-dir "$calib_dir"

if [[ "$quantize" -eq 1 ]]; then
    "${onnx_venv/#\~/$HOME}/bin/python" src/fm_adaptation/quantize_onnx.py \
        --fp32 "$out_dir/${name}_fp32.onnx" \
        --int8 "$out_dir/${name}_int8.onnx" \
        --calib-dir "$calib_dir" \
        --calib-limit "$calib_limit" \
        --reference "$debug_dir/${name}_reference.npz" \
        --report "$debug_dir/${name}_validation.json"
fi
