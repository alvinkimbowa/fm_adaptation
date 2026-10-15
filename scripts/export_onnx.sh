#!/usr/bin/env bash
# Writes a finished run's model as ONNX. Runs on the CPU, so no GPU is needed.
set -euo pipefail

config="${1:-configs/dinov3_convnext_upernet_ft_aug_nmus701.yaml}"
output="${2:-../knee_cat_seg/assets/convnextl701.onnx}"
checkpoint=${checkpoint:-best}   # best | final
# UperNet heads need mmseg, which only .venv-mm has.
venv=${venv:-.venv-mm}

export PYTHONPATH="${PYTHONPATH:-}:src"
export CUDA_VISIBLE_DEVICES=""

"${venv/#\~/$HOME}/bin/python" -m fm_adaptation.export_onnx \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output "$output"
