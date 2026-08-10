#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/sam3_linear.yaml}"
train=${train:-1}
predict=${predict:-1}
report=${report:-1}
checkpoint=final
overwrite=true
gpu_id=${gpu_id:-0}

export PYTHONPATH="${PYTHONPATH:-}:src"
export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export CUDA_VISIBLE_DEVICES="$gpu_id"

if [[ "$train" -eq 1 ]]; then
    python -m fm_adaptation.training --config "$config"
fi

if [[ "$predict" -eq 1 ]]; then
    predict_args=(--config "$config" --checkpoint "$checkpoint")
    if [[ "$overwrite" == true ]]; then
        predict_args+=(--overwrite)
    fi
    python -m fm_adaptation.predict "${predict_args[@]}"
fi

if [[ "$report" -eq 1 ]]; then
    bash scripts/run_report.sh
fi
