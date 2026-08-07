#!/usr/bin/env bash
set -euo pipefail


config="${1:-configs/sam3_linear.yaml}"
finetune=1
predict=1
report=1
checkpoint=final
overwrite=false
gpu_id=0

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"
export CUDA_VISIBLE_DEVICES="$gpu_id"

if [[ "$finetune" -eq 1 ]]; then
    python -m fm_adaptation.finetune --config "$config"
fi

if [[ "$predict" -eq 1 ]]; then
    predict_args=(--config "$config" --checkpoint "$checkpoint")
    if [[ "$overwrite" == true ]]; then
        predict_args+=(--overwrite)
    fi
    python -m fm_adaptation.predict "${predict_args[@]}"
fi

if [[ "$report" -eq 1 ]]; then
    python -m fm_adaptation.report \
        --results-dir models \
        --nnunet-results-dir ../nnUNet_fork/data/nnUNet_results
fi
