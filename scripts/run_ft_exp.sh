#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/sam3_linear.yaml}"
finetune=1
predict=1
report=1
overwrite=false

export PYTHONPATH="${PYTHONPATH:-}:src"

if [[ "$finetune" -eq 1 ]]; then
    python -m fm_adaptation.finetune --config "$config"
fi

if [[ "$predict" -eq 1 ]]; then
    predict_args=(--config "$config")
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
