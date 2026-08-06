#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/pe_linear.yaml}"
train=1
predict=1
report=1
overwrite=false

export PYTHONPATH="${PYTHONPATH:-}:src"

if [[ "$train" -eq 1 ]]; then
    python -m fm_adaptation.training --config "$config"
fi

if [[ "$predict" -eq 1 ]]; then
    predict_args=(--config "$config")
    if [[ "$overwrite" == true ]]; then
        predict_args+=(--overwrite)
    fi
    python -m fm_adaptation.predict "${predict_args[@]}"
fi

if [[ "$report" -eq 1 ]]; then
    python -m fm_adaptation.report --results-dir models
fi
