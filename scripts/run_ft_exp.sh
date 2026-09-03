#!/usr/bin/env bash
set -euo pipefail


config="${1:-configs/sam3_linear.yaml}"
finetune=${finetune:-1}
predict=${predict:-1}
report=${report:-1}
checkpoint=final
overwrite=false
gpu_id=${gpu_id:-0}

# The DINOv3 ViT-Adapter encoders build through mmsegmentation, so those configs run from the
# mmseg venv: `venv=.venv-mm bash scripts/run_ft_exp.sh <config>`.
venv=${venv:-~/UltrAi/projects/sam3/.venv}
export PATH="$(eval echo "$venv")/bin:$PATH"
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
    bash scripts/run_report.sh
fi
