#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/sam3_linear.yaml}"
train=${train:-1}
predict=${predict:-1}
report=${report:-1}
checkpoint=${checkpoint:-final}   # final | best | last
# Redoing a dataset that already has predictions costs a full forward pass per case for a result
# that cannot change, so this defaults to reusing them; set overwrite=1 to force everything afresh.
overwrite=${overwrite:-true}
gpu_id=${gpu_id:-0}
# The adapter decoders need mmseg's UPerHead and DINOv3's compiled deformable attention, which only
# .venv-mm has; every probe and finetune run stays on the SAM3 environment.
venv=${venv:-~/UltrAi/projects/sam3/.venv}

export PYTHONPATH="${PYTHONPATH:-}:src"
export PATH="${venv/#\~/$HOME}/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu_id"

if [[ "$train" -eq 1 ]]; then
    train_args=(--config "$config")
    # resume=1 continues the run from its own final.pt rather than starting over.
    [[ "${resume:-0}" -eq 1 ]] && train_args+=(--resume)
    python -m fm_adaptation.training "${train_args[@]}"
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
