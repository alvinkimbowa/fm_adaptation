#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/sam3_linear.yaml}"
train=${train:-1}
predict=${predict:-1}
compute_metrics=${compute_metrics:-1}
report=${report:-1}
checkpoint=${checkpoint:-final}   # final | best | last
# Redoing a dataset that already has predictions costs a full forward pass per case for a result
# that cannot change, so this defaults to reusing them; set overwrite=1 to force everything afresh.
overwrite=${overwrite:-false}
gpu_id=${gpu_id:-0}
# Empty keeps the fold the config names; set it to train and score another one of the split.
fold=${fold:-}
# The adapter decoders need mmseg's UPerHead and DINOv3's compiled deformable attention, which only
# .venv-mm has; every probe and finetune run stays on the SAM3 environment.
venv=${venv:-~/UltrAi/projects/sam3/.venv}

export PYTHONPATH="${PYTHONPATH:-}:src"
export PATH="${venv/#\~/$HOME}/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$gpu_id"

if [[ "$train" -eq 1 ]]; then
    train_args=(--config "$config")
    [[ -n "$fold" ]] && train_args+=(--fold "$fold")
    # resume=1 continues the run from its own final.pt rather than starting over.
    [[ "${resume:-0}" -eq 1 ]] && train_args+=(--resume)
    python -m fm_adaptation.training "${train_args[@]}"
fi

if [[ "$predict" -eq 1 ]]; then
    predict_args=(--config "$config" --checkpoint "$checkpoint")
    [[ -n "$fold" ]] && predict_args+=(--fold "$fold")
    if [[ "$overwrite" == true ]]; then
        predict_args+=(--overwrite)
    fi
    python -m fm_adaptation.predict "${predict_args[@]}"
fi

# Scoring is its own stage, on its own interpreter -- see scripts/compute_metrics.sh. Restricted to
# the run this config names, so `train=0 predict=0 compute_metrics=1` rescores just that one, with no
# GPU involved.
if [[ "$compute_metrics" -eq 1 ]]; then
    "${metrics_venv:-$PWD/.venv-mm}/bin/python" -m fm_adaptation.compute_metrics \
        --config "$config" --overwrite ${fold:+--folds "$fold"}
fi

if [[ "$report" -eq 1 ]]; then
    bash scripts/run_report.sh
fi
