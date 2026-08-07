#!/usr/bin/env bash
set -euo pipefail

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

datasets=(
    Dataset080_BUSBRA_GE_Logiq_5
    Dataset084_KidneyUS_Philips
    Dataset086_MMOTU_2D
)
experiments=(
    linear
    # nonlinear
    linear_finetune
    # nonlinear_finetune
    linear_finetune_wd*    # every weight-decay sweep of the linear finetune
)

interval=30

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m fm_adaptation.plot_history \
    --datasets "${datasets[@]}" \
    --experiments "${experiments[@]}" \
    --watch \
    --interval "$interval"
