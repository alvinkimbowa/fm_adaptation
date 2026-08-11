#!/usr/bin/env bash
# DINOv3 ViT-Adapter heads (UperNet, Mask2Former) through mmsegmentation's own training loop.
set -euo pipefail

config="${1:-configs/dinov3_linear_sci204.yaml}"
head=${head:-upernet}      # upernet | m2f
injector=${injector:-0}    # 1 restores ViT-Adapter's injector; the run is named <head>_inj
epochs=${epochs:-40}
batch_size=${batch_size:-2}
lr=${lr:-0.0001}
crop_size=${crop_size:-896}
train=${train:-1}
report=${report:-1}
gpu_id=${gpu_id:-0}

export PYTHONPATH="${PYTHONPATH:-}:src"
export PATH=/home/ultrai/UltrAi/projects/fm_adaptation/.venv-mm/bin:$PATH
export CUDA_VISIBLE_DEVICES="$gpu_id"

python -m fm_adaptation.mmseg_run \
    --config "$config" \
    --head "$head" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --lr "$lr" \
    --crop-size "$crop_size" \
    --train "$train" \
    --injector "$injector"

if [[ "$report" -eq 1 ]]; then
    bash scripts/run_report.sh
fi
