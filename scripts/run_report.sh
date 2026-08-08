#!/usr/bin/env bash
set -euo pipefail

results_dir=models
folds=0
# For comparison with MonoUNet and nnUNet
nnunet_dir=../nnUNet_fork/data/nnUNet_results
monounet_dirs=(
    ../monounetv2/models_v2/MonoUNetE123V2GatedDA
    ../monounetv2/models_v2/MonoUNetE123V2GatedS8DA
    ../monounetv2/models_v2/MonoUNetE123V2GatedS32DA
)

export PYTHONPATH="${PYTHONPATH:-}:src"
export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH

python -m fm_adaptation.report \
    --results-dir "$results_dir" \
    --folds "$folds" \
    --nnunet-results-dir "$nnunet_dir" \
    --monounet-results-dir "${monounet_dirs[@]}"
