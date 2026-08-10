#!/usr/bin/env bash
set -euo pipefail

results_dir=models
folds=0
# For comparison with MonoUNet and nnUNet
nnunet_dirs=(
    ../nnUNet_fork/data/nnUNet_results
)
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
    --nnunet-results-dir "${nnunet_dirs[@]}" \
    --monounet-results-dir "${monounet_dirs[@]}"

# Training curves and qualitative figures alongside the tables. Both only redraw what has changed,
# so this stays cheap enough to run after every experiment.
if [[ "${figures:-1}" -eq 1 ]]; then
    bash scripts/plot_curves.sh
    skip_unchanged=1 bash scripts/plot_qualitative.sh
fi
