#!/usr/bin/env bash
# Score a finished run under flips, rotations, rescalings and a dropped stain, to see whether it
# learned the anatomy or the orientation of its training set. Trains nothing; reads a checkpoint.
set -euo pipefail

# Runs to check. The point is the comparison, so keep the augmented runs beside the baseline they
# are meant to improve on -- a row only means something next to the one it changed.
configs=(
    configs/dinov3_upernet_inj_ft_balanced_sci208.yaml
    configs/dinov3_upernet_inj_ft_balanced_aug_sci208.yaml
    configs/dinov3_upernet_inj_ft_balanced_dropsmi_aug_sci208.yaml
)

output_dir=${output_dir:-results_analysis/robustness}
checkpoint=${checkpoint:-final}
figures=${figures:-1}
gpu_id=${gpu_id:-0}
# The adapter decoder needs mmseg's UPerHead, which only .venv-mm has; the figures need scikit-image,
# which only the SAM3 environment has. Two stages, two interpreters, as run_probe_exp.sh does.
venv=${venv:-$PWD/.venv-mm}
figure_venv=${figure_venv:-~/UltrAi/projects/sam3/.venv}

export PYTHONPATH="${PYTHONPATH:-}:src"
export CUDA_VISIBLE_DEVICES="$gpu_id"

for config in "${configs[@]}"; do
    "${venv/#\~/$HOME}/bin/python" -m fm_adaptation.robustness \
        --config "$config" --checkpoint "$checkpoint" --output-dir "$output_dir" \
        ${overwrite:+--overwrite}
done

# Qualitative figures beside each transform's predictions, drawn now rather than left to a later
# pass: `--splits` is emptied because these columns are named after the transform, not after a split.
if [[ "$figures" -eq 1 ]]; then
    "${figure_venv/#\~/$HOME}/bin/python" -m fm_adaptation.plot_qualitative \
        --results-dir "$output_dir" --splits
fi
