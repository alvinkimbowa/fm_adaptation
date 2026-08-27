#!/usr/bin/env bash
# Score a finished run under flips, rotations, rescalings and a dropped stain, to see whether it
# learned the anatomy or the orientation of its training set. Trains nothing; reads a checkpoint.
set -euo pipefail

# Runs to check. The point is the comparison, so keep the augmented runs beside the baseline they
# are meant to improve on -- a row only means something next to the one it changed.
configs=(
    # The MYKE sweep, kept for reference -- these three already have their full twelve-row tables.
    # configs/dinov3_upernet_inj_ft_balanced_sci208.yaml
    # configs/dinov3_upernet_inj_ft_balanced_aug_sci208.yaml
    # configs/dinov3_upernet_inj_ft_balanced_dropsmi_aug_sci208.yaml
    configs/dinov3_upernet_inj_ft_balanced_aug_sci219.yaml
    configs/dinov3_upernet_inj_ft_balanced_dropsmi_aug_sci219.yaml
    configs/dinov3_upernet_inj_ft_balanced_dropsmi_aug_sci207.yaml
)

# Rows to evaluate. Empty runs the whole sweep; naming rows runs just those, always alongside `none`,
# and writes to its own file so a partial table cannot replace a full one. Names must match
# `robustness.TRANSFORMS` exactly, spaces included, which is why this is an array.
transforms=(
    "drop SMI"
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
    # A run still training has no checkpoint to read. Skipping rather than failing lets the same
    # command be re-run as runs land, picking up whichever are finished.
    run_dir=$(PYTHONPATH=src "${venv/#\~/$HOME}/bin/python" -c "
from fm_adaptation.config import ExperimentConfig
print(ExperimentConfig.from_yaml('$config').run_dir)")
    if [[ ! -f "$run_dir/$checkpoint.pt" ]]; then
        echo "skipping $config -- no $checkpoint.pt in $run_dir yet"
        continue
    fi
    "${venv/#\~/$HOME}/bin/python" -m fm_adaptation.robustness \
        --config "$config" --checkpoint "$checkpoint" --output-dir "$output_dir" \
        ${transforms:+--transforms "${transforms[@]}"} \
        ${overwrite:+--overwrite}
done

# Qualitative figures beside each transform's predictions, drawn now rather than left to a later
# pass: `--splits` is emptied because these columns are named after the transform, not after a split.
if [[ "$figures" -eq 1 ]]; then
    "${figure_venv/#\~/$HOME}/bin/python" -m fm_adaptation.plot_qualitative \
        --results-dir "$output_dir" --splits
fi
