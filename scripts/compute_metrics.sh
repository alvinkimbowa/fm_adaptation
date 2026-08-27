#!/usr/bin/env bash
set -euo pipefail

# Scores saved predictions against the labels on disk. Prediction and scoring are separate stages:
# a prediction is written back at the case's native resolution, so scoring needs only two directories
# of label maps -- no model, no GPU. Run this after `predict`, or on its own to rescore anything.
#
results_dir=models
metrics_venv=${metrics_venv:-.venv-mm}
# A column whose metrics are newer than its predictions is left alone; set overwrite=1 to redo it.
overwrite=${overwrite:-0}
dry_run=${dry_run:-0}

# Which runs to score, selected the way run_report.sh and plot_qualitative.sh do it -- exact names,
# globs, or a `_suffix` tag. Leave a list empty to keep everything it selects.
models=(
    dinov3
    sam3
    # The baselines ship their own per-case metrics, so there is nothing here to compute for them.
)

datasets=(          # what a run was trained on
    # Dataset080_BUSBRA_GE_Logiq_5
    # Dataset082_BUSBRA_Toshiba_Aplio_300
    # Dataset083_BUSBRA_U_Systems
    # Dataset084_KidneyUS_Philips
    # Dataset086_MMOTU_2D
    # Dataset090_Echo_EchoCP
    Dataset105_lesion_eric_gfap_resized
    Dataset208_lesion_MYKE_smi_gfap
    Dataset209_lesion_MYE_smi_gfap
    Dataset213_lesion_KE_smi_gfap
    Dataset217_lesion_MY_smi_gfap
    Dataset218_lesion_eric_smi_gfap
    Dataset219_lesion_MYK_smi_gfap
    # Dataset203_neurites_yvonne_smi_2px_scaleaug
)

experiments=(       # the adaptation, i.e. the run name
    # linear
    # upernet
    # upernet_inj
    upernet_ours
    upernet_inj_ours
    upernet_inj_ft_ours
    upernet_inj_ft_dropsmi_ours
    upernet_inj_ft_dropany_ours
    upernet_inj_ft_balanced_ours
    upernet_inj_ft_balanced_dropany_ours
    upernet_inj_ft_init_ours
    # upernet_inj_ft_vit*_ours      # every ViT sweep at once
    # _ours                        # every run whose name ends in `ours`
    # m2f
)

splits=(
    validation
    test
)

folds=(0)

# nnU-Net baselines, which are trained outside this project and carry none of its config. They are
# selected from their own directory layout instead: `datasets` above picks what a run was trained on,
# `nnunet_tested_on` which of its evaluation sets to measure. Leave `nnunet_dirs` empty to skip them.
nnunet_dirs=(
    ~/GAA/spinal_cord_injury/data/nnUNet_results
)
nnunet_raw_data_dir=~/GAA/spinal_cord_injury/data/nnUNet_raw
nnunet_tested_on=(
    Dataset105_lesion_eric_gfap_resized
    Dataset207_lesion_katie_contusion_smi_gfap
    Dataset210_lesion_interrater_MY_smi_gfap
    Dataset211_lesion_paul_widefield_smi_gfap
    Dataset214_lesion_mohammad_smi_gfap
    Dataset215_lesion_yvonne_smi_gfap
    Dataset218_lesion_eric_smi_gfap
    # The older lesion and traced sets these runs also predicted are left out: they belong to
    # families no table here shows.
)

export PYTHONPATH="${PYTHONPATH:-}:src"

# `set -u` makes an empty array expansion an error, so each selection is only passed when it has one.
args=()
[[ ${#models[@]} -gt 0 ]] && args+=(--models "${models[@]}")
[[ ${#datasets[@]} -gt 0 ]] && args+=(--datasets "${datasets[@]}")
[[ ${#experiments[@]} -gt 0 ]] && args+=(--experiments "${experiments[@]}")
[[ ${#splits[@]} -gt 0 ]] && args+=(--splits "${splits[@]}")
[[ ${#folds[@]} -gt 0 ]] && args+=(--folds "${folds[@]}")
[[ "$overwrite" -eq 1 ]] && args+=(--overwrite)
[[ "$dry_run" -eq 1 ]] && args+=(--dry-run)

python=${metrics_venv/#\~/$HOME}/bin/python

"$python" -m fm_adaptation.compute_metrics \
    --results-dir "$results_dir" \
    ${args[@]+"${args[@]}"}

if [[ ${#nnunet_dirs[@]} -gt 0 ]]; then
    nnunet_args=()
    [[ ${#datasets[@]} -gt 0 ]] && nnunet_args+=(--datasets "${datasets[@]}")
    [[ ${#folds[@]} -gt 0 ]] && nnunet_args+=(--folds "${folds[@]}")
    [[ ${#nnunet_tested_on[@]} -gt 0 ]] && nnunet_args+=(--tested-on "${nnunet_tested_on[@]}")
    [[ "$overwrite" -eq 1 ]] && nnunet_args+=(--overwrite)
    [[ "$dry_run" -eq 1 ]] && nnunet_args+=(--dry-run)

    "$python" -m fm_adaptation.compute_metrics \
        --nnunet-results-dir "${nnunet_dirs[@]}" \
        --raw-data-dir "$nnunet_raw_data_dir" \
        ${nnunet_args[@]+"${nnunet_args[@]}"}
fi
