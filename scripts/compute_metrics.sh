#!/usr/bin/env bash
set -euo pipefail

# Scores saved predictions against the labels on disk. Prediction and scoring are separate stages:
# a prediction is written back at the case's native resolution, so scoring needs only two directories
# of label maps -- no model, no GPU. Run this after `predict`, or on its own to rescore anything.
#
results_dir=models
metrics_venv=${metrics_venv:-.venv-mm}
# The two stages are independent: runs under `results_dir`, then the nnU-Net trees named below.
# `score_runs=0` scores the baselines alone; an empty `nnunet_dirs` scores our runs alone.
score_runs=${score_runs:-1}
# A column whose metrics are newer than its predictions is left alone; set overwrite=1 to redo it.
overwrite=${overwrite:-0}
dry_run=${dry_run:-0}

# Which runs to score, selected the way run_report.sh and plot_qualitative.sh do it -- exact names,
# globs, or a `_suffix` tag. Leave a list empty to keep everything it selects.
models=(
    dinov3
    # sam3
    # nnU-Net is not selected here -- it lives outside `results_dir` and is picked up by
    # `nnunet_dirs` below, by its own directory layout.
)

datasets=(          # what a run was trained on
    # Dataset105_lesion_eric_gfap_resized
    # Dataset208_lesion_MYKE_smi_gfap
    # Dataset209_lesion_MYE_smi_gfap
    # Dataset213_lesion_KE_smi_gfap
    # Dataset217_lesion_MY_smi_gfap
    # Dataset218_lesion_eric_smi_gfap
    # Dataset219_lesion_MYK_smi_gfap
    Dataset214_lesion_mohammad_smi_gfap
    Dataset215_lesion_yvonne_smi_gfap
    Dataset203_neurites_yvonne_smi_2px_scaleaug
    Dataset301_neurite_yvonne_b2_smi
    Dataset302_neurite_yvonne_b2_smi_1px
    Dataset304_neurite_yvonne_b2_smi_1px_scaleaug
    Dataset070_Clarius_L15
    Dataset071_Sonix-Touch
    Dataset072_GE_LQP9
    Dataset073_GE_LE
)

experiments=(       # the adaptation, i.e. the run name
    linear
    upernet
    m2f
    convnext_upernet_p512_ours
    convnext_upernet_ft_p512_ours
    upernet_inj_ft_ours
    upernet_inj_ft_balanced_aug_ours
    upernet_inj_ft_p512_ours
    convnext_upernet_ft_aug_p512_ours
    convnextb_upernet_ft_aug_p512_ours
    convnexts_upernet_ft_aug_p512_ours
    convnextt_upernet_ft_aug_p512_ours
    convnextt_upernet_ft_aug_p512_skelrec_ours
    convnextt_upernet_ft_aug_p512_red_skelrec_ours
    convnextt_upernet_ft_aug_p512_red_distw_ours
    convnextt_upernet_ft_aug_p512_red_skelrec_distw_ours
    convnextt_upernet_ft_aug_p512_red_distw10_ours
    convnextt_upernet_ft_aug_p512_red_skelrec_distw10_ours
    convnextt_upernet_ft_aug_p512_red_ours
    convnextb_upernet_aug_p512_ours
    convnextt_upernet_aug_p512_ours
    # _ours                        # every run whose name ends in `ours`
)

splits=(
    validation
    test
)

folds=(0)

# Results trees belonging to other projects: nnU-Net baselines, and MonoUNet laid out
# `<architecture>/<trained on>/fold_N/test/<tested on>/preds` -- the same arrangement with the
# architecture where nnU-Net keeps a trainer. They carry none of this project's config, so they are
# selected from their own layout instead: `datasets` above picks what a run was trained on,
# `tested_on` below which of its evaluation sets to measure. Leave a list empty to skip those trees.
# A tree with nothing for the current selection says so and is passed over.
nnunet_dirs=(
    # ~/GAA/spinal_cord_injury/data/nnUNet_results
    ../knee_us_segmentation/data/nnUNet_results
)
monounet_dirs=(
    ../monounet/models/MonoUNetE123V2GatedDA
)

# Where the datasets those trees were trained and evaluated on live. Each column is measured against
# the labels of the set it was evaluated on, and a dataset is found by its number, so the roots are
# searched rather than paired with a tree.
raw_data_dirs=(
    ../knee_us_segmentation/data/nnUNet_raw
    ~/GAA/spinal_cord_injury/data/nnUNet_raw
)

tested_on=(
    # Dataset105_lesion_eric_gfap_resized
    # Dataset207_lesion_katie_contusion_smi_gfap
    # Dataset210_lesion_interrater_MY_smi_gfap
    # Dataset211_lesion_paul_widefield_smi_gfap
    # Dataset214_lesion_mohammad_smi_gfap
    # Dataset215_lesion_yvonne_smi_gfap
    # Dataset218_lesion_eric_smi_gfap
    # Dataset203_neurites_yvonne_smi_2px_scaleaug
    # Dataset300_neurite_yvonne_smi
    # Dataset301_neurite_yvonne_b2_smi
    Dataset070_Clarius_L15
    Dataset071_Sonix-Touch
    Dataset072_GE_LQP9
    Dataset073_GE_LE
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

if [[ "$score_runs" -eq 1 ]]; then
    "$python" -m fm_adaptation.compute_metrics \
        --results-dir "$results_dir" \
        ${args[@]+"${args[@]}"}
fi

if [[ ${#nnunet_dirs[@]} -gt 0 || ${#monounet_dirs[@]} -gt 0 ]]; then
    foreign_args=()
    [[ ${#nnunet_dirs[@]} -gt 0 ]] && foreign_args+=(--nnunet-results-dir "${nnunet_dirs[@]}")
    [[ ${#monounet_dirs[@]} -gt 0 ]] && foreign_args+=(--monounet-results-dir "${monounet_dirs[@]}")
    [[ ${#datasets[@]} -gt 0 ]] && foreign_args+=(--datasets "${datasets[@]}")
    [[ ${#folds[@]} -gt 0 ]] && foreign_args+=(--folds "${folds[@]}")
    [[ ${#tested_on[@]} -gt 0 ]] && foreign_args+=(--tested-on "${tested_on[@]}")
    [[ "$overwrite" -eq 1 ]] && foreign_args+=(--overwrite)
    [[ "$dry_run" -eq 1 ]] && foreign_args+=(--dry-run)

    "$python" -m fm_adaptation.compute_metrics \
        --raw-data-dir "${raw_data_dirs[@]}" \
        ${foreign_args[@]+"${foreign_args[@]}"}
fi
