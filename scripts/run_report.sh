#!/usr/bin/env bash
set -euo pipefail

results_dir=models

# Which rows to tabulate. The four lists below name the parts a run directory is built from --
# <model>/<train dataset>/<configuration>/fold_<n> -- so comment a line out to drop the rows that
# carry it. An empty list keeps every value that part can take. Entries match exactly, as a glob or
# as a `_suffix` tag, and a dataset by its number alone.
models=(
    nnUNet
    sam3
    dinov3
)

train_datasets=(
    # Dataset105_lesion_eric_gfap_resized
    # Dataset218_lesion_eric_smi_gfap
    Dataset207_lesion_katie_contusion_smi_gfap
    # Dataset208_lesion_MYKE_smi_gfap
    # Dataset209_lesion_MYE_smi_gfap
    # Dataset213_lesion_KE_smi_gfap
    # Dataset217_lesion_MY_smi_gfap
    Dataset219_lesion_MYK_smi_gfap
    Dataset203_neurites_yvonne_smi_2px_scaleaug
    # Dataset301_neurite_yvonne_b2_smi
    Dataset302_neurite_yvonne_b2_smi_1px
    Dataset304_neurite_yvonne_b2_smi_1px_scaleaug
)

# Columns, in this order. A `Test` column -- each row's own held-out split -- is always present and is
# never part of the cross-dataset average. The average column appears when more than one set is
# selected. A cell whose set shares images with the row's training set is greyed and left out of both
# the average and the best-value marking.
test_datasets=(
    Dataset207_lesion_katie_contusion_smi_gfap
    Dataset218_lesion_eric_smi_gfap
    Dataset214_lesion_mohammad_smi_gfap
    Dataset215_lesion_yvonne_smi_gfap
    Dataset211_lesion_paul_widefield_smi_gfap
    Dataset301_neurite_yvonne_b2_smi
    Dataset300_neurite_yvonne_smi
    Dataset302_neurite_yvonne_b2_smi_1px
    Dataset304_neurite_yvonne_b2_smi_1px_scaleaug
)


# A baseline carries no configuration, so this list does not reach it; drop those with `models`.
configs=(
    # linear
    # linear_finetune
    # nonlinear
    # nonlinear_finetune
    # upernet
    # upernet_inj
    # upernet_ours
    # upernet_inj_ours
    upernet_inj_ft_ours
    # upernet_inj_ft_dropsmi_ours
    # upernet_inj_ft_dropany_ours
    # upernet_inj_ft_balanced_ours
    # upernet_inj_ft_balanced_dropany_ours
    # upernet_inj_ft_balanced_aug_ours
    upernet_inj_ft_balanced_dropsmi_aug_ours
    upernet_inj_ft_balanced_aug_gfap_ours
    # upernet_inj_ft_poly_ours
    # upernet_inj_ft_init_ours
    # upernet_inj_ft_vitb_ours
    # upernet_inj_ft_vits_ours
    # m2f
    # upernet_inj_ft_p512_ours
    # convnext_upernet_p512_ours
    # convnext_upernet_ft_p512_ours
    # convnext_upernet_ft_init_p512_ours
    # convnext_upernet_ft_aug_p512_ours
    # convnextb_upernet_ft_aug_p512_ours
    # convnexts_upernet_ft_aug_p512_ours
    # convnextb_upernet_aug_p512_ours
    # convnextt_upernet_aug_p512_ours
    # convnextt_upernet_ft_aug_p512_ours
    convnextt_upernet_ft_aug_p512_red_ours
)

# One fold gives that fold's row; several are pooled into a single row, labelled with the folds each
# run actually holds. Empty keeps every fold as its own row.
folds=(
    0
)

nnunet_dirs=(
    ../xtinyunet/data/nnUNet_results
    ~/GAA/spinal_cord_injury/data/nnUNet_results
)
nnunet_raw_data_dir=~/GAA/spinal_cord_injury/data/nnUNet_raw
export PYTHONPATH="${PYTHONPATH:-}:src"

# The params columns come from models/parameter_counts.json, which nothing else writes, so a new run
# name would report no size at all. Fill in whatever the file does not cover yet; a run already
# counted is never rebuilt, so this costs seconds once every configuration is in there. Needs the
# mmseg venv (the mmseg heads) and a visible card (SAM3's builder refuses to build without one).
if [[ "${params:-1}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${params_gpu:-0}" .venv-mm/bin/python -m fm_adaptation.count_params \
        --only-missing \
        --results-dir "$results_dir" \
        --nnunet-results-dir "${nnunet_dirs[@]}"
fi

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH

# `set -u` makes an empty array expansion an error, so each selection is only passed when it has one.
report_args=()
[[ ${#models[@]} -gt 0 ]] && report_args+=(--models "${models[@]}")
[[ ${#train_datasets[@]} -gt 0 ]] && report_args+=(--train-datasets "${train_datasets[@]}")
[[ ${#configs[@]} -gt 0 ]] && report_args+=(--configs "${configs[@]}")
[[ ${#test_datasets[@]} -gt 0 ]] && report_args+=(--test-datasets "${test_datasets[@]}")
[[ ${#folds[@]} -gt 0 ]] && report_args+=(--folds "${folds[@]}")

python -m fm_adaptation.report \
    --results-dir "$results_dir" \
    --nnunet-results-dir "${nnunet_dirs[@]}" \
    --nnunet-raw-data-dir "$nnunet_raw_data_dir" \
    ${report_args[@]+"${report_args[@]}"}
