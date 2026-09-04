#!/usr/bin/env bash
set -euo pipefail

results_dir=models

# Which rows to tabulate. The four lists below name the parts a run directory is built from --
# <model>/<train dataset>/<configuration>/fold_<n> -- so comment a line out to drop the rows that
# carry it. An empty list keeps every value that part can take. Entries match exactly, as a glob or
# as a `_suffix` tag, and a dataset by its number alone.
# A plans is a network of its own, so each is named here separately: `nnUNet` takes the stock 2d
# plans, `nnUNetResEncM` the residual-encoder preset, and `nnUNet*` every plans at once.
models=(
    nnUNet
    # nnUNetResEncM
    XTinyUNet     # its own model, so no `nnUNet` entry above selects it
    MonoUNet
    sam3
    dinov3
)

# Group rows by their training dataset and rank best/second-best values within each group. Set to 0
# to sort by model/configuration and rank across all selected training datasets instead.
group_by_train_dataset="${group_by_train_dataset:-1}"

# Row order within each group: empty follows the lists below -- `models` first, then `configs` and
# `train_datasets`. `params` or `trainable` orders by network size instead, and `sort_descending=1`
# puts the largest first. The columns are fixed by `test_datasets` either way.
sort_by="${sort_by:-params}"
sort_descending="${sort_descending:-0}"

train_datasets=(
    Dataset072_GE_LQP9
    Dataset073_GE_LE
    Dataset070_Clarius_L15
    # Dataset071_Sonix-Touch
    # Dataset080_BUSBRA_GE_Logiq_5
    # Dataset082_BUSBRA_Toshiba_Aplio_300
    # # Dataset083_BUSBRA_U_Systems
    # Dataset084_KidneyUS_Philips
    # Dataset086_MMOTU_2D
    # Dataset089_Echo_CardiacUDA
    # Dataset090_Echo_EchoCP
    # Dataset093_Echo_CardiacNet
)

# Columns, in this order. A `Test` column -- each row's own held-out split -- is always present and is
# never part of the cross-dataset average. The average column appears when more than one set is
# selected. A cell whose set shares images with the row's training set is greyed and left out of both
# the average and the best-value marking.
test_datasets=(
    Dataset072_GE_LQP9
    Dataset073_GE_LE
    Dataset070_Clarius_L15
    # Dataset071_Sonix-Touch
    Dataset080_BUSBRA_GE_Logiq_5
    Dataset082_BUSBRA_Toshiba_Aplio_300
    Dataset083_BUSBRA_U_Systems
    Dataset084_KidneyUS_Philips
    Dataset086_MMOTU_2D
    Dataset089_Echo_CardiacUDA
    Dataset090_Echo_EchoCP
    Dataset093_Echo_CardiacNet
)


# A baseline carries no configuration, so this list does not reach it; drop those with `models`.
configs=(
    # linear
    # linear_finetune
    convnextt_upernet_ft_ours
    # convnexts_upernet_ft_ours
    upernet_inj_ft_vits_ours
    upernet_inj_ft_vits_aug_ours
    # upernet_inj_ft_vitb_ours
    # upernet_ours
    # upernet_inj_ours
    # upernet_inj_ft_ours
    # upernet_inj_ft_poly_ours
    # upernet_inj_ft_init_ours
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
)

# One fold gives that fold's row; several are pooled into a single row, labelled with the folds each
# run actually holds. Empty keeps every fold as its own row.
folds=(
    0
)

nnunet_dirs=(
    ../xtinyunet/data/nnUNet_results
    ~/GAA/spinal_cord_injury/data/nnUNet_results
    ../knee_us_segmentation/data/nnUNet_results
)
nnunet_raw_data_dir=~/GAA/spinal_cord_injury/data/nnUNet_raw
# MonoUNet keeps one architecture per directory and no configuration level, so each directory named
# here is one model, taking its row label from the architecture the directory is called after.
monounet_dirs=(
    ../monounet/models/MonoUNetE123V2GatedDA
)
export PYTHONPATH="${PYTHONPATH:-}:src"

# The params columns come from models/parameter_counts.json, which nothing else writes, so a new run
# name would report no size at all. Fill in whatever the file does not cover yet; a run already
# counted is never rebuilt, so this costs seconds once every configuration is in there. Needs the
# mmseg venv (the mmseg heads) and a visible card (SAM3's builder refuses to build without one).
if [[ "${params:-1}" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES="${params_gpu:-0}" .venv-mm/bin/python -m fm_adaptation.count_params \
        --only-missing \
        --results-dir "$results_dir" \
        --nnunet-results-dir "${nnunet_dirs[@]}" \
        --monounet-results-dir "${monounet_dirs[@]}"
fi

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH

# `set -u` makes an empty array expansion an error, so each selection is only passed when it has one.
report_args=()
[[ ${#models[@]} -gt 0 ]] && report_args+=(--models "${models[@]}")
[[ ${#train_datasets[@]} -gt 0 ]] && report_args+=(--train-datasets "${train_datasets[@]}")
[[ ${#configs[@]} -gt 0 ]] && report_args+=(--configs "${configs[@]}")
[[ ${#test_datasets[@]} -gt 0 ]] && report_args+=(--test-datasets "${test_datasets[@]}")
[[ ${#folds[@]} -gt 0 ]] && report_args+=(--folds "${folds[@]}")
[[ -n "$sort_by" ]] && report_args+=(--sort-by "$sort_by" --sort-descending "$sort_descending")

python -m fm_adaptation.report \
    --results-dir "$results_dir" \
    --group-by-train-dataset "$group_by_train_dataset" \
    --nnunet-results-dir "${nnunet_dirs[@]}" \
    --monounet-results-dir "${monounet_dirs[@]}" \
    --nnunet-raw-data-dir "$nnunet_raw_data_dir" \
    ${report_args[@]+"${report_args[@]}"}
