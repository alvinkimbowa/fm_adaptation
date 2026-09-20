#!/usr/bin/env bash
set -euo pipefail

# Runs side by side on the same cases: a row per case, a column per run, one figure per evaluation
# set, written to results/qualitative/. Each run's own figures live beside its predictions under
# models/ and are drawn by scripts/plot_qualitative.sh, which selects runs the same way -- this
# script differs only in that each selected run is a column rather than a figure of its own.

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

# Which runs to draw, one column each. The four lists below name the parts a run directory is built
# from -- <model>/<train dataset>/<configuration>/fold_<n> -- so comment a line out to drop those
# columns. An empty list keeps every value that part can take. Entries match exactly, as a glob or as
# a `_suffix` tag, and a dataset by its number alone. Columns are ordered as the lists order them:
# model first, then training set, then configuration.
models=(
    # nnunet
    # sam3
    dinov3
)

train_datasets=(
    # Dataset070_Clarius_L15
    # Dataset072_GE_LQP9
    # Dataset073_GE_LE
    Dataset701_nmus_median_rgb
)

# Evaluation sets to draw. Empty takes every set all the chosen runs have predictions for.
# A set that one column trained on is still comparable: that column holds its imagesTs while the
# others hold the whole dataset, and the figure is drawn on the cases they share.
test_datasets=(
    # Dataset070_Clarius_L15
    # Dataset072_GE_LQP9
    # Dataset073_GE_LE
    Dataset701_nmus_median_rgb
)

configs=(
    convnexts_upernet_ft_aug_ours
    convnexts_upernet_ft_aug_gateonly_ours
    convnexts_upernet_ft_aug_promptenc_ours
    convnextb_upernet_ft_aug_ours
    convnextb_upernet_ft_aug_promptenc_ours
    convnext_upernet_ft_aug_ours
)

# Cases to draw, in this order; empty samples across the reference run's Dice range instead.
cases=(
    11.9_033
    11.9_048
)

folds=(
    0
)

# none          : one figure per evaluation set, every selected run that has it a column.
# train_dataset : one figure per training set as well, so a figure compares the configurations
#                 trained on one set instead of mixing training sets into the same columns.
group_by=${group_by:-train_dataset}

# Result trees to draw from. An external trainer lays its runs out the same way, so naming its tree
# here is enough for its runs to be named in `experiments` like any other. Runs with no config.yaml
# of their own read their datasets from raw_data_dir.
results_dirs=(
    models
    # ../knee_us_segmentation/data/nnUNet_results
    # ~/GAA/spinal_cord_injury/data/nnUNet_results
)
raw_data_dir=../nmus_segmentation/data/nnUNet_raw


splits=(
    test
)

rows=2            # cases down the figure
per_row=1         # cases side by side, each with its own image / gt / model columns
output_dir=results/qualitative
format=${format:-png}      # png | svg | pdf
# -1 draws a new sample of cases every run, overwriting the previous figure. Set a number to pin one.
seed=-1

# overlay  : image + pred overlay + gt contour        (1 panel per sample)
# pair     : image, image + gt + pred                 (2 panels)
# mask_pair: image, gt + pred on black                (2 panels)
# split    : image, image + gt, image + pred          (3 panels)
# masks    : image, gt mask, pred mask                (3 panels)
layout=split

# How a mask is painted, in every layout -- a layout only arranges the panels and decides
# whether a mask sits on black or over the image.
# contour | overlay | centerline
gt_style=${gt_style:-overlay}
pred_style=${pred_style:-overlay}
# red | green | blue | yellow | magenta | cyan | white, or `auto` to follow each class's own colour
gt_color=${gt_color:-cyan}
pred_color=${pred_color:-red}
gt_width=1
pred_width=1
alpha=0.8

crop=420          # auto (patch size for patchwise runs, whole image otherwise) | full | pixels

args=()
[[ "${skip_unchanged:-0}" -eq 1 ]] && args+=(--skip-unchanged)

python -m fm_adaptation.compare_qualitative \
    "${args[@]}" \
    --results-dir "${results_dirs[@]/#\~/$HOME}" \
    --raw-data-dir "${raw_data_dir/#\~/$HOME}" \
    ${models[@]+--models "${models[@]}"} \
    ${train_datasets[@]+--train-datasets "${train_datasets[@]}"} \
    ${configs[@]+--configs "${configs[@]}"} \
    ${folds[@]+--folds "${folds[@]}"} \
    ${test_datasets[@]+--test-datasets "${test_datasets[@]}"} \
    --splits "${splits[@]}" \
    --group-by "${group_by//_/-}" \
    --rows "$rows" \
    --per-row "$per_row" \
    --output-dir "$output_dir" \
    --format "$format" \
    --layout "$layout" \
    --gt-style "$gt_style" \
    --pred-style "$pred_style" \
    --gt-color "$gt_color" \
    --pred-color "$pred_color" \
    --gt-width "$gt_width" \
    --pred-width "$pred_width" \
    --alpha "$alpha" \
    --crop "$crop" \
    ${cases[@]+--cases "${cases[@]}"} \
    --seed "$seed"
