#!/usr/bin/env bash
set -euo pipefail

# Per-image results by image-metadata group, as box plots with every image drawn over them, and the
# training set's images per group as bars. Written to results/plots/<task>/, one pair per training
# set: <split>__<metric>_by_<field>__<dataset>.png and training_by_<field>__<dataset>.png.

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

models=(
    dinov3
)

train_datasets=(
    Dataset701_nmus_median_rgb
)

# One box per configuration inside each group, in this order and colour.
configs=(
    convnextt_upernet_ft_ours
    convnexts_upernet_ft_aug_ours
    convnextb_upernet_ft_aug_ours
    convnext_upernet_ft_aug_ours
    # convnext_upernet_ft_aug_ctxearly_ours
    # convnext_upernet_ft_aug_ctxmid_ours
    # convnext_upernet_ft_aug_ctxlate_ours
    # convnext_upernet_ft_aug_promptenc_ours
    # upernet_inj_ft_vits_aug_ours
    # upernet_inj_ft_vitb_aug_ours
    # upernet_inj_ft_aug_ours
)

folds=(
    0
)

results_dirs=(
    models
)
raw_data_dir=../nmus_segmentation/data/nnUNet_raw

# test | validation: which of a run's own held-out parts the boxes read.
split=${split:-test}
metric=${metric:-dice}

# Group by this image-metadata field, showing these values in this order; empty keeps every value.
# `all` pools the values shown and the images the field says nothing about.
split_by=${split_by:-location}
groups=(
    wrist
    elbow
    mid-arm
    mid-forearm
    "distal forearm"
)

output_dir=results/plots
format=${format:-png}      # png | svg | pdf
seed=0

python -m fm_adaptation.plot_results \
    --results-dir "${results_dirs[@]}" \
    --raw-data-dir "$raw_data_dir" \
    ${models[@]+--models "${models[@]}"} \
    ${train_datasets[@]+--train-datasets "${train_datasets[@]}"} \
    ${configs[@]+--configs "${configs[@]}"} \
    ${folds[@]+--folds "${folds[@]}"} \
    --split "$split" \
    --metric "$metric" \
    --split-by "$split_by" \
    ${groups[@]+--groups "${groups[@]}"} \
    --output-dir "$output_dir" \
    --format "$format" \
    --seed "$seed"
