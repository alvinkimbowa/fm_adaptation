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
    nnunet
    # sam3
    dinov3
)

train_datasets=(
    # Dataset105_lesion_eric_gfap_resized
    # Dataset207_lesion_katie_contusion_smi_gfap
    # Dataset208_lesion_MYKE_smi_gfap
    # Dataset209_lesion_MYE_smi_gfap
    # Dataset213_lesion_KE_smi_gfap
    # Dataset217_lesion_MY_smi_gfap
    # Dataset218_lesion_eric_smi_gfap
    # Dataset219_lesion_MYK_smi_gfap
    Dataset203_neurites_yvonne_smi_2px_scaleaug
    Dataset301_neurite_yvonne_b2_smi
    Dataset302_neurite_yvonne_b2_smi_1px
)

# Evaluation sets to draw. Empty takes every set all the chosen runs have predictions for.
# A set that one column trained on is still comparable: that column holds its imagesTs while the
# others hold the whole dataset, and the figure is drawn on the cases they share.
test_datasets=(
    # Dataset207_lesion_katie_contusion_smi_gfap
    # Dataset211_lesion_paul_widefield_smi_gfap
    # Dataset214_lesion_mohammad_smi_gfap
    Dataset203_neurites_yvonne_smi_2px_scaleaug
    Dataset301_neurite_yvonne_b2_smi
    Dataset300_neurite_yvonne_smi
    Dataset302_neurite_yvonne_b2_smi_1px
)

configs=(
    # linear
    # upernet
    # upernet_inj
    # upernet_inj_ft_balanced_dropsmi_aug_ours
    # upernet_inj_ft_balanced_aug_gfap_ours
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
    convnextt_upernet_ft_aug_p512_ours
)

folds=(
    0
)

# Result trees to draw from. An external trainer lays its runs out the same way, so naming its tree
# here is enough for its runs to be named in `experiments` like any other. Runs with no config.yaml
# of their own read their datasets from raw_data_dir.
results_dirs=(
    models
    ~/GAA/spinal_cord_injury/data/nnUNet_results
)
raw_data_dir=~/GAA/spinal_cord_injury/data/nnUNet_raw


splits=(
    test
)

rows=8            # cases down the figure
per_row=2         # cases side by side, each with its own image / gt / model columns
output_dir=results/qualitative
# -1 draws a new sample of cases every run, overwriting the previous figure. Set a number to pin one.
seed=-1

# overlay  : image + pred overlay + gt contour        (1 panel per sample)
# pair     : image, image + gt + pred                 (2 panels)
# mask_pair: image, gt + pred on black                (2 panels)
# split    : image, image + gt, image + pred          (3 panels)
# masks    : image, gt mask, pred mask                (3 panels)
layout=masks

# How a mask is painted, in every layout -- a layout only arranges the panels and decides
# whether a mask sits on black or over the image.
# contour | overlay | centerline
gt_style=${gt_style:-contour}
pred_style=${pred_style:-contour}
# red | green | blue | yellow | magenta | cyan | white, or `auto` to follow each class's own colour
gt_color=${gt_color:-white}
pred_color=${pred_color:-yellow}
gt_width=1
pred_width=2
alpha=0.5

crop=auto         # auto (patch size for patchwise runs, whole image otherwise) | full | pixels

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
    --rows "$rows" \
    --per-row "$per_row" \
    --output-dir "$output_dir" \
    --layout "$layout" \
    --gt-style "$gt_style" \
    --pred-style "$pred_style" \
    --gt-color "$gt_color" \
    --pred-color "$pred_color" \
    --gt-width "$gt_width" \
    --pred-width "$pred_width" \
    --alpha "$alpha" \
    --crop "$crop" \
    --seed "$seed"
