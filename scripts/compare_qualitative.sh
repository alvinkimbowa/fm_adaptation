#!/usr/bin/env bash
set -euo pipefail

# Runs side by side on the same cases: a row per case, a column per run, one figure per evaluation
# set, written to results/qualitative/. Each run's own figures live beside its predictions under
# models/ and are drawn by scripts/plot_qualitative.sh -- this script is only about comparisons.

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

# One column each, in this order, named the way models/ names them:
#   <model>/<trained on>/<configuration>/fold_<n>
# Any combination of runs can be compared this way -- they need not share a training set or a
# configuration. Globs work: */Dataset219*/*aug*/fold_0 takes a family.
experiments=(
    dinov3/Dataset208_lesion_MYKE_smi_gfap/upernet_inj_ft_balanced_aug_ours/fold_0
    dinov3/Dataset219_lesion_MYK_smi_gfap/upernet_inj_ft_balanced_aug_ours/fold_0
    dinov3/Dataset219_lesion_MYK_smi_gfap/upernet_inj_ft_balanced_aug_gfap_ours/fold_0
    dinov3/Dataset207_lesion_katie_contusion_smi_gfap/upernet_inj_ft_balanced_dropsmi_aug_ours/fold_0
)

# Evaluation sets to draw. Empty takes every set all the chosen runs have predictions for.
# A set that one row trained on is still comparable: that row holds its imagesTs while the others
# hold the whole dataset, and the figure is drawn on the cases they share.
datasets=(
    Dataset210_lesion_interrater_MY_smi_gfap
    Dataset212_lesion_katie_dorsal_column_smi_gfap
    Dataset211_lesion_paul_widefield_smi_gfap
)

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
layout=split

# contour | overlay | centerline
gt_style=contour
pred_style=contour
# red | green | blue | yellow | magenta | cyan | white, or `auto` to follow each class's own colour
gt_color=white
pred_color=yellow
gt_width=1
pred_width=2
alpha=0.5

crop=auto         # auto (patch size for patchwise runs, whole image otherwise) | full | pixels

args=()
[[ "${skip_unchanged:-0}" -eq 1 ]] && args+=(--skip-unchanged)

python -m fm_adaptation.compare_qualitative \
    "${args[@]}" \
    --experiments "${experiments[@]}" \
    ${datasets[@]+--datasets "${datasets[@]}"} \
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
