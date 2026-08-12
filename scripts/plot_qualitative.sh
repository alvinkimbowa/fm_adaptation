#!/usr/bin/env bash
set -euo pipefail

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

datasets=(
    Dataset080_BUSBRA_GE_Logiq_5
    Dataset082_BUSBRA_Toshiba_Aplio_300
    Dataset083_BUSBRA_U_Systems
    Dataset084_KidneyUS_Philips
    Dataset086_MMOTU_2D
    Dataset203_neurite_2px_scaleaug
    Dataset204_lesion_czi_B
    Dataset205_neurite_2px_scaleaug_red
)
experiments=(
    linear
    upernet
    upernet_inj
    upernet_inj_ours
    upernet_inj_ft_ours
    m2f
    # nonlinear
    linear_finetune
    # nonlinear_finetune
)
splits=(
    validation
    test
)

rows=3
cols=4          # samples per row

# overlay  : image + pred overlay + gt contour        (1 panel per sample)
# pair     : image, image + gt + pred                 (2 panels)
# split    : image, image + gt, image + pred          (3 panels)
# masks    : image, gt mask, pred mask                (3 panels)
layout=masks

# contour | overlay | centerline
gt_style=contour
pred_style=overlay
gt_color=green
pred_color=red
gt_width=2
pred_width=2
alpha=0.4

crop=auto       # auto (patch size for patchwise runs, whole image otherwise) | full | pixels
seed=0

# Running this script by hand always redraws, so edits to the options above take effect. The automatic
# refresh in run_report.sh sets skip_unchanged=1 to leave figures whose results have not moved.
args=()
[[ "${skip_unchanged:-0}" -eq 1 ]] && args+=(--skip-unchanged)

python -m fm_adaptation.plot_qualitative \
    "${args[@]}" \
    --datasets "${datasets[@]}" \
    --experiments "${experiments[@]}" \
    --splits "${splits[@]}" \
    --rows "$rows" \
    --cols "$cols" \
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
