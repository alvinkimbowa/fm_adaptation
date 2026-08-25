#!/usr/bin/env bash
set -euo pipefail

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

comparison_models=(
    dinov3
    sam3
)

datasets=(
    # Dataset080_BUSBRA_GE_Logiq_5
    # Dataset082_BUSBRA_Toshiba_Aplio_300
    # Dataset083_BUSBRA_U_Systems
    # Dataset084_KidneyUS_Philips
    # Dataset086_MMOTU_2D
    # Dataset090_Echo_EchoCP
    Dataset204_lesion_czi_B
    Dataset206_lesion_120_czi_B
    Dataset208_combined_MYKE_smi_gfap
    Dataset209_combined_MYE_smi_gfap
    # Dataset203_neurite_2px_scaleaug
    # Dataset205_neurite_2px_scaleaug_red
)
experiments=(
    # linear
    # upernet
    # upernet_inj
    # upernet_ours
    # upernet_inj_ours
    upernet_inj_ft_ours
    upernet_inj_ft_dropsmi_ours
    upernet_inj_ft_dropany_ours
    upernet_inj_ft_balanced_ours
    upernet_inj_ft_balanced_dropany_ours
    # upernet_inj_ft_poly_ours
    upernet_inj_ft_init_ours
    # upernet_inj_ft_vitb_ours
    # upernet_inj_ft_vits_ours
    # upernet_inj_ft_vitb_poly_ours
    # upernet_inj_ft_vits_poly_ours
    # m2f
    # nonlinear
    # linear_finetune
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
# mask_pair: image, gt + pred on black                (2 panels)
# split    : image, image + gt, image + pred          (3 panels)
# masks    : image, gt mask, pred mask                (3 panels)
layout=masks

# contour | overlay | centerline
gt_style=contour
pred_style=overlay
# red | green | blue | yellow | magenta | cyan | white, or `auto` to follow each class's own colour
gt_color=white
pred_color=red
gt_width=1
pred_width=2
alpha=0.5

crop=auto       # auto (patch size for patchwise runs, whole image otherwise) | full | pixels
seed=0

# The comparison figure: the same cases drawn across every selected run, one column per report row,
# written to results/qualitative/<trained-on>/. It reuses everything above -- the datasets, the runs,
# the styles, the layout -- so there is one place to change what gets drawn. `comparison_rows` is the
# number of cases, since a comparison row is a case rather than a grid cell.
comparison_rows=8
# Runs to put in the comparison figure. Empty follows `experiments` above; narrowing it here keeps the
# per-run figures broad while the comparison stays to the handful of runs worth reading side by side.
comparison_experiments=(
    # upernet_inj_ft_balanced_ours     # narrow to this when comparison_across=1
)
# Cases side by side on one row, each with its own image / gt / model columns. `comparison_rows` is
# still the number of cases, so 5 cases at 2 per row is 3 rows, the last one half empty.
comparison_per_row=2
comparison_dir=results/qualitative
# 1 merges the training sets into one figure per evaluation set, so a column is the same
# configuration trained on a different dataset -- written to <comparison_dir>/across_training_sets/.
comparison_across=${comparison_across:-0}
# Set to 1 to skip the per-run figures and redraw only the comparisons -- the quick loop when tuning
# how a comparison looks, since the per-run figures are the slow half.
comparison_only=1
# -1 draws a new sample of cases every run, overwriting the previous figure. Set a number to pin one.
comparison_seed=-1

# Running this script by hand always redraws, so edits to the options above take effect. The automatic
# refresh in run_report.sh sets skip_unchanged=1 to leave figures whose results have not moved.
args=()
[[ "${skip_unchanged:-0}" -eq 1 ]] && args+=(--skip-unchanged)

if [[ "$comparison_only" -ne 1 ]]; then
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
fi

# The same runs again on shared cases: one figure per (trained-on, split, tested-on), a row per case
# and a column per report row. Selected exactly as the per-run figures are -- every evaluation set the
# chosen runs have predictions for.
compare_args=()
[[ ${#comparison_models[@]} -gt 0 ]] && compare_args+=(--models "${comparison_models[@]}")
[[ "${comparison_across:-0}" -eq 1 ]] && compare_args+=(--across-training-sets)
# Across training sets every training set contributes its own column, so carrying several adaptations
# as well multiplies the columns until the headers overlap into noise. Narrow to one unless asked
# otherwise; the balanced run is the one every training set has.
if [[ "$comparison_across" -eq 1 && ${#comparison_experiments[@]} -eq 0 ]]; then
    comparison_experiments=(upernet_inj_ft_balanced_ours)
fi
compare_experiments=("${comparison_experiments[@]:-${experiments[@]}}")
python -m fm_adaptation.compare_qualitative \
    "${args[@]}" \
    ${compare_args[@]+"${compare_args[@]}"} \
    --datasets "${datasets[@]}" \
    --experiments "${compare_experiments[@]}" \
    --splits "${splits[@]}" \
    --rows "$comparison_rows" \
    --per-row "$comparison_per_row" \
    --output-dir "$comparison_dir" \
    --layout "$layout" \
    --gt-style "$gt_style" \
    --pred-style "$pred_style" \
    --gt-color "$gt_color" \
    --pred-color "$pred_color" \
    --gt-width "$gt_width" \
    --pred-width "$pred_width" \
    --alpha "$alpha" \
    --crop "$crop" \
    --seed "$comparison_seed"
