#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# One case as a publication plate: a 2x2 of bare panels, lettered a to d, written to results/cihr_proposal_2026_09_05/figures/neurites/.
# Nothing is drawn on a panel but the picture itself -- no titles, no legend, no scores. Grids of many
# cases are scripts/plot_qualitative.sh and scripts/plot_comparison_qualitative.sh; one script per
# question, rather than one script with a mode switch.

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

# Which run supplies the prediction panel. The four lists below name the parts a run directory is
# built from -- <model>/<train dataset>/<configuration>/fold_<n>. Entries match exactly, as a glob or
# as a `_suffix` tag, and a dataset by its number alone. With two `prediction` panels, name two
# configurations; they fill the panels in the order the lists order them.
models=(
    dinov3
)

train_datasets=(
    Dataset302_neurite_yvonne_b2_smi_1px
)

configs=(
    convnextt_upernet_ft_aug_p512_red_ours
)

folds=(
    0
)

# Evaluation sets to draw, one figure each. Empty takes every set the run has predictions for.
test_datasets=(
    Dataset303_neurite_yvonne_in_vitro_smi
)

splits=(
    test
)

# What each panel holds, reading left to right then top to bottom:
#   image      : the case's image, in the stain planes the run was trained on
#   manual     : the annotator's traces
#   other      : the next directory from other_predictions, black when there is none
#   prediction : the next selected run's traces
panels=(image manual other prediction)

# Directories of another method's masks, named after the same cases. Any folder will do -- no run
# directory and no config.yaml. An `other` panel with nothing to show stays black.
other_predictions=(
)

# Which case to draw. Empty draws one at random from those the run predicted, a fresh one each time
# the script is run unless a seed is pinned below.
case=""

# Centre crop applied identically to every panel, in source pixels: one number for a square, two for
# height and width, empty for the whole image.
size=()

# Degrees anticlockwise. The slides are taller than they are wide and the plate reads as landscape.
rotate=90

# skeleton reduces every mask to its one-pixel centreline, so the annotator and the model are
# compared at the same stroke width. none draws each at the width it was stored at.
postprocess=skeleton
line_width=1.5
mask_color=white

letters=abcd
letter_size=42
letter_color=white
# Space between panels, as a fraction of a panel. The white figure behind it makes the dividing lines.
gutter=0.01

output_dir=results/cihr_proposal_2026_09_05/figures/neurites
format=png
dpi=""
# -1 draws a new case every run. Set a number to keep drawing the same one.
seed=-1

args=()
[[ -n "$case" ]] && args+=(--case "$case")
[[ -n "$dpi" ]] && args+=(--dpi "$dpi")
[[ ${#size[@]} -gt 0 ]] && args+=(--size "${size[@]}")
[[ ${#other_predictions[@]} -gt 0 ]] && args+=(--other-predictions "${other_predictions[@]/#\~/$HOME}")

python results/cihr_proposal_2026_09_05/panel_qualitative.py \
    "${args[@]}" \
    --results-dir models \
    ${models[@]+--models "${models[@]}"} \
    ${train_datasets[@]+--train-datasets "${train_datasets[@]}"} \
    ${configs[@]+--configs "${configs[@]}"} \
    ${folds[@]+--folds "${folds[@]}"} \
    ${test_datasets[@]+--test-datasets "${test_datasets[@]}"} \
    --splits "${splits[@]}" \
    --panels "${panels[@]}" \
    --rotate "$rotate" \
    --postprocess "$postprocess" \
    --line-width "$line_width" \
    --mask-color "$mask_color" \
    --letters "$letters" \
    --letter-size "$letter_size" \
    --letter-color "$letter_color" \
    --gutter "$gutter" \
    --output-dir "$output_dir" \
    --format "$format" \
    --seed "$seed"
