#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/UltrAi/projects/sam3/.venv/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:src"

# Sample one case with DINOv3 Dice above min_dice. Random draws can repeat in a small pool.
# -1: fresh random draw each run; 0 or above: reproducible selection.
# Overrides: dataset=yvonne seed=0 bash results/cihr_proposal_2026_09_05/plot_lesion_panel_qualitative.sh
seed=${seed:--3}
min_dice=${min_dice:-0.88}    # strict lower cutoff; 0.88 = >88%
dataset=${dataset:-yvonne}    # katie | mohammad | yvonne
case_prefix=""
case "$dataset" in
    katie)
        test_dataset=Dataset213_lesion_KE_smi_gfap
        nnunet_dataset=Dataset207_lesion_katie_contusion_smi_gfap
        source_split=Ts
        case_prefix=Katie_contusion__
        ;;
    mohammad)
        test_dataset=Dataset214_lesion_mohammad_smi_gfap
        nnunet_dataset="$test_dataset"
        source_split=Tr
        ;;
    yvonne)
        test_dataset=Dataset215_lesion_yvonne_smi_gfap
        nnunet_dataset="$test_dataset"
        source_split=Tr
        ;;
    *) echo "Unknown dataset: $dataset" >&2; exit 1 ;;
esac
raw_dataset="$HOME/GAA/spinal_cord_injury/data/nnUNet_raw/$test_dataset"
dino_results="models/dinov3/Dataset213_lesion_KE_smi_gfap/upernet_inj_ft_balanced_ours/fold_0/test/$test_dataset"
nnunet_results="$HOME/GAA/spinal_cord_injury/data/nnUNet_results/nnunet/Dataset105_lesion_eric_gfap_resized/nnUNetTrainer__nnUNetResEncUNetMPlans__2d/fold_0/test/$nnunet_dataset"
rotate=0
line_width=3.0
pred_style=mask    # mask: white on black | contour: white contour over image
output_dir=results/cihr_proposal_2026_09_05/figures/lesion

python results/cihr_proposal_2026_09_05/lesion_panel_qualitative.py \
    --raw-dataset "$raw_dataset" \
    --source-split "$source_split" --case-prefix "$case_prefix" \
    --dino-results "$dino_results" \
    --nnunet-results "$nnunet_results" \
    --seed "$seed" --min-dice "$min_dice" --rotate "$rotate" --line-width "$line_width" \
    --pred-style "$pred_style" \
    --output-dir "$output_dir"
