#!/usr/bin/env bash
set -euo pipefail

results_dir=models
folds=0

# Which rows to tabulate. Leave a list empty to keep everything it selects; otherwise list entries the
# way plot_curves.sh and plot_qualitative.sh do -- an exact name, a glob, or a `_suffix` tag, and for a
# dataset its number alone. `models` picks the models to compare, `datasets` what a run was trained on,
# and `experiments` the adaptation.
# `experiments` narrows the foundation-model rows only, so the baselines stay in as the comparison;
# use `models` to drop those.
models=(
    dinov3
    sam3
    nnunet
)

datasets=(
    # Dataset080_BUSBRA_GE_Logiq_5
    # Dataset082_BUSBRA_Toshiba_Aplio_300
    # Dataset083_BUSBRA_U_Systems
    # Dataset084_KidneyUS_Philips
    # Dataset086_MMOTU_2D
    # Dataset090_Echo_EchoCP
    Dataset105_lesion_eric_gfap_resized
    Dataset204_lesion_mohammad_eric_gfap
    Dataset206_lesion_yvonne_gfap
    Dataset208_lesion_MYKE_smi_gfap
    Dataset209_lesion_MYE_smi_gfap
    Dataset213_lesion_KE_smi_gfap
    Dataset217_lesion_MY_smi_gfap
    # Dataset203_neurites_yvonne_smi_2px_scaleaug
)

experiments=(
    # linear
    # linear_finetune
    # linear_finetune_wd*    # every weight-decay sweep of the linear finetune
    # nonlinear
    # nonlinear_finetune
    # upernet
    # upernet_inj
    upernet_ours
    upernet_inj_ours
    upernet_inj_ft_ours
    upernet_inj_ft_dropsmi_ours
    upernet_inj_ft_dropany_ours
    upernet_inj_ft_balanced_ours
    upernet_inj_ft_balanced_dropany_ours
    # upernet_inj_ft_poly_ours
    # upernet_inj_ft_init_ours
    # upernet_inj_ft_vitb_ours
    # upernet_inj_ft_vits_ours
    # upernet_inj_ft_vitb_poly_ours
    # upernet_inj_ft_vits_poly_ours
    # m2f
)

# For comparison with MonoUNet and nnUNet
nnunet_dirs=(
    ../xtinyunet/data/nnUNet_results
    ~/GAA/spinal_cord_injury/data/nnUNet_results
)
nnunet_raw_data_dir=~/GAA/spinal_cord_injury/data/nnUNet_raw
monounet_dirs=(
    ../monounetv2/models_v2/MonoUNetE123V2GatedDA
    ../monounetv2/models_v2/MonoUNetE123V2GatedS8DA
    ../monounetv2/models_v2/MonoUNetE123V2GatedS32DA
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
[[ ${#datasets[@]} -gt 0 ]] && report_args+=(--datasets "${datasets[@]}")
[[ ${#experiments[@]} -gt 0 ]] && report_args+=(--experiments "${experiments[@]}")

python -m fm_adaptation.report \
    --results-dir "$results_dir" \
    --folds "$folds" \
    --nnunet-results-dir "${nnunet_dirs[@]}" \
    --nnunet-raw-data-dir "$nnunet_raw_data_dir" \
    --monounet-results-dir "${monounet_dirs[@]}" \
    ${report_args[@]+"${report_args[@]}"}

# Training curves and qualitative figures alongside the tables. Both only redraw what has changed,
# so this stays cheap enough to run after every experiment.
if [[ "${figures:-1}" -eq 1 ]]; then
    bash scripts/plot_curves.sh
    skip_unchanged=1 bash scripts/plot_qualitative.sh
fi
