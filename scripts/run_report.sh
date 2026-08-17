#!/usr/bin/env bash
set -euo pipefail

results_dir=models
folds=0

# Which rows to tabulate. Leave a list empty to keep everything it selects; otherwise list entries the
# way plot_curves.sh and plot_qualitative.sh do -- exact names, globs, or a `_suffix` tag. `models`
# picks the models to compare, `datasets` what a run was trained on, and `experiments` the adaptation.
# `experiments` narrows the foundation-model rows only, so the baselines stay in as the comparison;
# use `models` to drop those.
models=(
    dinov3
    sam3
    nnU-Net       # case, hyphens and underscores are ignored, so `nnunet` works too
    XTinyUNet     # its own model, so `nnU-Net` above does not select it
    # MonoUNet-t
    # MonoUNet-B
    # MonoUNet-L
    monounet*     # all three MonoUNets
)

datasets=(
    Dataset080_BUSBRA_GE_Logiq_5
    Dataset082_BUSBRA_Toshiba_Aplio_300
    Dataset083_BUSBRA_U_Systems
    Dataset084_KidneyUS_Philips
    Dataset086_MMOTU_2D
    Dataset090_Echo_EchoCP
    Dataset203_neurite_2px_scaleaug
    Dataset204_lesion_czi_B
    Dataset205_neurite_2px_scaleaug_red
)

experiments=(
    # linear
    # linear_finetune
    # linear_finetune_wd*    # every weight-decay sweep of the linear finetune
    # nonlinear
    # nonlinear_finetune
    # upernet
    # upernet_inj
    # upernet_ours
    # upernet_inj_ours
    # upernet_inj_ft_ours
    upernet_inj_ft_poly_ours
    # upernet_inj_ft_init_ours
    upernet_inj_ft_vitb_ours
    upernet_inj_ft_vits_ours
    upernet_inj_ft_vitb_poly_ours
    upernet_inj_ft_vits_poly_ours
    # m2f
)

# For comparison with MonoUNet and nnUNet
nnunet_dirs=(
    ../nnUNet_fork/data/nnUNet_results
)
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
    --monounet-results-dir "${monounet_dirs[@]}" \
    ${report_args[@]+"${report_args[@]}"}

# Training curves and qualitative figures alongside the tables. Both only redraw what has changed,
# so this stays cheap enough to run after every experiment.
if [[ "${figures:-1}" -eq 1 ]]; then
    bash scripts/plot_curves.sh
    skip_unchanged=1 bash scripts/plot_qualitative.sh
fi
