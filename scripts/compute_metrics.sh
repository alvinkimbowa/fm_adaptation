#!/usr/bin/env bash
set -euo pipefail

# Scores saved predictions against the labels on disk. Prediction and scoring are separate stages:
# a prediction is written back at the case's native resolution, so scoring needs only two directories
# of label maps -- no model, no GPU. Run this after `predict`, or on its own to rescore anything.
#
results_dir=models
metrics_venv=${metrics_venv:-.venv-mm}
# A column whose metrics are newer than its predictions is left alone; set overwrite=1 to redo it.
overwrite=${overwrite:-0}
dry_run=${dry_run:-0}

# Which runs to score, selected the way run_report.sh and plot_qualitative.sh do it -- exact names,
# globs, or a `_suffix` tag. Leave a list empty to keep everything it selects.
models=(
    dinov3
    sam3
    # The baselines ship their own per-case metrics, so there is nothing here to compute for them.
)

datasets=(          # what a run was trained on
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
    Dataset213_combined_KE_smi_gfap
    # Dataset203_neurite_2px_scaleaug
    # Dataset205_neurite_2px_scaleaug_red
)

experiments=(       # the adaptation, i.e. the run name
    # linear
    # upernet
    # upernet_inj
    upernet_ours
    upernet_inj_ours
    upernet_inj_ft_ours
    upernet_inj_ft_dropsmi_ours
    upernet_inj_ft_dropany_ours
    upernet_inj_ft_balanced_ours
    upernet_inj_ft_balanced_dropany_ours
    upernet_inj_ft_init_ours
    # upernet_inj_ft_vit*_ours      # every ViT sweep at once
    # _ours                        # every run whose name ends in `ours`
    # m2f
)

splits=(
    validation
    test
)

folds=(0)

export PYTHONPATH="${PYTHONPATH:-}:src"

# `set -u` makes an empty array expansion an error, so each selection is only passed when it has one.
args=()
[[ ${#models[@]} -gt 0 ]] && args+=(--models "${models[@]}")
[[ ${#datasets[@]} -gt 0 ]] && args+=(--datasets "${datasets[@]}")
[[ ${#experiments[@]} -gt 0 ]] && args+=(--experiments "${experiments[@]}")
[[ ${#splits[@]} -gt 0 ]] && args+=(--splits "${splits[@]}")
[[ ${#folds[@]} -gt 0 ]] && args+=(--folds "${folds[@]}")
[[ "$overwrite" -eq 1 ]] && args+=(--overwrite)
[[ "$dry_run" -eq 1 ]] && args+=(--dry-run)

"${metrics_venv/#\~/$HOME}/bin/python" -m fm_adaptation.compute_metrics \
    --results-dir "$results_dir" \
    ${args[@]+"${args[@]}"}
