#!/usr/bin/env bash
set -euo pipefail
config=${1:?config required}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${gpu_id:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/fm-instance-matplotlib
python=${venv:-$PWD/.venv-mm}/bin/python
if [[ ${overfit_steps:-0} -gt 0 ]]; then
    "$python" -m fm_adaptation.instance_training --config "$config" --overfit-steps "$overfit_steps"
    exit
fi
args=()
[[ ${resume:-0} == 1 ]] && args+=(--resume)
"$python" -m fm_adaptation.instance_training --config "$config" "${args[@]}"
"$python" -m fm_adaptation.instance_predict --config "$config" --subset val
"$python" -m fm_adaptation.instance_predict --config "$config" --subset test
