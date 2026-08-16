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
    Dataset090_Echo_EchoCP
    Dataset203_neurite_2px_scaleaug
    Dataset204_lesion_czi_B
    Dataset205_neurite_2px_scaleaug_red
)
experiments=(
    linear
    upernet
    upernet_inj
    upernet_ours
    upernet_inj_ours
    upernet_inj_ft_ours
    upernet_inj_ft_init_ours
    upernet_inj_ft_vits_ours
    # m2f
    # nonlinear
    linear_finetune
    # nonlinear_finetune
    linear_finetune_wd*    # every weight-decay sweep of the linear finetune
)

interval=30

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m fm_adaptation.plot_history \
    --datasets "${datasets[@]}" \
    --experiments "${experiments[@]}" \
    --interval "$interval"
