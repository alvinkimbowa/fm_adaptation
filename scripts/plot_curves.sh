#!/usr/bin/env bash
set -euo pipefail

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH
export PYTHONPATH="${PYTHONPATH:-}:src"

datasets=(
    # Dataset080_BUSBRA_GE_Logiq_5
    # Dataset082_BUSBRA_Toshiba_Aplio_300
    # Dataset083_BUSBRA_U_Systems
    # Dataset084_KidneyUS_Philips
    # Dataset086_MMOTU_2D
    # Dataset090_Echo_EchoCP
    Dataset072_GE_LQP9
    Dataset073_GE_LE
    Dataset070_Clarius_L15
)
experiments=(
    # linear
    # linear_finetune
    # nonlinear
    # upernet
    # upernet_inj
    # upernet_inj_ft_balanced_ours
    # upernet_inj_ft_vitb_ours
    # upernet_inj_ft_ours
    upernet_inj_ft_p512_ours
    # m2f   # Mask2Former
    convnextt_upernet_ft_ours
    convnexts_upernet_ft_ours
    upernet_inj_ft_vits_ours
)

interval=30

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m fm_adaptation.plot_history \
    --datasets "${datasets[@]}" \
    --experiments "${experiments[@]}" \
    --interval "$interval"
