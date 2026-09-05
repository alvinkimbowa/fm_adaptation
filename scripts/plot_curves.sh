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
    # Dataset072_GE_LQP9
    # Dataset073_GE_LE
    # Dataset070_Clarius_L15
    Dataset208_lesion_MYKE_smi_gfap
    Dataset209_lesion_MYE_smi_gfap
    Dataset213_lesion_KE_smi_gfap
    Dataset217_lesion_MY_smi_gfap
    Dataset218_lesion_eric_smi_gfap
    Dataset219_lesion_MYK_smi_gfap
    Dataset203_neurites_yvonne_smi_2px_scaleaug
    Dataset301_neurite_yvonne_b2_smi
    Dataset302_neurite_yvonne_b2_smi_1px
    Dataset304_neurite_yvonne_b2_smi_1px_scaleaug
)
experiments=(
    # linear
    # linear_finetune
    # nonlinear
    # upernet
    # upernet_inj
    # upernet_inj_ft_balanced_ours
    # upernet_inj_ft_vitb_ours
    upernet_inj_ft_ours
    upernet_inj_ft_p512_ours
    # m2f   # Mask2Former
    convnextt_upernet_ft_ours
    convnexts_upernet_ft_ours
    upernet_inj_ft_vits_ours
    # convnextt_upernet_p512_ours
    convnextt_upernet_ft_aug_p512_ours
    convnextt_upernet_ft_aug_p512_red_ours
)

interval=30

export PYTHONPATH="${PYTHONPATH:-}:src"

python -m fm_adaptation.plot_history \
    --datasets "${datasets[@]}" \
    --experiments "${experiments[@]}" \
    --interval "$interval"
