from pathlib import Path
from fm_adaptation.plot_history import _read_history, _plot
root = Path("models/dinov3/Dataset311_neurite_yvonne_in_vitro_sim_smi_slices")
out = Path(__file__).parent
for tag, stem in (("T", "convnextt"), ("L", "convnext")):
    curves = [(f"{name} head", _read_history(root / f"{stem}_upernet_fullres{infix}_ft_aug_p512_red_s3_wmapdice_poly_bs8_ours/fold_0/history.csv"))
              for name, infix in (("BN", ""), ("LN", "_ln"))]
    _plot(curves, f"Dataset311 ConvNeXt-{tag} UperNet fullres poly bs8: BN vs LN head", out / f"convnext{tag.lower()}_bn_vs_ln.png")
