from pathlib import Path
from fm_adaptation.plot_history import _read_history, _plot
root = Path("models/dinov3/Dataset311_neurite_yvonne_in_vitro_sim_smi_slices")
out = Path(__file__).parent
for tag, stem in (("T", "convnextt"), ("L", "convnext")):
    runs = (("stride-4 head, bs2", "_ft_aug_p512_red_s3_wmapdice_ours"),
            ("fullres BN head", "_fullres_ft_aug_p512_red_s3_wmapdice_poly_bs8_ours"),
            ("fullres LN head", "_fullres_ln_ft_aug_p512_red_s3_wmapdice_poly_bs8_ours"))
    curves = [(name, _read_history(root / f"{stem}_upernet{suffix}/fold_0/history.csv")) for name, suffix in runs]
    _plot(curves, f"Dataset311 ConvNeXt-{tag} UperNet s3 wmapdice: stride-4 vs fullres BN vs fullres LN", out / f"convnext{tag.lower()}_bn_vs_ln.png")
