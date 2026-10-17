import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root = Path.home() / "GAA/spinal_cord_injury/data/nnUNet_raw/Dataset311_neurite_yvonne_in_vitro_sim_smi_slices"
train = json.load(open(root / "splits_final.json"))[0]["train"]
case = sorted(c for c in train if c.endswith("_z048"))[0]
stem, z = case[:-3], int(case[-3:])
members = [f"{stem}{k:03d}" for k in (z - 1, z, z + 1)]
img = np.max([np.array(Image.open(root / "imagesTr" / f"{m}_0000.png")) for m in members], axis=0)
lab = np.max([np.array(Image.open(root / "labelsTr" / f"{m}.png")) for m in members], axis=0)
if img.ndim == 3:
    img = img[..., 0]
H, W = img.shape
P = 512
# Centre the example patch on the labelled pixel nearest the image centre.
ys, xs = np.nonzero(lab)
k = np.argmin((ys - H / 2) ** 2 + (xs - W / 2) ** 2)
y0 = int(np.clip(ys[k] - P // 2, 0, H - P)); x0 = int(np.clip(xs[k] - P // 2, 0, W - P))

fig, ax = plt.subplots(1, 3, figsize=(19, 7), gridspec_kw={"width_ratios": [W / H, 1, 1]})
ax[0].imshow(img, cmap="gray")
for gy in range(0, H - P + 1, P // 2):
    for gx in range(0, W - P + 1, P // 2):
        ax[0].add_patch(Rectangle((gx, gy), P, P, fill=False, ec="cyan", lw=0.4, alpha=0.5))
ax[0].add_patch(Rectangle((x0, y0), P, P, fill=False, ec="red", lw=2.5))
ax[0].set_title(f"{case}  3-slice max projection, {W}x{H} px\ncyan: 512 grid at 50% overlap, red: the patch shown right")
crop = img[y0:y0 + P, x0:x0 + P]
ax[1].imshow(crop, cmap="gray"); ax[1].set_title(f"model input: {P}x{P} patch")
ax[2].imshow(crop, cmap="gray"); ax[2].imshow(np.ma.masked_equal(lab[y0:y0 + P, x0:x0 + P], 0), cmap="autumn", alpha=0.6, interpolation="nearest")
ax[2].set_title("same patch, label overlay")
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.tight_layout(); fig.savefig(Path(__file__).with_suffix(".png"), dpi=110)
print(case, img.shape, img.dtype, "patch at", (x0, y0), "patch covers %.1f%% of image area" % (100 * P * P / (H * W)))
