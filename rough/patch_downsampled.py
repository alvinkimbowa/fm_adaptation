import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Same patch as patch_context.py: centred on the labelled pixel nearest the image centre.
ys, xs = np.nonzero(lab)
k = np.argmin((ys - H / 2) ** 2 + (xs - W / 2) ** 2)
y0 = int(np.clip(ys[k] - P // 2, 0, H - P)); x0 = int(np.clip(xs[k] - P // 2, 0, W - P))

x = torch.from_numpy(img[y0:y0 + P, x0:x0 + P].astype(np.float32))[None, None]
y = torch.from_numpy((lab[y0:y0 + P, x0:x0 + P] > 0).astype(np.float32))[None, None]
levels = [(x, y)]
for _ in range(4):
    # Plain bilinear halving (no antialias), as F.interpolate does inside a network.
    x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    y = F.interpolate(y, scale_factor=0.5, mode="bilinear", align_corners=False)
    levels.append((x, y))

fig, ax = plt.subplots(2, len(levels), figsize=(4.2 * len(levels), 9.2))
contrast = []
for xi, yi in levels:
    # Neurite contrast: mean intensity of pixels at least a quarter covered by label, over background noise.
    a, m = xi[0, 0].numpy(), yi[0, 0].numpy()
    bg = a[(m == 0) & (a < 100)]
    contrast.append((a[m >= 0.25].mean() - bg.mean()) / bg.std())
for i, (xi, yi) in enumerate(levels):
    s = xi.shape[-1]
    ax[0, i].imshow(xi[0, 0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax[0, i].set_title(f"{s}x{s}  (1/{P // s}, {P // s} px per pixel)" if s < P else f"{s}x{s}  original")
    ax[0, i].set_xlabel(f"neurite contrast {contrast[i]:.1f} background SD")
    ax[1, i].imshow(yi[0, 0], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax[1, i].set_title(f"label, bilinear  (max {yi.max():.2f})", pad=4)
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
fig.subplots_adjust(hspace=0.15, wspace=0.05)
fig.suptitle(f"{case} 512 patch at {(x0, y0)}, halved 4 times with bilinear interpolation, shown at equal size")
fig.savefig(Path(__file__).with_suffix(".png"), dpi=110, bbox_inches="tight")
for (xi, yi), c in zip(levels, contrast):
    print(xi.shape[-1], "label max %.2f, contrast %.1f SD" % (yi.max(), c))
