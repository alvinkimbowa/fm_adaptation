import json
import cv2
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
# What a DINOv3 ViT is fed: `_resize_and_pad` resizes the patch to 896, cubic for the image and
# nearest for the label, and the loss is taken there.
V = 896
up = (torch.from_numpy(cv2.resize(x[0, 0].numpy(), (V, V), interpolation=cv2.INTER_CUBIC).clip(0, 255))[None, None],
      torch.from_numpy(cv2.resize(y[0, 0].numpy(), (V, V), interpolation=cv2.INTER_NEAREST))[None, None])
levels = [up, (x, y)]
for _ in range(4):
    # Plain bilinear halving (no antialias), as F.interpolate does inside a network.
    x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
    y = F.interpolate(y, scale_factor=0.5, mode="bilinear", align_corners=False)
    levels.append((x, y))

# The zoom row shows one 64x64 region of the original patch in every column, centred on the labelled
# pixel nearest the patch centre.
Z = 64
py, px = np.nonzero(levels[1][1][0, 0].numpy())
c = np.argmin((py - P / 2) ** 2 + (px - P / 2) ** 2)
zy, zx = int(np.clip(py[c] - Z // 2, 0, P - Z)), int(np.clip(px[c] - Z // 2, 0, P - Z))

fig, ax = plt.subplots(3, len(levels), figsize=(4.2 * len(levels), 13.8))
contrast = []
for xi, yi in levels:
    # Neurite contrast: mean intensity of pixels at least a quarter covered by label, over background noise.
    a, m = xi[0, 0].numpy(), yi[0, 0].numpy()
    bg = a[(m == 0) & (a < 100)]
    contrast.append((a[m >= 0.25].mean() - bg.mean()) / bg.std())
for i, (xi, yi) in enumerate(levels):
    s = xi.shape[-1]
    ax[0, i].imshow(xi[0, 0], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    if s > P:
        ax[0, i].set_title(f"{s}x{s}  ViT input (x{s / P:.2f}, cubic)")
    elif s < P:
        ax[0, i].set_title(f"{s}x{s}  (1/{P // s}, {P // s} px per pixel)")
    else:
        ax[0, i].set_title(f"{s}x{s}  original")
    ax[0, i].set_xlabel(f"neurite contrast {contrast[i]:.1f} background SD")
    ax[1, i].imshow(yi[0, 0], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    ax[1, i].set_title(f"label, {'nearest, as trained' if s > P else 'bilinear'}  (max {yi.max():.2f})", pad=4)
    f = s / P
    r = slice(round(zy * f), round((zy + Z) * f)), slice(round(zx * f), round((zx + Z) * f))
    ax[2, i].imshow(xi[0, 0].numpy()[r], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax[2, i].contour(yi[0, 0].numpy()[r], levels=[0.5], colors="red", linewidths=0.8)
    ax[2, i].set_title(f"zoom {Z}x{Z} px, red: label 0.5", pad=4)
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
fig.subplots_adjust(hspace=0.15, wspace=0.05)
fig.suptitle(f"{case} 512 patch at {(x0, y0)}, upsampled to the ViT's 896 and halved 4 times with bilinear interpolation, shown at equal size")
fig.savefig(Path(__file__).with_suffix(".png"), dpi=110, bbox_inches="tight")
for (xi, yi), c in zip(levels, contrast):
    print(xi.shape[-1], "label max %.2f, contrast %.1f SD" % (yi.max(), c))
