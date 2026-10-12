"""A fixed handful of validation samples drawn with the model's prediction, a figure every few epochs.

Each row is one sample: the image, the image under its ground truth, the image under the prediction.
The samples are chosen once and never change, so two epochs' figures differ only in what the model
learned between them.
"""

from contextlib import nullcontext

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Epochs between figures; the first epoch always gets one.
SAMPLE_EVERY = 5
GT_COLOR = (0.0, 1.0, 0.0)
PRED_COLOR = (1.0, 0.0, 1.0)
ALPHA = 0.7
# Width a panel is sure to get out of the three inches its column is given, once labels and spacing
# have taken theirs.
PANEL_INCHES = 2.4
# Items looked at per sample wanted before settling for fewer; a whole-image dataset decodes a full
# image per item, so the search has to be bounded.
SCAN_PER_SAMPLE = 40


def pick_samples(dataset, count=5):
    """Up to `count` items of `dataset` that hold foreground, as (image, mask, title) on the CPU.

    Drawn in a shuffled order of its own, seeded apart from the global generators, so the choice is
    the same on every run and on a resume, and taking it does not move the training trajectory.
    """
    samples = []
    order = np.random.default_rng(0).permutation(len(dataset))
    for index in order[: count * SCAN_PER_SAMPLE]:
        image, mask, meta = dataset[int(index)]
        if not (mask > 0).any():
            continue
        title = str(meta.get("case_id", index))
        if "y" in meta and "x" in meta:
            title += f"  y={meta['y']} x={meta['x']}"
        samples.append((image, mask, title))
        if len(samples) == count:
            break
    return samples


def _display(image):
    """A preprocessed CxHxW tensor as an HxWx3 float image stretched to [0, 1]."""
    array = image.float().numpy()
    array = array[:3] if array.shape[0] >= 3 else np.repeat(array[:1], 3, axis=0)
    low, high = array.min(), array.max()
    return ((array - low) / max(high - low, 1e-6)).transpose(1, 2, 0)


def _overlaid(image, mask, color):
    canvas = image.copy()
    canvas[mask] = (1.0 - ALPHA) * canvas[mask] + ALPHA * np.array(color, dtype=np.float32)
    return canvas


def render_samples(module, samples, device, path, title="", forward=None):
    """Predict `samples` with `module` as it stands and write the figure to `path`."""
    forward = forward or (lambda m, x, y: m(x, y.shape[-2:]))
    images = torch.stack([image for image, _, _ in samples])
    masks = torch.stack([mask for _, mask, _ in samples])
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    module.eval()
    with torch.no_grad(), amp:
        predictions = forward(module, images.to(device), masks.to(device)).argmax(1).cpu().numpy()

    fig, axes = plt.subplots(len(samples), 3, figsize=(9, 3 * len(samples)), squeeze=False)
    for row, ((image, mask, name), prediction) in enumerate(zip(samples, predictions)):
        shown = _display(image)
        panels = (
            ("image", shown),
            ("ground truth", _overlaid(shown, mask.numpy() > 0, GT_COLOR)),
            ("prediction", _overlaid(shown, prediction > 0, PRED_COLOR)),
        )
        for ax, (label, panel) in zip(axes[row], panels):
            ax.imshow(panel, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label)
        axes[row][0].set_ylabel(name, fontsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # A panel drawn smaller than its patch drops rows and columns, which breaks a one-pixel line
    # into dashes; this resolution gives every patch pixel at least one figure pixel.
    fig.savefig(path, dpi=max(110, int(np.ceil(images.shape[-1] / PANEL_INCHES))))
    plt.close(fig)
