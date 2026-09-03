#!/usr/bin/env bash
# Fetch a DINOv3 ConvNeXt trunk into foundational_models/dinov3/ckpts/, where `_load_dinov3_backbone`
# looks for it. Meta's own download URL is a signed link, so the weights come from Hugging Face in
# transformers' naming and are renamed onto the naming DINOv3's own `ConvNeXt` class uses.
#
# Runs on the SAM3 environment: it is the one with `safetensors`.
set -euo pipefail

size=${1:-large}

export PATH=~/UltrAi/projects/sam3/.venv/bin:$PATH

python - "$size" <<'PY'
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

sys.path.insert(0, "foundational_models/dinov3")
from dinov3.models.convnext import ConvNeXt, convnext_sizes

size = sys.argv[1]
letter = {"tiny": "t", "small": "s", "base": "b", "large": "l"}[size]
output = f"foundational_models/dinov3/ckpts/dinov3_convnext{letter}_pretrain_lvd1689m.pth"

path = hf_hub_download(f"facebook/dinov3-convnext-{size}-pretrain-lvd1689m", "model.safetensors")
published = load_file(path)

# Both sides hold the same modules in the same order -- the stem is a convolution then a norm, every
# later downsampler a norm then a convolution -- so this is a rename and nothing else.
BLOCK = {
    "depthwise_conv": "dwconv",
    "layer_norm": "norm",
    "pointwise_conv1": "pwconv1",
    "pointwise_conv2": "pwconv2",
}

state = {}
for key, tensor in published.items():
    if key.startswith("layer_norm."):
        state["norm." + key.split(".", 1)[1]] = tensor
    elif ".downsample_layers." in key:
        stage, _, rest = key.removeprefix("stages.").partition(".downsample_layers.")
        state[f"downsample_layers.{stage}.{rest}"] = tensor
    elif ".layers." in key:
        stage, _, rest = key.removeprefix("stages.").partition(".layers.")
        block, _, part = rest.partition(".")
        name, _, suffix = part.partition(".")
        state[f"stages.{stage}.{block}.{BLOCK.get(name, name)}" + (f".{suffix}" if suffix else "")] = tensor
    else:
        raise SystemExit(f"unrecognised published key: {key}")

# The final norm is registered twice, as `norm` and as the last entry of `norms`, so a strict load
# wants it under both names.
state.update({f"norms.3.{part}": state[f"norm.{part}"] for part in ("weight", "bias")})

model = ConvNeXt(**convnext_sizes[size])
model.load_state_dict(state, strict=True)
torch.save(state, output)
print(f"wrote {output}  ({sum(t.numel() for t in state.values()) / 1e6:.1f}M parameters)")
PY
