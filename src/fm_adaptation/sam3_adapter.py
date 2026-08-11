"""ViT-Adapter over SAM3's PE trunk, so the adapter arms can be run on either foundation model.

The adapter itself is DINOv3's, unchanged -- Spatial Prior Module, extractors, output norms and the
injector restored in `dinov3_injector` all carry over. Only the backbone loop differs, because SAM3's
ViTDet blocks keep the token grid in 4-D `(B, H, W, C)` so they can window-partition inside the block,
while the injector and extractor work on `(B, N, C)` sequences. The reshape between the two is the whole
adaptation.

Two constraints follow from the adapter's fixed strides:

* the input must divide by 32 (the SPM's coarsest level) *and* by 14 (SAM3's patch), so 896 rather than
  SAM3's native 1008 -- SAM3 interpolates its absolute position embedding, so this is supported;
* SAM3's trunk exposes neither `embed_dim` nor `patch_size`, which the adapter reads, so they are
  attached before it is built.
"""

import sys
from pathlib import Path

import torch
import torch.utils.checkpoint as cp

from .dinov3_injector import DINOv3AdapterWithInjector, deform_inputs

SAM3_ROOT = Path(__file__).resolve().parents[2] / "foundational_models" / "sam3"

EMBED_DIM = 1024
PATCH_SIZE = 14


def _retune_rope(trunk, image_size: int) -> int:
    """Re-fit the global blocks' rotary tables to a new input size.

    Each attention precomputes `freqs_cis` from its own token grid: the window for the windowed blocks,
    which never changes, and the whole image for the four global ones, which is sized for SAM3's native
    1008. `rope_pt_size` keeps the pretrained grid, so re-running the setup with the new `input_size`
    takes the module's own interpolation path (`scale_pos = rope_pt_size / input_size`) rather than
    silently changing the frequency scale.
    """
    grid = image_size // PATCH_SIZE
    retuned = 0
    for block in trunk.blocks:
        attn = block.attn
        if getattr(block, "window_size", 0) == 0 and getattr(attn, "use_rope", False):
            if tuple(attn.input_size) != (grid, grid):
                attn.input_size = (grid, grid)
                attn._setup_rope_freqs()
                retuned += 1
    return retuned


def load_sam3_trunk(checkpoint: str | None, image_size: int | None = None):
    """SAM3's PE trunk, with the attributes the adapter expects attached."""
    sys.path.insert(0, str(SAM3_ROOT))
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device="cpu",
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=checkpoint is None,
        enable_segmentation=False,
    )
    trunk = model.backbone.vision_backbone.trunk
    trunk.embed_dim = EMBED_DIM
    trunk.patch_size = PATCH_SIZE
    if image_size is not None:
        _retune_rope(trunk, image_size)
    return trunk


class SAM3Adapter(DINOv3AdapterWithInjector):
    """`DINOv3_Adapter` + injector, driving SAM3's ViTDet trunk instead of DINOv3's."""

    def __init__(self, backbone, interaction_indexes=None, use_injector=True, **kwargs):
        # SAM3's four global-attention blocks are the natural taps; for the 32-block ViT they are
        # (7, 15, 23, 31), giving ranges (0,7) (8,15) (16,23) (24,31).
        interaction_indexes = list(interaction_indexes or backbone.full_attn_ids)
        super().__init__(backbone, interaction_indexes=interaction_indexes, **kwargs)
        self.use_injector = use_injector
        if not use_injector:
            self.injectors = None

    def _run_blocks(self, tokens, start, stop, rope_sincos=None):
        """SAM3's blocks take the 4-D grid and no positional argument -- RoPE lives inside attention."""
        for block in self.backbone.blocks[start : stop + 1]:
            if self.checkpoint_backbone and self.training:
                tokens = cp.checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        return tokens

    def forward(self, x):
        from sam3.model.vitdet import get_abs_pos

        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs = x.shape[0]

        backbone = self.backbone
        tokens = backbone.patch_embed(x)  # (B, H, W, C)
        h, w = tokens.shape[1], tokens.shape[2]
        if backbone.pos_embed is not None:
            tokens = tokens + get_abs_pos(
                backbone.pos_embed,
                backbone.pretrain_use_cls_token,
                (h, w),
                backbone.retain_cls_token,
                tiling=backbone.tile_abs_pos,
            )
        tokens = backbone.ln_pre(tokens)

        outs = []
        last_tap = self.block_ranges[-1][1]
        for i, (start, stop) in enumerate(self.block_ranges):
            if self.use_injector:
                sequence = self.injectors[i](
                    query=tokens.flatten(1, 2),
                    reference_points=deform_inputs1[0],
                    feat=c,
                    spatial_shapes=deform_inputs1[1],
                    level_start_index=deform_inputs1[2],
                )
                tokens = sequence.view(bs, h, w, -1)
            tokens = self._run_blocks(tokens, start, stop)
            # SAM3 applies ln_post only after its final tap, and only when configured -- it is Identity
            # in the released image model, so this matches the trunk's own output either way.
            feature = backbone.ln_post(tokens) if stop == last_tap else tokens
            x_i = feature.flatten(1, 2)
            _, c, _ = self.interactions[i](
                x_i, c, None, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks
            )
            outs.append(x_i.transpose(1, 2).view(bs, -1, H_toks, W_toks).contiguous())

        return self._assemble(c, c1, c2, c3, c4, outs, bs, H_c, W_c)
