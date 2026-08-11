"""DINOv3's ViT-Adapter with ViT-Adapter's injector put back.

`DINOv3_Adapter` is ViT-Adapter with the injector removed: only the extractor survives, so the Spatial
Prior Module's convolutional features never reach the transformer and the backbone can be run once under
`no_grad`. This restores the two-way interaction, following how the original drives it in
`ViT-Adapter/segmentation/mmseg_custom/models/backbones/adapter_modules.py` -- injector, then the ViT
blocks of that interaction, then the extractor.

Everything else is inherited: the SPM, the extractors, `up`, the output norms and the whole feature
assembly are the base class's. The backbone's weights stay frozen; only its activations now carry
gradient, because an injector at block 5 can only be trained through the blocks that follow it.
"""

import sys
from pathlib import Path

import torch
import torch.utils.checkpoint as cp
from torch import nn

DINOV3_ROOT = Path(__file__).resolve().parents[2] / "foundational_models" / "dinov3"
sys.path.insert(0, str(DINOV3_ROOT))

from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (  # noqa: E402
    DINOv3_Adapter,
    deform_inputs,
)
from dinov3.eval.segmentation.models.utils.ms_deform_attn import MSDeformAttn  # noqa: E402


class Injector(nn.Module):
    """Writes the spatial features into the ViT tokens. Ported from ViT-Adapter, using DINOv3's own
    compiled `MSDeformAttn` so no second deformable-attention op is involved."""

    def __init__(self, dim, num_heads=16, n_points=4, n_levels=3, deform_ratio=0.5,
                 norm_layer=nn.LayerNorm, init_values=1e-6, with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio
        )
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):
        def _inner_forward(query, feat):
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class DINOv3AdapterWithInjector(DINOv3_Adapter):
    """`DINOv3_Adapter` plus one injector per interaction."""

    def __init__(self, backbone, interaction_indexes, deform_num_heads=16, n_points=4,
                 deform_ratio=0.5, injector_init_values=1e-6, checkpoint_backbone=True, **kwargs):
        # `init_values` is deliberately left at the base class's default so the extractors stay exactly as
        # they were in the existing runs; only the injector takes upstream's 1e-6.
        super().__init__(
            backbone,
            interaction_indexes=interaction_indexes,
            deform_num_heads=deform_num_heads,
            n_points=n_points,
            deform_ratio=deform_ratio,
            **kwargs,
        )
        self.checkpoint_backbone = checkpoint_backbone
        # The taps name the last block of each interaction; the injector needs the whole range, since it
        # writes into the tokens that those blocks then consume.
        self.block_ranges = list(zip([0] + [i + 1 for i in interaction_indexes[:-1]], interaction_indexes))
        embed_dim = self.backbone.embed_dim
        self.injectors = nn.ModuleList(
            [
                Injector(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    n_levels=3,
                    deform_ratio=deform_ratio,
                    init_values=injector_init_values,
                )
                for _ in interaction_indexes
            ]
        )
        # The base class ran its own init in `super().__init__()`, before these existed. Without this the
        # injectors would keep PyTorch's default Linear init and, worse, an MSDeformAttn whose sampling
        # offsets were never given their direction grid -- which does not train.
        self.injectors.apply(self._init_weights)
        self.injectors.apply(self._init_deform_weights)

    def _run_blocks(self, x, start, stop, rope_sincos):
        for block in self.backbone.blocks[start : stop + 1]:
            if self.checkpoint_backbone and self.training:
                x = cp.checkpoint(block, x, rope_sincos, use_reentrant=False)
            else:
                x = block(x, rope_sincos)
        return x

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs = x.shape[0]

        backbone = self.backbone
        tokens, (H, W) = backbone.prepare_tokens_with_masks(x)
        rope_sincos = backbone.rope_embed(H=H, W=W) if backbone.rope_embed is not None else None
        n_prefix = backbone.n_storage_tokens + 1  # cls + storage tokens, which are not spatial

        outs = []
        for i, (injector, (start, stop)) in enumerate(zip(self.injectors, self.block_ranges)):
            # The injector is spatial: it only touches patch tokens, and `deform_inputs1`'s reference
            # points are laid out over patch positions alone.
            prefix, patches = tokens[:, :n_prefix], tokens[:, n_prefix:]
            patches = injector(
                query=patches,
                reference_points=deform_inputs1[0],
                feat=c,
                spatial_shapes=deform_inputs1[1],
                level_start_index=deform_inputs1[2],
            )
            tokens = torch.cat([prefix, patches], dim=1)
            tokens = self._run_blocks(tokens, start, stop, rope_sincos)
            # `get_intermediate_layers(norm=True)` is what the extractor used to be fed, so normalise
            # here too and keep the only difference to the injector itself.
            normed = backbone.norm(tokens)
            x_i, cls = normed[:, n_prefix:], normed[:, 0]
            _, c, _ = self.interactions[i](
                x_i, c, cls, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks
            )
            outs.append(x_i.transpose(1, 2).view(bs, -1, H_toks, W_toks).contiguous())

        return self._assemble(c, c1, c2, c3, c4, outs, bs, H_c, W_c)

    def _assemble(self, c, c1, c2, c3, c4, outs, bs, H_c, W_c):
        """Split, reshape, optionally add the ViT features, and norm -- the base class's tail, verbatim."""
        import torch.nn.functional as F

        dim = c.shape[-1]
        c2_len, c3_len = c2.size(1), c3.size(1)
        c2 = c[:, 0:c2_len, :]
        c3 = c[:, c2_len : c2_len + c3_len, :]
        c4 = c[:, c2_len + c3_len :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        return {"1": self.norm1(c1), "2": self.norm2(c2), "3": self.norm3(c3), "4": self.norm4(c4)}
