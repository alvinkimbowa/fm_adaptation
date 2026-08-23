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


def _block_schedule(depth, repeat_mode, repeat_times):
    """The order the trunk's blocks are executed in, as indices into `backbone.blocks`.

    Running a block more than once buys depth without buying parameters: the computation graph gets
    longer while the weights stay exactly as many as before. The two modes differ in what is repeated.

    * `adjacent` -- `0 0 1 1 2 2 ...`, each block applied twice in a row, so a stage refines its own
      output at one level of abstraction before handing on.
    * `stack` -- `0 1 2 ... 0 1 2 ...`, the whole trunk run over its own output, so the second pass
      sees features the first pass already built.

    `None` is the ordinary one-pass order, which is what every run before this used.
    """
    if repeat_mode is None or repeat_times == 1:
        return list(range(depth))
    if repeat_mode == "adjacent":
        return [index for index in range(depth) for _ in range(repeat_times)]
    if repeat_mode == "stack":
        return list(range(depth)) * repeat_times
    raise ValueError(f"unknown block_repeat_mode {repeat_mode!r}; expected 'adjacent' or 'stack'")


class DINOv3AdapterWithInjector(DINOv3_Adapter):
    """`DINOv3_Adapter` plus one injector per interaction."""

    def __init__(self, backbone, interaction_indexes, deform_num_heads=16, n_points=4,
                 deform_ratio=0.5, injector_init_values=1e-6, checkpoint_backbone=True,
                 adapter_dim=None, block_repeat_mode=None, block_repeat_times=1, **kwargs):
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
        # Which blocks run, in what order. Without a repeat this is `range(depth)` and everything below
        # behaves exactly as it did when the trunk was executed straight through.
        self.block_schedule = _block_schedule(
            len(self.backbone.blocks), block_repeat_mode, block_repeat_times
        )
        # The taps name the last block of each interaction; the injector needs the whole range, since it
        # writes into the tokens that those blocks then consume. Under a repeat the taps are indices into
        # the schedule rather than into `blocks`, spread evenly over it so the interactions still divide
        # the trunk's computation into equal stages -- and so their count, and the adapter's parameter
        # count with it, does not change.
        taps = list(interaction_indexes)
        if len(self.block_schedule) != len(self.backbone.blocks):
            count = len(taps)
            length = len(self.block_schedule)
            taps = [round(length * (i + 1) / count) - 1 for i in range(count)]
        self.block_ranges = list(zip([0] + [i + 1 for i in taps[:-1]], taps))
        embed_dim = self.backbone.embed_dim
        # The adapter runs at the trunk's width unless told otherwise. A student that narrows it pays a
        # projection at every crossing between the two streams, which is why `_narrow` exists.
        self.adapter_dim = embed_dim if adapter_dim is None else adapter_dim
        self.narrow = self.adapter_dim != embed_dim
        if self.narrow:
            self._narrow(
                embed_dim, interaction_indexes, deform_num_heads, n_points, deform_ratio, kwargs
            )
        self.injectors = nn.ModuleList(
            [
                Injector(
                    dim=self.adapter_dim,
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

    def _narrow(self, embed_dim, interaction_indexes, deform_num_heads, n_points, deform_ratio,
                base_kwargs):
        """Rebuild the adapter at `self.adapter_dim` and add the projections it needs to reach the trunk.

        The base class sizes everything off `backbone.embed_dim`, because upstream's adapter is always as
        wide as the ViT it wraps. At the sizes the distillation students use that is the wrong default:
        the adapter would be most of the model. Everything with a width of its own is rebuilt narrower
        here, and the two streams are bridged explicitly.

        The output norms are deliberately *not* rebuilt. `_assemble` lifts the pyramid back to the
        trunk's width before them, because `add_vit_feature` adds the ViT's own features to it.
        """
        from functools import partial

        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import (
            InteractionBlockWithCls,
            SpatialPriorModule,
        )

        dim = self.adapter_dim
        count = len(interaction_indexes)
        # Every one of these has to be taken from the same place the base class takes it, defaults
        # included. Rebuilding the interactions with a bare constructor silently drops the base's
        # `drop_path_rate=0.3` -- the adapter's stochastic depth, and the students' main regulariser
        # against a training set of a few hundred images.
        drop_path_rate = base_kwargs.get("drop_path_rate", 0.3)
        init_values = base_kwargs.get("init_values", 0.0)
        with_cffn = base_kwargs.get("with_cffn", True)
        cffn_ratio = base_kwargs.get("cffn_ratio", 0.25)
        with_cp = base_kwargs.get("with_cp", True)
        use_extra_extractor = base_kwargs.get("use_extra_extractor", True)
        self.level_embed = nn.Parameter(torch.zeros(3, dim))
        self.spm = SpatialPriorModule(
            inplanes=base_kwargs.get("conv_inplane", 64), embed_dim=dim, with_cp=False
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlockWithCls(
                    dim=dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((i == count - 1) and use_extra_extractor),
                    with_cp=with_cp,
                )
                for i in range(count)
            ]
        )
        self.up = nn.ConvTranspose2d(dim, dim, 2, 2)
        # One pair per interaction: `token_down` lets the injector read the trunk's tokens, `token_up`
        # carries its answer back, and `feat_down` lets the extractor read them. The pyramid is lifted
        # back to the trunk's width by `out_proj`.
        self.token_down = nn.ModuleList([nn.Linear(embed_dim, dim) for _ in range(count)])
        self.token_up = nn.ModuleList([nn.Linear(dim, embed_dim) for _ in range(count)])
        self.feat_down = nn.ModuleList([nn.Linear(embed_dim, dim) for _ in range(count)])
        self.out_proj = nn.ModuleList([nn.Conv2d(dim, embed_dim, 1) for _ in range(4)])

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        for projections in (self.token_down, self.token_up, self.feat_down, self.out_proj):
            projections.apply(self._init_weights)
        self.interactions.apply(self._init_deform_weights)
        torch.nn.init.normal_(self.level_embed)

    def _run_blocks(self, x, start, stop, rope_sincos):
        # `start`/`stop` index the schedule, so a repeated block is simply visited more than once.
        for index in self.block_schedule[start : stop + 1]:
            block = self.backbone.blocks[index]
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
            if self.narrow:
                # The injector runs in the adapter's stream, so the tokens come down to its width and
                # only its *contribution* goes back up. Adding the injector's whole output would replace
                # the trunk's full-width tokens with a rank-`adapter_dim` reconstruction of themselves,
                # which throws away most of what the blocks below it computed.
                query = self.token_down[i](patches)
                injected = injector(
                    query=query,
                    reference_points=deform_inputs1[0],
                    feat=c,
                    spatial_shapes=deform_inputs1[1],
                    level_start_index=deform_inputs1[2],
                )
                patches = patches + self.token_up[i](injected - query)
            else:
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
            # `cls` is passed through the interaction untouched, so only the features it reads need
            # bringing down to the adapter's width.
            feat = self.feat_down[i](x_i) if self.narrow else x_i
            _, c, _ = self.interactions[i](
                feat, c, cls, deform_inputs1, deform_inputs2, H_c, W_c, H_toks, W_toks
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

        if self.narrow:
            # Back to the trunk's width, which is what `add_vit_feature`, the output norms and the
            # decoder all expect.
            c1, c2, c3, c4 = (proj(level) for proj, level in zip(self.out_proj, (c1, c2, c3, c4)))

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        return {"1": self.norm1(c1), "2": self.norm2(c2), "3": self.norm3(c3), "4": self.norm4(c4)}
