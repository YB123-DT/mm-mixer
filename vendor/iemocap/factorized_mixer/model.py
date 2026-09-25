from __future__ import annotations
import math
import re

import torch
import re
from torch import nn

from factorized_mixer.registry import C4_TUNE30_ARCH_VARIANTS, FINAL_TUNE_VARIANTS
from structured_cross.model import (
    CSS_AV_INPUT_DIMS,
    CrossResidualIntegrator,
    StructuredHighOrderCross,
    _ForwardContext,
    _load_base,
    build_structured_cross_model,
    optimizer_parameter_groups as p3_groups,
)


class QueryBypassScaleWrapper(nn.Module):
    """Compensate RawAux's fixed query bypass without editing its base model."""

    def __init__(self, original, target_scale, base_scale=.2):
        super().__init__()
        self.original = original
        self.target_scale = float(target_scale)
        self.base_scale = float(base_scale)

    def forward(self, query, key, value, **kwargs):
        output, weights = self.original(query=query, key=key, value=value, **kwargs)
        return output + (self.target_scale - self.base_scale) * query, weights


class ChannelGateResidualWrapper(nn.Module):
    """Compensate RawAux's fixed ChannelGate residual without editing its base model."""

    def __init__(self, original, target_residual, base_residual=.1):
        super().__init__()
        self.original = original
        self.target_residual = float(target_residual)
        self.base_residual = float(base_residual)

    def forward(self, inputs):
        return self.original(inputs) + (self.target_residual - self.base_residual)


def _apply_final_tune_overrides(base, overrides):
    """Apply only named tuning controls to a recursively built HO_WO_TAV model."""

    allowed = {
        "x3_rank", "x3_residual_scale", "x3_lr_multiplier", "x3_branch_dropout",
        "ca_heads", "ca_dropout", "ca_query_bypass", "ca_lr_multiplier",
        "adaptive_dropout", "channel_dropout", "channel_residual",
        "mixer_sub_scale", "mixer_mod_scale", "mixer_ffn_scale",
    }
    unknown = set(overrides).difference(allowed)
    if unknown:
        raise ValueError(f"unknown final-tune overrides: {sorted(unknown)}")

    cross = base.transformer_encoder.cross
    if "x3_rank" in overrides and int(overrides["x3_rank"]) != cross.rank:
        base.transformer_encoder.cross = StructuredHighOrderCross(
            cross.dim, int(overrides["x3_rank"]), cross.active_interactions
        )
    if "x3_residual_scale" in overrides:
        base.feature_integrator.residual_scale = float(overrides["x3_residual_scale"])
    if "x3_branch_dropout" in overrides:
        base.feature_integrator.branch_dropout = float(overrides["x3_branch_dropout"])
    if "x3_lr_multiplier" in overrides:
        base.x3_lr_multiplier = float(overrides["x3_lr_multiplier"])

    ca_heads = overrides.get("ca_heads")
    ca_dropout = overrides.get("ca_dropout")
    if ca_heads is not None or ca_dropout is not None:
        for attention in base.cross_attn.values():
            if ca_heads is not None:
                heads = int(ca_heads)
                if attention.embed_dim % heads:
                    raise ValueError("ca_heads must divide the attention embedding width")
                attention.num_heads = heads
                attention.head_dim = attention.embed_dim // heads
            if ca_dropout is not None:
                attention.dropout = float(ca_dropout)
    if "ca_query_bypass" in overrides:
        target = float(overrides["ca_query_bypass"])
        if target != .2:
            for modality in base.modalities:
                base.cross_attn[modality] = QueryBypassScaleWrapper(
                    base.cross_attn[modality], target
                )
    if "ca_lr_multiplier" in overrides:
        base.ca_lr_multiplier = float(overrides["ca_lr_multiplier"])

    if "adaptive_dropout" in overrides:
        base.adaptive_fusion.context_net[3].p = float(overrides["adaptive_dropout"])
    if "channel_dropout" in overrides:
        for gate in base.gates.values():
            gate[3].p = float(overrides["channel_dropout"])
    if "channel_residual" in overrides:
        target = float(overrides["channel_residual"])
        if target != .1:
            for modality in base.modalities:
                base.gates[modality] = ChannelGateResidualWrapper(base.gates[modality], target)

    mixer_scales = {
        "mixer_sub_scale": "alpha_sub",
        "mixer_mod_scale": "alpha_mod",
        "mixer_ffn_scale": "alpha_ffn",
    }
    for key, attribute in mixer_scales.items():
        if key not in overrides:
            continue
        scale = float(overrides[key])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"{key} must be positive finite")
        for block in base.transformer_encoder.blocks:
            getattr(block, attribute).fill_(scale)

    base.tuning_overrides = dict(overrides)
    return base


class TopologyAlignedCrossResidual(nn.Module):
    """Explicit VA/VT/AT/VAT interactions mapped back to three modality branches."""

    def __init__(self, dim=256, rank=32, modalities=("v", "a", "t")):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.modalities = tuple(modalities)
        self.pair_projections = nn.ModuleDict({m: nn.Linear(dim, rank) for m in self.modalities})
        self.triple_projections = nn.ModuleDict({m: nn.Linear(dim, rank) for m in self.modalities})
        self.output_heads = nn.ModuleDict({m: nn.Linear(3 * rank, dim) for m in self.modalities})
        for head in self.output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (3, self.dim):
            raise ValueError(f"expected [B, 3, {self.dim}] modality tokens")
        values = {name: tokens[:, index] for index, name in enumerate(self.modalities)}
        pair = {name: self.pair_projections[name](values[name]) for name in self.modalities}
        triple = {name: self.triple_projections[name](values[name]) for name in self.modalities}
        va = pair["v"] * pair["a"]
        vt = pair["v"] * pair["t"]
        at = pair["a"] * pair["t"]
        vat = triple["v"] * triple["a"] * triple["t"]
        return {
            "v": self.output_heads["v"](torch.cat((va, vt, vat), dim=-1)),
            "a": self.output_heads["a"](torch.cat((va, at, vat), dim=-1)),
            "t": self.output_heads["t"](torch.cat((vt, at, vat), dim=-1)),
        }


class RelationSpecificCrossResidual(nn.Module):
    """Relation-specific low-rank AV/VT/AT products with a clean residual output."""

    RELATION_ENDPOINTS = {"av": (1, 0), "tv": (2, 0), "ta": (2, 1)}

    def __init__(self, dim=256, rank=32, normalize=True, learnable_scale=False,
                 active_interactions=None, branch_dropout=0.0):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        self.active_interactions = frozenset(
            self.RELATION_ENDPOINTS if active_interactions is None else active_interactions
        )
        unknown = self.active_interactions.difference(self.RELATION_ENDPOINTS)
        if unknown:
            raise ValueError(f"unknown relation-specific interactions: {sorted(unknown)}")
        self.branch_dropout = float(branch_dropout)
        self.endpoint_projections = nn.ModuleDict({
            relation: nn.ModuleList((nn.Linear(dim, rank), nn.Linear(dim, rank)))
            for relation in self.RELATION_ENDPOINTS
        })
        norm = lambda: nn.LayerNorm(rank) if normalize else nn.Identity()
        self.endpoint_norms = nn.ModuleDict({
            relation: nn.ModuleList((norm(), norm())) for relation in self.RELATION_ENDPOINTS
        })
        self.output_heads = nn.ModuleDict({
            relation: nn.Linear(rank, dim) for relation in self.RELATION_ENDPOINTS
        })
        for head in self.output_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.layer_scales = (nn.ParameterDict({
            relation: nn.Parameter(torch.ones(())) for relation in self.RELATION_ENDPOINTS
        }) if learnable_scale else None)
        self.last_branch_mask = None

    def decomposed(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (3, self.dim):
            raise ValueError(f"expected exactly three modality tokens of width {self.dim}")
        outputs = {}
        for relation, (left_index, right_index) in self.RELATION_ENDPOINTS.items():
            projections = self.endpoint_projections[relation]
            norms = self.endpoint_norms[relation]
            left = norms[0](projections[0](tokens[:, left_index]))
            right = norms[1](projections[1](tokens[:, right_index]))
            value = self.output_heads[relation](left * right)
            if self.layer_scales is not None:
                value = self.layer_scales[relation] * value
            if relation not in self.active_interactions:
                value = torch.zeros_like(value)
            outputs[relation] = value
        outputs["tav"] = torch.zeros_like(outputs["av"])
        outputs["bias"] = torch.zeros_like(outputs["av"])
        return outputs

    def forward(self, tokens):
        components = self.decomposed(tokens)
        residual = sum(components[name] for name in self.active_interactions)
        self.last_branch_mask = None
        if self.training and self.branch_dropout > 0:
            keep = (torch.rand((tokens.shape[0], 1), device=tokens.device)
                    >= self.branch_dropout).to(residual.dtype)
            self.last_branch_mask = keep
            residual = residual * keep / (1.0 - self.branch_dropout)
        return residual


class CrossAttentionResidualWrapper(nn.Module):
    """Add the query-modality interaction residual to one directional CA output."""

    def __init__(self, original, residual, query_modality, modalities=("v", "a", "t")):
        super().__init__()
        self.original = original
        object.__setattr__(self, "_residual", residual)
        self.query_modality = query_modality
        self.modalities = tuple(modalities)

    def forward(self, query, key, value, **kwargs):
        output, weights = self.original(query=query, key=key, value=value, **kwargs)
        others = [name for name in self.modalities if name != self.query_modality]
        features = {self.query_modality: query.squeeze(1)}
        features.update({name: key[:, index] for index, name in enumerate(others)})
        tokens = torch.stack([features[name] for name in self.modalities], dim=1)
        residual = self._residual(tokens)[self.query_modality].unsqueeze(1)
        return output + residual, weights


class RawAuxCaptureSelector(nn.Module):
    """Capture the exact FeatureGating output used by the auxiliary heads."""

    def __init__(self, selector, context, modality, modalities):
        super().__init__()
        self.selector = selector
        self.modality = modality
        self.modalities = tuple(modalities)
        object.__setattr__(self, "_context", context)

    def forward(self, inputs):
        output = self.selector(inputs)
        context = self._context
        if self.modality == self.modalities[0]:
            context.raw_aux_parts = {}
            context.raw_aux = None
        context.raw_aux_parts[self.modality] = output
        if len(context.raw_aux_parts) == len(self.modalities):
            context.raw_aux = torch.stack(
                [context.raw_aux_parts[name] for name in self.modalities], dim=1
            )
        return output


class NoAdaptiveFusion(nn.Module):
    """Parameter-free placeholder when no cross-modal context is consumed."""

    def forward(self, modality_features):
        batch = modality_features[0].shape[0]
        context = torch.zeros_like(modality_features[0])
        weights = modality_features[0].new_full(
            (batch, len(modality_features)), 1.0 / len(modality_features)
        )
        return context, weights


class ModalitySubspaceTransformerSelector(nn.Module):
    """Refine one modality by attention over four learned feature subspaces."""

    def __init__(self, dim=256, token_count=4, token_dim=64, dropout=0.2):
        super().__init__()
        if token_count * token_dim != dim:
            raise ValueError("token_count * token_dim must equal the modality width")
        self.token_count = int(token_count)
        self.token_dim = int(token_dim)
        self.split = nn.Linear(dim, dim)
        layer = nn.TransformerEncoderLayer(
            token_dim, 4, token_dim * 4, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.output = nn.Linear(dim, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value):
        tokens = self.split(value).reshape(value.shape[0], self.token_count, self.token_dim)
        update = self.output(self.encoder(tokens).reshape(value.shape[0], -1))
        return value + update


class FeatureGatingCapacityModule(nn.Module):
    """FeatureGating with a configurable bottleneck and unchanged gating semantics."""

    def __init__(self, dim=256, hidden=128):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, inputs):
        return inputs * self.gate(inputs)


class MixerBlock(nn.Module):
    def __init__(self, k=6, d=256, routed=False, hidden=1536, reverse=False, use_sub=True, use_mod=True, use_ffn=True, layer_scale=None, sub_expansion=2, modal_expansion=None):
        super().__init__()
        if int(sub_expansion) < 1:
            raise ValueError("subspace expansion must be positive")
        self.norm_sub = nn.LayerNorm(d)
        sub_hidden = k * int(sub_expansion)
        self.sub = nn.Sequential(nn.Linear(k, sub_hidden), nn.GELU(), nn.Linear(sub_hidden, k))
        self.norm_mod = nn.LayerNorm(d)
        if routed and modal_expansion is not None:
            modal_hidden = 3 * int(modal_expansion)
            self.route = nn.Sequential(
                nn.Linear(3, modal_hidden, bias=False),
                nn.GELU(),
                nn.Linear(modal_hidden, 3, bias=False),
            )
        else:
            self.route = nn.Linear(3, 3, bias=False) if routed else nn.Identity()
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.reverse = reverse
        self.use_sub, self.use_mod, self.use_ffn = use_sub, use_mod, use_ffn
        if layer_scale == "learnable":
            self.alpha_sub = nn.Parameter(torch.full((d,), 0.1)); self.alpha_mod = nn.Parameter(torch.full((d,), 0.1)); self.alpha_ffn = nn.Parameter(torch.full((d,), 0.1))
        else:
            self.register_buffer("alpha_sub", torch.full((d,), float(layer_scale or 1.0))); self.register_buffer("alpha_mod", torch.full((d,), float(layer_scale or 1.0))); self.register_buffer("alpha_ffn", torch.full((d,), float(layer_scale or 1.0)))

    def forward(self, z):
        # z: [B, M, K, D]. Mix K within each modality, then M at each K.
        def subspace(value):
            x = self.norm_sub(value).transpose(-1, -2)
            return value + self.sub(x).transpose(-1, -2)
        def modality(value):
            x = self.norm_mod(value).permute(0, 2, 3, 1)
            x = self.route(x).permute(0, 3, 1, 2)
            return value + x
        if self.reverse:
            if self.use_mod: z = z + self.alpha_mod.view(1,1,1,-1) * (modality(z)-z)
            if self.use_sub: z = z + self.alpha_sub.view(1,1,1,-1) * (subspace(z)-z)
        else:
            if self.use_sub: z = z + self.alpha_sub.view(1,1,1,-1) * (subspace(z)-z)
            if self.use_mod: z = z + self.alpha_mod.view(1,1,1,-1) * (modality(z)-z)
        return z + self.alpha_ffn.view(1,1,1,-1) * self.ffn(self.norm_ffn(z)) if self.use_ffn else z


class FactorizedMixerEncoder(nn.Module):
    def __init__(self, dropout=.1, layers=2, routed=False, alternating=False, context=None, hidden=1536, token_count=6, token_dim=256, use_sub=True, use_mod=True, use_ffn=True, layer_scale=None, fusion_dim=256, sub_expansion=2, modal_expansion=None):
        super().__init__(); self.token_count=token_count; self.token_dim=token_dim
        self.fusion_dim = int(fusion_dim)
        self.split = nn.Linear(self.fusion_dim, token_count * token_dim)
        self.blocks = nn.ModuleList(MixerBlock(token_count, token_dim, routed, hidden=hidden, reverse=alternating and i % 2 == 1, use_sub=use_sub, use_mod=use_mod, use_ffn=use_ffn, layer_scale=layer_scale, sub_expansion=sub_expansion, modal_expansion=modal_expansion) for i in range(layers))
        self.output_projection = nn.Linear(token_dim, self.fusion_dim) if token_dim != self.fusion_dim else nn.Identity()
        self.output_adapter = nn.Identity(); object.__setattr__(self, "_forward_context", context)

    def forward(self, x):
        b, m, _ = x.shape
        z = self.split(x).reshape(b, 3, self.token_count, self.token_dim)
        for block in self.blocks: z = block(z)
        grouped = self.output_projection(self.output_adapter(z.mean(dim=2)))
        post_refinement = getattr(self, "post_refinement", None)
        if post_refinement is not None:
            grouped = post_refinement(grouped)
        self._forward_context.grouped = grouped
        cross = getattr(self, "cross", None)
        if cross is not None:
            cross_input = grouped
            if getattr(self, "cross_input_source", "grouped") == "raw_aux":
                cross_input = self._forward_context.raw_aux
                if cross_input is None:
                    raise RuntimeError("raw auxiliary features were not captured before Mixer encoding")
            self._forward_context.residual = cross(cross_input)
            self._forward_context.components = cross.decomposed(cross_input)
        return grouped


class PostMixerTransformerBlock(nn.Module):
    """A standard single PreNorm Transformer block over three modality tokens."""

    def __init__(self, dim=256, heads=8, hidden=512, dropout=.1):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            dim, heads, hidden, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )

    def forward(self, inputs):
        return self.block(inputs)


class PostMixerResidualMLP(nn.Module):
    """A conventional PreNorm residual MLP over each modality token."""

    def __init__(self, dim=256, hidden=512, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, inputs):
        return inputs + self.mlp(self.norm(inputs))


class IdentityEncoderWithCross(nn.Module):
    """Remove the Mixer while preserving the X3 input/output contract."""

    def __init__(self, context, cross):
        super().__init__()
        self.token_count = 1
        self.token_dim = 256
        self.cross = cross
        object.__setattr__(self, "_forward_context", context)

    def forward(self, x):
        self._forward_context.grouped = x
        self._forward_context.residual = self.cross(x)
        self._forward_context.components = self.cross.decomposed(x)
        return x


class TransformerThenMixerEncoder(nn.Module):
    """Keep X3 on the two-layer Transformer output, then refine only the main path."""

    def __init__(self, transformer_encoder, context, hidden=1536):
        super().__init__()
        self.token_count = transformer_encoder.token_count
        self.token_dim = transformer_encoder.token_dim
        self.split = transformer_encoder.split
        self.transformer = transformer_encoder.transformer
        self.output_adapter = transformer_encoder.output_adapter
        self.cross = transformer_encoder.cross
        self.mixer_split = nn.Linear(256, self.token_count * self.token_dim)
        self.mixers = nn.ModuleList([
            MixerBlock(self.token_count, self.token_dim, routed=True, hidden=hidden)
        ])
        self.last_x3_input = None
        self.last_mixer_output = None
        object.__setattr__(self, "_forward_context", context)

    def forward(self, x):
        batch, modalities, _ = x.shape
        tokens = self.split(x).reshape(batch, modalities * self.token_count, self.token_dim)
        encoded = self.transformer(tokens)
        transformer_grouped = self.output_adapter(
            encoded.reshape(batch, modalities, self.token_count, self.token_dim).mean(dim=2)
        )

        # Preserve the original X3 computation exactly at the Transformer output.
        self._forward_context.residual = self.cross(transformer_grouped)
        self._forward_context.components = self.cross.decomposed(transformer_grouped)

        z = self.mixer_split(transformer_grouped).reshape(
            batch, modalities, self.token_count, self.token_dim
        )
        for mixer in self.mixers:
            z = mixer(z)
        grouped = z.mean(dim=2)
        self._forward_context.grouped = grouped
        self.last_x3_input = transformer_grouped.detach()
        self.last_mixer_output = grouped.detach()
        return grouped


class HybridEncoder(nn.Module):
    def __init__(self, dropout=.1, reverse=False, context=None, transformer_layers=2,
                 mixer_layers=1, alternating=False):
        super().__init__()
        self.token_count = 6; self.token_dim = 256
        self.split = nn.Linear(256, self.token_count * self.token_dim)
        layer = nn.TransformerEncoderLayer(256, 8, 1024, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, transformer_layers)
        self.mixers = nn.ModuleList(
            MixerBlock(6, 256, False, 1536, alternating and index % 2 == 1)
            for index in range(mixer_layers)
        )
        self.output_adapter = nn.Identity(); self.reverse = reverse
        object.__setattr__(self, "_forward_context", context)

    def forward(self, x):
        batch, _, _ = x.shape
        z = self.split(x).reshape(batch, 18, 256)
        if self.reverse:
            for mixer in self.mixers:
                z = mixer(z.reshape(batch, 3, 6, 256)).reshape(batch, 18, 256)
            z = self.transformer(z)
        else:
            z = self.transformer(z)
            for mixer in self.mixers:
                z = mixer(z.reshape(batch, 3, 6, 256)).reshape(batch, 18, 256)
        grouped = self.output_adapter(z.reshape(batch, 3, 6, 256).mean(dim=2))
        self._forward_context.grouped = grouped
        self._forward_context.residual = self.cross(grouped)
        self._forward_context.components = self.cross.decomposed(grouped)
        return grouped


def build_factorized_mixer_model(experiment_id="M1", dropout=.1):
    if experiment_id in FINAL_TUNE_VARIANTS:
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        _apply_final_tune_overrides(base, FINAL_TUNE_VARIANTS[experiment_id])
        base.capacity_variant = f"P3_FINAL_TUNE_{experiment_id}"
        return base
    if experiment_id == "FINAL_INFO_GATE_ON":
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        rawaux = _load_base()
        base.info_gates = nn.ModuleDict({
            modality: rawaux.InformationGateModule(base.fusion_dim)
            for modality in base.modalities
        })
        base.capacity_variant = "P3_FINAL_INFO_GATE_ON"
        return base
    if experiment_id == "HO_WO_TAV_CSSAV":
        base = build_structured_cross_model("X3", 0.2, input_dims=CSS_AV_INPUT_DIMS)
        context = base.transformer_encoder._forward_context
        encoder = FactorizedMixerEncoder(
            0.2, 2, True, True, context, hidden=1536,
            token_count=6, token_dim=256,
        )
        encoder.cross = base.transformer_encoder.cross
        encoder.cross.active_interactions = frozenset({"av", "tv", "ta"})
        base.transformer_encoder = encoder
        base.capacity_variant = "P3_FACTORIZED_MIXER_M4_LR2_D2_H1536_HO_WO_TAV_CSSAV"
        base.mixer_lr_multiplier = 2.0
        return base
    c4_tune30 = re.fullmatch(
        r"LC30_TR_H(4|8|16)_F(256|512|768|1024)_D(00|05|10|15|20)",
        experiment_id,
    )
    if c4_tune30 and experiment_id in C4_TUNE30_ARCH_VARIANTS:
        heads, hidden, dropout_code = c4_tune30.groups()
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        base.transformer_encoder.post_refinement = PostMixerTransformerBlock(
            heads=int(heads), hidden=int(hidden), dropout=int(dropout_code) / 100.0,
        )
        base.capacity_variant = f"P3_C4_TUNE30_{experiment_id}"
        return base
    light_capacity = {
        "LC_C1_H1792": {"layers": 2, "hidden": 1792},
        "LC_C2_H2048": {"layers": 2, "hidden": 2048},
        "LC_C3_L3": {"layers": 3, "hidden": 1536},
        "LC_C4_POST_TR": {"refinement": "transformer"},
        "LC_C5_POST_MLP": {"refinement": "mlp"},
    }
    if experiment_id in light_capacity:
        spec = light_capacity[experiment_id]
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        encoder = base.transformer_encoder
        refinement = spec.get("refinement")
        if refinement is not None:
            encoder.post_refinement = (
                PostMixerTransformerBlock() if refinement == "transformer"
                else PostMixerResidualMLP()
            )
        else:
            replacement = FactorizedMixerEncoder(
                0.2, int(spec["layers"]), True, True,
                encoder._forward_context, hidden=int(spec["hidden"]),
                token_count=6, token_dim=256,
            )
            replacement.cross = encoder.cross
            base.transformer_encoder = replacement
        base.capacity_variant = f"P3_LIGHT_CAPACITY_{experiment_id}"
        return base
    modal_expansion = re.fullmatch(r"HO_WO_TAV_MR(2|4|8)", experiment_id)
    if modal_expansion:
        ratio = int(modal_expansion.group(1))
        base = build_structured_cross_model("X3", 0.2)
        context = base.transformer_encoder._forward_context
        encoder = FactorizedMixerEncoder(
            0.2, 2, True, True, context, hidden=1536,
            token_count=6, token_dim=256, modal_expansion=ratio,
        )
        encoder.cross = base.transformer_encoder.cross
        encoder.cross.active_interactions = frozenset({"av", "tv", "ta"})
        base.transformer_encoder = encoder
        base.capacity_variant = f"P3_FACTORIZED_MIXER_HO_WO_TAV_MR{ratio}"
        base.mixer_lr_multiplier = 2.0
        return base
    k_expansion = re.fullmatch(r"HO_WO_TAV_KR(1|4|8|16)", experiment_id)
    if k_expansion:
        ratio = int(k_expansion.group(1))
        base = build_structured_cross_model("X3", 0.2)
        context = base.transformer_encoder._forward_context
        encoder = FactorizedMixerEncoder(
            0.2, 2, True, True, context, hidden=1536,
            token_count=6, token_dim=256, sub_expansion=ratio,
        )
        encoder.cross = base.transformer_encoder.cross
        encoder.cross.active_interactions = frozenset({"av", "tv", "ta"})
        base.transformer_encoder = encoder
        base.capacity_variant = f"P3_FACTORIZED_MIXER_HO_WO_TAV_KR{ratio}"
        base.mixer_lr_multiplier = 2.0
        return base
    frozen_variants = {
        "FINAL_F0_FULL", "FINAL_F1_NO_SUBSPACE", "FINAL_F2_NO_ROUTING",
        "FINAL_F3_NO_FFN", "FINAL_F4_NO_ALTERNATING", "FINAL_F5_NO_MIXER",
        "FINAL_F6_NO_X3", "FINAL_F7_TRANSFORMER", "FINAL_F8_TR_X3_MIXER1",
        "FINAL_F9_X3_TRANSFORMER_REPLAY", "FINAL_F10_NO_CA",
    }
    if experiment_id in frozen_variants:
        if experiment_id == "FINAL_F9_X3_TRANSFORMER_REPLAY":
            base = build_structured_cross_model("X3", dropout)
            base.mixer_lr_multiplier = 1.0
        elif experiment_id == "FINAL_F7_TRANSFORMER":
            base = build_structured_cross_model("X3", dropout)
            base.transformer_encoder.cross.active_interactions = frozenset({"av", "tv", "ta"})
            base.mixer_lr_multiplier = 2.0
        elif experiment_id == "FINAL_F8_TR_X3_MIXER1":
            base = build_structured_cross_model("X3", dropout)
            original = base.transformer_encoder
            context = original._forward_context
            base.transformer_encoder = TransformerThenMixerEncoder(original, context, hidden=1536)
            base.mixer_lr_multiplier = 1.0
        else:
            base = build_factorized_mixer_model("HO_WO_TAV", dropout)
            encoder = base.transformer_encoder
            if experiment_id == "FINAL_F1_NO_SUBSPACE":
                for block in encoder.blocks:
                    block.use_sub = False
            elif experiment_id == "FINAL_F2_NO_ROUTING":
                for block in encoder.blocks:
                    block.route = nn.Identity()
            elif experiment_id == "FINAL_F3_NO_FFN":
                for block in encoder.blocks:
                    block.use_ffn = False
            elif experiment_id == "FINAL_F4_NO_ALTERNATING":
                for block in encoder.blocks:
                    block.reverse = False
            elif experiment_id == "FINAL_F5_NO_MIXER":
                base.transformer_encoder = IdentityEncoderWithCross(
                    encoder._forward_context, encoder.cross
                )
            elif experiment_id == "FINAL_F6_NO_X3":
                base.feature_integrator.residual_scale = 0.0
            elif experiment_id == "FINAL_F10_NO_CA":
                base.disable_channel_attention()
        base.capacity_variant = experiment_id
        return base
    if experiment_id == "AUX_PRE_FEATURE_GATE":
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        base.aux_before_feature_selection = True
        base.capacity_variant = "P3_AUX_PRE_FEATURE_GATE"
        return base
    scaled_cores = {
        "CORE_C320": (320, 1920, 40),
        "CORE_C384": (384, 2304, 48),
        "CORE_C512": (512, 3072, 64),
    }
    if experiment_id in scaled_cores:
        width, hidden, rank = scaled_cores[experiment_id]
        rawaux = _load_base()
        base = rawaux.HierarchicalAttentionFusion(
            embed_dims={"v": 342, "a": 1024, "t": 1024}, num_classes=6,
            modalities=["v", "a", "t"], fusion_dim=width,
            num_transformer_layers=3, num_heads=8, dropout=0.2,
            modality_importance={"t": .65, "a": .2, "v": .15},
            use_moe=False, use_contrastive=False, aux_from_raw_projected=True,
            align_self_weight=.95,
        )
        base.disable_alignment()
        from structured_cross.model import IdentityInformationGate
        base.info_gates = nn.ModuleDict({m: IdentityInformationGate() for m in base.modalities})
        context = _ForwardContext()
        encoder = FactorizedMixerEncoder(
            0.2, 2, True, True, context, hidden, 6, width,
            fusion_dim=width,
        )
        encoder.cross = StructuredHighOrderCross(
            width, rank, active_interactions={"av", "tv", "ta"}
        )
        base.transformer_encoder = encoder
        base.feature_integrator = CrossResidualIntegrator(
            base.feature_integrator, context, residual_scale=1.0
        )
        base.capacity_variant = f"P3_{experiment_id}"
        base.mixer_lr_multiplier = 2.0
        base.cross_lr_multiplier = 2.0
        return base
    feature_gate_capacity = re.fullmatch(r"FG_CAP_H(128|256|384|512)", experiment_id)
    if feature_gate_capacity:
        hidden = int(feature_gate_capacity.group(1))
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        if hidden != 128:
            for modality in base.modalities:
                base.feature_selectors[modality] = FeatureGatingCapacityModule(256, hidden)
        base.capacity_variant = f"P3_{experiment_id}"
        return base
    if experiment_id in {"FG_IDENTITY", "FG_TRANSFORMER"}:
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        for modality in base.modalities:
            base.feature_selectors[modality] = (
                nn.Identity() if experiment_id == "FG_IDENTITY"
                else ModalitySubspaceTransformerSelector(dropout=0.2)
            )
        base.capacity_variant = f"P3_{experiment_id}"
        return base
    relation_variants = {
        "RS_R1": dict(rank=32, normalize=False),
        "RS_R2": dict(rank=32, normalize=True),
        "RS_R3": dict(rank=32, normalize=True, learnable_scale=True),
        "RS_R4": dict(rank=64, normalize=True),
        "RS_R5": dict(rank=16, normalize=True),
        "RS_R6": dict(rank=32, normalize=True, active_interactions={"av", "tv"}),
        "RS_R7": dict(rank=32, normalize=True, branch_dropout=0.05),
    }
    if experiment_id in relation_variants:
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        base.transformer_encoder.cross = RelationSpecificCrossResidual(
            dim=256, **relation_variants[experiment_id]
        )
        base.capacity_variant = f"P3_{experiment_id}"
        return base
    if experiment_id in {"BLOCK_TRANSFORMER_CTRL", "BLOCK_RESIDUAL_ONLY"}:
        cross_variant = "CTRL_K6D256L2" if experiment_id.endswith("CTRL") else "X3"
        base = build_structured_cross_model(cross_variant, 0.2)
        base.transformer_encoder.cross.active_interactions = frozenset({"av", "tv", "ta"})
        base.capacity_variant = f"P3_{experiment_id}"
        base.mixer_lr_multiplier = 2.0
        return base
    if experiment_id == "BLOCK_MIXER_ONLY":
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        base.feature_integrator.residual_scale = 0.0
        base.capacity_variant = "P3_BLOCK_MIXER_ONLY"
        return base
    if experiment_id == "M4_WOTAV_NO_ADAPTIVE_CHANNEL":
        base = build_factorized_mixer_model("HO_WO_TAV", dropout)
        base.adaptive_fusion = NoAdaptiveFusion()
        base.disable_gates()
        base.capacity_variant = "P3_M4_WOTAV_NO_ADAPTIVE_CHANNEL"
        return base
    batch64_lr_scan = {
        "M4_BS64_LR1": 2.0,
        "M4_BS64_LRSQRT2": 2.0 * (2.0 ** 0.5),
        "M4_BS64_LR2": 4.0,
        "M4_BS64_LR4": 8.0,
    }
    if experiment_id in batch64_lr_scan:
        base = build_factorized_mixer_model("M4_LR2_D2_H1536", dropout)
        base.mixer_lr_multiplier = batch64_lr_scan[experiment_id]
        base.capacity_variant = f"P3_FACTORIZED_MIXER_{experiment_id}"
        return base
    if experiment_id == "HO_RAW_AUX_FULL":
        base = build_factorized_mixer_model("M4_LR2_D2_H1536", dropout)
        context = base.transformer_encoder._forward_context
        for name in base.modalities:
            base.feature_selectors[name] = RawAuxCaptureSelector(
                base.feature_selectors[name], context, name, base.modalities
            )
        base.transformer_encoder.cross_input_source = "raw_aux"
        base.capacity_variant = "P3_FACTORIZED_MIXER_M4_LR2_D2_H1536_HO_RAW_AUX_FULL"
        return base
    order_ablation = {
        "HO_AV_ONLY": {"av"},
        "HO_TV_ONLY": {"tv"},
        "HO_TA_ONLY": {"ta"},
        "HO_TAV_ONLY": {"tav"},
        "HO_WO_AV": {"tv", "ta", "tav"},
        "HO_WO_TV": {"av", "ta", "tav"},
        "HO_WO_TA": {"av", "tv", "tav"},
        "HO_WO_TAV": {"av", "tv", "ta"},
    }
    if experiment_id in order_ablation:
        base = build_factorized_mixer_model("M4_LR2_D2_H1536", dropout)
        base.transformer_encoder.cross.active_interactions = frozenset(order_ablation[experiment_id])
        base.capacity_variant = f"P3_FACTORIZED_MIXER_M4_LR2_D2_H1536_{experiment_id}"
        return base
    if experiment_id == "M4_CA_PARALLEL_X3":
        base = build_structured_cross_model("X3", 0.2)
        context = base.transformer_encoder._forward_context
        # Move the structured branch from the post-integrator position to CA outputs.
        base.feature_integrator = base.feature_integrator.base_integrator
        encoder = FactorizedMixerEncoder(0.2, 2, False, True, context, 1536, 6, 256)
        base.transformer_encoder = encoder
        residual = TopologyAlignedCrossResidual(256, 32, tuple(base.modalities))
        base.ca_parallel_residual = residual
        for modality in base.modalities:
            base.cross_attn[modality] = CrossAttentionResidualWrapper(
                base.cross_attn[modality], residual, modality, tuple(base.modalities)
            )
        base.capacity_variant = "P3_FACTORIZED_MIXER_M4_CA_PARALLEL_X3"
        base.mixer_lr_multiplier = 2.0
        return base
    if experiment_id == "HYB_M4_LR2_D2_H1536":
        base = build_structured_cross_model("X3", 0.2)
        context = base.transformer_encoder._forward_context
        encoder = HybridEncoder(0.2, reverse=False, context=context, transformer_layers=2,
                                mixer_layers=2, alternating=True)
        encoder.cross = base.transformer_encoder.cross
        base.transformer_encoder = encoder
        base.capacity_variant = "P3_HYB_M4_LR2_D2_H1536"
        base.mixer_lr_multiplier = 2.0
        return base
    no_aux = experiment_id == "M4_NOAUX_LR2_D2_H1536"
    if no_aux:
        experiment_id = "M4_LR2_D2_H1536"
    variants = {"M1": (False, False), "M2": (True, False), "M3": (True, True), "M4": (False, True),
                "M1_LR05": (False, False), "M1_LR2": (False, False),
                "M4_LR05": (False, True), "M4_LR2": (False, True),
                "M2_LR2": (True, False), "M3_LR2": (True, True),
                **{f"{base}_LR{lr}_D{drop}_H{hidden}": (base == "M4", base == "M4")
                   for base in ("M1", "M4") for lr in ("05", "10", "2")
                   for drop in ("0", "2") for hidden in ("1024", "1536")}}
    param = re.search(r"^(M1|M4)_K(4|6|8|12|16)_D(128|256|384)_L(2|4)_LR(1|2)_P(0|2)$", experiment_id)
    if param:
        base_name,k,d,l,lr,p=param.groups(); base=build_structured_cross_model("X3",float(p)/10.)
        context=base.transformer_encoder._forward_context
        encoder=FactorizedMixerEncoder(float(p)/10.,int(l),False,base_name=="M4",context,hidden=1536,token_count=int(k),token_dim=int(d)); encoder.cross=base.transformer_encoder.cross
        base.transformer_encoder=encoder; base.capacity_variant=f"P3_{experiment_id}"; base.mixer_lr_multiplier=float(lr); return base
    if experiment_id in ("H1", "H2"):
        base = build_structured_cross_model("X3", dropout)
        context = base.transformer_encoder._forward_context
        encoder = HybridEncoder(dropout, reverse=experiment_id == "H2", context=context)
        encoder.cross = base.transformer_encoder.cross
        base.transformer_encoder = encoder
        base.capacity_variant = f"P3_HYBRID_{experiment_id}"
        base.mixer_lr_multiplier = 1.
        return base
    if experiment_id.startswith("ABL_"):
        code = experiment_id[4:]; base = build_structured_cross_model("X3", 0.0)
        context = base.transformer_encoder._forward_context
        flags = {"S":(True,False,False),"C":(False,True,False),"F":(False,False,True),"SC":(True,True,False),"SF":(True,False,True),"CF":(False,True,True),"SCF":(True,True,True)}[code]
        encoder = FactorizedMixerEncoder(0.0, 2, False, False, context, 1536, 6, 256, *flags)
        encoder.cross = base.transformer_encoder.cross; base.transformer_encoder = encoder; base.capacity_variant=f"P3_ABL_{code}"; return base
    if experiment_id.startswith("LS_"):
        base=build_structured_cross_model("X3",0.0); context=base.transformer_encoder._forward_context; scale={"LS_01":0.1,"LS_001":0.01,"LS_learnable":"learnable"}[experiment_id]
        encoder=FactorizedMixerEncoder(0.0,2,False,False,context,1536,6,256,True,True,True,scale); encoder.cross=base.transformer_encoder.cross; base.transformer_encoder=encoder; base.capacity_variant=f"P3_{experiment_id}"; return base
    if experiment_id in ("M4_BD05", "M4_BD10"):
        cross_id = "X3BD05" if experiment_id.endswith("05") else "X3BD10"
        base=build_structured_cross_model(cross_id,0.2); context=base.transformer_encoder._forward_context
        encoder=FactorizedMixerEncoder(0.2,2,False,True,context,1536,6,256); encoder.cross=base.transformer_encoder.cross; base.transformer_encoder=encoder; base.capacity_variant=f"P3_{experiment_id}"; base.mixer_lr_multiplier=2.; return base
    if experiment_id not in variants:
        raise ValueError(f"unknown mixer variant {experiment_id}")
    match = re.search(r"_D(0|2)_H(1024|1536)$", experiment_id)
    if match:
        dropout = float(match.group(1)) / 10.
        hidden = int(match.group(2))
    else:
        hidden = 1536
    if experiment_id.endswith("D0"):
        dropout = 0.
    elif experiment_id.endswith("D2"):
        dropout = .2
    routed, alternating = variants[experiment_id]
    base = build_structured_cross_model("X3", dropout)
    context = base.transformer_encoder._forward_context
    encoder = FactorizedMixerEncoder(dropout, 2, routed, alternating, context, hidden=hidden)
    encoder.cross = base.transformer_encoder.cross
    base.transformer_encoder = encoder
    base.capacity_variant = f"P3_FACTORIZED_MIXER_{experiment_id}"
    base.mixer_lr_multiplier = .5 if ("_LR05_" in experiment_id or experiment_id.endswith("LR05")) else 2. if ("_LR2_" in experiment_id or experiment_id.endswith("LR2")) else 1.
    if no_aux:
        base.disable_aux_loss = True
    return base


def optimizer_parameter_groups(model, lr):
    return p3_groups(model, lr * getattr(model, "mixer_lr_multiplier", 1.))
