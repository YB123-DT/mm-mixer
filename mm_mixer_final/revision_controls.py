"""Controlled replacements for the revision experiments.

Call ``apply_revision_control`` on a freshly built Full model before creating
its optimizer. The original encoder publishes the post-encoder residual, so
these controls retain that object, its context/channel, and the main pooling.
No controls add unused parameters to manufacture an exact budget match.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn


REVISION_CONTROL_VARIANTS = (
    "amm_mlp",
    "amm_attention",
    "amm_cubemlp",
    "single_projection_view",
    "no_feature_and_adaptive_gating",
    "pairwise_mlp_residual",
    "amm_mlp_no_aux",
)
NO_AUXILIARY_LOSS_VARIANTS = ("no_auxiliary_loss", "amm_mlp_no_aux")

REVISION_CONTROL_SPECS = {
    "amm_mlp": {
        "operation": "two_joint_residual_mlps",
        "budget": "nearest_full_amm_parameter_count",
        "tokens": 1,
        "blocks": 2,
        "description": "Flatten the three input branches, apply two PreNorm residual MLPs, restore three branches.",
    },
    "amm_attention": {
        "operation": "two_joint_self_attention_blocks",
        "budget": "nearest_full_amm_parameter_count",
        "heads": 8,
        "blocks": 2,
        "description": "Retain the learned projection views; jointly attend over the modality-view tokens with two PreNorm blocks.",
    },
    "amm_cubemlp": {
        "operation": "cubemlp_style_fixed_axis_postnorm",
        "budget": "nearest_full_amm_parameter_count",
        "blocks": 2,
        "source": "https://arxiv.org/html/2207.14087",
        "source_equations": [2, 3, 4],
        "description": "CubeMLP-style S-M-D two-affine residual units with normalization on the mixed axis after residual addition.",
        "adaptation": "Uses MM-Mixer's learned S views, fixed axis sizes, GELU, and mean-view pooling, not CubeMLP's temporal input or complete predictor.",
    },
    "single_projection_view": {
        "operation": "full_amm_with_one_projection_view",
        "tokens": 1,
        "subspace_hidden": 2,
        "description": "Retain both AMM blocks and all axes, replacing D-to-6D by D-to-D and S MLPs by 1-to-2-to-1.",
    },
    "no_feature_and_adaptive_gating": {
        "operation": "remove_feature_and_adaptive_gating",
        "description": "Combine the existing feature-selector identity and adaptive-context/channel-gate deletions.",
    },
    "pairwise_mlp_residual": {
        "operation": "ordinary_joint_mlp_residual",
        "budget": "nearest_live_pairwise_parameter_count",
        "description": "A zero-output-initialized ordinary 3D-to-hidden-to-D MLP at the same residual add position, matching active pairwise parameters.",
    },
    "amm_mlp_no_aux": {
        "operation": "two_joint_residual_mlps",
        "budget": "nearest_full_amm_parameter_count",
        "tokens": 1,
        "blocks": 2,
        "auxiliary_loss": False,
        "description": "The same amm_mlp replacement with auxiliary losses disabled by the dataset trainer.",
    },
}


def parameter_count(module: nn.Module) -> int:
    """Count registered trainable scalars (including inactive legacy slots)."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def pairwise_live_parameter_count(cross: nn.Module) -> int:
    """Count the exact active slots of Full's three-pair HO_WO_TAV branch.

    Full retains three unused triple projections and an unused fourth output
    slice. Matching its registered count would give an ordinary MLP much more
    active capacity; both counts are therefore included in control metadata.
    """
    active = getattr(cross, "active_interactions", frozenset(("va", "vt", "at")))
    if set(active) not in ({"av", "tv", "ta"}, {"va", "vt", "at"}):
        raise ValueError("pairwise budget requires exactly the three active pair interactions")
    if len(cross.pair_projections) != 3 or cross.output.in_features != 4 * cross.rank:
        raise ValueError("unexpected Full pairwise parameterization")
    count = parameter_count(cross.pair_projections)
    if cross.output.weight.requires_grad:
        count += cross.output.out_features * 3 * cross.rank
    if cross.output.bias is not None and cross.output.bias.requires_grad:
        count += cross.output.bias.numel()
    return count


def _nearest_hidden(target: int, constant: int, slope: int) -> int:
    ideal = (target - constant) / slope
    candidates = {max(1, math.floor(ideal)), max(1, math.ceil(ideal))}
    return min(candidates, key=lambda hidden: (abs(constant + slope * hidden - target), hidden))


def _mlp(width: int, hidden: int, output: int | None = None) -> nn.Sequential:
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width if output is None else output))


class JointResidualMLPBlock(nn.Module):
    """Mix all three branches directly; no learned view expansion."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = int(dim)
        self.norm = nn.LayerNorm(3 * dim)
        self.ffn = _mlp(3 * dim, hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if tuple(value.shape[1:]) != (3, 1, self.dim):
            raise ValueError(f"expected [B, 3, 1, {self.dim}] for the joint MLP")
        flattened = value.reshape(value.shape[0], 3 * self.dim)
        return (flattened + self.ffn(self.norm(flattened))).reshape_as(value)


class JointAttentionBlock(nn.Module):
    """Attend across all modality-view tokens, then apply a feature FFN."""

    def __init__(self, dim: int, hidden: int, heads: int = 8):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0.0, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = _mlp(dim, hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, modalities, views, dim = value.shape
        tokens = value.reshape(batch, modalities * views, dim)
        normalized = self.norm_attention(tokens)
        update, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + update
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens.reshape(batch, modalities, views, dim)


class CubeMLPStyleBlock(nn.Module):
    """Adapt the fixed S-M-D, axis-wise post-residual LN in CubeMLP Eq. 2-4.

    The inherited learned views and downstream mean pooling are MM-Mixer's
    interface adaptations. This class is not a reproduction of the original
    temporal feature pipeline, classifier, or dimensionality-reduction setup.
    """

    def __init__(self, tokens: int, dim: int, hidden: int):
        super().__init__()
        self.sub = _mlp(tokens, 2 * tokens)
        self.route = _mlp(3, 6)
        self.ffn = _mlp(dim, hidden)
        self.norm_sub = nn.LayerNorm(tokens)
        self.norm_mod = nn.LayerNorm(3)
        self.norm_ffn = nn.LayerNorm(dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        sequence = value.transpose(-1, -2)
        value = self.norm_sub(sequence + self.sub(sequence)).transpose(-1, -2)
        modality = value.permute(0, 2, 3, 1)
        value = self.norm_mod(modality + self.route(modality)).permute(0, 3, 1, 2)
        return self.norm_ffn(value + self.ffn(value))


class OrdinaryResidualMLP(nn.Module):
    """Ordinary nonlinear joint residual, with no multiplicative pair terms."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = int(dim)
        self.mlp = _mlp(3 * dim, hidden, dim)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (3, self.dim):
            raise ValueError(f"expected [B, 3, {self.dim}] modality branches")
        return self.mlp(tokens.reshape(tokens.shape[0], 3 * self.dim))

    def decomposed(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        # Preserve the context's diagnostic interface without calling this
        # ordinary MLP a sum of explicit pairwise products.
        return {"mlp": self(tokens)}


class NoAdaptiveContext(nn.Module):
    """The existing no-AG placeholder, accepting either dataset signature."""

    def forward(self, modality_features, bias=None):
        del bias
        reference = modality_features[0]
        return torch.zeros_like(reference), reference.new_full(
            (reference.shape[0], len(modality_features)), 1.0 / len(modality_features)
        )


def revision_control_metadata(model: nn.Module) -> dict:
    return copy.deepcopy(getattr(model, "revision_control", {}))


def apply_revision_control(model: nn.Module, variant: str) -> nn.Module:
    """Apply one named control to Full in place; other variants are exact no-ops."""
    if variant not in REVISION_CONTROL_VARIANTS:
        return model
    if getattr(model, "revision_control", None):
        raise ValueError("revision controls require a fresh Full model; do not stack controls")
    encoder = model.transformer_encoder
    if len(encoder.blocks) != 2 or encoder.token_count != 6:
        raise ValueError("revision controls require the Full two-block, six-view encoder")
    reference = encoder.split.weight
    dim, views = int(encoder.token_dim), int(encoder.token_count)
    if encoder.split.in_features != dim:
        raise ValueError("revision controls expect equal external and hidden feature widths")

    metadata = copy.deepcopy(REVISION_CONTROL_SPECS[variant])
    metadata.update(variant=variant, model_parameters_before=parameter_count(model))
    original_core = parameter_count(encoder.split) + parameter_count(encoder.blocks)
    metadata["original_amm_parameters"] = original_core

    if variant in {"amm_mlp", "amm_mlp_no_aux"}:
        width = 3 * dim
        hidden = _nearest_hidden(original_core, 2 * 3 * width, 2 * (2 * width + 1))
        encoder.split = nn.Identity()
        encoder.token_count = 1
        encoder.blocks = nn.ModuleList(JointResidualMLPBlock(dim, hidden) for _ in range(2)).to(reference)
        metadata.update(hidden=hidden, target_parameters=original_core)
    elif variant == "amm_attention":
        heads = 8
        if dim % heads:
            raise ValueError("attention control requires width divisible by eight")
        block_target = parameter_count(encoder.blocks)
        # Per block: MHA 4D²+4D; two LN 4D; FFN bias D.
        hidden = _nearest_hidden(block_target, 2 * (4 * dim * dim + 9 * dim), 2 * (2 * dim + 1))
        encoder.blocks = nn.ModuleList(JointAttentionBlock(dim, hidden, heads) for _ in range(2)).to(reference)
        metadata.update(hidden=hidden, target_parameters=original_core, internal_dropout=0.0)
    elif variant == "amm_cubemlp":
        # S MLP: 4S²+3S; M MLP: 45; axis LNs: 2(S+3+D);
        # feature MLP adds (2D+1)*hidden + D.
        constant = 4 * views * views + 3 * views + 45 + 2 * (views + 3 + dim) + dim
        hidden = _nearest_hidden(parameter_count(encoder.blocks), 2 * constant, 2 * (2 * dim + 1))
        encoder.blocks = nn.ModuleList(CubeMLPStyleBlock(views, dim, hidden) for _ in range(2)).to(reference)
        metadata.update(hidden=hidden, target_parameters=original_core, axis_order=["S", "M", "D"])
    elif variant == "single_projection_view":
        encoder.token_count = 1
        encoder.split = nn.Linear(dim, dim).to(reference)
        for block in encoder.blocks:
            attribute = "subspace_mlp" if hasattr(block, "subspace_mlp") else "sub"
            setattr(block, attribute, _mlp(1, 2).to(reference))
    elif variant == "no_feature_and_adaptive_gating":
        model.feature_selectors = nn.ModuleDict({name: nn.Identity() for name in model.modalities})
        model.adaptive_fusion = NoAdaptiveContext()
        model.disable_gates()
    elif variant == "pairwise_mlp_residual":
        original_cross = encoder.cross
        live_budget = pairwise_live_parameter_count(original_cross)
        hidden = _nearest_hidden(live_budget, dim, 4 * dim + 1)
        encoder.cross = OrdinaryResidualMLP(dim, hidden).to(reference)
        metadata.update(
            hidden=hidden,
            target_parameters=live_budget,
            original_pairwise_registered_parameters=parameter_count(original_cross),
            original_pairwise_live_parameters=live_budget,
            replacement_parameters=parameter_count(encoder.cross),
            zero_output_initialized=True,
        )

    if variant in {"amm_mlp", "amm_attention", "amm_cubemlp", "amm_mlp_no_aux"}:
        metadata["replacement_parameters"] = parameter_count(encoder.split) + parameter_count(encoder.blocks)
    if "target_parameters" in metadata:
        metadata["parameter_difference"] = metadata["replacement_parameters"] - metadata["target_parameters"]
    metadata.update(
        model_parameters_after=parameter_count(model),
        actual_tokens=encoder.token_count,
        actual_dim=encoder.token_dim,
        actual_blocks=len(encoder.blocks),
    )
    model.revision_control = metadata
    model.capacity_variant = f"{model.capacity_variant}_{variant.upper()}"
    return model
