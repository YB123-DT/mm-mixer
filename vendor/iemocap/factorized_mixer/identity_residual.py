"""Identity-first residual wrappers shared with the MELD residual mainline."""

from __future__ import annotations

import torch
from torch import nn


RESIDUAL_VARIANTS = frozenset(
    {
        "residual_no_pairwise",
        "residual_no_feature_gating",
        "residual_no_adaptive_gating",
        "residual_no_cross_attention",
        "residual_no_mixer",
        "residual_no_feature_and_adaptive_gating",
    }
)


class LearnableResidualFeatureGate(nn.Module):
    def __init__(self, gate: nn.Module, initial_scale: float = 0.1):
        super().__init__()
        self.gate = gate
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, value):
        return value + self.residual_scale * self.gate(value)


class LearnableResidualAdaptiveGate(nn.Module):
    """Turn the caller's ``x*gate + 0.1*x`` into ``x + alpha*x*gate``."""

    def __init__(self, gate: nn.Module, initial_scale: float = 0.1):
        super().__init__()
        self.gate = gate
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, combined):
        return 0.9 + self.residual_scale * self.gate(combined)


class LearnableResidualCrossAttention(nn.Module):
    """Turn the caller's ``attn + 0.2*q`` into ``q + alpha*attn``."""

    def __init__(self, attention: nn.Module, initial_scale: float = 0.1):
        super().__init__()
        self.attention = attention
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, query, key, value, **kwargs):
        update, weights = self.attention(query, key, value, **kwargs)
        return 0.8 * query + self.residual_scale * update, weights


class LearnableResidualQueryIntegrator(nn.Module):
    """Match the tenth learnable residual used by the MELD mainline."""

    def __init__(
        self,
        integrator: nn.Module,
        queries: int = 3,
        dim: int = 256,
        initial_scale: float = 0.1,
    ):
        super().__init__()
        self.integrator = integrator
        self.queries = int(queries)
        self.dim = int(dim)
        self.residual_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, pooled):
        expected = self.queries * self.dim
        if pooled.ndim != 2 or pooled.shape[-1] != expected:
            raise ValueError(f"expected flattened [{self.queries}, {self.dim}] queries")
        main = pooled.reshape(pooled.shape[0], self.queries, self.dim).mean(dim=1)
        return main + self.residual_scale * self.integrator(pooled)


def install_identity_first_residuals(model, initial_scale: float = 0.1):
    """Install the same ten learnable residual scales as the MELD mainline."""
    if model.cross_attn is None:
        raise ValueError("identity-first MCA residuals require cross-attention modules")
    for modality in model.modalities:
        model.feature_selectors[modality] = LearnableResidualFeatureGate(
            model.feature_selectors[modality], initial_scale
        )
        model.gates[modality] = LearnableResidualAdaptiveGate(
            model.gates[modality], initial_scale
        )
        model.cross_attn[modality] = LearnableResidualCrossAttention(
            model.cross_attn[modality], initial_scale
        )
    model.feature_integrator = LearnableResidualQueryIntegrator(
        model.feature_integrator,
        queries=3,
        dim=model.fusion_dim,
        initial_scale=initial_scale,
    )
    model.identity_first_residuals = True
    return model


def apply_identity_first_ablation(model, variant: str, no_adaptive_fusion: nn.Module):
    if variant == "residual_no_feature_gating":
        model.feature_selectors = nn.ModuleDict(
            {name: nn.Identity() for name in model.modalities}
        )
    elif variant == "residual_no_adaptive_gating":
        model.adaptive_fusion = no_adaptive_fusion
        model.disable_gates()
    elif variant == "residual_no_cross_attention":
        model.cross_attn = None
    elif variant == "residual_no_mixer":
        model.transformer_encoder = nn.Identity()
    elif variant == "residual_no_feature_and_adaptive_gating":
        model.feature_selectors = nn.ModuleDict(
            {name: nn.Identity() for name in model.modalities}
        )
        model.adaptive_fusion = no_adaptive_fusion
        model.disable_gates()
    return model
