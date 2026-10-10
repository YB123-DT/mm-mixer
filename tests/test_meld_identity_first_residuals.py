from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MELD = ROOT / "vendor" / "meld"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MELD))

import model as meld_model
from mm_mixer_final.config import get_config


class _OnesGate(torch.nn.Module):
    def forward(self, value):
        return torch.ones_like(value)


class _EchoAttention(torch.nn.Module):
    def forward(self, query, key, value, **kwargs):
        del key, value, kwargs
        return 3.0 * query, None


class _AdaptiveOnesGate(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, value):
        return torch.ones_like(value[..., : self.dim])


def test_zero_scales_are_exact_identity_paths():
    value = torch.randn(2, 5)
    feature = meld_model.LearnableResidualFeatureGate(_OnesGate(), 0.0)
    assert torch.equal(feature(value), value)

    combined = torch.randn(2, 10)
    adaptive = meld_model.LearnableResidualAdaptiveGate(
        _AdaptiveOnesGate(5), dim=5, initial_scale=0.0
    )
    # Reproduce the legacy caller around the wrapped gate.
    assert torch.allclose(value * adaptive(combined) + 0.1 * value, value)

    query = torch.randn(2, 1, 5)
    cross = meld_model.LearnableResidualCrossAttention(_EchoAttention(), 0.0)
    adapted, _ = cross(query, query, query)
    assert torch.allclose(adapted + 0.2 * query, query)

    integrator = meld_model.LearnableResidualQueryIntegrator(
        torch.nn.Linear(15, 5), queries=3, dim=5, initial_scale=0.0
    )
    pooled = torch.randn(2, 15)
    assert torch.allclose(integrator(pooled), pooled.reshape(2, 3, 5).mean(dim=1))


def test_residual_no_pairwise_installs_only_missing_identity_paths():
    model = meld_model.build_model("M4_NO_PAIR_RESIDUAL", dropout=0.0)

    assert model.identity_first_residuals is True
    assert isinstance(model.transformer_encoder, meld_model.FactorizedMixerOnlyEncoder)
    assert not hasattr(model.transformer_encoder, "cross")
    assert all(
        isinstance(module, meld_model.LearnableResidualFeatureGate)
        for module in model.feature_selectors.values()
    )
    assert all(
        isinstance(module, meld_model.LearnableResidualAdaptiveGate)
        for module in model.gates.values()
    )
    assert all(
        isinstance(module, meld_model.LearnableResidualCrossAttention)
        for module in model.cross_attn.values()
    )
    assert isinstance(
        model.feature_integrator, meld_model.LearnableResidualQueryIntegrator
    )

    scales = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("residual_scale")
    ]
    assert len(scales) == 10
    assert all(parameter.item() == pytest.approx(0.1) for parameter in scales)

    # AMM already has one residual per axis and must not receive another wrapper.
    assert all(isinstance(block, meld_model.MixerBlock) for block in model.transformer_encoder.blocks)

    config = get_config("meld", "residual_no_pairwise", 2025)
    assert config.switches["no_pairwise"] is True
    assert sum(config.switches.values()) == 1


def test_residual_scales_receive_finite_nonzero_gradients():
    torch.manual_seed(17)
    model = meld_model.build_model("M4_NO_PAIR_RESIDUAL", dropout=0.0)
    features = {
        "v": torch.randn(4, 342),
        "a": torch.randn(4, 1024),
        "t": torch.randn(4, 1024),
    }
    output = model(features)
    logits = output[0] if isinstance(output, tuple) else output
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()

    scale_gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.endswith("residual_scale")
    ]
    assert len(scale_gradients) == 10
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in scale_gradients)
    assert all(gradient.abs().sum().item() > 0 for gradient in scale_gradients)
