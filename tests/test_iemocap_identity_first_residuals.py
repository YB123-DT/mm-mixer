from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
IEMOCAP = ROOT / "vendor" / "iemocap"
sys.path[:0] = [str(ROOT), str(IEMOCAP)]

from dataset_runners.iemocap import build_variant_model
from factorized_mixer.identity_residual import (
    LearnableResidualAdaptiveGate,
    LearnableResidualCrossAttention,
    LearnableResidualFeatureGate,
    LearnableResidualQueryIntegrator,
)
from mm_mixer_final.config import get_config


def test_residual_mainline_matches_meld_identity_first_contract():
    model = build_variant_model("residual_no_pairwise", dropout=0.0)
    assert model.identity_first_residuals is True
    assert not hasattr(model.transformer_encoder, "cross")
    assert all(
        isinstance(module, LearnableResidualFeatureGate)
        for module in model.feature_selectors.values()
    )
    assert all(
        isinstance(module, LearnableResidualAdaptiveGate)
        for module in model.gates.values()
    )
    assert all(
        isinstance(module, LearnableResidualCrossAttention)
        for module in model.cross_attn.values()
    )
    assert isinstance(model.feature_integrator, LearnableResidualQueryIntegrator)
    scales = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("residual_scale")
    ]
    assert len(scales) == 10
    assert all(parameter.item() == pytest.approx(0.1) for parameter in scales)
    config = get_config("iemocap", "residual_no_pairwise", 2025)
    assert config.switches["no_pairwise"] is True


def test_residual_scales_receive_gradients():
    torch.manual_seed(19)
    model = build_variant_model("residual_no_pairwise", dropout=0.0)
    output = model(
        {
            "v": torch.randn(4, 342),
            "a": torch.randn(4, 1024),
            "t": torch.randn(4, 1024),
        }
    )
    logits = output[0] if isinstance(output, tuple) else output
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    gradients = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.endswith("residual_scale")
    ]
    assert len(gradients) == 10
    assert all(value is not None and torch.isfinite(value).all() for value in gradients)
    assert all(value.abs().sum().item() > 0 for value in gradients)


@pytest.mark.parametrize(
    ("variant", "attribute"),
    (
        ("residual_no_feature_gating", "feature_selectors"),
        ("residual_no_adaptive_gating", "gates"),
        ("residual_no_cross_attention", "cross_attn"),
        ("residual_no_mixer", "transformer_encoder"),
        ("residual_no_feature_and_adaptive_gating", "feature_selectors"),
    ),
)
def test_residual_ablation_variants_build_and_run(variant, attribute):
    model = build_variant_model(variant, dropout=0.0)
    assert hasattr(model, attribute)
    output = model(
        {
            "v": torch.randn(2, 342),
            "a": torch.randn(2, 1024),
            "t": torch.randn(2, 1024),
        }
    )
    logits = output[0] if isinstance(output, tuple) else output
    assert logits.shape == (2, 6)
