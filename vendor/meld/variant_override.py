from __future__ import annotations

import contextlib
import types

import torch
from torch import nn

import model as model_module


CANDIDATES = (
    "MX_LR1",
    "MX_SCALE05",
    "MX_H1024",
    "MX_H768",
    "MX_L1",
    "MX_DROP10",
)
ORIGINAL_FUSION = model_module.HierarchicalAttentionFusion
_ACTIVE_CANDIDATE = None
_ACTIVE_MODALITIES = ("t", "a", "v")
_ACTIVE_STRUCTURAL_VARIANT = "full"


def _branch_delta_sub(block, value):
    return block.subspace_mlp(
        block.norm_sub(value).transpose(-1, -2)
    ).transpose(-1, -2)


def _branch_delta_modal(block, value):
    return block.route(
        block.norm_modal(value).permute(0, 2, 3, 1)
    ).permute(0, 3, 1, 2)


def _scaled_forward(block, value):
    scale = block.residual_scale
    if block.reverse:
        value = value + scale * _branch_delta_modal(block, value)
        value = value + scale * _branch_delta_sub(block, value)
    else:
        value = value + scale * _branch_delta_sub(block, value)
        value = value + scale * _branch_delta_modal(block, value)
    return value + scale * block.ffn(block.norm_ffn(value))


def _dropout_forward(block, value):
    dropout = block.branch_dropout
    if block.reverse:
        value = value + dropout(_branch_delta_modal(block, value))
        value = value + dropout(_branch_delta_sub(block, value))
    else:
        value = value + dropout(_branch_delta_sub(block, value))
        value = value + dropout(_branch_delta_modal(block, value))
    return value + dropout(block.ffn(block.norm_ffn(value)))


def _replace_ffn_width(value, hidden):
    for block in value.transformer_encoder.blocks:
        first, last = block.ffn[0], block.ffn[-1]
        block.ffn = nn.Sequential(
            nn.Linear(first.in_features, hidden, bias=first.bias is not None),
            nn.GELU(),
            nn.Linear(hidden, last.out_features, bias=last.bias is not None),
        ).to(device=first.weight.device, dtype=first.weight.dtype)


def apply_candidate(value, candidate):
    if candidate != "FULL" and candidate not in CANDIDATES:
        raise ValueError(f"unknown candidate: {candidate}")
    if candidate == "FULL":
        return value
    if candidate == "MX_LR1":
        value.experiment_lr_multiplier = 1.0
    elif candidate == "MX_SCALE05":
        for block in value.transformer_encoder.blocks:
            block.residual_scale = 0.5
            block.forward = types.MethodType(_scaled_forward, block)
    elif candidate == "MX_H1024":
        _replace_ffn_width(value, 1024)
    elif candidate == "MX_H768":
        _replace_ffn_width(value, 768)
    elif candidate == "MX_L1":
        value.transformer_encoder.blocks = nn.ModuleList(
            [value.transformer_encoder.blocks[0]]
        )
    elif candidate == "MX_DROP10":
        for block in value.transformer_encoder.blocks:
            block.branch_dropout = nn.Dropout(0.1)
            block.forward = types.MethodType(_dropout_forward, block)
    value.capacity_variant = f"{value.capacity_variant}_{candidate}"
    return value


class CandidateHierarchicalAttentionFusion(ORIGINAL_FUSION):
    """Importable candidate class so the formal trainer can pickle the model."""

    def __init__(self, *args, **kwargs):
        requested = kwargs.get("capacity_variant")
        super().__init__(*args, **kwargs)
        if requested in {
            "M4_PAIR",
            "M4_PAIR_NO_MIXER",
            "M4_NO_PAIR",
            "M4_PAIR_NO_ADAPTIVE",
            "M4_PAIR_NO_CA",
        }:
            if _ACTIVE_CANDIDATE is None:
                raise RuntimeError("candidate context is not active")
            apply_candidate(self, _ACTIVE_CANDIDATE)
            from mm_mixer_final.structural_ablations import apply_structural_ablation

            apply_structural_ablation(self, _ACTIVE_STRUCTURAL_VARIANT)
            if frozenset(_ACTIVE_MODALITIES) != frozenset(("t", "a", "v")):
                from mm_mixer_final.modalities import install_training_input_mask

                install_training_input_mask(self, _ACTIVE_MODALITIES)


@contextlib.contextmanager
def candidate_model_context(
    candidate, active_modalities=("t", "a", "v"), structural_variant="full"
):
    global _ACTIVE_CANDIDATE, _ACTIVE_MODALITIES, _ACTIVE_STRUCTURAL_VARIANT
    original_class = model_module.HierarchicalAttentionFusion
    original_candidate = _ACTIVE_CANDIDATE
    original_modalities = _ACTIVE_MODALITIES
    original_structural_variant = _ACTIVE_STRUCTURAL_VARIANT
    _ACTIVE_CANDIDATE = candidate
    _ACTIVE_MODALITIES = tuple(active_modalities)
    _ACTIVE_STRUCTURAL_VARIANT = structural_variant
    model_module.HierarchicalAttentionFusion = CandidateHierarchicalAttentionFusion
    try:
        yield
    finally:
        model_module.HierarchicalAttentionFusion = original_class
        _ACTIVE_CANDIDATE = original_candidate
        _ACTIVE_MODALITIES = original_modalities
        _ACTIVE_STRUCTURAL_VARIANT = original_structural_variant
