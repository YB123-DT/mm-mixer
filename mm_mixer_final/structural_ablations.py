from __future__ import annotations

from torch import nn


STRUCTURAL_ABLATION_VARIANTS = (
    "no_sequence_mixing",
    "no_modality_mixing",
    "no_feature_mixing",
    "one_mixer_block",
    "no_feature_gating",
)


def apply_structural_ablation(model: nn.Module, variant: str) -> nn.Module:
    """Apply one post-initialization structural ablation to a Full model."""
    if variant not in STRUCTURAL_ABLATION_VARIANTS:
        return model

    blocks = model.transformer_encoder.blocks
    if variant == "no_sequence_mixing":
        for block in blocks:
            block.use_sub = False
    elif variant == "no_modality_mixing":
        for block in blocks:
            block.use_mod = False
    elif variant == "no_feature_mixing":
        for block in blocks:
            block.use_ffn = False
    elif variant == "one_mixer_block":
        model.transformer_encoder.blocks = nn.ModuleList([blocks[0]])
    elif variant == "no_feature_gating":
        model.feature_selectors = nn.ModuleDict(
            {name: nn.Identity() for name in model.modalities}
        )

    model.capacity_variant = f"{model.capacity_variant}_{variant.upper()}"
    return model
