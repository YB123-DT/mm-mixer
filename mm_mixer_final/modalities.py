from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


MODALITY_VARIANTS = (
    "modal_t",
    "modal_a",
    "modal_v",
    "modal_ta",
    "modal_tv",
    "modal_av",
)

_ACTIVE_MODALITIES = {
    "full": ("t", "a", "v"),
    "modal_t": ("t",),
    "modal_a": ("a",),
    "modal_v": ("v",),
    "modal_ta": ("t", "a"),
    "modal_tv": ("t", "v"),
    "modal_av": ("a", "v"),
}


@dataclass(frozen=True)
class TrainingInputMask:
    enabled: frozenset[str]

    def __call__(self, _module, args):
        if not args or not isinstance(args[0], dict):
            raise TypeError("modality mask expects a feature dictionary")
        features = {
            name: value if name in self.enabled else torch.zeros_like(value)
            for name, value in args[0].items()
        }
        return (features, *args[1:])


def active_modalities(variant: str) -> tuple[str, ...]:
    try:
        return _ACTIVE_MODALITIES[variant]
    except KeyError as error:
        if variant in MODALITY_VARIANTS:
            raise AssertionError(f"missing modality contract for {variant}") from error
        return _ACTIVE_MODALITIES["full"]


def install_training_input_mask(
    model: nn.Module, active: Iterable[str]
) -> nn.Module:
    """Remove unavailable raw inputs throughout training and evaluation.

    The three modality slots remain present because MM-Mixer has a fixed
    three-axis architecture. Unavailable inputs carry no sample information:
    they are zeroed before the first projection/gating/cross-modal operation.
    """
    enabled = frozenset(active)
    all_modalities = frozenset(("t", "a", "v"))
    if enabled == all_modalities:
        return model
    if not enabled or not enabled <= all_modalities:
        raise ValueError(f"invalid active modalities: {sorted(enabled)}")

    model.register_forward_pre_hook(TrainingInputMask(enabled))
    model.active_input_modalities = tuple(
        name for name in ("t", "a", "v") if name in enabled
    )
    return model
