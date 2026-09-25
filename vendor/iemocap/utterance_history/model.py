from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ABLATION48 = Path(__file__).resolve().parents[1]


class PackedLinear3(nn.Module):
    """One Linear(256,18) whose three six-logit slices are averaged."""

    def __init__(self, heads: nn.ModuleList):
        super().__init__()
        if len(heads) != 3 or not all(isinstance(head, nn.Linear) for head in heads):
            raise ValueError("PackedLinear3 requires exactly three Linear heads")
        rng = torch.random.get_rng_state()
        try:
            self.linear = nn.Linear(heads[0].in_features, 3 * heads[0].out_features)
        finally:
            torch.random.set_rng_state(rng)
        with torch.no_grad():
            self.linear.weight.copy_(torch.cat([head.weight for head in heads], dim=0))
            self.linear.bias.copy_(torch.cat([head.bias for head in heads], dim=0))
        self.classes = heads[0].out_features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).reshape(len(features), 3, self.classes).mean(dim=1)


def pack_linear3(base: nn.Module) -> nn.Module:
    packed = PackedLinear3(base.classifiers)
    base.classifiers = nn.ModuleList([packed])
    base.num_classifier_heads = 1
    return base


class HistoryResidualEncoder(nn.Module):
    """Encode one causal projected prefix into its last valid state."""

    def __init__(self, dim: int = 256, heads: int = 8, ffn_dim: int = 1024):
        super().__init__()
        self.causal = True
        layer = nn.TransformerEncoderLayer(
            dim, heads, ffn_dim, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 1)
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, projected: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if projected.ndim != 3 or valid.shape != projected.shape[:2]:
            raise ValueError("projected history and mask shapes do not match")
        if not valid.bool().any(dim=1).all():
            raise ValueError("every history prefix must contain a valid turn")
        width = projected.size(1)
        causal_mask = torch.triu(
            torch.ones(width, width, dtype=torch.bool, device=projected.device), diagonal=1
        )
        encoded = self.encoder(
            projected, mask=causal_mask, src_key_padding_mask=~valid.bool()
        )
        last = valid.long().sum(dim=1) - 1
        selected = encoded[torch.arange(len(encoded), device=encoded.device), last]
        return self.output_norm(selected)


class UtteranceHistoryResidualModel(nn.Module):
    """Inject causal history deltas while preserving the base utterance forward."""

    def __init__(self, base: nn.Module, modalities=(), force_identity: bool = False):
        super().__init__()
        self.base = base
        self.history_modalities = tuple(modalities)
        self.force_identity = bool(force_identity)
        self.history_encoders = nn.ModuleDict(
            {modality: HistoryResidualEncoder() for modality in self.history_modalities}
        )
        self.history_alpha = nn.ParameterDict(
            {modality: nn.Parameter(torch.zeros(())) for modality in self.history_modalities}
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def _project_prefix(self, modality: str, history: torch.Tensor) -> torch.Tensor:
        # Reuse the S0 Linear/LN/GELU weights but intentionally skip its dropout.
        flat = history.reshape(-1, history.size(-1))
        projection = self.base.proj[modality]
        projected = projection[2](projection[1](projection[0](flat)))
        return projected.reshape(*history.shape[:2], -1)

    @staticmethod
    def _inject(delta: torch.Tensor, alpha: torch.Tensor):
        return lambda _module, _inputs, output: output + alpha * delta

    def forward(self, features, return_attention=False, labels=None):
        current = {modality: features[modality] for modality in ("t", "a", "v")}
        if self.force_identity or not self.history_modalities:
            return self.base(current, return_attention=return_attention, labels=labels)

        valid = features["history_mask"].bool()
        deltas = {
            modality: self.history_encoders[modality](
                self._project_prefix(modality, features[f"{modality}_history"]), valid
            )
            for modality in self.history_modalities
        }
        handles = []
        try:
            for modality, delta in deltas.items():
                handles.append(self.base.proj[modality].register_forward_hook(
                    self._inject(delta, self.history_alpha[modality])
                ))
            return self.base(current, return_attention=return_attention, labels=labels)
        finally:
            for handle in handles:
                handle.remove()


def _build_s0(dropout: float):
    path = str(ABLATION48)
    if path not in sys.path:
        sys.path.insert(0, path)
    from ablation48.builder import build_fusion_model
    return build_fusion_model("S0", dropout)


def build_history_model(experiment_id: str, dropout: float = 0.1):
    definitions = {
        "H0": ((), False),
        "H00": (("a", "v"), True),
        "HA": (("a",), False),
        "HV": (("v",), False),
        "HAV": (("a", "v"), False),
        "SL": ((), False),
    }
    if experiment_id not in definitions:
        raise ValueError(f"unknown history experiment: {experiment_id}")
    modalities, force_identity = definitions[experiment_id]
    base = _build_s0(dropout)
    if experiment_id == "SL":
        pack_linear3(base)
    return UtteranceHistoryResidualModel(base, modalities, force_identity)
