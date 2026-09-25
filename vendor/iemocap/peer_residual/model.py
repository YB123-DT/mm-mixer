from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

from .modules import PeerResidualFusion
from .registry import VARIANTS


ABLATION48 = Path(__file__).resolve().parents[1]


def _build_s0(dropout: float):
    path = str(ABLATION48)
    if path not in sys.path:
        sys.path.insert(0, path)
    from ablation48.builder import build_fusion_model
    return build_fusion_model("S0", dropout)


def _copy_gate_initialization(base, peer_fusion, experiment_id: str):
    for modality in base.modalities:
        if modality in peer_fusion.info_gates:
            source = base.info_gates[modality].gate
            target = peer_fusion.info_gates[modality].gate
            if experiment_id == "V3":
                with torch.no_grad():
                    target[0].weight[:, :source[0].in_features].copy_(source[0].weight)
                    target[0].weight[:, source[0].in_features:].zero_()
                    target[0].bias.copy_(source[0].bias)
                    target[1].load_state_dict(source[1].state_dict())
                    target[3].load_state_dict(source[3].state_dict())
            else:
                target.load_state_dict(source.state_dict(), strict=True)
        if modality in peer_fusion.channel_gates:
            peer_fusion.channel_gates[modality].gate.load_state_dict(
                base.gates[modality].state_dict(), strict=True
            )


class PeerFusionModel(nn.Module):
    """Run the pinned S0 backbone with one explicit peer-gating intervention."""

    def __init__(self, base: nn.Module, peer_fusion: PeerResidualFusion):
        super().__init__()
        self.base = base
        self.peer_fusion = peer_fusion
        self.last_diagnostics = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def set_epoch(self, epoch: int):
        self.peer_fusion.set_epoch(epoch)

    def _aux_logits(self, projected_raw):
        return {modality: self.base.mod_classifiers[modality](projected_raw[modality])
                for modality in self.base.modalities}

    def forward(self, features, return_attention=False, labels=None):
        base = self.base
        if hasattr(base.transformer_encoder, "epoch"):
            self.peer_fusion.set_epoch(base.transformer_encoder.epoch)
        projected = {}
        for modality in base.modalities:
            if modality not in features:
                raise ValueError("peer-residual screening requires complete T/A/V inputs")
            projected[modality] = base.feature_selectors[modality](base.proj[modality](features[modality]))
            # Preserve the pinned S0 RNG schedule even though quality is diagnostic-only.
            base.assess_modality_quality(features[modality], modality)

        projected_raw = {modality: projected[modality].clone() for modality in base.modalities}
        modality_list = [projected[modality] for modality in base.modalities]
        global_context, _ = base.adaptive_fusion(modality_list)
        aux_logits = self._aux_logits(projected_raw) if self.peer_fusion.variant.aux_sufficiency else None
        gated, diagnostics = self.peer_fusion(
            projected, aux_logits=aux_logits, global_context=global_context
        )
        self.last_diagnostics = diagnostics

        cross_out, attentions = [], {}
        for query_modality in base.modalities:
            others = [gated[modality] for modality in base.modalities if modality != query_modality]
            key_value = torch.stack(others, dim=1)
            query = gated[query_modality].unsqueeze(1)
            attention_out, attention_weights = base.cross_attn[query_modality](
                query=query, key=key_value, value=key_value
            )
            attentions[query_modality] = attention_weights
            cross_out.append(attention_out + query * .2)

        fused_representation = torch.cat(cross_out, dim=1)
        transformed = base.transformer_encoder(fused_representation)
        batch_size = transformed.size(0)
        queries = base.pool_queries.expand(batch_size, -1, -1)
        pooled, pool_weights = base.pool(queries, transformed, transformed)
        fused_features = base.feature_integrator(pooled.reshape(batch_size, -1))
        logits = torch.stack([classifier(fused_features) for classifier in base.classifiers]).mean(dim=0)
        if aux_logits is None:
            aux_logits = self._aux_logits(projected_raw)

        if return_attention:
            confidence = base.confidence_estimator(fused_features)
            uncertainty = base.uncertainty_estimator(fused_features)
            return logits, aux_logits, attentions, pool_weights, confidence, uncertainty, None
        return logits, aux_logits


def build_peer_model(experiment_id: str, dropout: float = .1, *, preserve_peer_rng: bool = True):
    if experiment_id not in VARIANTS:
        raise ValueError(f"unknown peer-residual experiment: {experiment_id}")
    base = _build_s0(dropout)
    rng = torch.random.get_rng_state()
    peer_fusion = PeerResidualFusion(
        VARIANTS[experiment_id], base.fusion_dim, tuple(base.modalities), dropout
    )
    if experiment_id == "V9":
        # The pinned formal loop updates this existing attribute once per epoch.
        base.transformer_encoder.epoch = 0
    _copy_gate_initialization(base, peer_fusion, experiment_id)
    # Remove unused original gates so ablations and optimizer partitions are honest.
    base.info_gates = nn.ModuleDict()
    base.gates = nn.ModuleDict()
    if preserve_peer_rng:
        torch.random.set_rng_state(rng)
    return PeerFusionModel(base, peer_fusion)


def optimizer_parameter_groups(model, lr: float):
    partitions = (
        ("base.proj.", lr * .5),
        ("base.transformer_encoder.", lr),
        ("base.classifiers.", lr * 2),
    )
    groups, used = [], set()
    named = list(model.named_parameters())
    for prefix, group_lr in partitions:
        parameters = [parameter for name, parameter in named
                      if name.startswith(prefix) and parameter.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": group_lr})
            used.update(id(parameter) for parameter in parameters)
    remaining = [parameter for _, parameter in named
                 if parameter.requires_grad and id(parameter) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": lr})
    flattened = [id(parameter) for group in groups for parameter in group["params"]]
    required = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(flattened) != len(set(flattened)) or set(flattened) != required:
        raise RuntimeError("optimizer groups do not partition trainable parameters")
    return groups
