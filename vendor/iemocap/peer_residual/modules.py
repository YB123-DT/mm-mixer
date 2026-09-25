from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .registry import PeerVariant


class LearnedPeerReference(nn.Module):
    def __init__(self, dim: int, modalities, dropout: float, *, uniform_init: bool):
        super().__init__()
        self.modalities = tuple(modalities)
        self.networks = nn.ModuleDict()
        for target in self.modalities:
            network = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(dim, 2),
            )
            if uniform_init:
                nn.init.zeros_(network[-1].weight)
                nn.init.zeros_(network[-1].bias)
            self.networks[target] = network

    def forward(self, features):
        references, weights = {}, {}
        for target in self.modalities:
            peers = [features[name] for name in self.modalities if name != target]
            weight = torch.softmax(self.networks[target](torch.cat(peers, dim=-1)), dim=-1)
            references[target] = peers[0] * weight[:, :1] + peers[1] * weight[:, 1:]
            weights[target] = weight
        return references, weights


class FixedPeerReference(nn.Module):
    def __init__(self, modalities):
        super().__init__(); self.modalities = tuple(modalities)

    def forward(self, features):
        references, weights = {}, {}
        for target in self.modalities:
            peers = [features[name] for name in self.modalities if name != target]
            references[target] = (peers[0] + peers[1]) / 2
            weights[target] = features[target].new_full((len(features[target]), 2), .5)
        return references, weights


class AuxSufficiencyStats(nn.Module):
    def forward(self, logits):
        probability = torch.softmax(logits.detach(), dim=-1)
        confidence = probability.max(dim=-1).values
        top2 = probability.topk(2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1) / math.log(probability.size(-1))
        return torch.stack((entropy, confidence, margin), dim=-1)


class UtilityGate(nn.Module):
    def __init__(self, dim: int, dropout: float, aux: bool):
        super().__init__()
        self.aux = bool(aux)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2 + (3 if aux else 0), dim),
            nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid(),
        )

    def forward(self, feature, context, stats=None):
        values = [feature, context]
        if self.aux:
            if stats is None:
                raise ValueError("auxiliary sufficiency statistics are required")
            values.append(stats)
        return self.gate(torch.cat(values, dim=-1))


class ChannelGate(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim, dim), nn.Sigmoid(),
        )

    def forward(self, feature, context):
        return self.gate(torch.cat((feature, context), dim=-1))


@dataclass
class PeerDiagnostics:
    reference_weights: dict[str, torch.Tensor]
    utility: dict[str, torch.Tensor]
    channel: dict[str, torch.Tensor]
    residual_norm: dict[str, torch.Tensor]
    transition: dict[str, torch.Tensor]
    aux_stats: dict[str, torch.Tensor]

    def detached(self):
        return {name: {key: value.detach() for key, value in values.items()}
                for name, values in self.__dict__.items()}


class PeerResidualFusion(nn.Module):
    def __init__(self, variant: PeerVariant, dim: int, modalities, dropout: float):
        super().__init__()
        self.variant = variant
        self.dim = int(dim)
        self.modalities = tuple(modalities)
        if len(self.modalities) != 3:
            raise ValueError("peer-residual screening requires exactly three modalities")
        if variant.reference == "fixed_peer":
            self.reference = FixedPeerReference(self.modalities)
        elif variant.reference == "learned_peer":
            self.reference = LearnedPeerReference(
                dim, self.modalities, dropout,
                uniform_init=variant.stability == "uniform_peer_init",
            )
        elif variant.reference == "global":
            self.reference = None
        else:
            raise ValueError(f"unknown reference: {variant.reference}")
        if variant.utility_gate:
            self.info_gates = nn.ModuleDict({
                modality: UtilityGate(dim, dropout, variant.aux_sufficiency)
                for modality in self.modalities
            })
        else:
            self.info_gates = nn.ModuleDict()
        if variant.channel_gate:
            self.channel_gates = nn.ModuleDict({
                modality: ChannelGate(dim, dropout) for modality in self.modalities
            })
        else:
            self.channel_gates = nn.ModuleDict()
        self.aux_stats = AuxSufficiencyStats()
        self.epoch = 0
        if variant.stability == "learned_transition":
            self.transition_logits = nn.ParameterDict({
                modality: nn.Parameter(torch.tensor(-2.1972246)) for modality in self.modalities
            })
        else:
            self.transition_logits = nn.ParameterDict()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _references(self, features, global_context):
        if self.variant.reference == "global":
            if global_context is None:
                raise ValueError("global reference requires the original adaptive context")
            refs = {modality: global_context for modality in self.modalities}
            weights = {modality: global_context.new_full((len(global_context), 3), float("nan"))
                       for modality in self.modalities}
            return refs, weights
        return self.reference(features)

    def _eta(self, modality, feature, reference, stats):
        if not self.variant.utility_gate:
            return feature.new_ones((len(feature), 1))
        return self.info_gates[modality](feature, reference, stats)

    def _q(self, modality, feature, reference):
        if not self.variant.channel_gate:
            return feature.new_ones(feature.shape)
        return self.channel_gates[modality](feature, reference)

    def _v1(self, modality, feature, reference, stats):
        eta = self._eta(modality, feature, reference, stats)
        gated = eta * feature
        q = self._q(modality, gated, reference)
        output = gated if not self.variant.channel_gate else (q + .1) * gated
        return output, eta, q, feature - reference

    def _v2(self, modality, feature, reference, stats):
        if self.variant.stability == "rms_match":
            feature_rms = feature.square().mean(dim=-1, keepdim=True).sqrt()
            reference_rms = reference.square().mean(dim=-1, keepdim=True).sqrt()
            scale = (feature_rms / (reference_rms + 1e-6)).detach().clamp(.5, 2.)
            reference = reference * scale
        residual = feature - reference
        eta = self._eta(modality, feature, reference, stats)
        gated = eta * residual
        q = self._q(modality, gated, reference)
        if not self.variant.channel_gate:
            output = reference + gated
        else:
            multiplier = .1 + .9 * q if self.variant.stability == "bounded_channel" else q + .1
            output = reference + multiplier * gated
        return output, eta, q, residual

    def forward(self, features, *, aux_logits=None, global_context=None):
        references, weights = self._references(features, global_context)
        outputs, utilities, channels, residual_norms, transitions, stats_by_modality = {}, {}, {}, {}, {}, {}
        for modality in self.modalities:
            stats = None
            if self.variant.aux_sufficiency:
                if aux_logits is None or modality not in aux_logits:
                    raise ValueError("V3 requires unimodal auxiliary logits")
                stats = self.aux_stats(aux_logits[modality])
                stats_by_modality[modality] = stats
            feature, reference = features[modality], references[modality]
            if not self.variant.residual:
                output, eta, q, residual = self._v1(modality, feature, reference, stats)
            else:
                output, eta, q, residual = self._v2(modality, feature, reference, stats)
                if self.variant.stability in {"scheduled_transition", "learned_transition"}:
                    v1, _, _, _ = self._v1(modality, feature, reference, stats)
                    if self.variant.stability == "scheduled_transition":
                        transition = feature.new_tensor(min(1., self.epoch / 10.)).expand(len(feature), 1)
                    else:
                        transition = torch.sigmoid(self.transition_logits[modality]).expand(len(feature), 1)
                    output = (1 - transition) * v1 + transition * output
                    transitions[modality] = transition
            outputs[modality] = output
            utilities[modality] = eta
            channels[modality] = q
            residual_norms[modality] = residual.norm(dim=-1)
        diagnostics = PeerDiagnostics(weights, utilities, channels, residual_norms, transitions, stats_by_modality)
        return outputs, diagnostics.detached()
