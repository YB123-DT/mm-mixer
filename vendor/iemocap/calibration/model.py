from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .interventions import (
    InterventionSampler, contribution_loss, contribution_target,
    untouched_stability_loss,
)
from .registry import get_variant
from .roles import role_supervision_loss, role_target


ABLATION48 = Path(__file__).resolve().parents[1]


def _build_s0(dropout: float):
    path = str(ABLATION48)
    if path not in sys.path:
        sys.path.insert(0, path)
    from ablation48.builder import build_fusion_model
    return build_fusion_model("S0", dropout)


class ObservedInformationGate(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self.last_gate = None
        self.capture = False

    def forward(self, feature, context):
        output = self.original(feature, context)
        if self.capture:
            self.last_gate = self.original.gate(torch.cat((feature, context), dim=-1))
        return output


class ReplayAttention(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        self.original = original
        self.replay_logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))
        self.register_buffer("schedule_scale", torch.tensor(1.), persistent=False)

    @property
    def coefficient(self):
        return self.schedule_scale * .5 * torch.sigmoid(self.replay_logit)

    def forward(self, query, key, value, **kwargs):
        output, weights = self.original(query=query, key=key, value=value, **kwargs)
        return output + self.coefficient * query, weights


class CalibrationS0Model(nn.Module):
    """Observe S0 gates and optionally replay evidence without copying S0.forward."""

    def __init__(self, base: nn.Module, experiment_id: str):
        super().__init__()
        self.base = base
        self.variant = get_variant(experiment_id)
        self.last_gates: dict[str, torch.Tensor] = {}
        self.last_calibration: dict[str, torch.Tensor | str | float] = {}
        self.register_buffer("training_batches", torch.tensor(0, dtype=torch.long))
        self.steps_per_epoch = 182
        self.interventions = InterventionSampler(base.modalities)
        for modality in base.modalities:
            base.info_gates[modality] = ObservedInformationGate(base.info_gates[modality])
            if self.variant.replay:
                base.cross_attn[modality] = ReplayAttention(base.cross_attn[modality])

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def replay_coefficients(self):
        if not self.variant.replay:
            return {}
        return {name: self.base.cross_attn[name].coefficient for name in self.base.modalities}

    def replay_parameters(self):
        if not self.variant.replay:
            return []
        return [self.base.cross_attn[name].replay_logit for name in self.base.modalities]

    def schedule_scale(self):
        if not self.variant.staged:
            return 1.
        epoch_progress = float(self.training_batches) / self.steps_per_epoch
        return max(0., min(1., (epoch_progress - 10.) / 10.))

    def _set_replay_scale(self, scale):
        if self.variant.replay:
            for name in self.base.modalities:
                self.base.cross_attn[name].schedule_scale.fill_(scale)

    def _base_forward(self, features, return_attention=False, labels=None, capture_gates=False):
        for module in self.base.info_gates.values():
            module.capture = capture_gates
            if capture_gates:
                module.last_gate = None
        output = self.base(features, return_attention=return_attention, labels=labels)
        if capture_gates:
            self.last_gates = {
                name: self.base.info_gates[name].last_gate for name in self.base.modalities
            }
        for module in self.base.info_gates.values():
            module.capture = False
        return output

    def _isec_loss(self, features, labels, clean_logits, clean_gates):
        intervened_features, intervention = self.interventions.sample(features)
        intervened_output = self._base_forward(
            intervened_features, labels=labels, capture_gates=True
        )
        intervened_logits = intervened_output[0] if isinstance(intervened_output, tuple) else intervened_output
        intervened_gates = dict(self.last_gates)
        clean_per_example = F.cross_entropy(clean_logits, labels, reduction="none")
        intervened_per_example = F.cross_entropy(intervened_logits, labels, reduction="none")
        target, weight = contribution_target(
            clean_per_example, intervened_per_example, self.variant.target_temperature
        )
        zero = clean_logits.new_zeros(())
        contrib = contribution_loss(intervened_gates[intervention.modality], target, weight) \
            if self.variant.contribution else zero
        stable = untouched_stability_loss(clean_gates, intervened_gates, intervention.modality) \
            if self.variant.stability else zero
        intervention_ce = intervened_per_example.mean() \
            if self.variant.intervention_classification else zero
        additional = (
            self.variant.lambda_contribution * contrib
            + self.variant.lambda_stability * stable
            + .1 * intervention_ce
        )
        self.last_calibration = {
            "additional_loss": additional.detach(), "contribution_loss": contrib.detach(),
            "stability_loss": stable.detach(), "intervention_ce": intervention_ce.detach(),
            "target_mean": target.mean(), "target_weight_mean": weight.mean(),
            "intervention_modality": intervention.modality,
            "intervention_kind": intervention.kind, "intervention_strength": intervention.strength,
        }
        return additional

    def _role_loss(self, aux_logits, labels, gates):
        losses, targets, weights = [], {}, {}
        for modality in self.base.modalities:
            target, weight = role_target(
                aux_logits[modality], labels, self.variant.target_temperature
            )
            targets[modality], weights[modality] = target, weight
            losses.append(role_supervision_loss(gates[modality], target, weight))
        role = torch.stack(losses).mean()
        additional = self.variant.lambda_role * role
        self.last_calibration = {
            "additional_loss": additional.detach(), "role_loss": role.detach(),
            **{f"role_target_{name}": value.mean() for name, value in targets.items()},
            **{f"role_weight_{name}": value.mean() for name, value in weights.items()},
        }
        return additional

    def forward(self, features, return_attention=False, labels=None, *, capture_gates=False):
        training_batch = labels is not None and self.training
        if training_batch:
            self.training_batches.add_(1)
        scale = self.schedule_scale()
        self._set_replay_scale(scale)
        needs_supervision = labels is not None and self.training and (
            self.variant.family == "isec" or self.variant.role_supervision
        ) and scale > 0
        output = self._base_forward(
            features, return_attention=return_attention, labels=labels,
            capture_gates=capture_gates or needs_supervision,
        )
        if not needs_supervision:
            if labels is not None and self.training and not self.variant.auxiliary_supervision:
                return output[0], {}
            return output
        clean_gates = dict(self.last_gates)
        main_logits, aux_logits = output[:2]
        if self.variant.family == "isec":
            additional = self._isec_loss(features, labels, main_logits, clean_gates)
        else:
            additional = self._role_loss(aux_logits, labels, clean_gates)
        additional = additional * scale
        self.last_calibration["schedule_scale"] = main_logits.new_tensor(scale)
        # MultitaskFusionLoss applies contrastive_weight=0.1 to the third item.
        contrastive_slot = additional / .1
        returned_aux = aux_logits if self.variant.auxiliary_supervision else {}
        return main_logits, returned_aux, contrastive_slot


def build_calibration_model(experiment_id: str, dropout: float = .1):
    return CalibrationS0Model(_build_s0(dropout), experiment_id)


def optimizer_parameter_groups(model, lr: float):
    partitions = (
        ("base.proj.", lr * .5),
        ("base.transformer_encoder.", lr),
        ("base.classifiers.", lr * 2),
    )
    named = list(model.named_parameters()); groups = []; used = set()
    for prefix, group_lr in partitions:
        parameters = [p for name, p in named if name.startswith(prefix) and p.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": group_lr})
            used.update(id(p) for p in parameters)
    remaining = [p for _, p in named if p.requires_grad and id(p) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": lr})
    return groups
