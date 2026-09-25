from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Intervention:
    modality: str
    kind: str
    strength: float


class InterventionSampler:
    kinds = ("zero", "permute", "gaussian")

    def __init__(self, modalities=("text", "audio", "visual"), noise_range=(.25, 1.0)):
        self.modalities = tuple(modalities)
        self.noise_range = tuple(float(value) for value in noise_range)

    def sample(self, features: dict[str, torch.Tensor], generator=None):
        device = next(iter(features.values())).device
        modality_index = int(torch.randint(len(self.modalities), (), generator=generator, device=device))
        kind_index = int(torch.randint(len(self.kinds), (), generator=generator, device=device))
        low, high = self.noise_range
        strength = float(torch.empty((), device=device).uniform_(low, high, generator=generator))
        intervention = Intervention(self.modalities[modality_index], self.kinds[kind_index], strength)
        return self.apply(features, intervention, generator=generator), intervention

    def apply(self, features, intervention: Intervention, generator=None):
        output = dict(features)
        value = features[intervention.modality]
        if intervention.kind == "zero":
            changed = torch.zeros_like(value)
        elif intervention.kind == "permute":
            permutation = torch.randperm(len(value), device=value.device, generator=generator)
            changed = value[permutation]
        elif intervention.kind == "gaussian":
            rms = value.detach().square().mean().sqrt().clamp_min(1e-6)
            noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
            changed = value + noise * rms * intervention.strength
        else:
            raise ValueError(f"unknown intervention: {intervention.kind}")
        output[intervention.modality] = changed
        return output


def contribution_target(clean_loss, intervened_loss, temperature=1.):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    delta = (intervened_loss.detach() - clean_loss.detach()).clamp_min(0)
    positive = delta[delta > 0]
    scale = positive.median() if positive.numel() else delta.new_tensor(1.)
    normalized = delta / scale.clamp_min(1e-6)
    target = torch.exp(-normalized / temperature)
    weight = (1 - target).detach()
    return target.detach(), weight


def contribution_loss(gate, target, weight):
    gate = gate.squeeze(-1)
    loss = F.smooth_l1_loss(gate, target, reduction="none")
    return (loss * weight).sum() / weight.sum().clamp_min(1.)


def untouched_stability_loss(clean_gates, intervened_gates, target_modality):
    losses = [F.smooth_l1_loss(intervened_gates[name], clean_gates[name].detach())
              for name in clean_gates if name != target_modality]
    if not losses:
        raise ValueError("stability loss requires at least one untouched modality")
    return torch.stack(losses).mean()
