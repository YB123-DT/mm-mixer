from __future__ import annotations

import torch
import torch.nn.functional as F


def role_target(logits, labels, temperature=1.):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probability = torch.softmax(logits.detach(), dim=-1)
    true_probability = probability.gather(1, labels[:, None]).squeeze(1)
    mask = F.one_hot(labels, probability.size(-1)).bool()
    competitor = probability.masked_fill(mask, -1).max(dim=-1).values
    signed_margin = true_probability - competitor
    target = torch.sigmoid(signed_margin / temperature).detach()
    weight = (2 * (target - .5).abs()).detach()
    return target, weight


def role_supervision_loss(gate, target, weight):
    per_example = F.smooth_l1_loss(gate.squeeze(-1), target, reduction="none")
    return (per_example * weight).sum() / weight.sum().clamp_min(1.)
