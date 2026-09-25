from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LayerwiseVariant:
    name: str
    routes: Tuple[str, str, str]
    use_individual: bool
    use_marginal: bool
    alpha: float = 0.3
    gamma: float = 0.1
    margin: float = 0.0
    tau: float = 0.1


def resolve_layerwise_variant(variant: str) -> LayerwiseVariant:
    name = str(variant).lower()
    presets = {
        "a0": LayerwiseVariant("a0", ("late", "late", "late"), False, False),
        "a1": LayerwiseVariant("a1", ("late", "late", "late"), True, False),
        "a2": LayerwiseVariant("a2", ("early", "middle", "late"), True, False),
        "a3": LayerwiseVariant("a3", ("early", "middle", "late"), True, True),
    }
    if name not in presets:
        raise ValueError(f"head_stage2_variant must be one of A0-A3, got {variant!r}")
    return presets[name]


def negative_marginal_penalty(
    head_logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.0,
    tau: float = 0.1,
    head_index: int = None,
) -> torch.Tensor:
    if head_logits.ndim != 3 or head_logits.size(0) != 3:
        raise ValueError("head_logits must have shape [3, batch, classes]")
    if tau <= 0:
        raise ValueError("tau must be positive")
    indices = range(3) if head_index is None else (head_index,)
    penalties = []
    for current in indices:
        if current not in (0, 1, 2):
            raise ValueError("head_index must be 0, 1, or 2")
        peers = torch.stack(
            [head_logits[index].detach() for index in range(3) if index != current]
        ).mean(dim=0)
        student = (head_logits[current] + 2.0 * peers) / 3.0
        student_ce = F.cross_entropy(student, targets, reduction="none")
        baseline = F.cross_entropy(peers, targets, reduction="none")
        delta = baseline.detach() - student_ce
        penalties.append(tau * F.softplus((margin - delta) / tau))
    return torch.stack(penalties).mean()


def layerwise_marginal_diagnostics(
    head_logits: torch.Tensor,
    targets: torch.Tensor,
    head_names=("evidence", "interaction", "global"),
) -> dict:
    if head_logits.ndim != 3 or head_logits.size(0) != 3:
        raise ValueError("head_logits must have shape [3, batch, classes]")
    if len(head_names) != 3:
        raise ValueError("head_names must contain exactly three names")
    per_head = []
    for current, name in enumerate(head_names):
        peers = torch.stack(
            [head_logits[index].detach() for index in range(3) if index != current]
        ).mean(dim=0)
        student = (head_logits[current] + 2.0 * peers) / 3.0
        baseline_ce = F.cross_entropy(peers, targets, reduction="none")
        student_ce = F.cross_entropy(student, targets, reduction="none")
        delta = baseline_ce - student_ce
        per_head.append(
            {
                "name": str(name),
                "mean_delta": float(delta.mean().item()),
                "positive_delta_rate": float(delta.gt(0).float().mean().item()),
            }
        )
    return {"per_head": per_head}


def compose_layerwise_loss(
    head_logits: torch.Tensor,
    ensemble_logits: torch.Tensor,
    targets: torch.Tensor,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    variant: str,
    alpha: float = None,
    gamma: float = None,
    margin: float = None,
    tau: float = None,
    ensemble_loss: torch.Tensor = None,
) -> torch.Tensor:
    preset = resolve_layerwise_variant(variant)
    alpha = preset.alpha if alpha is None else float(alpha)
    gamma = preset.gamma if gamma is None else float(gamma)
    margin = preset.margin if margin is None else float(margin)
    tau = preset.tau if tau is None else float(tau)
    loss = criterion(ensemble_logits, targets) if ensemble_loss is None else ensemble_loss
    if preset.use_individual:
        individual = torch.stack([criterion(logits, targets) for logits in head_logits]).mean()
        loss = loss + alpha * individual
    if preset.use_marginal:
        loss = loss + gamma * negative_marginal_penalty(
            head_logits, targets, margin=margin, tau=tau
        )
    return loss
