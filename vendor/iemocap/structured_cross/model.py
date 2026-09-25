from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


BASE_PATH = Path(__file__).resolve().parents[1] / "base/multiattn.py"
DEFAULT_INPUT_DIMS = {"v": 342, "a": 1024, "t": 1024}
CSS_AV_INPUT_DIMS = {"v": 342, "a": 1582, "t": 1024}


def _load_base():
    spec = importlib.util.spec_from_file_location("rawaux_structured_cross_base", BASE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class IdentityInformationGate(nn.Module):
    def forward(self, x, context):
        return x


class StructuredHighOrderCross(nn.Module):
    """Low-rank pairwise and three-way products over modality-preserving tokens."""

    INTERACTION_NAMES = ("av", "tv", "ta", "tav")

    def __init__(self, dim=256, rank=64, active_interactions=None):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.active_interactions = frozenset(
            self.INTERACTION_NAMES if active_interactions is None else active_interactions
        )
        unknown = self.active_interactions.difference(self.INTERACTION_NAMES)
        if unknown:
            raise ValueError(f"unknown structured interactions: {sorted(unknown)}")
        self.pair_projections = nn.ModuleList(nn.Linear(dim, rank) for _ in range(3))
        self.triple_projections = nn.ModuleList(nn.Linear(dim, rank) for _ in range(3))
        self.output = nn.Linear(4 * rank, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _interactions(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (3, self.dim):
            raise ValueError(f"expected exactly three modality tokens of width {self.dim}")
        pair = [projection(tokens[:, index]) for index, projection in enumerate(self.pair_projections)]
        triple = [projection(tokens[:, index]) for index, projection in enumerate(self.triple_projections)]
        interactions = [
            pair[0] * pair[1], pair[0] * pair[2], pair[1] * pair[2],
            triple[0] * triple[1] * triple[2],
        ]
        return [
            value if name in self.active_interactions else torch.zeros_like(value)
            for name, value in zip(self.INTERACTION_NAMES, interactions)
        ]

    def decomposed(self, tokens):
        interactions = self._interactions(tokens)
        weights = self.output.weight.split(self.rank, dim=1)
        # Tokens follow base.modalities == ["v", "a", "t"].
        values = {name: F.linear(value, weight) for name, value, weight in zip(
            ("av", "tv", "ta", "tav"), interactions, weights,
        )}
        values["bias"] = self.output.bias.unsqueeze(0).expand(tokens.shape[0], -1)
        return values

    def forward(self, tokens):
        return self.output(torch.cat(self._interactions(tokens), dim=-1))


class _ForwardContext:
    residual = None
    components = None
    base_fused = None
    grouped = None
    raw_aux = None


class K6TokenEncoderWithCross(nn.Module):
    def __init__(self, dropout, context, rank=64, cross_input_source="grouped"):
        super().__init__()
        self.token_count = 6
        self.token_dim = 256
        self.split = nn.Linear(256, self.token_count * self.token_dim)
        layer = nn.TransformerEncoderLayer(
            self.token_dim, 8, 1024, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, 2)
        self.output_adapter = nn.Identity()
        self.cross = StructuredHighOrderCross(256, rank)
        self.cross_input_source = cross_input_source
        object.__setattr__(self, "_forward_context", context)

    def forward(self, x):
        batch, modalities, _ = x.shape
        tokens = self.split(x).reshape(batch, modalities * self.token_count, self.token_dim)
        encoded = self.transformer(tokens)
        grouped = encoded.reshape(batch, modalities, self.token_count, self.token_dim).mean(dim=2)
        grouped = self.output_adapter(grouped)
        self._forward_context.grouped = grouped
        cross_input = grouped
        if self.cross_input_source == "raw_aux":
            cross_input = self._forward_context.raw_aux
            if cross_input is None:
                raise RuntimeError("raw auxiliary features were not captured before token encoding")
        self._forward_context.residual = self.cross(cross_input)
        self._forward_context.components = self.cross.decomposed(cross_input)
        return grouped


class CrossResidualIntegrator(nn.Module):
    def __init__(self, base_integrator, context, residual_scale=1., branch_dropout=0.):
        super().__init__()
        self.base_integrator = base_integrator
        self.residual_scale = float(residual_scale)
        self.branch_dropout = float(branch_dropout)
        object.__setattr__(self, "_forward_context", context)

    def forward(self, pooled_flat):
        fused = self.base_integrator(pooled_flat)
        self._forward_context.base_fused = fused
        residual = self._forward_context.residual
        if residual is None:
            raise RuntimeError("structured cross integrator called before token encoder")
        self._forward_context.residual = None
        if self.training and self.branch_dropout > 0:
            residual = torch.nn.functional.dropout(residual, p=self.branch_dropout, training=True)
        return fused + self.residual_scale * residual


def build_structured_cross_model(experiment_id="X1", dropout=.1, input_dims=None):
    variants = {
        "CTRL_K6D256L2": (32, 0., 2.),
        "X1": (64, 1., 1.),
        "X2": (32, .5, 2.), "X3": (32, 1., 2.),
        "X4": (64, .5, 2.), "X5": (64, 1., 2.),
        "X6": (128, .5, 2.), "X7": (128, 1., 2.),
        "X8": (16, 1., 2.), "X9": (24, 1., 2.),
        "X10": (40, 1., 2.), "X11": (48, 1., 2.),
        "X12": (32, .75, 2.), "X13": (32, 1.25, 2.),
        "X14": (32, 1., 1.), "X15": (32, 1., 3.),
        "X3BD05": (32, 1., 2., .05), "X3BD10": (32, 1., 2., .1),
    }
    if experiment_id not in variants:
        raise ValueError(f"unknown structured-cross experiment: {experiment_id}")
    vals = variants[experiment_id]; rank, residual_scale, cross_lr_multiplier = vals[:3]; branch_dropout = vals[3] if len(vals) > 3 else 0.
    resolved_input_dims = dict(DEFAULT_INPUT_DIMS if input_dims is None else input_dims)
    if set(resolved_input_dims) != {"v", "a", "t"}:
        raise ValueError("input_dims must provide exactly v, a, and t")

    base = _load_base()
    model = base.HierarchicalAttentionFusion(
        embed_dims=resolved_input_dims, num_classes=6,
        modalities=["v", "a", "t"], fusion_dim=256, num_transformer_layers=3,
        num_heads=8, dropout=dropout,
        modality_importance={"t": .65, "a": .2, "v": .15},
        use_moe=False, use_contrastive=False, aux_from_raw_projected=True,
        align_self_weight=.95,
    )
    model.disable_alignment()
    model.info_gates = nn.ModuleDict({m: IdentityInformationGate() for m in model.modalities})
    context = _ForwardContext()
    model.transformer_encoder = K6TokenEncoderWithCross(dropout, context, rank=rank)
    model.feature_integrator = CrossResidualIntegrator(model.feature_integrator, context, residual_scale, branch_dropout)
    model.capacity_variant = f"K6D256L2M_STRUCTURED_CROSS_R{rank}_S{residual_scale}"
    model.cross_lr_multiplier = cross_lr_multiplier
    model.input_dims = resolved_input_dims
    return model


def optimizer_parameter_groups(model, lr):
    cross_lr = lr * getattr(model, "cross_lr_multiplier", 1.) * getattr(
        model, "x3_lr_multiplier", 1.
    )
    ca_lr = lr * getattr(model, "ca_lr_multiplier", 1.)
    partitions = (("proj.", lr * .5),
                  ("transformer_encoder.cross.", cross_lr),
                  ("cross_attn.", ca_lr),
                  ("transformer_encoder.", lr), ("classifiers.", lr * 2))
    named = list(model.named_parameters()); groups = []; used = set()
    for prefix, rate in partitions:
        params = [p for name, p in named if name.startswith(prefix) and p.requires_grad and id(p) not in used]
        if params:
            groups.append({"params": params, "lr": rate}); used.update(id(p) for p in params)
    remaining = [p for _, p in named if p.requires_grad and id(p) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": lr})
    return groups
