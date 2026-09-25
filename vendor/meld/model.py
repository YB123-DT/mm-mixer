"""Three controlled MELD replays over the frozen 68.27 RawAux/S15M trainer."""

from __future__ import annotations

import contextvars
import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
_spec = importlib.util.spec_from_file_location("_meld_threeway_base", BASE / "multiattn.py")
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
assert _spec.loader is not None
_spec.loader.exec_module(_base)

for _name in dir(_base):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_base, _name))


class ResidualChannel:
    def __init__(self):
        self._slot = contextvars.ContextVar(f"meld_pair_residual_{id(self)}", default=None)

    def __deepcopy__(self, memo):
        clone = type(self)()
        memo[id(self)] = clone
        return clone

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self.__init__()

    def publish(self, value):
        self._slot.set(value)

    def consume(self):
        value = self._slot.get()
        if value is None:
            raise RuntimeError("pairwise residual was not published or was already consumed")
        self._slot.set(None)
        return value


class PairwiseCross(nn.Module):
    """Exact HO_WO_TAV parameterization: VA/VT/AT active, TAV slot zeroed."""

    interaction_names = ("va", "vt", "at")

    def __init__(self, dim=256, rank=32):
        super().__init__()
        self.dim, self.rank = int(dim), int(rank)
        self.pair_projections = nn.ModuleList(nn.Linear(dim, rank) for _ in range(3))
        # Kept to match the IEMOCAP HO_WO_TAV module parameterization exactly.
        self.triple_projections = nn.ModuleList(nn.Linear(dim, rank) for _ in range(3))
        self.output = nn.Linear(4 * rank, dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, tokens):
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (3, self.dim):
            raise ValueError(f"expected [batch, 3, {self.dim}] modality tokens")
        projected = [layer(tokens[:, i]) for i, layer in enumerate(self.pair_projections)]
        va, vt, at = projected[0] * projected[1], projected[0] * projected[2], projected[1] * projected[2]
        tav_disabled = torch.zeros_like(va)
        return self.output(torch.cat((va, vt, at, tav_disabled), dim=-1))


class NoAdaptiveFusion(nn.Module):
    def forward(self, modality_features, bias=None):
        del bias
        batch = modality_features[0].shape[0]
        context = torch.zeros_like(modality_features[0])
        weights = modality_features[0].new_full(
            (batch, len(modality_features)), 1.0 / len(modality_features)
        )
        return context, weights


class EncoderPublisher(nn.Module):
    def __init__(self, encoder, channel, rank=32, cross=None):
        super().__init__()
        self.encoder = encoder
        self.cross = PairwiseCross(256, rank) if cross is None else cross
        object.__setattr__(self, "_channel", channel)

    def forward(self, inputs):
        grouped = self.encoder(inputs)
        self._channel.publish(self.cross(grouped))
        return grouped


class MixerBlock(nn.Module):
    def __init__(self, k=6, dim=256, hidden=1536, reverse=False):
        super().__init__()
        self.reverse = bool(reverse)
        self.norm_sub = nn.LayerNorm(dim)
        self.subspace_mlp = nn.Sequential(nn.Linear(k, 2 * k), nn.GELU(), nn.Linear(2 * k, k))
        self.norm_modal = nn.LayerNorm(dim)
        self.route = nn.Linear(3, 3, bias=False)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.use_sub = True
        self.use_mod = True
        self.use_ffn = True

    def _subspace(self, value):
        mixed = self.subspace_mlp(self.norm_sub(value).transpose(-1, -2)).transpose(-1, -2)
        return value + mixed

    def _modality(self, value):
        mixed = self.norm_modal(value).permute(0, 2, 3, 1)
        mixed = self.route(mixed).permute(0, 3, 1, 2)
        return value + mixed

    def forward(self, value):
        if self.reverse:
            if self.use_mod:
                value = self._modality(value)
            if self.use_sub:
                value = self._subspace(value)
        else:
            if self.use_sub:
                value = self._subspace(value)
            if self.use_mod:
                value = self._modality(value)
        return value + self.ffn(self.norm_ffn(value)) if self.use_ffn else value


class FactorizedMixerEncoder(nn.Module):
    def __init__(self, channel, dropout=0.2, layers=2, hidden=1536, rank=32):
        super().__init__()
        del dropout  # M4's validated Mixer block contains no internal dropout.
        self.token_count, self.token_dim = 6, 256
        self.split = nn.Linear(256, self.token_count * self.token_dim)
        self.blocks = nn.ModuleList(
            MixerBlock(self.token_count, self.token_dim, hidden, reverse=index % 2 == 1)
            for index in range(layers)
        )
        self.output_projection = nn.Identity()
        self.cross = PairwiseCross(256, rank)
        object.__setattr__(self, "_channel", channel)

    def forward(self, inputs):
        batch, modalities, _ = inputs.shape
        value = self.split(inputs).reshape(batch, modalities, self.token_count, self.token_dim)
        for block in self.blocks:
            value = block(value)
        grouped = self.output_projection(value.mean(dim=2))
        self._channel.publish(self.cross(grouped))
        return grouped


class FactorizedMixerOnlyEncoder(nn.Module):
    def __init__(self, dropout=0.2, layers=2, hidden=1536):
        super().__init__()
        del dropout
        self.token_count, self.token_dim = 6, 256
        self.split = nn.Linear(256, self.token_count * self.token_dim)
        self.blocks = nn.ModuleList(
            MixerBlock(self.token_count, self.token_dim, hidden, reverse=index % 2 == 1)
            for index in range(layers)
        )
        self.output_projection = nn.Identity()

    def forward(self, inputs):
        batch, modalities, _ = inputs.shape
        value = self.split(inputs).reshape(
            batch, modalities, self.token_count, self.token_dim
        )
        for block in self.blocks:
            value = block(value)
        return self.output_projection(value.mean(dim=2))


class ResidualIntegrator(nn.Module):
    def __init__(self, integrator, channel, scale=1.0):
        super().__init__()
        self.base_integrator = integrator
        self.scale = float(scale)
        object.__setattr__(self, "_channel", channel)

    def forward(self, pooled):
        return self.base_integrator(pooled) + self.scale * self._channel.consume()


def _install_pairwise(model, mixer):
    channel = ResidualChannel()
    if mixer:
        model.transformer_encoder = FactorizedMixerEncoder(channel, dropout=0.2, layers=2, hidden=1536, rank=32)
        model.experiment_lr_multiplier = 2.0
        model.capacity_variant = "M4_K6_D256_L2_H1536_PAIR_WO_TAV"
    else:
        model.transformer_encoder = EncoderPublisher(model.transformer_encoder, channel, rank=32)
        model.experiment_lr_multiplier = 1.0
        model.capacity_variant = "S15M_PAIR_WO_TAV"
    model.feature_integrator = ResidualIntegrator(model.feature_integrator, channel, scale=1.0)
    return model


def _install_mixer_only(model):
    model.transformer_encoder = FactorizedMixerOnlyEncoder(
        dropout=0.2, layers=2, hidden=1536
    )
    model.experiment_lr_multiplier = 2.0
    model.capacity_variant = "M4_K6_D256_L2_H1536_NO_PAIR"
    return model


def _remove_mixer_preserving_pairwise(model):
    """Replace only the Axis-wise Mixer with an identity token path."""
    encoder = model.transformer_encoder
    if not isinstance(encoder, FactorizedMixerEncoder):
        raise TypeError("no-mixer ablation requires the full factorized Mixer")
    model.transformer_encoder = EncoderPublisher(
        nn.Identity(), encoder._channel, rank=encoder.cross.rank, cross=encoder.cross
    )
    model.capacity_variant = "M4_PAIR_NO_MIXER"
    return model


class HierarchicalAttentionFusion(_base.HierarchicalAttentionFusion):
    def __init__(self, *args, **kwargs):
        requested = kwargs.get("capacity_variant")
        controlled = {
            "BASELINE",
            "BASE_PAIR",
            "M4_PAIR",
            "M4_PAIR_NO_MIXER",
            "M4_PAIR_NO_ADAPTIVE",
            "M4_PAIR_NO_CA",
            "M4_NO_PAIR",
            "M4_PAIR_NO_AUX",
        }
        if requested not in controlled:
            super().__init__(*args, **kwargs)
            return
        kwargs["capacity_variant"] = "S15M"
        kwargs["information_gate_enabled"] = False
        super().__init__(*args, **kwargs)
        self.experiment_lr_multiplier = 1.0
        if requested == "BASE_PAIR":
            rng_state = torch.random.get_rng_state()
            _install_pairwise(self, mixer=False)
            torch.random.set_rng_state(rng_state)
        elif requested in {
            "M4_PAIR",
            "M4_PAIR_NO_MIXER",
            "M4_PAIR_NO_ADAPTIVE",
            "M4_PAIR_NO_CA",
            "M4_PAIR_NO_AUX",
        }:
            rng_state = torch.random.get_rng_state()
            _install_pairwise(self, mixer=True)
            torch.random.set_rng_state(rng_state)
            if requested == "M4_PAIR_NO_MIXER":
                _remove_mixer_preserving_pairwise(self)
            elif requested == "M4_PAIR_NO_ADAPTIVE":
                self.adaptive_fusion = NoAdaptiveFusion()
                self.disable_gates()
                self.capacity_variant = requested
            elif requested == "M4_PAIR_NO_CA":
                self.disable_channel_attention()
                self.capacity_variant = requested
            elif requested == "M4_PAIR_NO_AUX":
                self.capacity_variant = requested
        elif requested == "M4_NO_PAIR":
            rng_state = torch.random.get_rng_state()
            _install_mixer_only(self)
            torch.random.set_rng_state(rng_state)
        else:
            self.capacity_variant = "S15M_BASELINE_REPLAY"


def build_model(experiment_id, dropout=0.2):
    return HierarchicalAttentionFusion(
        embed_dims={"v": 342, "a": 1024, "t": 1024},
        num_classes=7,
        modalities=["v", "a", "t"],
        fusion_dim=256,
        num_transformer_layers=2,
        num_heads=8,
        dropout=dropout,
        modality_importance={"t": 0.65, "a": 0.2, "v": 0.15},
        use_moe=False,
        use_contrastive=False,
        aux_from_raw_projected=True,
        layerwise_variant="a0",
        capacity_variant=experiment_id,
        information_gate_enabled=False,
    )
