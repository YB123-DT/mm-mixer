from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType


IEMOCAP = "/data2/yb/multimodalERC/IEMOCAP"


@dataclass(frozen=True)
class PeerVariant:
    experiment_id: str
    reference: str = "learned_peer"
    residual: bool = True
    utility_gate: bool = True
    channel_gate: bool = True
    aux_sufficiency: bool = False
    stability: str = "none"


def _variant(experiment_id: str, **changes) -> PeerVariant:
    return dataclasses.replace(PeerVariant(experiment_id), **changes)


VARIANTS = MappingProxyType({
    "V1": _variant("V1", residual=False),
    "V2": _variant("V2"),
    "V3": _variant("V3", aux_sufficiency=True),
    "V4": _variant("V4", utility_gate=False),
    "V5": _variant("V5", channel_gate=False),
    "V6": _variant("V6", reference="fixed_peer"),
    "V7": _variant("V7", reference="global"),
    "V8": _variant("V8", stability="uniform_peer_init"),
    "V9": _variant("V9", stability="scheduled_transition"),
    "V10": _variant("V10", stability="learned_transition"),
    "V11": _variant("V11", stability="rms_match"),
    "V12": _variant("V12", stability="bounded_channel"),
})


@dataclass(frozen=True)
class PeerRunConfig:
    experiment_id: str
    protocol: str = "STRICT_PEAK_TEST_DIAGNOSTIC_PEER_RESIDUAL"
    seed: int = 2025
    epochs: int = 100
    batch_size: int = 32
    batch_protocol: str = "utterance"
    gradient_accumulation: int = 1
    baseline_config: str = f"{IEMOCAP}/Model_rawaux_clean/configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
    pkl: str = f"{IEMOCAP}/external/CSS/data/iemocap_multimodal_features.pkl"
    features: str = f"{IEMOCAP}/Model_rawaux_textpeak_v27_fill53/artifacts/fill53_features.npz"

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


def validate_run_config(config: PeerRunConfig) -> None:
    if config.experiment_id not in VARIANTS:
        raise ValueError(f"unknown peer-residual experiment: {config.experiment_id}")
    expected = PeerRunConfig(config.experiment_id)
    for field in dataclasses.fields(expected):
        if getattr(config, field.name) != getattr(expected, field.name):
            raise ValueError(f"{field.name} differs from the frozen screening protocol")
