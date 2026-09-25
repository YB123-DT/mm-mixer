from __future__ import annotations

import dataclasses
from dataclasses import dataclass


IEMOCAP = "/data2/yb/multimodalERC/IEMOCAP"


@dataclass(frozen=True)
class CalibrationVariant:
    experiment_id: str
    family: str
    contribution: bool = False
    stability: bool = False
    intervention_classification: bool = False
    role_supervision: bool = False
    replay: bool = False
    auxiliary_supervision: bool = True
    staged: bool = False
    lambda_contribution: float = .1
    lambda_stability: float = .05
    lambda_role: float = .1
    target_temperature: float = 1.

    def __post_init__(self):
        if self.family not in {"isec", "saboteur_replay"}:
            raise ValueError(f"unknown calibration family: {self.family}")
        if self.target_temperature <= 0:
            raise ValueError("target_temperature must be positive")


VARIANTS = {
    "A1": CalibrationVariant("A1", "isec", contribution=True, stability=True),
    "A2": CalibrationVariant("A2", "isec", stability=True),
    "A3": CalibrationVariant("A3", "isec", contribution=True),
    "A4": CalibrationVariant("A4", "isec", intervention_classification=True),
    "B1": CalibrationVariant("B1", "saboteur_replay", role_supervision=True),
    "B2": CalibrationVariant("B2", "saboteur_replay", replay=True),
    "B3": CalibrationVariant("B3", "saboteur_replay", role_supervision=True, replay=True),
    "B4": CalibrationVariant(
        "B4", "saboteur_replay", replay=True, auxiliary_supervision=False
    ),
    "A5": CalibrationVariant(
        "A5", "isec", contribution=True, stability=True, staged=True
    ),
    "B5": CalibrationVariant(
        "B5", "saboteur_replay", role_supervision=True, replay=True, staged=True
    ),
}


def get_variant(experiment_id: str) -> CalibrationVariant:
    try:
        return VARIANTS[experiment_id]
    except KeyError as error:
        raise ValueError(f"unknown calibration experiment: {experiment_id}") from error


@dataclass(frozen=True)
class CalibrationRunConfig:
    experiment_id: str
    seed: int = 2025
    epochs: int = 100
    batch_size: int = 32
    gradient_accumulation: int = 1
    baseline_config: str = f"{IEMOCAP}/Model_rawaux_clean/configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
    pkl: str = f"{IEMOCAP}/external/CSS/data/iemocap_multimodal_features.pkl"
    features: str = f"{IEMOCAP}/Model_rawaux_textpeak_v27_fill53/artifacts/fill53_features.npz"

    def __post_init__(self):
        get_variant(self.experiment_id)

    def to_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**value)
