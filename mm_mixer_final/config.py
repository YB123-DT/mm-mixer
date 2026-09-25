from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any

from .modalities import MODALITY_VARIANTS
from .structural_ablations import STRUCTURAL_ABLATION_VARIANTS


DATASETS = ("iemocap", "meld")
VARIANTS = (
    "full",
    "no_mixer",
    "no_pairwise",
    "no_adaptive_gating",
    "no_cross_attention",
    "no_auxiliary_loss",
)
RUN_VARIANTS = VARIANTS + MODALITY_VARIANTS + STRUCTURAL_ABLATION_VARIANTS
_SWITCHES = (
    "no_mixer",
    "no_pairwise",
    "no_adaptive_gating",
    "no_cross_attention",
    "no_auxiliary_loss",
    *STRUCTURAL_ABLATION_VARIANTS,
)


@dataclass(frozen=True)
class FinalConfig:
    dataset: str
    variant: str
    seed: int
    seeds: tuple[int, ...]
    class_names: tuple[str, ...]
    input_dims: dict[str, int]
    mixer: dict[str, int]
    learning_rates: dict[str, float]
    loss: dict[str, Any]
    switches: dict[str, bool]
    epochs: int
    batch_size: int
    feature_paths: dict[str, Any]
    source_root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_IEMOCAP_ROOT = Path("/data2/yb/multimodalERC/IEMOCAP")
_MELD_ROOT = Path("/data2/yb/multimodalERC/MELD")
_MELD_LR = 2.0832826726482106e-5

_IEMOCAP = FinalConfig(
    dataset="iemocap",
    variant="full",
    seed=2025,
    seeds=(2025, 2066, 2088, 2118),
    class_names=("happiness", "sadness", "neutral", "anger", "excited", "frustration"),
    input_dims={"t": 1024, "a": 1024, "v": 342},
    mixer={"blocks": 2, "tokens": 6, "dim": 256, "ffn": 1536},
    learning_rates={
        "projection": 3e-5,
        "pairwise_cross": 1.2e-4,
        "mixer": 6e-5,
        "cross_attention": 6e-5,
        "remaining": 6e-5,
        "classifier": 1.2e-4,
    },
    loss={
        "kind": "iemocap_batch_focal_detached",
        "weights": {"main": 0.45, "t": 0.33, "a": 0.11, "v": 0.11},
        "main_class_weight": True,
        "aux_class_weight": True,
        "focal_gamma": 2.5,
        "focal_probability_detached": True,
    },
    switches={key: False for key in _SWITCHES},
    epochs=100,
    batch_size=32,
    feature_paths={
        "packed": str(
            _IEMOCAP_ROOT
            / "Model_rawaux_textpeak_cssv_v1/artifacts/"
              "iemocap_textpeak_audio_cssv.npz"
        ),
        "metadata": str(_IEMOCAP_ROOT / "external/CSS/data/iemocap_multimodal_features.pkl"),
    },
    source_root="vendor/iemocap",
)

_MELD = FinalConfig(
    dataset="meld",
    variant="full",
    seed=2025,
    seeds=(2025, 2028, 2069, 2101),
    class_names=("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"),
    input_dims={"t": 1024, "a": 1024, "v": 342},
    mixer={"blocks": 2, "tokens": 6, "dim": 256, "ffn": 1536},
    learning_rates={
        "projection": _MELD_LR * 0.5,
        "mixer": _MELD_LR,
        "remaining": _MELD_LR,
        "classifier": _MELD_LR * 2.0,
    },
    loss={
        "kind": "meld_per_sample_focal_no_detach",
        "main_class_weight": False,
        "aux_class_weight": True,
        # Historical ``log_vars`` were never part of the optimizer and stayed
        # exactly zero.  Freeze the effective semantics explicitly as unit
        # task weights instead of advertising fake learnable uncertainty.
        "task_weighting": "fixed_unit",
        "fixed_task_weights": {
            "main": 1.0, "t": 1.0, "a": 1.0, "v": 1.0,
        },
        "focal_gamma": 2.5,
        "focal_probability_detached": False,
    },
    switches={key: False for key in _SWITCHES},
    epochs=50,
    batch_size=32,
    feature_paths={
        split: {
            "v": str(_MELD_ROOT / f"Model/features_denseface/{split}_features/visual_features.json"),
            "a": str(_MELD_ROOT / f"Dataset/Data/{split}_features/audio_features.json"),
            "t": str(_MELD_ROOT / f"Model/features_roberta_large_ft/{split}_features/text_features.json"),
        }
        for split in ("train", "dev", "test")
    },
    source_root="vendor/meld",
)


def get_config(dataset: str, variant: str, seed: int) -> FinalConfig:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset}")
    if variant not in RUN_VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    base = _IEMOCAP if dataset == "iemocap" else _MELD
    if seed not in base.seeds:
        raise ValueError(f"{seed} is not a formal seed for {dataset}: {base.seeds}")
    switches = dict(base.switches)
    if variant in _SWITCHES:
        switches[variant] = True
    return replace(base, variant=variant, seed=int(seed), switches=switches)


def config_contract_sha256(config: FinalConfig) -> str:
    payload = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()
