from __future__ import annotations
import argparse, contextlib, dataclasses, json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import calibration.runner as shared
from peer_residual.runner import _formal_module, _sha
from .model import build_factorized_mixer_model, optimizer_parameter_groups
from .registry import FINAL_TUNE_VARIANTS, MIXER_VARIANTS
from structured_cross.model import CSS_AV_INPUT_DIMS, DEFAULT_INPUT_DIMS

IEMOCAP = "/data2/yb/multimodalERC/IEMOCAP"
RAWAUX_FILL53_FEATURES = (
    f"{IEMOCAP}/Model_rawaux_textpeak_cssv_v1/artifacts/"
    "iemocap_textpeak_audio_cssv.npz"
)
CSS_AV_CURRENT_TEXT_FEATURES = (
    f"{IEMOCAP}/Model_rawaux_css_av_current_text/artifacts/css_av_current_text.npz"
)
EFFECTIVE_TRAINING_KEYS = (
    "main_loss_weight", "aux_loss_weights", "poly_alpha", "poly_gamma",
    "focal_reweight_gamma", "label_smoothing", "lr_fusion",
)


def feature_spec_for_experiment(experiment_id: str) -> dict[str, object]:
    """Return the explicit input contract for an experiment variant."""
    if experiment_id == "HO_WO_TAV_CSSAV":
        return {
            "protocol": "css_official_av_current_text",
            "input_dims": dict(CSS_AV_INPUT_DIMS),
        }
    return {
        "protocol": "textpeak_audio_cssv",
        "input_dims": dict(DEFAULT_INPUT_DIMS),
    }

@dataclass(frozen=True)
class MixerRunConfig:
    experiment_id: str = "M1"
    seed: int = 2025
    epochs: int = 100
    batch_size: int = 32
    gradient_accumulation: int = 1
    scheduler_mode: str = "legacy"
    baseline_config: str = f"{IEMOCAP}/Model_rawaux_clean/configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
    pkl: str = f"{IEMOCAP}/external/CSS/data/iemocap_multimodal_features.pkl"
    features: str = RAWAUX_FILL53_FEATURES
    feature_protocol: str | None = None
    tuning_overrides: Mapping[str, object] | None = None
    run_id: str | None = None
    def __post_init__(self):
        if self.experiment_id not in MIXER_VARIANTS and not self.tuning_overrides:
            raise ValueError(f"unknown mixer experiment {self.experiment_id}")
        corrected_modes = {"corrected_cosine", "corrected_updates20"}
        if self.scheduler_mode not in {"legacy", *corrected_modes}: raise ValueError(f"unknown scheduler mode {self.scheduler_mode}")
        if self.scheduler_mode in corrected_modes and self.gradient_accumulation != 2:
            raise ValueError(f"{self.scheduler_mode} requires gradient_accumulation=2 to match the pinned training loop")
        expected_protocol = feature_spec_for_experiment(self.experiment_id)["protocol"]
        if self.feature_protocol is None:
            object.__setattr__(self, "feature_protocol", expected_protocol)
        elif self.feature_protocol != expected_protocol:
            raise ValueError(
                f"{self.experiment_id} requires feature_protocol={expected_protocol}"
            )
        if (self.experiment_id == "HO_WO_TAV_CSSAV"
                and self.features == RAWAUX_FILL53_FEATURES):
            object.__setattr__(self, "features", CSS_AV_CURRENT_TEXT_FEATURES)
        if self.tuning_overrides is not None:
            overrides = dict(self.tuning_overrides)
            if not overrides:
                raise ValueError("tuning_overrides must not be empty")
            object.__setattr__(self, "tuning_overrides", overrides)
        if self.run_id is not None:
            if not self.run_id or Path(self.run_id).name != self.run_id:
                raise ValueError("run_id must be a non-empty path-free name")
    def to_dict(self):
        value = dataclasses.asdict(self)
        return {key: item for key, item in value.items() if item is not None}


def normalize_config(config: MixerRunConfig | Mapping[str, object]) -> dict[str, object]:
    """Materialize default fields before comparing persisted queue configs."""
    value = config if isinstance(config, MixerRunConfig) else MixerRunConfig(**dict(config))
    return value.to_dict()


def validate_ho_wo_tav_config(config: MixerRunConfig | Mapping[str, object]) -> MixerRunConfig:
    """Reject a paired baseline unless it is the exact normalized Full config."""
    value = config if isinstance(config, MixerRunConfig) else MixerRunConfig(**dict(config))
    expected = MixerRunConfig(experiment_id="HO_WO_TAV", seed=value.seed)
    if value.to_dict() != expected.to_dict():
        raise ValueError("Full config does not normalize to the exact HO_WO_TAV baseline")
    return value


def effective_training_manifest(config: MixerRunConfig) -> dict[str, object]:
    baseline = Path(config.baseline_config).resolve()
    fixed = json.loads(baseline.read_text())["fixed_params"]
    return {
        "baseline_config": str(baseline),
        "baseline_config_sha256": _sha(baseline),
        "effective_training_params": {
            key: fixed[key] for key in EFFECTIVE_TRAINING_KEYS
        },
        "scheduler_mode": config.scheduler_mode,
        "gradient_accumulation": config.gradient_accumulation,
    }


@contextlib.contextmanager
def _temporary_tuning_variant(config: MixerRunConfig):
    """Expose an R4 union to the existing Task 1 model builder for one run."""
    overrides = config.tuning_overrides
    if overrides is None:
        yield
        return
    existing = FINAL_TUNE_VARIANTS.get(config.experiment_id)
    if existing is not None:
        if dict(existing) != dict(overrides):
            raise ValueError("registered tuning variant conflicts with config overrides")
        yield
        return
    if not config.experiment_id.startswith("TUNE_R4_"):
        raise ValueError("only R4 may materialize an unregistered combined tuning variant")
    FINAL_TUNE_VARIANTS[config.experiment_id] = dict(overrides)
    MIXER_VARIANTS[config.experiment_id] = {}
    try:
        yield
    finally:
        FINAL_TUNE_VARIANTS.pop(config.experiment_id, None)
        MIXER_VARIANTS.pop(config.experiment_id, None)

def execute(config, output_root):
    spec = feature_spec_for_experiment(config.experiment_id)
    previous_builder = shared.build_calibration_model
    previous_groups = shared.optimizer_parameter_groups
    previous_protocol = shared.PROTOCOL
    previous_manifest_builder = shared.build_run_manifest
    shared.build_calibration_model = build_factorized_mixer_model
    shared.optimizer_parameter_groups = optimizer_parameter_groups
    shared.PROTOCOL = "P3_FACTORIZED_MIXER_PEAK_TEST_DIAGNOSTIC"
    previous_scheduler_mode = shared.SCHEDULER_MODE
    previous_input_dims = shared.INPUT_DIMS
    previous_feature_protocol = shared.FEATURE_PROTOCOL
    shared.SCHEDULER_MODE = config.scheduler_mode
    shared.INPUT_DIMS = dict(spec["input_dims"])
    shared.FEATURE_PROTOCOL = str(spec["protocol"])
    shared.build_run_manifest = lambda **kwargs: {
        **previous_manifest_builder(**kwargs),
        **effective_training_manifest(config),
    }
    try:
        with _temporary_tuning_variant(config):
            return shared.execute(config, output_root)
    finally:
        shared.build_calibration_model = previous_builder
        shared.optimizer_parameter_groups = previous_groups
        shared.PROTOCOL = previous_protocol
        shared.SCHEDULER_MODE = previous_scheduler_mode
        shared.INPUT_DIMS = previous_input_dims
        shared.FEATURE_PROTOCOL = previous_feature_protocol
        shared.build_run_manifest = previous_manifest_builder

def generate_configs(destination):
    formal = _formal_module(); destination.mkdir(parents=True, exist_ok=True)
    for eid in MIXER_VARIANTS:
        batch_size = 64 if eid.startswith("M4_BS64_") else 32
        kwargs = {"batch_size": batch_size}
        if eid == "HO_WO_TAV_CSSAV":
            kwargs["features"] = CSS_AV_CURRENT_TEXT_FEATURES
        formal.atomic_json(destination / f"{eid}.json", MixerRunConfig(eid, **kwargs).to_dict())

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--output-root"); parser.add_argument("--generate-configs"); args=parser.parse_args()
    if args.generate_configs: generate_configs(Path(args.generate_configs)); return
    config=MixerRunConfig(**json.loads(Path(args.config).read_text()))
    print(json.dumps(execute(config, Path(args.output_root)), indent=2), flush=True)
if __name__ == "__main__": main()
