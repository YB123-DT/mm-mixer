from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import calibration.runner as shared
from .model import build_structured_cross_model, optimizer_parameter_groups


IEMOCAP = "/data2/yb/multimodalERC/IEMOCAP"


@dataclass(frozen=True)
class StructuredCrossRunConfig:
    experiment_id: str = "X1"
    seed: int = 2025
    epochs: int = 100
    batch_size: int = 32
    gradient_accumulation: int = 1
    baseline_config: str = f"{IEMOCAP}/Model_rawaux_clean/configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
    pkl: str = f"{IEMOCAP}/external/CSS/data/iemocap_multimodal_features.pkl"
    features: str = f"{IEMOCAP}/Model_rawaux_textpeak_v27_fill53/artifacts/fill53_features.npz"

    def __post_init__(self):
        if self.experiment_id not in {"CTRL_K6D256L2", *(f"X{i}" for i in range(1, 16))}:
            raise ValueError(f"unknown structured-cross experiment: {self.experiment_id}")

    def to_dict(self):
        return dataclasses.asdict(self)


def execute(config, output_root):
    shared.build_calibration_model = build_structured_cross_model
    shared.optimizer_parameter_groups = optimizer_parameter_groups
    shared.PROTOCOL = "K6D256L2M_STRUCTURED_CROSS_PEAK_TEST_DIAGNOSTIC"
    return shared.execute(config, output_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config = StructuredCrossRunConfig(**json.loads(Path(args.config).read_text()))
    print(json.dumps(execute(config, Path(args.output_root)), indent=2), flush=True)


if __name__ == "__main__":
    main()
