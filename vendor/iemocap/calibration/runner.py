from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import time
import traceback
from pathlib import Path

import torch

from peer_residual.runner import _formal_module, _labels, _sha
from .model import build_calibration_model, optimizer_parameter_groups
from .registry import CalibrationRunConfig, VARIANTS


PROTOCOL = "S0_CALIBRATION_PEAK_TEST_DIAGNOSTIC"
SCHEDULER_MODE = "legacy"
INPUT_DIMS = {"v": 342, "a": 1024, "t": 1024}
FEATURE_PROTOCOL = "rawaux_fill53"


def build_loaders(config):
    from utterance_history.runner import HistoryConfig, build_loaders as history_loaders
    return history_loaders(HistoryConfig(
        "H0", pkl=config.pkl, features=config.features,
        baseline_config=config.baseline_config,
    ))


@contextlib.contextmanager
def calibration_train_module(experiment_id):
    formal = _formal_module(); original = formal.build_experiment_model
    formal.build_experiment_model = lambda _ignored, dropout: build_calibration_model(experiment_id, dropout)
    try:
        module = formal._patched_train_module(experiment_id, scheduler_mode=SCHEDULER_MODE)
        module.formal_optimizer_parameter_groups = optimizer_parameter_groups
        yield module
    finally:
        formal.build_experiment_model = original


def build_run_manifest(*, config, model, input_dims, selected, checkpoint_path: Path,
                       elapsed_seconds: float, train_samples: int = 5810,
                       test_samples: int = 1623):
    """Build the shared run manifest without hiding model-local controls."""
    return {
        "protocol": PROTOCOL,
        "experiment_id": config.experiment_id,
        "run_id": getattr(config, "run_id", None) or config.experiment_id,
        "seed": config.seed,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "feature_protocol": FEATURE_PROTOCOL,
        "input_dims": dict(input_dims),
        "feature_artifact_sha256": _sha(Path(config.features)),
        "tuning_overrides": dict(getattr(model, "tuning_overrides", {})),
        "weighted_f1": selected["weighted_f1"],
        "accuracy": selected["accuracy"],
        "macro_f1": selected["macro_f1"],
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "fresh_strict_replay_exact": True,
    }


def execute(config: CalibrationRunConfig, output_root: Path):
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils.class_weight import compute_class_weight

    run_id = getattr(config, "run_id", None) or config.experiment_id
    formal = _formal_module(); ledger = formal.RunLedger(output_root, run_id)
    with ledger.acquire(blocking=False):
        started = time.time(); ledger.status("running", protocol=PROTOCOL, started_at_unix=started)
        formal.atomic_json(ledger.run_dir / "config.json", config.to_dict())
        try:
            base_module = formal._load_base_module(); base_module.set_random_seed(config.seed)
            train_set, test_set, train_loader, test_loader = build_loaders(config)
            weights = compute_class_weight("balanced", classes=np.arange(6), y=np.asarray(_labels(train_set)))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = LabelEncoder(); encoder.classes_ = np.asarray(formal.CLASS_NAMES)
            fixed = formal._training_cfg(config)
            input_dims = dict(INPUT_DIMS)
            with calibration_train_module(config.experiment_id) as train_module:
                model, report, metrics = train_module.train_full_pipeline(
                    train_loader, test_loader, test_loader, device, encoder, ["v", "a", "t"],
                    input_dims,
                    torch.tensor(weights, dtype=torch.float32, device=device), fixed, config.epochs,
                )
            ledger.atomic_checkpoint(copy.deepcopy(model.state_dict()))
            selected, logits_a, predictions_a, labels_a = formal.replay_metrics(model, test_loader, device)
            dropout = float(fixed.get("fusion_dropout", fixed.get("dropout_rate", .15)))
            fresh = build_calibration_model(config.experiment_id, dropout).to(device)
            fresh.load_state_dict(torch.load(ledger.checkpoint_path, map_location=device, weights_only=True), strict=True)
            replay, logits_b, predictions_b, labels_b = formal.replay_metrics(fresh, test_loader, device)
            exact = all((torch.equal(logits_a, logits_b), torch.equal(predictions_a, predictions_b),
                         torch.equal(labels_a, labels_b)))
            if selected["samples"] != 1623 or not exact:
                raise RuntimeError("fresh calibration checkpoint replay failed")
            final = {**metrics, "selected": selected, "fresh_strict_replay": replay,
                     "fresh_strict_replay_exact": True}
            formal.atomic_json(ledger.run_dir / "metrics.json", final)
            formal.atomic_text(ledger.run_dir / "classification_report.txt", report + "\n")
            manifest = build_run_manifest(
                config=config, model=model, input_dims=input_dims, selected=selected,
                checkpoint_path=ledger.checkpoint_path, elapsed_seconds=time.time() - started,
                train_samples=len(train_set), test_samples=len(test_set),
            )
            formal.atomic_json(ledger.run_dir / "manifest.json", manifest)
            ledger.status("complete", protocol=PROTOCOL, fresh_strict_replay_exact=True,
                          checkpoint_sha256=manifest["checkpoint_sha256"])
            return final
        except BaseException as error:
            ledger.status("failed", protocol=PROTOCOL, error_type=type(error).__name__,
                          error=str(error), traceback=traceback.format_exc())
            raise


def generate_configs(destination):
    formal = _formal_module(); destination.mkdir(parents=True, exist_ok=True)
    for experiment_id in VARIANTS:
        formal.atomic_json(destination / f"{experiment_id}.json", CalibrationRunConfig(experiment_id).to_dict())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config")
    parser.add_argument("--configs", nargs="*"); parser.add_argument("--output-root", required=True)
    parser.add_argument("--generate-configs")
    args = parser.parse_args()
    if args.generate_configs:
        generate_configs(Path(args.generate_configs)); return
    paths = ([args.config] if args.config else args.configs) or []
    if not paths:
        raise ValueError("provide --config or --configs")
    for path in paths:
        config = CalibrationRunConfig.from_dict(json.loads(Path(path).read_text()))
        print(json.dumps(execute(config, Path(args.output_root)), indent=2), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
