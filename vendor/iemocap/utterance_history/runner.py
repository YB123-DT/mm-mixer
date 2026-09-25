from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import UtteranceHistoryDataset
from .model import build_history_model


ROOT = Path(__file__).resolve().parents[1]
IEMOCAP = Path("/data2/yb/multimodalERC/IEMOCAP")
ABLATION48 = ROOT
CLEAN = ROOT / "base"
PKL = IEMOCAP / "external/CSS/data/iemocap_multimodal_features.pkl"
FEATURES = (
    IEMOCAP
    / "Model_rawaux_textpeak_cssv_v1/artifacts/iemocap_textpeak_audio_cssv.npz"
)
DATASET_SOURCE = ROOT / "data/fill53_dataset.py"
BASELINE_CONFIG = CLEAN / "configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
EXPERIMENTS = ("H0", "H00", "HA", "HV", "HAV", "SL")
PROTOCOL = "STRICT_PEAK_TEST_DIAGNOSTIC_UTTERANCE_HISTORY"


def _formal_module():
    path = str(ABLATION48)
    if path not in sys.path:
        sys.path.insert(0, path)
    from ablation48 import formal_runner
    return formal_runner


@dataclass(frozen=True)
class HistoryConfig:
    experiment_id: str
    protocol: str = PROTOCOL
    baseline: str = "S0/E14 K4-d128-L1 fill53 TextPeak"
    baseline_config: str = str(BASELINE_CONFIG)
    pkl: str = str(PKL)
    features: str = str(FEATURES)
    seed: int = 2025
    epochs: int = 100
    batch_size: int = 32
    batch_protocol: str = "utterance"
    gradient_accumulation: int = 1
    history_modalities: tuple[str, ...] = ()
    force_identity: bool = False

    def to_dict(self):
        value = dataclasses.asdict(self)
        value["history_modalities"] = list(self.history_modalities)
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["history_modalities"] = tuple(value.get("history_modalities", ()))
        return cls(**value)


def build_default_config(experiment_id: str) -> HistoryConfig:
    definitions = {
        "H0": ((), False),
        "H00": (("a", "v"), True),
        "HA": (("a",), False),
        "HV": (("v",), False),
        "HAV": (("a", "v"), False),
        "SL": ((), False),
    }
    if experiment_id not in definitions:
        raise ValueError(f"unknown history experiment: {experiment_id}")
    modalities, identity = definitions[experiment_id]
    return HistoryConfig(experiment_id, history_modalities=modalities, force_identity=identity)


def validate_config(config: HistoryConfig) -> None:
    expected = build_default_config(config.experiment_id)
    if config.protocol != PROTOCOL:
        raise ValueError("history protocol mismatch")
    for field in ("batch_protocol", "batch_size", "epochs", "seed", "gradient_accumulation",
                  "history_modalities", "force_identity"):
        if getattr(config, field) != getattr(expected, field):
            raise ValueError(f"{field} differs from the frozen history protocol")
    if config.batch_protocol != "utterance" or config.gradient_accumulation != 1:
        raise ValueError("history experiments must use unaccumulated utterance batches")


def _dataset_class():
    spec = importlib.util.spec_from_file_location("history_fill53_dataset", DATASET_SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Fill53Dataset


def build_loaders(config: HistoryConfig):
    validate_config(config)
    dataset_cls = _dataset_class()
    train = dataset_cls(config.pkl, config.features, "train", True)
    test = dataset_cls(config.pkl, config.features, "test", False)
    if config.history_modalities:
        train = UtteranceHistoryDataset(train, config.history_modalities)
        test = UtteranceHistoryDataset(test, config.history_modalities)
    train_loader = DataLoader(train, config.batch_size, shuffle=True, collate_fn=train.collate_fn)
    test_loader = DataLoader(test, config.batch_size, shuffle=False, collate_fn=test.collate_fn)
    if (len(train), len(test), len(train_loader), len(test_loader)) != (5810, 1623, 182, 51):
        raise RuntimeError("utterance loader schedule drifted from S0")
    return train, test, train_loader, test_loader


def optimizer_parameter_groups(model, lr: float):
    groups, used = [], set()
    named = list(model.named_parameters())
    for prefix, group_lr in (("proj.", lr * 0.5), ("transformer_encoder.", lr),
                             ("classifiers.", lr * 2.0)):
        parameters = []
        for name, parameter in named:
            normalized = name[5:] if name.startswith("base.") else name
            if normalized.startswith(prefix) and parameter.requires_grad:
                parameters.append(parameter)
                used.add(id(parameter))
        if parameters:
            groups.append({"params": parameters, "lr": group_lr})
    remaining = [parameter for _, parameter in named
                 if parameter.requires_grad and id(parameter) not in used]
    if remaining:
        groups.append({"params": remaining, "lr": lr})
    flattened = [id(parameter) for group in groups for parameter in group["params"]]
    required = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if len(flattened) != len(set(flattened)) or set(flattened) != required:
        raise RuntimeError("optimizer groups do not exactly partition trainable parameters")
    return groups


def generate_configs(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for experiment in EXPERIMENTS:
        config = build_default_config(experiment)
        validate_config(config)
        _formal_module().atomic_json(destination / f"{experiment}.json", config.to_dict())


def synthetic_smoke(experiment_id: str):
    torch.manual_seed(2025)
    model = build_history_model(experiment_id, 0.0).train()
    batch, width = 3, 4
    lengths = torch.tensor([4, 3, 1])
    features = {
        "t": torch.randn(batch, 1024), "a": torch.randn(batch, 1024),
        "v": torch.randn(batch, 342),
        "a_history": torch.randn(batch, width, 1024),
        "v_history": torch.randn(batch, width, 342),
        "history_mask": torch.arange(width)[None] < lengths[:, None],
    }
    logits = model(features)[0]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 5]))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return {
        "experiment_id": experiment_id, "shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()), "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": bool(gradients and all(torch.isfinite(value).all() for value in gradients)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


@contextlib.contextmanager
def _history_train_module(experiment_id: str):
    formal = _formal_module()
    original_builder = formal.build_experiment_model
    formal.build_experiment_model = lambda _ignored, dropout: build_history_model(experiment_id, dropout)
    try:
        yield formal._patched_train_module(experiment_id)
    finally:
        formal.build_experiment_model = original_builder


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths():
    return [CLEAN / "train_erc.py", CLEAN / "multiattn.py", BASELINE_CONFIG,
            DATASET_SOURCE, *sorted((ROOT / "utterance_history").glob("*.py"))]


def _labels(dataset):
    base = dataset.base if isinstance(dataset, UtteranceHistoryDataset) else dataset
    return [base.labels[dialogue][turn] for dialogue, turn in base.index]


def execute(config: HistoryConfig, output_root: Path):
    validate_config(config)
    formal = _formal_module()
    ledger = formal.RunLedger(output_root, config.experiment_id)
    with ledger.acquire(blocking=False):
        started = time.time()
        ledger.status("running", protocol=PROTOCOL, started_at_unix=started)
        formal.atomic_json(ledger.run_dir / "config.json", config.to_dict())
        hashes = formal.compute_source_hashes(_source_paths())
        hashes.update({config.pkl: _sha(Path(config.pkl)), config.features: _sha(Path(config.features))})
        formal.atomic_json(ledger.run_dir / "source_sha256.json", hashes)
        try:
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            from sklearn.utils.class_weight import compute_class_weight

            base_module = formal._load_base_module()
            base_module.set_random_seed(config.seed)
            train_set, test_set, train_loader, test_loader = build_loaders(config)
            labels = np.asarray(_labels(train_set))
            weights = compute_class_weight("balanced", classes=np.arange(6), y=labels)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = LabelEncoder()
            encoder.classes_ = np.asarray(formal.CLASS_NAMES)
            fixed = formal._training_cfg(config)
            with _history_train_module(config.experiment_id) as train_module:
                train_module.formal_optimizer_parameter_groups = optimizer_parameter_groups
                model, report_text, metrics = train_module.train_full_pipeline(
                    train_loader, test_loader, test_loader, device, encoder, ["v", "a", "t"],
                    {"v": 342, "a": 1024, "t": 1024},
                    torch.tensor(weights, dtype=torch.float32, device=device), fixed, config.epochs,
                )
            ledger.atomic_checkpoint(copy.deepcopy(model.state_dict()))
            selected, logits_a, predictions_a, labels_a = formal.replay_metrics(model, test_loader, device)
            fresh = build_history_model(
                config.experiment_id,
                float(fixed.get("fusion_dropout", fixed.get("dropout_rate", 0.15))),
            ).to(device)
            fresh.load_state_dict(torch.load(ledger.checkpoint_path, map_location=device, weights_only=True), strict=True)
            replay, logits_b, predictions_b, labels_b = formal.replay_metrics(fresh, test_loader, device)
            exact = (torch.equal(logits_a, logits_b) and torch.equal(predictions_a, predictions_b)
                     and torch.equal(labels_a, labels_b))
            if selected["samples"] != 1623 or replay["samples"] != 1623 or not exact:
                raise RuntimeError("fresh 1623-sample checkpoint replay failed")
            logits_path = ledger.run_dir / "replay_logits.pt"
            temporary = logits_path.with_suffix(".tmp")
            torch.save({"logits": logits_b, "predictions": predictions_b, "labels": labels_b}, temporary)
            os.replace(temporary, logits_path)
            final_metrics = {**metrics, "selected": selected, "fresh_strict_replay": replay,
                             "fresh_strict_replay_exact": True,
                             "replay_logits_sha256": _sha(logits_path)}
            formal.atomic_json(ledger.run_dir / "metrics.json", final_metrics)
            formal.atomic_text(ledger.run_dir / "classification_report.txt", report_text + "\n")
            manifest = {
                "protocol": PROTOCOL, "selection_rule": "EMA at peak Test WF1 (diagnostic only)",
                "generalization_claim_allowed": False, "experiment_id": config.experiment_id,
                "batch_protocol": "utterance", "gradient_accumulation": 1, "seed": config.seed,
                "train_samples": 5810, "test_samples": 1623,
                "history_modalities": list(config.history_modalities), "force_identity": config.force_identity,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "history_alpha": {name: float(value.detach().cpu()) for name, value in model.history_alpha.items()},
                "checkpoint": str(ledger.checkpoint_path), "checkpoint_sha256": _sha(ledger.checkpoint_path),
                "source_sha256": hashes, "weighted_f1": selected["weighted_f1"],
                "accuracy": selected["accuracy"], "macro_f1": selected["macro_f1"],
                "per_class": selected["per_class"], "fresh_strict_replay_exact": True,
                "replay_logits_sha256": _sha(logits_path), "elapsed_seconds": time.time() - started,
                "argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            formal.atomic_json(ledger.run_dir / "manifest.json", manifest)
            ledger.status("complete", protocol=PROTOCOL, fresh_strict_replay_exact=True,
                          checkpoint_sha256=manifest["checkpoint_sha256"], finished_at_unix=time.time())
            return manifest
        except BaseException as error:
            ledger.status("failed", protocol=PROTOCOL, error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(), finished_at_unix=time.time())
            raise


def verify_run_artifacts(root: Path, experiment_id: str) -> bool:
    run = Path(root) / "runs" / experiment_id
    required = ("status.json", "manifest.json", "metrics.json", "replay_logits.pt",
                "best_peak_test_state_dict.pt", "config.json", "source_sha256.json")
    if any(not (run / name).is_file() for name in required):
        return False
    try:
        status = json.loads((run / "status.json").read_text())
        manifest = json.loads((run / "manifest.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        config = HistoryConfig.from_dict(json.loads((run / "config.json").read_text()))
        validate_config(config)
        return bool(
            status.get("state") == "complete"
            and status.get("fresh_strict_replay_exact") is True
            and manifest.get("protocol") == PROTOCOL
            and manifest.get("seed") == 2025
            and manifest.get("batch_protocol") == "utterance"
            and manifest.get("gradient_accumulation") == 1
            and metrics.get("fresh_strict_replay_exact") is True
            and manifest.get("checkpoint_sha256") == _sha(run / "best_peak_test_state_dict.pt")
            and manifest.get("replay_logits_sha256") == _sha(run / "replay_logits.pt")
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output-root", default=str(ROOT / "formal"))
    parser.add_argument("--generate-configs", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.generate_configs:
        generate_configs(Path(args.output_root) / "configs")
        return
    config = HistoryConfig.from_dict(json.loads(Path(args.config).read_text()))
    validate_config(config)
    if args.smoke:
        print(json.dumps(synthetic_smoke(config.experiment_id), sort_keys=True))
    elif args.execute:
        print(json.dumps(execute(config, Path(args.output_root)), indent=2))
    else:
        print(json.dumps({"dry_run": True, "config": config.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
