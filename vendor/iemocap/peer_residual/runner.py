from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from .model import build_peer_model, optimizer_parameter_groups
from .registry import PeerRunConfig, VARIANTS, validate_run_config


ROOT = Path(__file__).resolve().parents[1]
IEMOCAP = Path("/data2/yb/multimodalERC/IEMOCAP")
ABLATION48 = ROOT
PROTOCOL = "STRICT_PEAK_TEST_DIAGNOSTIC_PEER_RESIDUAL"


def _formal_module():
    path = str(ABLATION48)
    if path not in sys.path:
        sys.path.insert(0, path)
    from ablation48 import formal_runner
    return formal_runner


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths():
    clean = ROOT / "base"
    return [
        clean / "train_erc.py", clean / "multiattn.py",
        *sorted((ROOT / "peer_residual").glob("*.py")),
    ]


def generate_configs(destination: Path):
    formal = _formal_module()
    destination.mkdir(parents=True, exist_ok=True)
    for experiment in VARIANTS:
        formal.atomic_json(destination / f"{experiment}.json", PeerRunConfig(experiment).to_dict())


def build_loaders(config: PeerRunConfig):
    from utterance_history.runner import HistoryConfig, build_loaders as history_loaders
    history = HistoryConfig(
        "H0", pkl=config.pkl, features=config.features,
        baseline_config=config.baseline_config,
    )
    return history_loaders(history)


@contextlib.contextmanager
def _peer_train_module(experiment_id: str):
    formal = _formal_module()
    original = formal.build_experiment_model
    formal.build_experiment_model = lambda _ignored, dropout: build_peer_model(experiment_id, dropout)
    try:
        module = formal._patched_train_module(experiment_id)
        module.formal_optimizer_parameter_groups = optimizer_parameter_groups
        if experiment_id == "V9":
            original_evaluate = module.evaluate_model
            calls = {"value": 0}

            def scheduled_evaluate(model, *args, **kwargs):
                epoch = calls["value"] // 2 + 1
                model.set_epoch(epoch)
                calls["value"] += 1
                return original_evaluate(model, *args, **kwargs)

            module.evaluate_model = scheduled_evaluate
        yield module
    finally:
        formal.build_experiment_model = original


def synthetic_smoke(experiment_id: str):
    torch.manual_seed(2025)
    model = build_peer_model(experiment_id, 0).train()
    features = {"t": torch.randn(3, 1024), "a": torch.randn(3, 1024), "v": torch.randn(3, 342)}
    logits = model(features)[0]
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 5]))
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return {
        "experiment_id": experiment_id, "shape": list(logits.shape),
        "finite": bool(torch.isfinite(logits).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": bool(gradients and all(torch.isfinite(value).all() for value in gradients)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def _labels(dataset):
    return [dataset.labels[dialogue][turn] for dialogue, turn in dataset.index]


def _collect_diagnostics(model, loader, device, output_path: Path):
    collected = {}
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            model({name: value.to(device) for name, value in features.items()})
            for family, values in (model.last_diagnostics or {}).items():
                for modality, value in values.items():
                    collected.setdefault(f"{family}.{modality}", []).append(value.cpu())
    tensors = {key: torch.cat(values) for key, values in collected.items() if values}
    temporary = output_path.with_suffix(".tmp")
    torch.save(tensors, temporary); os.replace(temporary, output_path)
    summary = {}
    for key, value in tensors.items():
        finite = value[torch.isfinite(value)]
        summary[key] = {
            "shape": list(value.shape), "finite_count": int(finite.numel()),
            "mean": float(finite.mean()) if finite.numel() else None,
            "std": float(finite.std(unbiased=False)) if finite.numel() else None,
            "min": float(finite.min()) if finite.numel() else None,
            "max": float(finite.max()) if finite.numel() else None,
        }
    return summary


def execute(config: PeerRunConfig, output_root: Path):
    validate_run_config(config)
    formal = _formal_module()
    ledger = formal.RunLedger(output_root, config.experiment_id)
    with ledger.acquire(blocking=False):
        started = time.time()
        ledger.status("running", protocol=PROTOCOL, started_at_unix=started)
        formal.atomic_json(ledger.run_dir / "config.json", config.to_dict())
        hashes = formal.compute_source_hashes(_source_paths())
        hashes.update({config.pkl: _sha(config.pkl), config.features: _sha(config.features),
                       config.baseline_config: _sha(config.baseline_config)})
        formal.atomic_json(ledger.run_dir / "source_sha256.json", hashes)
        try:
            import numpy as np
            from sklearn.preprocessing import LabelEncoder
            from sklearn.utils.class_weight import compute_class_weight
            base_module = formal._load_base_module()
            base_module.set_random_seed(config.seed)
            train_set, test_set, train_loader, test_loader = build_loaders(config)
            weights = compute_class_weight("balanced", classes=np.arange(6), y=np.asarray(_labels(train_set)))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = LabelEncoder(); encoder.classes_ = np.asarray(formal.CLASS_NAMES)
            fixed = formal._training_cfg(config)
            with _peer_train_module(config.experiment_id) as train_module:
                model, report_text, metrics = train_module.train_full_pipeline(
                    train_loader, test_loader, test_loader, device, encoder, ["v", "a", "t"],
                    {"v": 342, "a": 1024, "t": 1024},
                    torch.tensor(weights, dtype=torch.float32, device=device), fixed, config.epochs,
                )
            if config.experiment_id == "V9":
                model.set_epoch(10)
            ledger.atomic_checkpoint(copy.deepcopy(model.state_dict()))
            selected, logits_a, predictions_a, labels_a = formal.replay_metrics(model, test_loader, device)
            dropout = float(fixed.get("fusion_dropout", fixed.get("dropout_rate", .15)))
            fresh = build_peer_model(config.experiment_id, dropout).to(device)
            fresh.load_state_dict(torch.load(ledger.checkpoint_path, map_location=device, weights_only=True), strict=True)
            if config.experiment_id == "V9":
                fresh.set_epoch(10)
            replay, logits_b, predictions_b, labels_b = formal.replay_metrics(fresh, test_loader, device)
            exact = (torch.equal(logits_a, logits_b) and torch.equal(predictions_a, predictions_b)
                     and torch.equal(labels_a, labels_b))
            if selected["samples"] != 1623 or not exact:
                raise RuntimeError("fresh 1623-sample checkpoint replay failed")
            replay_path = ledger.run_dir / "replay_logits.pt"
            temporary = replay_path.with_suffix(".tmp")
            torch.save({"logits": logits_b, "predictions": predictions_b, "labels": labels_b}, temporary)
            os.replace(temporary, replay_path)
            diagnostic_path = ledger.run_dir / "diagnostics.pt"
            diagnostic_summary = _collect_diagnostics(fresh, test_loader, device, diagnostic_path)
            formal.atomic_json(ledger.run_dir / "diagnostics.json", diagnostic_summary)
            final_metrics = {**metrics, "selected": selected, "fresh_strict_replay": replay,
                             "fresh_strict_replay_exact": True,
                             "replay_logits_sha256": _sha(replay_path)}
            formal.atomic_json(ledger.run_dir / "metrics.json", final_metrics)
            formal.atomic_text(ledger.run_dir / "classification_report.txt", report_text + "\n")
            manifest = {
                "protocol": PROTOCOL, "selection_rule": "EMA at peak Test WF1 (diagnostic only)",
                "generalization_claim_allowed": False, "experiment_id": config.experiment_id,
                "seed": config.seed, "train_samples": 5810, "test_samples": 1623,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint": str(ledger.checkpoint_path), "checkpoint_sha256": _sha(ledger.checkpoint_path),
                "source_sha256": hashes, "weighted_f1": selected["weighted_f1"],
                "accuracy": selected["accuracy"], "macro_f1": selected["macro_f1"],
                "per_class": selected["per_class"], "fresh_strict_replay_exact": True,
                "replay_logits_sha256": _sha(replay_path), "diagnostics_sha256": _sha(diagnostic_path),
                "elapsed_seconds": time.time() - started,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            }
            formal.atomic_json(ledger.run_dir / "manifest.json", manifest)
            ledger.status("complete", protocol=PROTOCOL, fresh_strict_replay_exact=True,
                          checkpoint_sha256=manifest["checkpoint_sha256"], finished_at_unix=time.time())
            return manifest
        except BaseException as error:
            ledger.status("failed", protocol=PROTOCOL, error_type=type(error).__name__, error=str(error),
                          traceback=traceback.format_exc(), finished_at_unix=time.time())
            raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--output-root", default=str(ROOT / "formal/peer_screen"))
    parser.add_argument("--config-root", default=str(ROOT / "formal/peer_configs"))
    parser.add_argument("--generate-configs", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.generate_configs:
        generate_configs(Path(args.config_root)); return
    config = PeerRunConfig.from_dict(json.loads(Path(args.config).read_text()))
    validate_run_config(config)
    if args.smoke:
        print(json.dumps(synthetic_smoke(config.experiment_id), sort_keys=True))
    elif args.execute:
        print(json.dumps(execute(config, Path(args.output_root)), indent=2))
    else:
        print(json.dumps(config.to_dict(), indent=2))


if __name__ == "__main__":
    main()
