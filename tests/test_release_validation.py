from __future__ import annotations

import inspect

import json

import torch

from mm_mixer_final.artifacts import PeakArtifactStore, classification_metrics
from mm_mixer_final.audit import (
    config_payload_sha256,
    loaded_python_source_hashes,
    optimizer_group_audit,
    sha256,
)
from mm_mixer_final.config import config_contract_sha256, get_config


def test_loaded_source_hashes_include_final_common_modules():
    hashes = loaded_python_source_hashes()
    assert any(path.endswith("/mm_mixer_final/audit.py") for path in hashes)
    assert any(path.endswith("/mm_mixer_final/config.py") for path in hashes)


def test_iemocap_runtime_optimizer_audit_matches_actual_groups():
    from dataset_runners.iemocap import _build_full, optimizer_parameter_groups

    model = _build_full(0.2)
    groups = optimizer_parameter_groups(model, 3e-5)
    audit = optimizer_group_audit(model, groups)
    assert {item["name"]: item["lr"] for item in audit} == {
        "projection": 3e-5,
        "pairwise_cross": 1.2e-4,
        "cross_attention": 6e-5,
        "mixer_encoder": 6e-5,
        "classifier": 1.2e-4,
        "remaining": 6e-5,
    }


def test_iemocap_runner_does_not_publish_inside_training_loop():
    from dataset_runners import iemocap

    source = inspect.getsource(iemocap.run_variant)
    assert "FORMAL_PEAK_CALLBACK = publish_epoch_peak" not in source


def test_release_bundle_validation_rejects_changed_source(tmp_path):
    source = tmp_path / "source.py"
    feature = tmp_path / "feature.bin"
    source.write_text("x = 1\n")
    feature.write_bytes(b"feature")
    config = {"dataset": "iemocap", "seed": 2025}
    cfg = get_config("iemocap", "full", 2025)
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])
    store = PeakArtifactStore(tmp_path / "run", ("a", "b"))
    manifest = {
        "dataset": "iemocap",
        "variant": "full",
        "seed": 2025,
        "fresh_strict_replay_exact": True,
        "config_sha256": config_payload_sha256(config),
        "config_contract_sha256": config_contract_sha256(cfg),
        "source_hashes": {str(source.resolve()): sha256(source)},
        "feature_hashes": {str(feature.resolve()): sha256(feature)},
    }
    store.publish_if_better({
        "epoch": 1,
        "state_dict": {"x": torch.tensor(1)},
        "logits": logits,
        "labels": labels,
        "metrics": classification_metrics(logits, labels, ("a", "b")),
        "history": {"epoch": 1},
        "config": config,
        "manifest": manifest,
    })
    assert store.verify_saved_predictions(logits, labels)
    expected = {
        "dataset": "iemocap",
        "variant": "full",
        "seed": 2025,
        "config_contract_sha256": config_contract_sha256(cfg),
    }
    assert store.validate_public_bundle(expected)
    checkpoint = store.public / "best_peak_test_state_dict.pt"
    original_checkpoint = checkpoint.read_bytes()
    checkpoint.write_bytes(original_checkpoint + b"tampered")
    assert not store.validate_public_bundle(expected)
    checkpoint.write_bytes(original_checkpoint)
    assert store.validate_public_bundle(expected)
    source.write_text("x = 2\n")
    assert not store.validate_public_bundle(expected)
