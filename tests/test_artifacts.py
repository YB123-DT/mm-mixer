from __future__ import annotations

import json

import torch

from mm_mixer_final.artifacts import PeakArtifactStore, classification_metrics


def _payload(epoch: int, logits: torch.Tensor, labels: torch.Tensor):
    return {
        "epoch": epoch,
        "state_dict": {"weight": torch.tensor([float(epoch)])},
        "logits": logits,
        "labels": labels,
        "metrics": classification_metrics(logits, labels, ("a", "b")),
        "history": [{"epoch": epoch}],
        "config": {"dataset": "toy"},
        "manifest": {"source_hashes": {"toy.py": "abc"}},
    }


def test_non_peak_does_not_replace_published_bundle(tmp_path):
    store = PeakArtifactStore(tmp_path, ("a", "b"))
    labels = torch.tensor([0, 1, 1])
    assert store.publish_if_better(_payload(1, torch.tensor([[3., 0.], [0., 3.], [0., 3.]]), labels))
    first = (tmp_path / "best_peak" / "peak_test_metrics.json").read_bytes()
    assert not store.publish_if_better(_payload(2, torch.zeros(3, 2), labels))
    assert (tmp_path / "best_peak" / "peak_test_metrics.json").read_bytes() == first


def test_peak_bundle_has_complete_schema_and_can_be_verified(tmp_path):
    store = PeakArtifactStore(tmp_path, ("a", "b"))
    logits = torch.tensor([[4., 0.], [0., 4.], [0., 3.]])
    labels = torch.tensor([0, 1, 1])
    assert store.publish_if_better(_payload(3, logits, labels))
    peak = tmp_path / "best_peak"
    assert {path.name for path in peak.iterdir()} == {
        "best_peak_test_state_dict.pt",
        "peak_test_predictions.pt",
        "peak_test_metrics.json",
        "classification_report.txt",
        "history.json",
        "config.json",
        "manifest.json",
        "status.json",
    }
    status = json.loads((peak / "status.json").read_text())
    assert status["state"] == "peak_published"
    assert store.verify_saved_predictions(logits, labels)


def test_fresh_replay_requires_exact_logits_predictions_labels_and_metrics(tmp_path):
    store = PeakArtifactStore(tmp_path, ("a", "b"))
    logits = torch.tensor([[4., 0.], [0., 4.]])
    labels = torch.tensor([0, 1])
    store.publish_if_better(_payload(1, logits, labels))
    assert store.verify_saved_predictions(logits, labels)
    assert not store.verify_saved_predictions(logits + 0.01, labels)


def test_every_strict_refresh_atomically_points_to_a_complete_bundle(tmp_path):
    store = PeakArtifactStore(tmp_path, ("a", "b"))
    labels = torch.tensor([0, 1])
    first = _payload(1, torch.tensor([[1., 0.], [1., 0.]]), labels)
    second = _payload(2, torch.tensor([[3., 0.], [0., 3.]]), labels)
    assert store.publish_if_better(first)
    first_target = store.public.resolve()
    assert store.publish_if_better(second)
    assert store.public.resolve() != first_target
    assert {path.name for path in store.public.iterdir()} == {
        "best_peak_test_state_dict.pt",
        "peak_test_predictions.pt",
        "peak_test_metrics.json",
        "classification_report.txt",
        "history.json",
        "config.json",
        "manifest.json",
        "status.json",
    }
