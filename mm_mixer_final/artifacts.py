from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Sequence

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)

from .audit import config_payload_sha256, sha256


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, class_names: Sequence[str]
) -> dict[str, Any]:
    predictions = logits.argmax(dim=-1)
    y_true = labels.detach().cpu().numpy()
    y_pred = predictions.detach().cpu().numpy()
    labels_index = list(range(len(class_names)))
    _, _, class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_index, zero_division=0
    )
    return {
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "class_f1": {
            name: float(value) for name, value in zip(class_names, class_f1)
        },
        "support": {
            name: int(value) for name, value in zip(class_names, support)
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class PeakArtifactStore:
    """Versioned peak bundles with an atomically replaced public symlink."""

    def __init__(self, run_root: Path | str, class_names: Sequence[str]):
        self.run_root = Path(run_root)
        self.class_names = tuple(class_names)
        self.bundle_root = self.run_root / ".peak_bundles"
        self.bundle_root.mkdir(parents=True, exist_ok=True)
        self.public = self.run_root / "best_peak"

    @property
    def best_weighted_f1(self) -> float:
        if not self.public.exists():
            return float("-inf")
        return float(json.loads((self.public / "peak_test_metrics.json").read_text())["weighted_f1"])

    def publish_if_better(
        self, payload: dict[str, Any], *, force: bool = False
    ) -> bool:
        score = float(payload["metrics"]["weighted_f1"])
        if not force and score <= self.best_weighted_f1:
            return False

        bundle = self.bundle_root / f"epoch{int(payload['epoch']):04d}-{uuid.uuid4().hex}"
        bundle.mkdir()
        predictions = payload["logits"].argmax(dim=-1)
        torch.save(payload["state_dict"], bundle / "best_peak_test_state_dict.pt")
        torch.save(
            {
                "logits": payload["logits"].detach().cpu(),
                "predictions": predictions.detach().cpu(),
                "labels": payload["labels"].detach().cpu(),
            },
            bundle / "peak_test_predictions.pt",
        )
        metrics = {"epoch": int(payload["epoch"]), **payload["metrics"]}
        _atomic_json(bundle / "peak_test_metrics.json", metrics)
        report = classification_report(
            payload["labels"].detach().cpu().numpy(),
            predictions.detach().cpu().numpy(),
            labels=list(range(len(self.class_names))),
            target_names=self.class_names,
            zero_division=0,
        )
        (bundle / "classification_report.txt").write_text(report, encoding="utf-8")
        _atomic_json(bundle / "history.json", payload["history"])
        _atomic_json(bundle / "config.json", payload["config"])
        artifact_sha256 = {
            name: sha256(bundle / name)
            for name in (
                "best_peak_test_state_dict.pt",
                "peak_test_predictions.pt",
                "peak_test_metrics.json",
                "classification_report.txt",
                "history.json",
                "config.json",
            )
        }
        manifest = {
            **payload["manifest"],
            "artifact_sha256": artifact_sha256,
        }
        _atomic_json(bundle / "manifest.json", manifest)
        _atomic_json(
            bundle / "status.json",
            {
                "state": "peak_published",
                "epoch": int(payload["epoch"]),
                "weighted_f1": score,
                "fresh_strict_replay_exact": False,
                "checkpoint_sha256": artifact_sha256[
                    "best_peak_test_state_dict.pt"
                ],
                "predictions_sha256": artifact_sha256[
                    "peak_test_predictions.pt"
                ],
            },
        )

        temporary_link = self.run_root / f".best_peak.{uuid.uuid4().hex}.tmp"
        os.symlink(os.path.relpath(bundle, self.run_root), temporary_link)
        os.replace(temporary_link, self.public)
        return True

    def verify_saved_predictions(
        self, fresh_logits: torch.Tensor, fresh_labels: torch.Tensor
    ) -> bool:
        saved = torch.load(
            self.public / "peak_test_predictions.pt", map_location="cpu", weights_only=True
        )
        fresh_logits = fresh_logits.detach().cpu()
        fresh_labels = fresh_labels.detach().cpu()
        fresh_predictions = fresh_logits.argmax(dim=-1)
        exact = (
            torch.equal(saved["logits"], fresh_logits)
            and torch.equal(saved["predictions"], fresh_predictions)
            and torch.equal(saved["labels"], fresh_labels)
        )
        if not exact:
            return False
        fresh_metrics = classification_metrics(
            fresh_logits, fresh_labels, self.class_names
        )
        stored_metrics = json.loads(
            (self.public / "peak_test_metrics.json").read_text()
        )
        exact = all(
            stored_metrics[key] == fresh_metrics[key]
            for key in ("weighted_f1", "macro_f1", "accuracy", "class_f1", "support")
        )
        if exact:
            status = json.loads((self.public / "status.json").read_text())
            status["state"] = "complete"
            status["fresh_strict_replay_exact"] = True
            _atomic_json(self.public / "status.json", status)
        return exact

    def validate_public_bundle(self, expected: dict[str, Any]) -> bool:
        required = {
            "best_peak_test_state_dict.pt",
            "peak_test_predictions.pt",
            "peak_test_metrics.json",
            "classification_report.txt",
            "history.json",
            "config.json",
            "manifest.json",
            "status.json",
        }
        try:
            if not self.public.is_dir():
                return False
            if not required.issubset(path.name for path in self.public.iterdir()):
                return False
            status = json.loads((self.public / "status.json").read_text())
            manifest = json.loads((self.public / "manifest.json").read_text())
            config = json.loads((self.public / "config.json").read_text())
            saved_metrics = json.loads(
                (self.public / "peak_test_metrics.json").read_text()
            )
            if (
                status.get("state") != "complete"
                or status.get("fresh_strict_replay_exact") is not True
                or manifest.get("fresh_strict_replay_exact") is not True
            ):
                return False
            for key in (
                "dataset", "variant", "seed", "config_contract_sha256"
            ):
                if manifest.get(key) != expected.get(key):
                    return False
            if manifest.get("config_sha256") != config_payload_sha256(config):
                return False
            artifact_hashes = manifest.get("artifact_sha256")
            if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != {
                "best_peak_test_state_dict.pt",
                "peak_test_predictions.pt",
                "peak_test_metrics.json",
                "classification_report.txt",
                "history.json",
                "config.json",
            }:
                return False
            for name, digest in artifact_hashes.items():
                if sha256(self.public / name) != digest:
                    return False
            if (
                status.get("checkpoint_sha256")
                != artifact_hashes["best_peak_test_state_dict.pt"]
                or status.get("predictions_sha256")
                != artifact_hashes["peak_test_predictions.pt"]
            ):
                return False
            for field in ("source_hashes", "feature_hashes"):
                values = manifest.get(field)
                if not isinstance(values, dict) or not values:
                    return False
                for path, digest in values.items():
                    if not Path(path).is_file() or sha256(path) != digest:
                        return False
            checkpoint = torch.load(
                self.public / "best_peak_test_state_dict.pt",
                map_location="cpu",
                weights_only=True,
            )
            if not isinstance(checkpoint, dict) or not checkpoint:
                return False
            predictions = torch.load(
                self.public / "peak_test_predictions.pt",
                map_location="cpu",
                weights_only=True,
            )
            if not (
                torch.equal(
                    predictions["predictions"],
                    predictions["logits"].argmax(dim=-1),
                )
                and predictions["labels"].shape[0]
                == predictions["logits"].shape[0]
            ):
                return False
            computed = classification_metrics(
                predictions["logits"],
                predictions["labels"],
                self.class_names,
            )
            return all(
                saved_metrics[key] == computed[key]
                for key in (
                    "weighted_f1", "macro_f1", "accuracy",
                    "class_f1", "support",
                )
            )
        except (
            OSError, json.JSONDecodeError, KeyError, TypeError,
            ValueError, RuntimeError,
        ):
            return False
