#!/usr/bin/env python3
"""Recheck completed revision predictions and write descriptive result tables.

This CPU-only reader trusts the queue's prior full checkpoint/source/data audit,
then independently rechecks identity, completion, prediction/metric/config
hashes, and recomputed prediction metrics. It never runs or selects experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import statistics
from typing import Any

import torch
from sklearn.metrics import confusion_matrix

from mm_mixer_final.artifacts import classification_metrics
from mm_mixer_final.config import get_config
from mm_mixer_final.revision_results import REVISION_SEEDS


METRICS = ("weighted_f1", "accuracy", "macro_f1")
SELECTION = "strict_peak_test_wf1"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def _read_plan(path: Path, raw_content: bytes | None = None) -> list[dict[str, Any]]:
    content = json.loads(path.read_bytes() if raw_content is None else raw_content)
    raw_jobs = content.get("jobs") if isinstance(content, dict) else content
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("plan must contain a nonempty jobs list")
    jobs, identities = [], set()
    for raw in raw_jobs:
        job = dict(raw)
        dataset, variant, seed = job["dataset"], job["variant"], job["seed"]
        if dataset not in REVISION_SEEDS or type(seed) is not int or seed not in REVISION_SEEDS[dataset]:
            raise ValueError(f"not a revision dataset/seed: {dataset}/{seed}")
        if not isinstance(variant, str) or not variant or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in variant):
            raise ValueError(f"invalid variant: {variant}")
        identity = f"{dataset}/{variant}/seed{seed}"
        if identity in identities:
            raise ValueError(f"duplicate job: {identity}")
        identities.add(identity)
        if not Path(job["output_root"]).is_absolute():
            raise ValueError(f"{identity}: output_root must be absolute")
        if job.get("reuse_bundle") and not Path(job["reuse_bundle"]).is_absolute():
            raise ValueError(f"{identity}: reuse_bundle must be absolute")
        job["id"] = identity
        jobs.append(job)
    return jobs


def _checked_bytes(bundle: Path, filename: str, hashes: dict[str, str]) -> bytes:
    content = (bundle / filename).read_bytes()
    if hashes.get(filename) != _digest(content):
        raise ValueError(f"artifact hash mismatch: {filename}")
    return content


def verify_predictions(job: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Verify one already-completed queue record; never accept partial peaks."""
    if record.get("state") != "completed" or record.get("artifact_verified") is not True:
        raise ValueError("queue record is not completed and artifact_verified")
    if record.get("formal_protocol_verified") is not True:
        raise ValueError("queue record lacks formal_protocol_verified")
    for key in ("dataset", "variant", "seed"):
        if record.get(key) != job[key]:
            raise ValueError(f"queue identity mismatch: {key}")
    expected_bundle = Path(job.get("reuse_bundle") or Path(job["output_root"]) / job["id"] / "best_peak").resolve()
    if not record.get("bundle") or Path(record["bundle"]).resolve() != expected_bundle:
        raise ValueError("queue bundle differs from the planned result path")
    # Resolve best_peak once so an atomic link update cannot mix two bundles.
    bundle = expected_bundle
    manifest = json.loads((bundle / "manifest.json").read_text())
    status = json.loads((bundle / "status.json").read_text())
    if not isinstance(manifest, dict) or not isinstance(status, dict):
        raise ValueError("bundle manifest and status must be objects")
    if status.get("state") != "complete" or status.get("fresh_strict_replay_exact") is not True or manifest.get("fresh_strict_replay_exact") is not True:
        raise ValueError("bundle is not complete with exact fresh replay")
    if manifest.get("selection") != SELECTION:
        raise ValueError("bundle selection protocol mismatch")
    for key in ("dataset", "variant", "seed"):
        if manifest.get(key) != job[key]:
            raise ValueError(f"bundle identity mismatch: {key}")
    contract = manifest.get("config_contract_sha256")
    if not isinstance(contract, str) or len(contract) != 64 or record.get("verified_contract") != contract:
        raise ValueError("bundle contract differs from the queue-verified contract")
    if job.get("expected_contract_sha256") and job["expected_contract_sha256"] != contract:
        raise ValueError("bundle contract differs from the planned contract")
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, dict):
        raise ValueError("missing artifact hashes")
    pred_bytes = _checked_bytes(bundle, "peak_test_predictions.pt", hashes)
    metrics_bytes = _checked_bytes(bundle, "peak_test_metrics.json", hashes)
    config = json.loads(_checked_bytes(bundle, "config.json", hashes))
    if not isinstance(config, dict):
        raise ValueError("saved configuration must be an object")
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    if manifest.get("config_sha256") != _digest(config_bytes):
        raise ValueError("config content hash mismatch")
    formal = get_config(job["dataset"], "full", job["seed"])
    budget_key = "epochs" if job["dataset"] == "iemocap" else "num_epochs"
    if type(config.get(budget_key)) is not int or config[budget_key] != formal.epochs:
        raise ValueError(f"formal epoch budget mismatch: {budget_key}")
    if job["dataset"] == "iemocap":
        correct_selection = config.get("selection") == SELECTION
    else:
        fixed, audit = config.get("fixed_params"), config.get("runtime_audit")
        correct_selection = (isinstance(fixed, dict) and isinstance(audit, dict)
                             and fixed.get("selection_mode") == "test" and audit.get("selection") == SELECTION)
    if not correct_selection:
        raise ValueError("saved configuration selection protocol mismatch")
    if status.get("predictions_sha256") != _digest(pred_bytes):
        raise ValueError("status prediction hash mismatch")
    names = tuple(get_config(job["dataset"], "full", job["seed"]).class_names)
    predictions = torch.load(io.BytesIO(pred_bytes), map_location="cpu", weights_only=True)
    if not isinstance(predictions, dict):
        raise ValueError("prediction payload must be an object")
    logits, labels, saved_predictions = (predictions[key] for key in ("logits", "labels", "predictions"))
    if not all(isinstance(value, torch.Tensor) for value in (logits, labels, saved_predictions)):
        raise ValueError("prediction payload must contain tensors")
    if logits.ndim != 2 or logits.shape[1] != len(names) or logits.shape[0] == 0:
        raise ValueError("invalid logits shape or empty test predictions")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0] or saved_predictions.shape != labels.shape:
        raise ValueError("prediction/label shape mismatch")
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("labels must be integer class indices")
    if not bool(torch.isfinite(logits).all()) or bool((labels < 0).any()) or bool((labels >= len(names)).any()):
        raise ValueError("nonfinite logits or out-of-range labels")
    if not torch.equal(logits.argmax(dim=-1), saved_predictions):
        raise ValueError("saved predictions differ from logits argmax")
    recomputed = classification_metrics(logits, labels, names)
    saved_metrics = json.loads(metrics_bytes)
    if not isinstance(saved_metrics, dict) or not isinstance(record.get("metrics"), dict):
        raise ValueError("saved and queue metrics must be objects")
    for key in (*METRICS, "class_f1", "support"):
        if saved_metrics.get(key) != recomputed[key]:
            raise ValueError(f"recomputed metric mismatch: {key}")
        if record.get("metrics", {}).get(key) != saved_metrics[key]:
            raise ValueError(f"queue metric differs from verified artifact: {key}")
    if status.get("epoch") != saved_metrics.get("epoch") or status.get("weighted_f1") != recomputed["weighted_f1"]:
        raise ValueError("status metrics/epoch mismatch")
    matrix = confusion_matrix(labels.numpy(), saved_predictions.numpy(), labels=list(range(len(names))))
    return {"bundle": str(bundle), "epoch": saved_metrics["epoch"], "metrics": recomputed,
            "class_names": list(names), "samples": int(labels.numel()),
            "labels_sha256": _digest(labels.to(torch.int64).contiguous().numpy().tobytes()),
            "confusion_counts": matrix.tolist(),
            "classes_without_true_examples": [name for name in names if recomputed["support"][name] == 0],
            "artifact_sha256": {name: hashes[name] for name in ("peak_test_predictions.pt", "peak_test_metrics.json", "config.json")},
            "verified_contract": contract}


def analyze(plan_path: Path, state_path: Path) -> dict[str, Any]:
    plan_bytes, state_bytes = plan_path.read_bytes(), state_path.read_bytes()
    jobs = _read_plan(plan_path, plan_bytes)
    state = json.loads(state_bytes)
    if not isinstance(state.get("jobs"), dict):
        raise ValueError("state must contain a jobs mapping")
    if state.get("selection") not in (None, SELECTION):
        raise ValueError("queue selection protocol mismatch")
    indexed = {}
    for job in jobs:
        record = state["jobs"].get(job["id"], {})
        row = {key: job[key] for key in ("id", "dataset", "variant", "seed")}
        row.update(state=record.get("state", "missing"), included=False,
                   artifact_verified=record.get("artifact_verified", False),
                   formal_protocol_verified=record.get("formal_protocol_verified", False),
                   reused=record.get("reused", False), error=record.get("error"), metrics=None)
        if record.get("state") == "completed" and record.get("artifact_verified") is True:
            try:
                row.update(verify_predictions(job, record))
                row.update(included=True, verification="prediction_metrics_rechecked")
            except (OSError, ValueError, TypeError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as exc:
                row.update(state="invalid_artifact", error=str(exc), verification="failed")
        else:
            row["verification"] = "not_eligible"
        indexed[job["id"]] = row
    groups = []
    for dataset, variant in sorted({(job["dataset"], job["variant"]) for job in jobs}):
        names = tuple(get_config(dataset, "full", REVISION_SEEDS[dataset][0]).class_names)
        rows = []
        for seed in REVISION_SEEDS[dataset]:
            identity = f"{dataset}/{variant}/seed{seed}"
            row = dict(indexed.get(identity, {"id": identity, "dataset": dataset, "variant": variant,
                                             "seed": seed, "state": "not_planned", "included": False,
                                             "metrics": None, "error": "seed omitted from plan"}))
            row["delta_to_full"] = None
            baseline = indexed.get(f"{dataset}/full/seed{seed}", {})
            if row["included"] and baseline.get("included"):
                if row["labels_sha256"] == baseline["labels_sha256"]:
                    row["delta_to_full"] = {key: row["metrics"][key] - baseline["metrics"][key] for key in METRICS}
                else:
                    row["pairing_error"] = "Full and variant test label order differs"
            elif row["included"]:
                row["pairing_error"] = "same-seed verified Full unavailable in this plan"
            rows.append(row)
        available = [row for row in rows if row["included"]]
        paired = [row for row in rows if row["delta_to_full"] is not None]
        means, deviations = [], []
        for true_index in range(len(names)):
            mean_row, std_row = [], []
            for predicted_index in range(len(names)):
                summary = _stats([row["confusion_counts"][true_index][predicted_index] for row in available])
                mean_row.append(summary["mean"])
                std_row.append(summary["sample_std"])
            means.append(mean_row)
            deviations.append(std_row)
        groups.append({"dataset": dataset, "variant": variant, "class_names": list(names),
                       "complete_three_seeds": len(available) == 3, "n": len(available),
                       "missing_or_failed_seeds": [row["seed"] for row in rows if not row["included"]],
                       "seeds": rows,
                       "metrics": {key: _stats([row["metrics"][key] for row in available]) for key in METRICS},
                       "class_f1": {name: _stats([row["metrics"]["class_f1"][name] for row in available
                                                   if row["metrics"]["support"][name] > 0]) for name in names},
                       "confusion_counts": {"n": len(available), "mean": means, "sample_std": deviations},
                       "paired_delta_to_full": {key: _stats([row["delta_to_full"][key] for row in paired]) for key in METRICS}})
    return {"schema_version": 1, "selection": SELECTION, "metric_scale": "fraction_0_to_1",
            "standard_deviation": "sample (ddof=1); null when n<2",
            "class_f1_missing_policy": "Seed-level metrics retain the stored zero_division=0 convention; class averages exclude seeds with zero true support and report their n.",
            "confusion_orientation": "rows=true class; columns=predicted class; raw counts",
            "paired_delta_direction": "variant_minus_full; same dataset/seed and identical test label order",
            "verification_scope": "Prior queue checkpoint/source/data verification plus fresh identity, completion, config/prediction/metric hash and metric recomputation checks; no training or checkpoint selection.",
            "expected_seeds": {key: list(value) for key, value in REVISION_SEEDS.items()},
            "inputs": {"plan": str(plan_path.resolve()), "state": str(state_path.resolve()),
                       "plan_sha256": _digest(plan_bytes), "state_sha256": _digest(state_bytes)},
            "ignored_state_jobs": sorted(set(state["jobs"]) - set(indexed)),
            "complete": all(group["complete_three_seeds"] for group in groups), "groups": groups}


def _cell(summary: dict[str, Any], signed=False) -> str:
    if summary["mean"] is None:
        return "— (n=0)"
    mean = f"{summary['mean'] * 100:+.2f}" if signed else f"{summary['mean'] * 100:.2f}"
    std = "—" if summary["sample_std"] is None else f"{summary['sample_std'] * 100:.2f}"
    return f"{mean} ± {std} (n={summary['n']})"


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# MM-Mixer revision experiment results", "",
             "Selection: `strict_peak_test_wf1`. Metrics below are percentages; paired differences are percentage points. Standard deviations use ddof=1. Partial groups are explicitly marked and are not three-seed results.", "",
             "| Dataset | Variant | Verified seeds | WF1 | ACC | Macro-F1 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for group in result["groups"]:
        status = f"{group['n']}/3" + (" complete" if group["complete_three_seeds"] else " INCOMPLETE")
        lines.append(f"| {group['dataset']} | {group['variant']} | {status} | " + " | ".join(_cell(group["metrics"][key]) for key in METRICS) + " |")
    lines += ["", "## Same-seed differences from Full", "",
              "| Dataset | Variant | ΔWF1 | ΔACC | ΔMacro-F1 |", "| --- | --- | --- | --- | --- |"]
    for group in result["groups"]:
        if group["variant"] != "full":
            lines.append(f"| {group['dataset']} | {group['variant']} | " + " | ".join(_cell(group["paired_delta_to_full"][key], signed=True) for key in METRICS) + " |")
    for group in result["groups"]:
        lines += ["", f"## {group['dataset']} / {group['variant']}", "",
                  "| Seed | State | WF1 | ACC | Macro-F1 |", "| --- | --- | --- | --- | --- |"]
        notes = []
        for row in group["seeds"]:
            cells = [f"{row['metrics'][key] * 100:.2f}" if row["included"] else "—" for key in METRICS]
            lines.append(f"| {row['seed']} | {row['state']} | " + " | ".join(cells) + " |")
            if row.get("error") and not row["included"]:
                error = str(row["error"]).replace("\n", " ").replace("|", "\\|")
                notes.append(f"Seed {row['seed']}: {error}")
            if row.get("pairing_error"):
                notes.append(f"Seed {row['seed']} pairing: {row['pairing_error']}.")
        for note in notes:
            lines += ["", note]
        lines += ["", "| Class | F1 |", "| --- | --- |"]
        for name in group["class_names"]:
            lines.append(f"| {name} | {_cell(group['class_f1'][name])} |")
    lines += ["", "Per-seed confusion counts and their elementwise mean/sample SD are retained in the JSON. Rows are true classes and columns are predictions; class order is explicit. Classes with no true examples remain missing in class-level averages. No statistical significance or improvement claims are inferred.", ""]
    return "\n".join(lines)


def write_outputs(result: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
    paths = [json_path, markdown_path]
    if json_path.resolve() == markdown_path.resolve():
        raise ValueError("JSON and Markdown output paths must differ")
    # Reserve both destinations with O_EXCL before writing either. On failure,
    # remove only files created here; existing files are never overwritten.
    descriptors, created = [], []
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptors.append(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644))
            created.append(path)
        payloads = [json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", render_markdown(result)]
        for index, (descriptor, payload) in enumerate(zip(descriptors, payloads)):
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptors[index] = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
    except BaseException:
        for descriptor in descriptors:
            if descriptor is not None:
                os.close(descriptor)
        for path in created:
            path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    for path in (args.output_json, args.output_md):
        if path.exists() or path.is_symlink():
            parser.error(f"output already exists: {path}")
    result = analyze(args.plan, args.state)
    write_outputs(result, args.output_json, args.output_md)
    print(json.dumps({"complete": result["complete"], "groups": len(result["groups"]),
                      "output_json": str(args.output_json), "output_md": str(args.output_md)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
