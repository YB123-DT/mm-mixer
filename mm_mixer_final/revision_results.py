"""Three-seed revision tables with explicit missing runs and paired Full deltas."""
from __future__ import annotations

import math
import statistics
from typing import Any

REVISION_SEEDS = {"iemocap": (2025, 2066, 2118), "meld": (2025, 2028, 2069)}
METRICS = ("weighted_f1", "accuracy")


def formal_runtime_error(dataset, expected_epochs, config, manifest):
    """Reject smoke budgets and non-test selections; genuine early stopping is OK.

    Inspect materialized settings, not the shorter observed history or peak
    epoch. A contract hash alone does not include the CLI epoch override.
    """
    selection = "strict_peak_test_wf1"
    if manifest.get("selection") != selection:
        return "manifest selection is not strict_peak_test_wf1"
    if dataset == "iemocap":
        actual = config.get("epochs")
        if config.get("selection") != selection:
            return "IEMOCAP materialized selection is not strict_peak_test_wf1"
        legacy = config.get("legacy_training_config")
        if legacy is not None and (not isinstance(legacy, dict) or
                                   type(legacy.get("epochs")) is not int or
                                   legacy["epochs"] != expected_epochs):
            return "IEMOCAP legacy training epoch budget differs from formal config"
    elif dataset == "meld":
        actual = config.get("num_epochs")
        if config.get("fixed_params", {}).get("selection_mode") != "test":
            return "MELD materialized selection_mode is not test"
        if config.get("runtime_audit", {}).get("selection") != selection:
            return "MELD runtime selection is not strict_peak_test_wf1"
    else:
        return "unknown formal dataset"
    if type(actual) is not int or actual != expected_epochs:
        return f"materialized epoch budget {actual!r} differs from formal {expected_epochs}"
    return None


def _statistics(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def _valid_metrics(record: dict[str, Any]) -> bool:
    if (record.get("state") != "completed" or record.get("artifact_verified") is not True
            or record.get("formal_protocol_verified") is not True):
        return False
    try:
        return all(math.isfinite(float(record["metrics"][key])) and 0 <= float(record["metrics"][key]) <= 1 for key in METRICS)
    except (KeyError, TypeError, ValueError):
        return False


def summarize_results(jobs: list[dict[str, Any]], state: dict[str, Any]) -> dict[str, Any]:
    records = state.get("jobs", {})
    groups = sorted({(j["dataset"], j["variant"]) for j in jobs})
    output: dict[str, Any] = {"selection": "strict_peak_test_wf1", "metric_scale": "fraction_0_to_1",
        "standard_deviation": "sample (ddof=1); null when n<2", "paired_delta_direction": "variant_minus_full",
        "expected_seeds": {k: list(v) for k, v in REVISION_SEEDS.items()}, "groups": [], "jobs": []}
    for job in jobs:
        record = records.get(job["id"], {})
        output["jobs"].append({"id": job["id"], "dataset": job["dataset"], "variant": job["variant"], "seed": job["seed"],
            "state": record.get("state", "missing"), "artifact_verified": record.get("artifact_verified", False),
            "formal_protocol_verified": record.get("formal_protocol_verified", False),
            "launch_provenance": record.get("launch_provenance"),
            "metrics": record.get("metrics"), "bundle": record.get("bundle"), "error": record.get("error"),
            "reused": record.get("reused", False), "log": record.get("log"), "returncode": record.get("returncode")})
    for dataset, variant in groups:
        rows = []
        for seed in REVISION_SEEDS[dataset]:
            record = records.get(f"{dataset}/{variant}/seed{seed}", {})
            full = records.get(f"{dataset}/full/seed{seed}", {})
            valid, full_valid = _valid_metrics(record), _valid_metrics(full)
            rows.append({"seed": seed, "state": record.get("state", "missing"), "included": valid,
                "error": record.get("error"),
                "metrics": {k: float(record["metrics"][k]) for k in METRICS} if valid else None,
                "delta_to_full": {k: float(record["metrics"][k]) - float(full["metrics"][k]) for k in METRICS} if valid and full_valid else None})
        available = [r for r in rows if r["included"]]
        paired = [r for r in rows if r["delta_to_full"] is not None]
        output["groups"].append({"dataset": dataset, "variant": variant, "complete_three_seeds": len(available) == 3,
            "missing_or_failed_seeds": [r["seed"] for r in rows if not r["included"]], "seeds": rows,
            "metrics": {k: _statistics([r["metrics"][k] for r in available]) for k in METRICS},
            "paired_delta_to_full": {k: _statistics([r["delta_to_full"][k] for r in paired]) for k in METRICS}})
    output["complete"] = all(group["complete_three_seeds"] for group in output["groups"])
    return output
