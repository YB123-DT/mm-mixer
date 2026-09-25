from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RESULT_FIELDS = {
    "complete", "finite", "replay_exact", "config_sha256", "source_sha256",
    "checkpoint_sha256", "replay_sha256", "matched_delta_wf1",
    "matched_control", "matched_control_wf1",
}


def canonical_lock_sha256(lock: Mapping) -> str:
    content = {key: value for key, value in lock.items() if key != "lock_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_results_lock(lock: Mapping) -> None:
    if set(lock) != {"schema_version", "results", "lock_sha256"}:
        raise ValueError("results lock has unexpected or missing top-level fields")
    if lock["schema_version"] != 1:
        raise ValueError("unsupported results lock schema")
    if lock["lock_sha256"] != canonical_lock_sha256(lock):
        raise ValueError("results lock canonical SHA mismatch")
    expected = {f"{i:02d}" for i in range(1, 47)}
    if set(lock["results"]) != expected:
        raise ValueError("results lock must contain exactly experiments 01-46")
    for experiment_id, result in lock["results"].items():
        if set(result) != _RESULT_FIELDS:
            raise ValueError(f"{experiment_id}: invalid result fields")
        if not all(result[key] is True for key in ("complete", "finite", "replay_exact")):
            raise ValueError(f"{experiment_id}: incomplete, non-finite, or replay-inexact")
        for key in ("config_sha256", "source_sha256", "checkpoint_sha256", "replay_sha256"):
            if (not isinstance(result[key], str) or not _SHA.fullmatch(result[key])
                    or len(set(result[key])) == 1):
                raise ValueError(f"{experiment_id}: malformed {key}")
        delta = result["matched_delta_wf1"]
        if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta):
            raise ValueError(f"{experiment_id}: matched delta must be finite")
        expected_control = "DIALOGUE_NULL" if experiment_id in {"41","42","43","44"} else (
            "S1P" if experiment_id in {"13","14","15","16"} else "S0")
        if result["matched_control"] != expected_control:
            raise ValueError(f"{experiment_id}: matched control must be {expected_control}")
        score=result["matched_control_wf1"]
        if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):
            raise ValueError(f"{experiment_id}: matched control WF1 must be finite")
