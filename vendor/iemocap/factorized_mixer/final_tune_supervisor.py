"""Fail-closed supervisor for immutable final-tuning queue manifests.

Candidate generation belongs to :mod:`factorized_mixer.final_tune_queue`.
This module only consumes a prepared manifest after independently proving that
its roster, configs, smoke evidence, artifacts, and process ownership still
match the final ``HO_WO_TAV`` contract.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from .registry import FINAL_TUNE_VARIANTS
from .runner import MixerRunConfig, RAWAUX_FILL53_FEATURES, normalize_config


GPU_IDS = (5, 6, 7)
CPU_THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
DEFAULT_PYTHON = "/data2/yb/reproduction_envs/s0/bin/python"
LEGACY_SMOKE_SCHEMA_VERSION = 1
SMOKE_SCHEMA_VERSION = 2
SMOKE_MEMORY_BUDGET_MIB = 30 * 1024


@dataclass(frozen=True)
class PreparedJob:
    """One immutable, manifest-backed final-tuning job."""

    round_name: str
    experiment_id: str
    run_id: str
    seed: int
    gpu_id: int
    config_path: Path
    output_root: Path
    log_path: Path
    tuning_overrides: Mapping[str, object]
    config_sha256: str
    baseline_config_path: Path | None = None
    baseline_run_dir: Path | None = None


@dataclass(frozen=True)
class CompletedArtifact:
    run_dir: Path
    weighted_f1: float


@dataclass(frozen=True)
class SupervisorResult:
    exit_code: int
    phase: str
    total: int
    complete: int
    running: int
    failed: int
    waiting: int


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _rawaux_feature_sha256() -> str:
    path = Path(RAWAUX_FILL53_FEATURES)
    if not path.is_file():
        raise ValueError(f"exact RawAux feature artifact is missing: {path}")
    return _sha256(path)


def _config_sha256(config: Mapping[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _required(mapping: Mapping[str, object], key: str, label: str) -> object:
    if key not in mapping:
        raise ValueError(f"{label} is missing {key}")
    return mapping[key]


def _normalised_config(path: Path, label: str) -> dict[str, object]:
    try:
        return normalize_config(_read_json(path, label))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a valid MixerRunConfig: {path}") from error


def _queue_module():
    """Import queue definitions lazily to keep process supervision isolated."""
    from . import final_tune_queue
    return final_tune_queue


def _canonical_job_config(job: PreparedJob) -> dict[str, object]:
    """Reconstruct the only configuration this queue is allowed to execute."""
    queue = _queue_module()
    if job.round_name in queue.ROUND_PREFIXES:
        if (not job.experiment_id.startswith(queue.ROUND_PREFIXES[job.round_name])
                or job.experiment_id not in FINAL_TUNE_VARIANTS
                or dict(FINAL_TUNE_VARIANTS[job.experiment_id]) != dict(job.tuning_overrides)):
            raise ValueError("job does not belong to its canonical final-tune override family")
        config = MixerRunConfig(
            experiment_id=job.experiment_id,
            seed=job.seed,
            tuning_overrides=dict(job.tuning_overrides),
        )
    elif job.round_name == "R4":
        if not job.experiment_id.startswith("TUNE_R4_") or not job.tuning_overrides:
            raise ValueError("R4 job does not carry a canonical combined override")
        config = MixerRunConfig(
            experiment_id=job.experiment_id,
            seed=job.seed,
            run_id=job.run_id,
            tuning_overrides=dict(job.tuning_overrides),
        )
    else:
        raise ValueError(f"unknown final-tune round: {job.round_name}")
    expected = normalize_config(config)
    pinned = {
        "epochs": 100,
        "batch_size": 32,
        "gradient_accumulation": 1,
        "scheduler_mode": "legacy",
        "feature_protocol": "rawaux_fill53",
        "features": RAWAUX_FILL53_FEATURES,
    }
    if any(expected.get(key) != value for key, value in pinned.items()):
        raise ValueError("canonical final-tune config violates the pinned training contract")
    return expected


def _validate_immutable_config(job: PreparedJob) -> dict[str, object]:
    expected = _canonical_job_config(job)
    try:
        observed = _normalised_config(job.config_path, "prepared job config")
    except ValueError as error:
        raise ValueError("prepared job config differs from the canonical final-tune config") from error
    if observed != expected:
        raise ValueError("prepared job config differs from the canonical final-tune config")
    if _config_sha256(observed) != job.config_sha256:
        raise ValueError("prepared job config SHA differs from its manifest row")
    return expected


def _optional_path(row: Mapping[str, object], key: str) -> Path | None:
    value = row.get(key)
    return None if value is None else Path(str(value)).resolve()


def _prepared_job(row: Mapping[str, object], *, round_root: Path,
                  round_name: str) -> PreparedJob:
    label = "prepared queue job"
    experiment_id = _required(row, "experiment_id", label)
    run_id = _required(row, "run_id", label)
    seed = _required(row, "seed", label)
    gpu_id = _required(row, "gpu_id", label)
    config_path = Path(str(_required(row, "config", label))).resolve()
    output_root = Path(str(_required(row, "output_root", label))).resolve()
    log_path = Path(str(_required(row, "log", label))).resolve()
    tuning_overrides = _required(row, "tuning_overrides", label)
    config_sha256 = _required(row, "config_sha256", label)
    thread_caps = _required(row, "thread_caps", label)
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("prepared queue job has an invalid experiment_id")
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ValueError("prepared queue job has an invalid run_id")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("prepared queue job has an invalid seed")
    if gpu_id not in GPU_IDS:
        raise ValueError("prepared queue job uses a non-approved GPU")
    if output_root != round_root:
        raise ValueError("prepared queue job does not use the canonical round root")
    if config_path != (round_root / "configs" / f"{run_id}.json").resolve():
        raise ValueError("prepared queue job config is outside the canonical configs directory")
    if log_path != (round_root / "logs" / f"{run_id}_gpu{gpu_id}.log").resolve():
        raise ValueError("prepared queue job log is outside the canonical logs directory")
    if not isinstance(tuning_overrides, dict):
        raise ValueError("prepared queue job tuning_overrides must be an object")
    if not isinstance(config_sha256, str) or len(config_sha256) != 64:
        raise ValueError("prepared queue job is missing a valid config SHA")
    if not isinstance(thread_caps, dict) or thread_caps != CPU_THREAD_CAPS:
        raise ValueError("prepared queue job does not pin all required CPU thread pools")
    if not config_path.is_file():
        raise ValueError(f"prepared queue job config is missing: {config_path}")
    job = PreparedJob(
        round_name=round_name,
        experiment_id=experiment_id,
        run_id=run_id,
        seed=seed,
        gpu_id=gpu_id,
        config_path=config_path,
        output_root=output_root,
        log_path=log_path,
        tuning_overrides=dict(tuning_overrides),
        config_sha256=config_sha256,
        baseline_config_path=_optional_path(row, "baseline_config_source"),
        baseline_run_dir=_optional_path(row, "baseline_run_dir"),
    )
    _validate_immutable_config(job)
    return job


def _job_signature(job: PreparedJob) -> tuple[object, ...]:
    return (
        job.experiment_id, job.run_id, job.seed, job.gpu_id,
        str(job.config_path), str(job.output_root), str(job.log_path),
        dict(job.tuning_overrides),
    )


def _validate_screen_roster(round_root: Path, round_name: str,
                            jobs: Sequence[PreparedJob]) -> None:
    expected_jobs = _queue_module().build_round_jobs(round_root.parent, round_name)
    expected = {
        item.run_id: (
            item.experiment_id, item.run_id, item.seed, item.gpu_id,
            str(item.config_path.resolve()), str(item.output_root.resolve()),
            str(item.log_path.resolve()), dict(item.tuning_overrides),
        )
        for item in expected_jobs
    }
    observed = {job.run_id: _job_signature(job) for job in jobs}
    if observed != expected:
        raise ValueError("prepared queue roster differs from its deterministic canonical screen")


def _r4_experiment_id(component_ids: Sequence[str]) -> str:
    return "TUNE_R4_" + "__".join(name.removeprefix("TUNE_") for name in component_ids)


def _validate_r4_roster(manifest: Mapping[str, object], jobs: Sequence[PreparedJob]) -> None:
    queue = _queue_module()
    winners = manifest.get("screen_winners")
    seeds = manifest.get("paired_midrange_baseline_seeds")
    expected_counts = {"R1": 3, "R2": 2, "R3": 2}
    if not isinstance(winners, dict) or set(winners) != set(expected_counts):
        raise ValueError("R4 manifest is missing its canonical screen winners")
    if not isinstance(seeds, list) or len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("R4 manifest must name exactly four paired baseline seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
        raise ValueError("R4 paired baseline seeds are invalid")
    winner_ids: dict[str, list[str]] = {}
    for round_name, count in expected_counts.items():
        values = winners[round_name]
        if not isinstance(values, list) or len(values) != count or len(set(values)) != count:
            raise ValueError(f"R4 winner list for {round_name} is invalid")
        prefix = queue.ROUND_PREFIXES[round_name]
        if any(not isinstance(value, str) or not value.startswith(prefix)
               or value not in FINAL_TUNE_VARIANTS for value in values):
            raise ValueError(f"R4 winner list for {round_name} is not a canonical screen subset")
        winner_ids[round_name] = list(values)
    pending: list[tuple[str, int, dict[str, object]]] = []
    for r1 in winner_ids["R1"]:
        for r2 in winner_ids["R2"]:
            for r3 in winner_ids["R3"]:
                components = (r1, r2, r3)
                overrides: dict[str, object] = {}
                for component in components:
                    overlap = set(overrides).intersection(FINAL_TUNE_VARIANTS[component])
                    if overlap:
                        raise ValueError("R4 components overlap in their tuning controls")
                    overrides.update(FINAL_TUNE_VARIANTS[component])
                experiment_id = _r4_experiment_id(components)
                for seed in seeds:
                    pending.append((experiment_id, seed, overrides))
    if len(pending) != 48:
        raise ValueError("R4 canonical roster did not materialize 48 jobs")
    random.Random(queue.SHUFFLE_SEED).shuffle(pending)
    expected = {}
    for index, (experiment_id, seed, overrides) in enumerate(pending):
        gpu_id = GPU_IDS[index % len(GPU_IDS)]
        run_id = f"{experiment_id}_seed{seed}"
        expected[run_id] = (experiment_id, seed, gpu_id, dict(overrides))
    observed = {job.run_id: (job.experiment_id, job.seed, job.gpu_id, dict(job.tuning_overrides))
                for job in jobs}
    if observed != expected:
        raise ValueError("R4 roster, seed pairing, or GPU assignment differs from its manifest winners")
    for job in jobs:
        if job.baseline_config_path is None or job.baseline_run_dir is None:
            raise ValueError("R4 job lacks paired-baseline provenance")
        try:
            queue._validate_baseline_seed(queue.BaselineSeed(
                seed=job.seed,
                config_path=job.baseline_config_path,
                run_dir=job.baseline_run_dir,
            ))
        except (TypeError, ValueError) as error:
            raise ValueError("R4 paired baseline no longer proves strict completion") from error


def load_prepared_round(round_root: Path) -> list[PreparedJob]:
    """Load an exact queue manifest and reject every form of contract drift."""
    canonical_root = Path(round_root).resolve()
    manifest = _read_json(canonical_root / "queue_manifest.json", "queue manifest")
    round_name = manifest.get("round")
    queue = _queue_module()
    if not isinstance(round_name, str) or round_name not in (*queue.ROUND_PREFIXES, "R4"):
        raise ValueError("queue manifest has an invalid round")
    if canonical_root.name != round_name:
        raise ValueError("queue manifest round does not match the canonical round root")
    if manifest.get("shuffle_seed") != queue.SHUFFLE_SEED:
        raise ValueError("queue manifest shuffle seed differs from the canonical allocation")
    if manifest.get("gpu_ids") != list(GPU_IDS):
        raise ValueError("queue manifest must name exactly GPUs 5, 6, and 7")
    if manifest.get("cpu_thread_caps") != CPU_THREAD_CAPS:
        raise ValueError("queue manifest does not pin all required CPU thread pools")
    rows = manifest.get("jobs")
    if not isinstance(rows, list) or len(rows) != 48:
        raise ValueError("prepared queue manifest must contain exactly 48 jobs")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("prepared queue manifest contains a non-object job")
    jobs = [_prepared_job(row, round_root=canonical_root, round_name=round_name) for row in rows]
    if len({job.run_id for job in jobs}) != len(jobs):
        raise ValueError("prepared queue manifest contains duplicate run ids")
    if [sum(job.gpu_id == gpu for job in jobs) for gpu in GPU_IDS] != [16, 16, 16]:
        raise ValueError("prepared queue manifest must assign exactly 16 jobs to each approved GPU")
    if round_name in queue.ROUND_PREFIXES:
        _validate_screen_roster(canonical_root, round_name, jobs)
    else:
        _validate_r4_roster(manifest, jobs)
    return jobs


def job_environment(job: PreparedJob, inherited: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pin every worker to its assigned physical GPU and one CPU thread pool."""
    environment = dict(os.environ if inherited is None else inherited)
    environment.update(CPU_THREAD_CAPS)
    environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
    return environment


def validate_completed_job(job: PreparedJob) -> CompletedArtifact:
    """Require strict, immutable artifacts before a job may count as complete."""
    expected_config = _validate_immutable_config(job)
    run_dir = job.output_root / "runs" / job.run_id
    active_pids = _matching_live_runner_pids(
        config_path=job.config_path, output_root=job.output_root
    )
    if active_pids:
        raise ValueError(
            "active factorized_mixer.runner still matches the immutable launch "
            f"contract (PIDs: {', '.join(map(str, active_pids))})"
        )
    status = _read_json(run_dir / "status.json", "status")
    metrics = _read_json(run_dir / "metrics.json", "metrics")
    manifest = _read_json(run_dir / "manifest.json", "manifest")
    observed_config = _normalised_config(run_dir / "config.json", "run config")
    if observed_config != expected_config:
        raise ValueError("run config differs from the immutable queued config")
    if status.get("state") != "complete":
        raise ValueError("status does not report complete")
    for label, payload in (("status", status), ("metrics", metrics), ("manifest", manifest)):
        if payload.get("fresh_strict_replay_exact") is not True:
            raise ValueError(f"{label} does not prove a fresh strict replay")
    checkpoint = run_dir / "best_peak_test_state_dict.pt"
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint is missing: {checkpoint}")
    checkpoint_sha = _sha256(checkpoint)
    if status.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("status checkpoint SHA does not match the checkpoint")
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("manifest checkpoint SHA does not match the checkpoint")
    if manifest.get("experiment_id") != job.experiment_id:
        raise ValueError("manifest experiment identity differs from the queued job")
    if manifest.get("run_id") != job.run_id:
        raise ValueError("manifest run identity differs from the queued job")
    if manifest.get("seed") != job.seed:
        raise ValueError("manifest seed differs from the queued job")
    if manifest.get("tuning_overrides") != dict(job.tuning_overrides):
        raise ValueError("manifest tuning overrides differ from the queued job")
    if manifest.get("feature_artifact_sha256") != _rawaux_feature_sha256():
        raise ValueError("manifest feature artifact SHA does not match exact RawAux fill53 features")
    selected = metrics.get("selected")
    if not isinstance(selected, dict) or "weighted_f1" not in selected:
        raise ValueError("metrics are missing selected weighted_f1")
    weighted_f1 = float(selected["weighted_f1"])
    if not math.isfinite(weighted_f1):
        raise ValueError("metrics selected weighted_f1 is not finite")
    return CompletedArtifact(run_dir=run_dir, weighted_f1=weighted_f1)


def gpu_compute_processes() -> dict[int, set[int]] | None:
    """Return every observed compute PID on each approved GPU, or ``None``."""
    try:
        gpu_lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        ).splitlines()
        uuid_to_gpu: dict[str, int] = {}
        for line in gpu_lines:
            if not line.strip() or "," not in line:
                raise ValueError("malformed GPU identity row")
            index_text, uuid = line.split(",", 1)
            uuid_to_gpu[uuid.strip()] = int(index_text.strip())
        if not all(gpu in uuid_to_gpu.values() for gpu in GPU_IDS):
            raise ValueError("one or more approved GPUs are absent")
        app_lines = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        ).splitlines()
        processes = {gpu: set() for gpu in GPU_IDS}
        for line in app_lines:
            normalized = line.strip()
            if not normalized or normalized.lower().startswith("no running processes"):
                continue
            if "," not in normalized:
                raise ValueError("malformed compute-process row")
            uuid, pid_text = normalized.split(",", 1)
            gpu = uuid_to_gpu.get(uuid.strip())
            pid = int(pid_text.strip())
            if gpu in processes:
                processes[gpu].add(pid)
        return processes
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def gpu_process_counts() -> dict[int, int] | None:
    """Compatibility helper; scheduling itself uses PID-level isolation."""
    processes = gpu_compute_processes()
    return None if processes is None else {gpu: len(pids) for gpu, pids in processes.items()}


def _proc_cmdline(pid: int) -> tuple[str, ...] | None:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    tokens = tuple(token.decode(errors="surrogateescape") for token in raw.split(b"\0") if token)
    return tokens or None


def _proc_cwd(pid: int) -> Path | None:
    """Resolve a process working directory for relative argv path arguments."""
    try:
        return Path(os.readlink(Path("/proc") / str(pid) / "cwd"))
    except OSError:
        return None


def _proc_start_ticks(pid: int) -> str | None:
    try:
        raw = (Path("/proc") / str(pid) / "stat").read_text()
    except OSError:
        return None
    if ")" not in raw:
        return None
    fields_after_comm = raw.rsplit(")", 1)[1].split()
    return fields_after_comm[19] if len(fields_after_comm) > 19 else None


def _capture_process_identity(job: PreparedJob, pid: int) -> dict[str, object] | None:
    start_ticks = _proc_start_ticks(pid)
    command = _proc_cmdline(pid)
    if start_ticks is None or command is None:
        return None
    return {
        "pid": pid,
        "start_ticks": start_ticks,
        "config_path": str(job.config_path),
        "output_root": str(job.output_root),
    }


def _runner_command_matches(command: Sequence[str], *, config_path: Path,
                            output_root: Path, cwd: Path | None = None) -> bool:
    """Match only the exact runner module and immutable launch arguments."""
    try:
        module_index = command.index("-m")
    except ValueError:
        return False
    if module_index + 1 >= len(command) or command[module_index + 1] != "factorized_mixer.runner":
        return False

    def single_value(flag: str) -> str | None:
        values = [command[index + 1] for index, token in enumerate(command[:-1])
                  if token == flag]
        return values[0] if len(values) == 1 else None

    config_value = single_value("--config")
    output_root_value = single_value("--output-root")
    if config_value is None or output_root_value is None:
        return False
    actual_config = _canonical_launch_path(config_value, cwd=cwd)
    actual_output_root = _canonical_launch_path(output_root_value, cwd=cwd)
    return (actual_config == config_path.resolve()
            and actual_output_root == output_root.resolve())


def _canonical_launch_path(value: str, *, cwd: Path | None) -> Path | None:
    candidate = Path(value)
    if not candidate.is_absolute():
        if cwd is None:
            return None
        candidate = cwd / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return None


def _matching_live_runner_pids(*, config_path: Path, output_root: Path) -> tuple[int, ...]:
    """Return only live runner PIDs for this exact immutable launch contract."""
    try:
        process_entries = tuple(Path("/proc").iterdir())
    except OSError:
        return ()
    matches = []
    for process_entry in process_entries:
        try:
            pid = int(process_entry.name)
        except ValueError:
            continue
        command = _proc_cmdline(pid)
        if command is not None and _runner_command_matches(
                command, config_path=config_path, output_root=output_root, cwd=_proc_cwd(pid)):
            matches.append(pid)
    return tuple(sorted(matches))


def _owned_process_identity_matches(job: PreparedJob, item: Mapping[str, object]) -> bool:
    pid = item.get("pid")
    identity = item.get("process_identity")
    if not isinstance(pid, int) or pid <= 0 or not isinstance(identity, dict):
        return False
    if identity.get("pid") != pid or identity.get("config_path") != str(job.config_path):
        return False
    if identity.get("output_root") != str(job.output_root):
        return False
    start_ticks = _proc_start_ticks(pid)
    if start_ticks is None or start_ticks != identity.get("start_ticks"):
        return False
    command = _proc_cmdline(pid)
    return command is not None and _runner_command_matches(
        command, config_path=job.config_path, output_root=job.output_root, cwd=_proc_cwd(pid)
    )


def _state_path(round_root: Path) -> Path:
    return round_root / "supervisor" / "state.json"


def _initial_state(jobs: Sequence[PreparedJob], *, round_root: Path,
                   safe_per_gpu_limit: int, max_attempts: int) -> dict[str, Any]:
    return {
        "version": 2,
        "round_root": str(round_root),
        "gpu_ids": list(GPU_IDS),
        "safe_per_gpu_limit": safe_per_gpu_limit,
        "max_attempts": max_attempts,
        "phase": "starting",
        "reason": None,
        "foreign_gpu_pids": {str(gpu): [] for gpu in GPU_IDS},
        "jobs": {
            job.run_id: {
                "experiment_id": job.experiment_id,
                "gpu_id": job.gpu_id,
                "attempts": 0,
                "pid": None,
                "process_identity": None,
                "status": "pending",
                "last_exit_code": None,
            }
            for job in jobs
        },
    }


def _load_state(jobs: Sequence[PreparedJob], *, round_root: Path,
                safe_per_gpu_limit: int, max_attempts: int) -> dict[str, Any]:
    path = _state_path(round_root)
    if not path.exists():
        return _initial_state(
            jobs,
            round_root=round_root,
            safe_per_gpu_limit=safe_per_gpu_limit,
            max_attempts=max_attempts,
        )
    state = _read_json(path, "supervisor state")
    expected_ids = {job.run_id for job in jobs}
    if set(state.get("jobs", {})) != expected_ids:
        raise ValueError("supervisor state does not match the prepared queue manifest")
    if state.get("gpu_ids") != list(GPU_IDS):
        raise ValueError("supervisor state uses non-approved GPUs")
    if state.get("round_root") != str(round_root):
        raise ValueError("supervisor state belongs to a different round root")
    for job in jobs:
        item = state["jobs"][job.run_id]
        if not isinstance(item, dict) or item.get("experiment_id") != job.experiment_id:
            raise ValueError("supervisor state job identity differs from the manifest")
        if item.get("gpu_id") != job.gpu_id:
            raise ValueError("supervisor state GPU assignment differs from the manifest")
        item.setdefault("process_identity", None)
        item.setdefault("pid", None)
        item.setdefault("last_exit_code", None)
    return state


def _counts_from_state(state: Mapping[str, Any]) -> tuple[int, int, int]:
    jobs = state["jobs"]
    complete = sum(item["status"] == "complete" for item in jobs.values())
    running = sum(item["status"] == "running" for item in jobs.values())
    failed = sum(item["status"] == "failed" for item in jobs.values())
    return complete, running, failed


def _persist_state(round_root: Path, state: dict[str, Any]) -> None:
    complete, running, failed = _counts_from_state(state)
    state.update({
        "total": len(state["jobs"]),
        "complete": complete,
        "running": running,
        "failed": failed,
        "waiting": len(state["jobs"]) - complete - running - failed,
        "updated_at_unix": time.time(),
    })
    supervisor_root = _state_path(round_root).parent
    _atomic_json(_state_path(round_root), state)
    _atomic_json(supervisor_root / "attempts.json", {
        run_id: item["attempts"] for run_id, item in state["jobs"].items()
    })
    _atomic_json(supervisor_root / "pids.json", {
        run_id: item["pid"] for run_id, item in state["jobs"].items()
    })


def _close_process_log(process: object) -> None:
    stream = getattr(process, "_final_tune_log", None)
    if stream is not None:
        stream.close()


def launch_job(job: PreparedJob, *, project_root: Path,
               python_executable: str) -> subprocess.Popen[str]:
    """Launch exactly one immutable worker; the caller records its identity."""
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log = job.log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                python_executable,
                "-u",
                "-m",
                "factorized_mixer.runner",
                "--config",
                str(job.config_path),
                "--output-root",
                str(job.output_root),
            ],
            cwd=project_root,
            env=job_environment(job),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except BaseException:
        log.close()
        raise
    setattr(process, "_final_tune_log", log)
    return process


def _invalidate_completion(item: dict[str, object], max_attempts: int) -> None:
    item["pid"] = None
    item["process_identity"] = None
    item["last_exit_code"] = None
    item["status"] = "failed" if int(item["attempts"]) >= max_attempts else "pending"


def _mark_completed_artifacts(state: dict[str, Any], jobs: Sequence[PreparedJob],
                              max_attempts: int) -> None:
    """Revalidate every alleged completion on every loop; state is never proof."""
    for job in jobs:
        item = state["jobs"][job.run_id]
        try:
            validate_completed_job(job)
        except ValueError:
            if item["status"] == "complete":
                _invalidate_completion(item, max_attempts)
            continue
        item.update({
            "status": "complete",
            "pid": None,
            "process_identity": None,
            "last_exit_code": 0,
        })


def _reconcile_running(state: dict[str, Any], local_processes: dict[str, object],
                       jobs_by_id: Mapping[str, PreparedJob], max_attempts: int) -> None:
    for run_id, process in list(local_processes.items()):
        exit_code = process.poll()
        if exit_code is None:
            continue
        _close_process_log(process)
        del local_processes[run_id]
        item = state["jobs"][run_id]
        item["pid"] = None
        item["process_identity"] = None
        item["last_exit_code"] = int(exit_code)
        try:
            validate_completed_job(jobs_by_id[run_id])
        except ValueError:
            item["status"] = "failed" if int(item["attempts"]) >= max_attempts else "pending"
        else:
            item.update({
                "status": "complete",
                "pid": None,
                "process_identity": None,
                "last_exit_code": 0,
            })
    for run_id, item in state["jobs"].items():
        if item["status"] != "running" or run_id in local_processes:
            continue
        if _owned_process_identity_matches(jobs_by_id[run_id], item):
            continue
        item["pid"] = None
        item["process_identity"] = None
        item["last_exit_code"] = None
        try:
            validate_completed_job(jobs_by_id[run_id])
        except ValueError:
            item["status"] = "failed" if int(item["attempts"]) >= max_attempts else "pending"
        else:
            item.update({
                "status": "complete",
                "pid": None,
                "process_identity": None,
                "last_exit_code": 0,
            })


def _owned_pids_by_gpu(state: Mapping[str, Any], jobs_by_id: Mapping[str, PreparedJob],
                       local_processes: Mapping[str, object]) -> dict[int, set[int]]:
    owned = {gpu: set() for gpu in GPU_IDS}
    for run_id, item in state["jobs"].items():
        if item["status"] != "running":
            continue
        pid = item.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            continue
        if run_id in local_processes:
            if local_processes[run_id].poll() is None:
                owned[jobs_by_id[run_id].gpu_id].add(pid)
            continue
        if _owned_process_identity_matches(jobs_by_id[run_id], item):
            owned[jobs_by_id[run_id].gpu_id].add(pid)
    return owned


def _mark_exhausted(state: dict[str, Any], max_attempts: int) -> None:
    for item in state["jobs"].values():
        if item["status"] == "pending" and int(item["attempts"]) >= max_attempts:
            item["status"] = "failed"


def _result(state: Mapping[str, Any], *, exit_code: int, phase: str) -> SupervisorResult:
    complete, running, failed = _counts_from_state(state)
    return SupervisorResult(
        exit_code=exit_code,
        phase=phase,
        total=len(state["jobs"]),
        complete=complete,
        running=running,
        failed=failed,
        waiting=len(state["jobs"]) - complete - running - failed,
    )


def _validate_limit(safe_per_gpu_limit: int) -> None:
    if isinstance(safe_per_gpu_limit, bool) or not isinstance(safe_per_gpu_limit, int):
        raise ValueError("safe_per_gpu_limit must be an integer")
    if not 1 <= safe_per_gpu_limit <= 16:
        raise ValueError("safe_per_gpu_limit must be between 1 and 16")


def _smoke_gpu_ids(payload: Mapping[str, object]) -> tuple[int, ...]:
    """Return the GPU lanes certified by either supported smoke schema."""
    schema_version = payload.get("schema_version")
    gpu_ids = payload.get("gpu_ids")
    if schema_version == LEGACY_SMOKE_SCHEMA_VERSION:
        if gpu_ids != list(GPU_IDS):
            raise ValueError("legacy smoke evidence does not cover exactly GPUs 5, 6, and 7")
        return GPU_IDS
    if schema_version != SMOKE_SCHEMA_VERSION:
        raise ValueError("smoke evidence schema is invalid")
    if (not isinstance(gpu_ids, list) or not gpu_ids
            or any(isinstance(gpu, bool) or not isinstance(gpu, int) for gpu in gpu_ids)
            or tuple(gpu_ids) != tuple(sorted(set(gpu_ids)))
            or not set(gpu_ids).issubset(GPU_IDS)):
        raise ValueError("partial smoke evidence has invalid certified GPU IDs")
    return tuple(gpu_ids)


def _certified_gpu_subset(certified_gpu_ids: set[int] | None,
                          evidence_gpu_ids: Sequence[int]) -> tuple[int, ...]:
    if certified_gpu_ids is None:
        return tuple(evidence_gpu_ids)
    if not isinstance(certified_gpu_ids, set) or not certified_gpu_ids:
        raise ValueError("certified_gpu_ids must be a non-empty set")
    if any(isinstance(gpu, bool) or not isinstance(gpu, int) for gpu in certified_gpu_ids):
        raise ValueError("certified_gpu_ids must contain GPU integers")
    if not certified_gpu_ids.issubset(set(evidence_gpu_ids)):
        raise ValueError("certified_gpu_ids are not covered by smoke evidence")
    return tuple(sorted(certified_gpu_ids))


def _smoke_values(payload: Mapping[str, object], key: str,
                  gpu_ids: Sequence[int]) -> dict[int, float]:
    raw = payload.get(key)
    if not isinstance(raw, dict) or set(raw) != {str(gpu) for gpu in gpu_ids}:
        raise ValueError(f"smoke evidence has invalid {key}")
    values: dict[int, float] = {}
    for gpu in gpu_ids:
        value = raw[str(gpu)]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"smoke evidence has a non-finite {key} value")
        values[gpu] = float(value)
    return values


def validate_smoke_evidence(smoke_path: Path, *, round_root: Path,
                            safe_per_gpu_limit: int,
                            certified_gpu_ids: set[int] | None = None) -> dict[str, object]:
    """Verify a round-bound memory measurement before any full worker launches."""
    evidence = _read_json(Path(smoke_path), "smoke evidence")
    evidence_gpu_ids = _smoke_gpu_ids(evidence)
    selected_gpu_ids = _certified_gpu_subset(certified_gpu_ids, evidence_gpu_ids)
    if evidence.get("round") != round_root.name:
        raise ValueError("smoke evidence does not match this queue round")
    if evidence.get("feature_protocol") != "rawaux_fill53":
        raise ValueError("smoke evidence does not match the RawAux feature protocol")
    if evidence.get("queue_manifest_sha256") != _sha256(round_root / "queue_manifest.json"):
        raise ValueError("smoke evidence does not match the prepared queue manifest")
    worker_mib = _smoke_values(evidence, "worker_mib", evidence_gpu_ids)
    overhead_mib = _smoke_values(evidence, "overhead_mib", evidence_gpu_ids)
    if any(value <= 0 for value in worker_mib.values()) or any(value < 0 for value in overhead_mib.values()):
        raise ValueError("smoke evidence memory measurements are invalid")
    derived_caps: dict[int, int] = {}
    for gpu in evidence_gpu_ids:
        headroom = SMOKE_MEMORY_BUDGET_MIB - overhead_mib[gpu]
        if headroom <= 0:
            raise ValueError("smoke evidence overhead leaves no safe GPU memory")
        derived_caps[gpu] = min(16, int(headroom // worker_mib[gpu]))
    recorded = evidence.get("safe_per_gpu")
    if evidence.get("schema_version") == LEGACY_SMOKE_SCHEMA_VERSION:
        if isinstance(recorded, bool) or not isinstance(recorded, int):
            raise ValueError("smoke evidence safe_per_gpu is invalid")
        recorded_caps = {gpu: recorded for gpu in evidence_gpu_ids}
    else:
        if not isinstance(recorded, dict) or set(recorded) != {str(gpu) for gpu in evidence_gpu_ids}:
            raise ValueError("partial smoke evidence safe_per_gpu is invalid")
        recorded_caps = {}
        for gpu in evidence_gpu_ids:
            value = recorded[str(gpu)]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError("partial smoke evidence safe_per_gpu is invalid")
            recorded_caps[gpu] = value
    for gpu in evidence_gpu_ids:
        if not 1 <= recorded_caps[gpu] <= derived_caps[gpu]:
            raise ValueError("smoke evidence safe_per_gpu is invalid")
    if any(safe_per_gpu_limit > recorded_caps[gpu] for gpu in selected_gpu_ids):
        raise ValueError("requested per-GPU limit exceeds matching smoke evidence")
    return evidence


def _blocked_result(round_root: Path, state: dict[str, Any], reason: str) -> SupervisorResult:
    state.update({"phase": "blocked", "reason": reason})
    _persist_state(round_root, state)
    return _result(state, exit_code=2, phase="blocked")


def run_supervisor(*, round_root: Path, project_root: Path,
                   python_executable: str = DEFAULT_PYTHON,
                   safe_per_gpu_limit: int, smoke_path: Path | None = None,
                   certified_gpu_ids: set[int] | None = None,
                   max_attempts: int = 3, poll_seconds: float = 30.0,
                   max_cycles: int | None = None) -> SupervisorResult:
    """Run until strict completion, a fail-closed block, or exhausted retries."""
    _validate_limit(safe_per_gpu_limit)
    if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer")
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative")
    canonical_root = Path(round_root).resolve()
    canonical_project = Path(project_root).resolve()
    jobs = load_prepared_round(canonical_root)
    jobs_by_id = {job.run_id: job for job in jobs}
    state = _load_state(
        jobs,
        round_root=canonical_root,
        safe_per_gpu_limit=safe_per_gpu_limit,
        max_attempts=max_attempts,
    )
    state["safe_per_gpu_limit"] = safe_per_gpu_limit
    state["max_attempts"] = max_attempts
    evidence_path = (Path(smoke_path) if smoke_path is not None
                     else canonical_root.parent / "smoke" / f"{canonical_root.name}.json")
    try:
        evidence = validate_smoke_evidence(
            evidence_path,
            round_root=canonical_root,
            safe_per_gpu_limit=safe_per_gpu_limit,
            certified_gpu_ids=certified_gpu_ids,
        )
    except ValueError as error:
        return _blocked_result(canonical_root, state, f"smoke evidence rejected: {error}")
    certified_gpu_ids = _certified_gpu_subset(certified_gpu_ids, _smoke_gpu_ids(evidence))
    state["smoke_evidence"] = {
        "path": str(evidence_path.resolve()),
        "safe_per_gpu": evidence["safe_per_gpu"],
        "queue_manifest_sha256": evidence["queue_manifest_sha256"],
        "certified_gpu_ids": list(certified_gpu_ids),
    }
    state["certified_gpu_ids"] = list(certified_gpu_ids)
    local_processes: dict[str, object] = {}
    cycles = 0
    while True:
        _mark_completed_artifacts(state, jobs, max_attempts)
        _reconcile_running(state, local_processes, jobs_by_id, max_attempts)
        _mark_exhausted(state, max_attempts)
        complete, running, failed = _counts_from_state(state)
        if complete == len(jobs):
            state.update({"phase": "complete", "reason": None})
            _persist_state(canonical_root, state)
            return _result(state, exit_code=0, phase="complete")
        if complete + failed == len(jobs) and running == 0:
            state.update({"phase": "failed", "reason": "all remaining jobs exhausted their attempts"})
            _persist_state(canonical_root, state)
            return _result(state, exit_code=1, phase="failed")
        observed = gpu_compute_processes()
        if observed is None:
            return _blocked_result(
                canonical_root,
                state,
                "nvidia-smi GPU PID query failed; refusing to launch",
            )
        owned = _owned_pids_by_gpu(state, jobs_by_id, local_processes)
        foreign = {gpu: observed[gpu] - owned[gpu] for gpu in GPU_IDS}
        state["foreign_gpu_pids"] = {
            str(gpu): sorted(foreign[gpu]) for gpu in GPU_IDS
        }
        occupied = {
            gpu: max(
                len(observed[gpu]),
                sum(item["status"] == "running" and item["gpu_id"] == gpu
                    for item in state["jobs"].values()),
            )
            for gpu in GPU_IDS
        }
        for job in jobs:
            item = state["jobs"][job.run_id]
            if item["status"] != "pending" or int(item["attempts"]) >= max_attempts:
                continue
            if job.gpu_id not in certified_gpu_ids:
                continue
            if foreign[job.gpu_id] or occupied[job.gpu_id] >= safe_per_gpu_limit:
                continue
            try:
                process = launch_job(
                    job,
                    project_root=canonical_project,
                    python_executable=python_executable,
                )
            except OSError as error:
                item["attempts"] = int(item["attempts"]) + 1
                item["last_exit_code"] = None
                item["status"] = "failed" if item["attempts"] >= max_attempts else "pending"
                state["reason"] = f"launch failed for {job.run_id}: {type(error).__name__}"
                continue
            pid = int(process.pid)
            local_processes[job.run_id] = process
            item.update({
                "attempts": int(item["attempts"]) + 1,
                "pid": pid,
                "process_identity": _capture_process_identity(job, pid),
                "status": "running",
                "last_exit_code": None,
            })
            occupied[job.gpu_id] += 1
        state.update({"phase": "running", "reason": None})
        _persist_state(canonical_root, state)
        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            return _result(state, exit_code=3, phase="running")
        if poll_seconds:
            time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--python", dest="python_executable", default=DEFAULT_PYTHON)
    parser.add_argument("--per-gpu", dest="safe_per_gpu_limit", required=True, type=int)
    parser.add_argument("--smoke-path", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    result = run_supervisor(
        round_root=Path(args.round_root),
        project_root=Path(args.project_root),
        python_executable=args.python_executable,
        safe_per_gpu_limit=args.safe_per_gpu_limit,
        smoke_path=Path(args.smoke_path),
        max_attempts=args.max_attempts,
        poll_seconds=args.poll_seconds,
    )
    print(json.dumps({
        "exit_code": result.exit_code,
        "phase": result.phase,
        "total": result.total,
        "complete": result.complete,
        "running": result.running,
        "failed": result.failed,
        "waiting": result.waiting,
    }, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
