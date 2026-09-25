"""Persistent, fail-closed orchestration for the four final-tuning rounds.

This module intentionally owns no candidate-generation or worker-launch
semantics.  It only composes the immutable queue and fail-closed supervisor:
one prepared round is smoked on each clean approved GPU, supervised to strict
completion, and then permits the next round to begin.  It never sends a
signal to an existing process; occupied GPUs are recorded and polled instead.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from . import final_tune_queue as queue
from . import final_tune_supervisor as supervisor
from .runner import MixerRunConfig


ROUND_ORDER = ("R1", "R2", "R3", "R4")
PIPELINE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PipelineResult:
    """The durable pipeline status after a bounded or terminal invocation."""

    exit_code: int
    phase: str
    completed_rounds: tuple[str, ...]
    active_round: str | None
    reason: str | None


class _PipelineBlocked(RuntimeError):
    """A fail-closed condition that can be retried after external state changes."""


class _SmokeFailed(RuntimeError):
    """A smoke worker itself failed; retrying blindly would hide that failure."""


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


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


def _baseline_rows(baselines: Sequence[queue.BaselineSeed]) -> list[dict[str, object]]:
    return [
        {
            "seed": baseline.seed,
            "config_path": str(Path(baseline.config_path).resolve()),
            "run_dir": str(Path(baseline.run_dir).resolve()),
        }
        for baseline in baselines
    ]


def validate_baseline_descriptors(
        descriptors: Sequence[queue.BaselineSeed],
) -> list[queue.BaselineSeed]:
    """Validate baseline descriptor shape and paths without requiring completion."""
    baselines = list(descriptors)
    if len(baselines) != 4:
        raise ValueError("pipeline requires exactly four distinct paired baseline descriptors")
    if any(not isinstance(item, queue.BaselineSeed) for item in baselines):
        raise ValueError("pipeline baseline descriptors must be BaselineSeed values")
    if len({item.seed for item in baselines}) != 4:
        raise ValueError("pipeline requires exactly four distinct paired baseline descriptors")
    for baseline in baselines:
        if isinstance(baseline.seed, bool) or not isinstance(baseline.seed, int):
            raise ValueError("pipeline baseline descriptor has an invalid seed")
        if not Path(baseline.config_path).is_file():
            raise ValueError(f"pipeline baseline config is missing: {baseline.config_path}")
        if not Path(baseline.run_dir).is_dir():
            raise ValueError(f"pipeline baseline run directory is missing: {baseline.run_dir}")
    return baselines


def validate_completed_baseline_descriptors(
        descriptors: Sequence[queue.BaselineSeed],
) -> list[queue.BaselineSeed]:
    """Require strict baseline completion only at the R4 boundary or R4 resume."""
    baselines = validate_baseline_descriptors(descriptors)
    for baseline in baselines:
        queue._validate_baseline_seed(baseline)
    return baselines


def _state_path(queue_root: Path) -> Path:
    return queue_root / "pipeline" / "state.json"


def _initial_state(*, queue_root: Path, project_root: Path,
                   baselines: Sequence[queue.BaselineSeed],
                   requested_per_gpu: int) -> dict[str, Any]:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "queue_root": str(queue_root.resolve()),
        "project_root": str(project_root.resolve()),
        "gpu_ids": list(supervisor.GPU_IDS),
        "requested_per_gpu": requested_per_gpu,
        "baseline_descriptors": _baseline_rows(baselines),
        "phase": "starting",
        "reason": None,
        "active_round": "R1",
        "completed_rounds": [],
        "smoke": {},
        "foreign_gpu_pids": {str(gpu): [] for gpu in supervisor.GPU_IDS},
    }


def _load_state(*, queue_root: Path, project_root: Path,
                baselines: Sequence[queue.BaselineSeed],
                requested_per_gpu: int) -> dict[str, Any]:
    path = _state_path(queue_root)
    if not path.exists():
        return _initial_state(
            queue_root=queue_root,
            project_root=project_root,
            baselines=baselines,
            requested_per_gpu=requested_per_gpu,
        )
    state = _read_json(path, "pipeline state")
    expected = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "queue_root": str(queue_root.resolve()),
        "project_root": str(project_root.resolve()),
        "gpu_ids": list(supervisor.GPU_IDS),
        "requested_per_gpu": requested_per_gpu,
        "baseline_descriptors": _baseline_rows(baselines),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(f"pipeline state {key} differs from this invocation")
    completed = state.get("completed_rounds")
    if (not isinstance(completed, list)
            or any(round_name not in ROUND_ORDER for round_name in completed)
            or completed != list(ROUND_ORDER[:len(completed)])):
        raise ValueError("pipeline state has an invalid completed-round prefix")
    if not isinstance(state.get("smoke"), dict):
        raise ValueError("pipeline state has invalid smoke progress")
    return state


def _persist_state(queue_root: Path, state: dict[str, Any]) -> None:
    state["updated_at_unix"] = time.time()
    _atomic_json(_state_path(queue_root), state)


def _result(state: Mapping[str, Any], *, exit_code: int, phase: str) -> PipelineResult:
    active = state.get("active_round")
    return PipelineResult(
        exit_code=exit_code,
        phase=phase,
        completed_rounds=tuple(state.get("completed_rounds", [])),
        active_round=active if isinstance(active, str) else None,
        reason=state.get("reason") if isinstance(state.get("reason"), str) else None,
    )


def gpu_memory_mib() -> dict[int, float] | None:
    """Read total used memory on the approved physical cards, fail closed on drift."""
    try:
        lines = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            text=True,
        ).splitlines()
        values: dict[int, float] = {}
        for line in lines:
            if not line.strip() or "," not in line:
                raise ValueError("malformed GPU memory row")
            index_text, memory_text = line.split(",", 1)
            gpu = int(index_text.strip())
            memory = float(memory_text.strip().split()[0])
            if gpu in supervisor.GPU_IDS:
                if not math.isfinite(memory) or memory < 0:
                    raise ValueError("invalid GPU memory value")
                values[gpu] = memory
        if set(values) != set(supervisor.GPU_IDS):
            raise ValueError("one or more approved GPUs are absent from memory query")
        return values
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _smoke_root(round_root: Path) -> Path:
    return round_root.parent / "smoke"


def _smoke_evidence_path(round_root: Path) -> Path:
    return _smoke_root(round_root) / f"{round_root.name}.json"


def _smoke_config_path(round_root: Path, gpu_id: int) -> Path:
    return _smoke_root(round_root) / "configs" / f"{round_root.name}_gpu{gpu_id}.json"


def _smoke_run_id(round_root: Path, gpu_id: int) -> str:
    return f"SMOKE_{round_root.name}_GPU{gpu_id}"


def _write_smoke_config(round_root: Path, job: supervisor.PreparedJob, gpu_id: int) -> Path:
    """Derive a one-epoch config without ever mutating an immutable queue config."""
    payload = _read_json(job.config_path, "prepared job config")
    payload.update({"epochs": 1, "run_id": _smoke_run_id(round_root, gpu_id)})
    config = MixerRunConfig(**payload)
    if config.epochs != 1:
        raise ValueError("smoke config did not preserve its one-epoch cap")
    path = _smoke_config_path(round_root, gpu_id)
    _atomic_json(path, config.to_dict())
    return path


def _smoke_environment(gpu_id: int) -> dict[str, str]:
    if gpu_id not in supervisor.GPU_IDS:
        raise ValueError("smoke must use an approved GPU")
    environment = dict(os.environ)
    environment.update(supervisor.CPU_THREAD_CAPS)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return environment


def _launch_smoke(*, round_root: Path, job: supervisor.PreparedJob, gpu_id: int,
                  project_root: Path, python_executable: str) -> subprocess.Popen[str]:
    config_path = _write_smoke_config(round_root, job, gpu_id)
    smoke_root = _smoke_root(round_root)
    log_path = smoke_root / "logs" / f"{round_root.name}_gpu{gpu_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [
                python_executable, "-u", "-m", "factorized_mixer.runner",
                "--config", str(config_path), "--output-root", str(smoke_root / "work"),
            ],
            cwd=project_root,
            env=_smoke_environment(gpu_id),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except BaseException:
        log.close()
        raise
    setattr(process, "_final_tune_smoke_log", log)
    return process


def _close_smoke_log(process: object) -> None:
    stream = getattr(process, "_final_tune_smoke_log", None)
    if stream is not None:
        stream.close()


def _run_one_epoch_smoke(*, round_root: Path, job: supervisor.PreparedJob,
                         gpu_id: int, project_root: Path, python_executable: str,
                         poll_seconds: float) -> tuple[float, float] | None:
    """Return ``(overhead, worker)`` or ``None`` when a foreign PID contaminated it.

    Any query failure is remembered until the owned smoke worker exits; this
    avoids abandoning a child process while still refusing to certify evidence.
    """
    baseline = gpu_memory_mib()
    if baseline is None:
        raise _PipelineBlocked("nvidia-smi GPU memory query failed before smoke launch")
    observed = supervisor.gpu_compute_processes()
    if observed is None:
        raise _PipelineBlocked("nvidia-smi GPU PID query failed before smoke launch")
    if observed[gpu_id]:
        return None
    try:
        process = _launch_smoke(
            round_root=round_root,
            job=job,
            gpu_id=gpu_id,
            project_root=project_root,
            python_executable=python_executable,
        )
    except OSError as error:
        raise _SmokeFailed(f"smoke launch failed on GPU {gpu_id}: {type(error).__name__}") from error
    peak = baseline[gpu_id]
    unsafe_reason: str | None = None
    contaminated = False
    observed_own_pid = False
    exit_code: int | None = None
    try:
        while True:
            memory = gpu_memory_mib()
            if memory is None:
                unsafe_reason = "nvidia-smi GPU memory query failed during smoke"
            else:
                peak = max(peak, memory[gpu_id])
            observed = supervisor.gpu_compute_processes()
            if observed is None:
                unsafe_reason = "nvidia-smi GPU PID query failed during smoke"
            else:
                observed_own_pid = observed_own_pid or int(process.pid) in observed[gpu_id]
                foreign = observed[gpu_id] - {int(process.pid)}
                if foreign:
                    contaminated = True
            exit_code = process.poll()
            if exit_code is not None:
                break
            if poll_seconds:
                time.sleep(poll_seconds)
    finally:
        _close_smoke_log(process)
    if exit_code != 0:
        raise _SmokeFailed(f"one-epoch smoke failed on GPU {gpu_id} with exit code {exit_code}")
    if unsafe_reason is not None:
        raise _PipelineBlocked(unsafe_reason)
    if contaminated or not observed_own_pid:
        return None
    overhead = baseline[gpu_id]
    return overhead, max(1.0, peak - overhead)


def _record_foreign_processes(state: dict[str, Any], observed: Mapping[int, set[int]]) -> None:
    state["foreign_gpu_pids"] = {
        str(gpu): sorted(int(pid) for pid in observed[gpu])
        for gpu in supervisor.GPU_IDS
    }


def _sample_gpu_ids(samples: Mapping[str, Mapping[str, float]]) -> tuple[int, ...]:
    if not samples:
        return ()
    gpu_ids: list[int] = []
    for raw_gpu, sample in samples.items():
        try:
            gpu = int(raw_gpu)
        except (TypeError, ValueError) as error:
            raise ValueError("pipeline smoke sample has an invalid GPU ID") from error
        if str(gpu) != raw_gpu or gpu not in supervisor.GPU_IDS or not isinstance(sample, Mapping):
            raise ValueError("pipeline smoke sample has an invalid GPU ID")
        worker = sample.get("worker_mib")
        overhead = sample.get("overhead_mib")
        if (isinstance(worker, bool) or not isinstance(worker, (int, float))
                or not math.isfinite(float(worker)) or float(worker) <= 0
                or isinstance(overhead, bool) or not isinstance(overhead, (int, float))
                or not math.isfinite(float(overhead)) or float(overhead) < 0):
            raise ValueError("pipeline smoke sample has invalid memory measurements")
        gpu_ids.append(gpu)
    return tuple(sorted(gpu_ids))


def _smoke_payload(*, round_root: Path,
                   samples: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    gpu_ids = _sample_gpu_ids(samples)
    if not gpu_ids:
        raise ValueError("pipeline cannot write empty smoke evidence")
    worker_mib = {str(gpu): int(math.ceil(samples[str(gpu)]["worker_mib"]))
                  for gpu in gpu_ids}
    overhead_mib = {str(gpu): int(math.ceil(samples[str(gpu)]["overhead_mib"]))
                    for gpu in gpu_ids}
    caps: dict[int, int] = {}
    for gpu in gpu_ids:
        headroom = supervisor.SMOKE_MEMORY_BUDGET_MIB - overhead_mib[str(gpu)]
        if headroom <= 0:
            raise _PipelineBlocked("smoke overhead leaves no safe GPU memory")
        caps[gpu] = min(16, headroom // worker_mib[str(gpu)])
    return {
        "schema_version": supervisor.SMOKE_SCHEMA_VERSION,
        "round": round_root.name,
        "feature_protocol": "rawaux_fill53",
        "gpu_ids": list(gpu_ids),
        "queue_manifest_sha256": supervisor._sha256(round_root / "queue_manifest.json"),
        "worker_mib": worker_mib,
        "overhead_mib": overhead_mib,
        "safe_per_gpu": {str(gpu): caps[gpu] for gpu in gpu_ids},
    }


def _ensure_round_smoke(*, round_root: Path, jobs: Sequence[supervisor.PreparedJob],
                        project_root: Path, python_executable: str,
                        requested_per_gpu: int, state: dict[str, Any],
                        poll_seconds: float) -> set[int]:
    """Create manifest-bound per-GPU smoke certificates without touching busy cards."""
    evidence_path = _smoke_evidence_path(round_root)
    progress = state.setdefault("smoke", {}).setdefault(round_root.name, {"samples": {}})
    samples = progress.setdefault("samples", {})
    if not isinstance(samples, dict):
        raise ValueError("pipeline state smoke samples are invalid")
    if evidence_path.is_file():
        try:
            evidence = supervisor.validate_smoke_evidence(
                evidence_path,
                round_root=round_root,
                safe_per_gpu_limit=requested_per_gpu,
            )
            for gpu in supervisor._smoke_gpu_ids(evidence):
                samples[str(gpu)] = {
                    "overhead_mib": evidence["overhead_mib"][str(gpu)],
                    "worker_mib": evidence["worker_mib"][str(gpu)],
                }
        except ValueError:
            # A changed queue needs a newly bound sample; old samples are not proof.
            state.setdefault("smoke", {}).pop(round_root.name, None)
            progress = state.setdefault("smoke", {}).setdefault(round_root.name, {"samples": {}})
            samples = progress["samples"]
    if not jobs:
        raise ValueError("prepared round has no jobs for its smoke configuration")
    observed = supervisor.gpu_compute_processes()
    if observed is None:
        raise _PipelineBlocked("nvidia-smi GPU PID query failed; refusing smoke launch")
    _record_foreign_processes(state, observed)
    for gpu in supervisor.GPU_IDS:
        if str(gpu) in samples:
            continue
        observed = supervisor.gpu_compute_processes()
        if observed is None:
            raise _PipelineBlocked("nvidia-smi GPU PID query failed; refusing smoke launch")
        _record_foreign_processes(state, observed)
        if observed[gpu]:
            continue
        sample = _run_one_epoch_smoke(
            round_root=round_root,
            job=jobs[0],
            gpu_id=gpu,
            project_root=project_root,
            python_executable=python_executable,
            poll_seconds=poll_seconds,
        )
        if sample is None:
            continue
        overhead, worker = sample
        samples[str(gpu)] = {"overhead_mib": overhead, "worker_mib": worker}
    observed = supervisor.gpu_compute_processes()
    if observed is None:
        raise _PipelineBlocked("nvidia-smi GPU PID query failed after smoke")
    _record_foreign_processes(state, observed)
    certified_gpu_ids = set(_sample_gpu_ids(samples))
    if not certified_gpu_ids:
        return set()
    payload = _smoke_payload(round_root=round_root, samples=samples)
    safe_per_gpu = payload["safe_per_gpu"]
    if any(safe_per_gpu[str(gpu)] < requested_per_gpu for gpu in certified_gpu_ids):
        raise _PipelineBlocked(
            f"smoke supports fewer than {requested_per_gpu} workers per GPU, "
            f"below requested {requested_per_gpu}"
        )
    _atomic_json(evidence_path, payload)
    supervisor.validate_smoke_evidence(
        evidence_path,
        round_root=round_root,
        safe_per_gpu_limit=requested_per_gpu,
        certified_gpu_ids=certified_gpu_ids,
    )
    progress["certified_gpu_ids"] = sorted(certified_gpu_ids)
    return certified_gpu_ids


def _prepare_or_load_round(*, queue_root: Path, round_name: str,
                           baselines: Sequence[queue.BaselineSeed]) -> tuple[Path, list[supervisor.PreparedJob]]:
    round_root = queue_root / round_name
    manifest_path = round_root / "queue_manifest.json"
    if not manifest_path.is_file():
        if round_name == "R4":
            queue.prepare_r4(queue_root, baselines)
        else:
            queue.prepare_round(queue_root, round_name)
    return round_root, supervisor.load_prepared_round(round_root)


def _revalidate_round_completion(queue_root: Path, round_name: str) -> None:
    """State is never evidence: every completed round is proved again before use."""
    if round_name in queue.ROUND_PREFIXES:
        queue.validate_round_completion(queue_root, round_name)
        return
    round_root = queue_root / "R4"
    for job in supervisor.load_prepared_round(round_root):
        supervisor.validate_completed_job(job)


def _next_round(state: Mapping[str, Any]) -> str | None:
    completed = state["completed_rounds"]
    return ROUND_ORDER[len(completed)] if len(completed) < len(ROUND_ORDER) else None


def run_pipeline(*, queue_root: Path, project_root: Path,
                 paired_baselines: Sequence[queue.BaselineSeed],
                 python_executable: str = supervisor.DEFAULT_PYTHON,
                 requested_per_gpu: int = 16, poll_seconds: float = 30.0,
                 max_cycles: int | None = None,
                 supervisor_max_cycles: int | None = None) -> PipelineResult:
    """Advance only the next verified round; resume safely from persistent state."""
    supervisor._validate_limit(requested_per_gpu)
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative")
    if max_cycles is not None and (isinstance(max_cycles, bool) or max_cycles < 1):
        raise ValueError("max_cycles must be a positive integer when provided")
    baselines = validate_baseline_descriptors(paired_baselines)
    canonical_queue = Path(queue_root).resolve()
    canonical_project = Path(project_root).resolve()
    state = _load_state(
        queue_root=canonical_queue,
        project_root=canonical_project,
        baselines=baselines,
        requested_per_gpu=requested_per_gpu,
    )
    try:
        for round_name in state["completed_rounds"]:
            if round_name == "R4":
                validate_completed_baseline_descriptors(baselines)
            _revalidate_round_completion(canonical_queue, round_name)
    except ValueError as error:
        state.update({"phase": "blocked", "reason": f"completed round revalidation failed: {error}"})
        _persist_state(canonical_queue, state)
        return _result(state, exit_code=2, phase="blocked")

    cycles = 0
    while True:
        round_name = _next_round(state)
        if round_name is None:
            state.update({"phase": "complete", "active_round": None, "reason": None})
            _persist_state(canonical_queue, state)
            return _result(state, exit_code=0, phase="complete")
        state.update({"phase": "preparing", "active_round": round_name, "reason": None})
        if round_name == "R4":
            try:
                baselines = validate_completed_baseline_descriptors(baselines)
            except ValueError as error:
                state.update({"phase": "blocked", "reason": f"baseline revalidation failed: {error}"})
                _persist_state(canonical_queue, state)
                return _result(state, exit_code=2, phase="blocked")
        try:
            round_root, jobs = _prepare_or_load_round(
                queue_root=canonical_queue,
                round_name=round_name,
                baselines=baselines,
            )
            certified_gpu_ids = _ensure_round_smoke(
                round_root=round_root,
                jobs=jobs,
                project_root=canonical_project,
                python_executable=python_executable,
                requested_per_gpu=requested_per_gpu,
                state=state,
                poll_seconds=poll_seconds,
            )
        except _PipelineBlocked as error:
            state.update({"phase": "blocked", "reason": str(error)})
            _persist_state(canonical_queue, state)
            return _result(state, exit_code=2, phase="blocked")
        except _SmokeFailed as error:
            state.update({"phase": "failed", "reason": str(error)})
            _persist_state(canonical_queue, state)
            return _result(state, exit_code=1, phase="failed")
        except ValueError as error:
            state.update({"phase": "blocked", "reason": f"prepared round rejected: {error}"})
            _persist_state(canonical_queue, state)
            return _result(state, exit_code=2, phase="blocked")

        if not certified_gpu_ids:
            state.update({"phase": "waiting_for_clean_gpu", "reason": "one or more GPUs have foreign compute PIDs"})
            _persist_state(canonical_queue, state)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return _result(state, exit_code=3, phase="waiting_for_clean_gpu")
            if poll_seconds:
                time.sleep(poll_seconds)
            continue

        state.update({"phase": "supervising", "reason": None})
        _persist_state(canonical_queue, state)
        result = supervisor.run_supervisor(
            round_root=round_root,
            project_root=canonical_project,
            python_executable=python_executable,
            safe_per_gpu_limit=requested_per_gpu,
            smoke_path=_smoke_evidence_path(round_root),
            certified_gpu_ids=certified_gpu_ids,
            poll_seconds=poll_seconds,
            max_cycles=1 if supervisor_max_cycles is None else supervisor_max_cycles,
        )
        if result.phase == "complete" and result.exit_code == 0:
            try:
                _revalidate_round_completion(canonical_queue, round_name)
            except ValueError as error:
                state.update({"phase": "blocked", "reason": f"round completion rejected: {error}"})
                _persist_state(canonical_queue, state)
                return _result(state, exit_code=2, phase="blocked")
            state["completed_rounds"].append(round_name)
            state.update({"phase": "advancing", "reason": None})
            _persist_state(canonical_queue, state)
            continue
        if result.phase == "running":
            state.update({"phase": "waiting_for_supervisor", "reason": result.phase})
            _persist_state(canonical_queue, state)
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return _result(state, exit_code=result.exit_code, phase="waiting_for_supervisor")
            if poll_seconds:
                time.sleep(poll_seconds)
            continue
        if result.phase == "blocked":
            state.update({"phase": "waiting_for_supervisor", "reason": result.phase})
            _persist_state(canonical_queue, state)
            return _result(state, exit_code=result.exit_code, phase="waiting_for_supervisor")
        state.update({"phase": "failed", "reason": f"supervisor ended in {result.phase}"})
        _persist_state(canonical_queue, state)
        return _result(state, exit_code=result.exit_code or 1, phase="failed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--python", dest="python_executable", default=supervisor.DEFAULT_PYTHON)
    parser.add_argument("--per-gpu", dest="requested_per_gpu", type=int, default=16)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-cycles", type=int)
    parser.add_argument("--supervisor-max-cycles", type=int)
    parser.add_argument(
        "--baseline", action="append", nargs=3, metavar=("SEED", "CONFIG", "RUN_DIR"),
        required=True,
        help="repeat exactly four times: seed, exact Full config, and completed run directory",
    )
    args = parser.parse_args(argv)
    baselines = [
        queue.BaselineSeed(seed=int(seed), config_path=Path(config), run_dir=Path(run_dir))
        for seed, config, run_dir in args.baseline
    ]
    result = run_pipeline(
        queue_root=Path(args.queue_root),
        project_root=Path(args.project_root),
        paired_baselines=baselines,
        python_executable=args.python_executable,
        requested_per_gpu=args.requested_per_gpu,
        poll_seconds=args.poll_seconds,
        max_cycles=args.max_cycles,
        supervisor_max_cycles=args.supervisor_max_cycles,
    )
    print(json.dumps({
        "exit_code": result.exit_code,
        "phase": result.phase,
        "completed_rounds": list(result.completed_rounds),
        "active_round": result.active_round,
        "reason": result.reason,
    }, sort_keys=True))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
