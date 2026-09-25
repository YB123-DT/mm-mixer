"""Prepare the fixed-GPU, final ``HO_WO_TAV`` tuning screens.

This module deliberately owns its GPU policy instead of inheriting it from a
supervisor.  Preparation is side-effect free outside the caller-provided
queue root; launching is intentionally a separate future step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
from typing import Mapping, Sequence

from .registry import FINAL_TUNE_VARIANTS
from .runner import (
    RAWAUX_FILL53_FEATURES,
    MixerRunConfig,
    normalize_config,
    validate_ho_wo_tav_config,
)


GPU_IDS = (5, 6, 7)
SHUFFLE_SEED = 2025
CPU_THREAD_CAPS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
ROUND_PREFIXES = {"R1": "TUNE_R1_", "R2": "TUNE_R2_", "R3": "TUNE_R3_"}
ROUND_FACTORS = {
    "R1": ("x3_rank", "x3_residual_scale", "x3_lr_multiplier"),
    "R2": ("ca_heads", "ca_dropout", "ca_query_bypass"),
    "R3": ("adaptive_dropout", "channel_dropout", "channel_residual"),
}


@dataclass(frozen=True)
class QueuedJob:
    round_name: str
    experiment_id: str
    run_id: str
    seed: int
    gpu_id: int
    config_path: Path
    output_root: Path
    log_path: Path
    tuning_overrides: Mapping[str, object]
    baseline_config_path: Path | None = None
    baseline_run_dir: Path | None = None

    @property
    def thread_caps(self) -> dict[str, str]:
        return dict(CPU_THREAD_CAPS)


@dataclass(frozen=True)
class BaselineSeed:
    """A completed paired Full run eligible for R4 replication."""
    seed: int
    config_path: Path
    run_dir: Path


@dataclass(frozen=True)
class CompletedArtifact:
    run_dir: Path
    weighted_f1: float
    manifest: Mapping[str, object]
    metrics: Mapping[str, object]


def _round_name(round_name: str) -> str:
    normalized = str(round_name).upper()
    if normalized not in ROUND_PREFIXES:
        raise ValueError(f"unknown final-tune screen: {round_name}")
    return normalized


def screen_candidate_ids(round_name: str) -> list[str]:
    """Read, rather than recreate, the exact candidate screen from Task 1."""
    prefix = ROUND_PREFIXES[_round_name(round_name)]
    candidates = [name for name in FINAL_TUNE_VARIANTS if name.startswith(prefix)]
    if len(candidates) != 48:
        raise RuntimeError(f"{round_name} registry must contain exactly 48 candidates")
    return candidates


def _stratified_round_robin_assignments(round_name: str) -> list[tuple[int, str]]:
    """Balance every screen-factor marginal before shuffling within GPU lanes."""
    normalized = _round_name(round_name)
    factor_names = ROUND_FACTORS[normalized]
    candidates = screen_candidate_ids(normalized)
    factor_indices = {
        name: {
            value: index
            for index, value in enumerate(sorted({
                FINAL_TUNE_VARIANTS[candidate][name] for candidate in candidates
            }))
        }
        for name in factor_names
    }
    lanes = {gpu_id: [] for gpu_id in GPU_IDS}
    for candidate in candidates:
        overrides = FINAL_TUNE_VARIANTS[candidate]
        gpu_id = GPU_IDS[sum(factor_indices[name][overrides[name]] for name in factor_names)
                         % len(GPU_IDS)]
        lanes[gpu_id].append(candidate)
    randomizer = random.Random(SHUFFLE_SEED)
    for gpu_id in GPU_IDS:
        randomizer.shuffle(lanes[gpu_id])
    if any(len(lanes[gpu_id]) != 16 for gpu_id in GPU_IDS):
        raise RuntimeError("stratified final-tune allocation must produce 16 jobs per GPU")
    assignments = [
        (gpu_id, lanes[gpu_id][slot])
        for slot in range(16)
        for gpu_id in GPU_IDS
    ]
    _assert_factor_balance(normalized, assignments)
    return assignments


def _assert_factor_balance(round_name: str, assignments: Sequence[tuple[int, str]]) -> None:
    """Fail fast if a card would be correlated with a screen factor level."""
    for factor_name in ROUND_FACTORS[round_name]:
        levels = {FINAL_TUNE_VARIANTS[candidate][factor_name] for _, candidate in assignments}
        for level in levels:
            per_gpu = [sum(gpu_id == gpu and FINAL_TUNE_VARIANTS[candidate][factor_name] == level
                           for gpu_id, candidate in assignments) for gpu in GPU_IDS]
            total = sum(per_gpu)
            expected = sorted([total // len(GPU_IDS) + (index < total % len(GPU_IDS))
                               for index in range(len(GPU_IDS))])
            if sorted(per_gpu) != expected:
                raise RuntimeError(
                    f"{round_name} {factor_name}={level!r} is not GPU-stratified: {per_gpu}"
                )


def build_round_jobs(queue_root: Path, round_name: str) -> list[QueuedJob]:
    """Create a deterministic, factor-stratified, round-robin screen assignment."""
    normalized = _round_name(round_name)
    round_root = Path(queue_root) / normalized
    jobs = []
    for gpu_id, experiment_id in _stratified_round_robin_assignments(normalized):
        jobs.append(QueuedJob(
            round_name=normalized,
            experiment_id=experiment_id,
            run_id=experiment_id,
            seed=2025,
            gpu_id=gpu_id,
            config_path=round_root / "configs" / f"{experiment_id}.json",
            output_root=round_root,
            log_path=round_root / "logs" / f"{experiment_id}_gpu{gpu_id}.log",
            tuning_overrides=dict(FINAL_TUNE_VARIANTS[experiment_id]),
        ))
    partition_round_jobs(jobs)
    return jobs


def partition_round_jobs(jobs: list[QueuedJob]) -> tuple[tuple[QueuedJob, ...], ...]:
    """Return the fixed three-card queue partitions and reject drift early."""
    partitions = tuple(tuple(job for job in jobs if job.gpu_id == gpu_id) for gpu_id in GPU_IDS)
    if len(jobs) != 48 or any(len(partition) != 16 for partition in partitions):
        raise ValueError("each final-tune round must contain 16 queued jobs per GPU")
    if any(job.gpu_id not in GPU_IDS for job in jobs):
        raise ValueError("final-tune jobs may only use GPUs 5, 6, and 7")
    return partitions


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: MixerRunConfig | Mapping[str, object]) -> str:
    """Hash the normalized JSON contract exactly as the safe supervisor does."""
    canonical = json.dumps(
        normalize_config(config), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


@lru_cache(maxsize=None)
def _feature_artifact_sha(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_file():
        raise ValueError(f"expected feature artifact is missing: {path}")
    return _sha256(path)


def _validated_config(path: Path, label: str) -> dict[str, object]:
    try:
        return normalize_config(_read_json(path, label))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not a valid MixerRunConfig: {path}") from error


def _proc_cmdline(pid: int) -> tuple[str, ...] | None:
    """Read one live process argv without treating inaccessible processes as errors."""
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except (OSError, RuntimeError):
        return None
    command = tuple(token.decode(errors="surrogateescape")
                    for token in raw.split(b"\0") if token)
    return command or None


def _proc_cwd(pid: int) -> Path | None:
    """Resolve a process working directory for relative argv path arguments."""
    try:
        return Path(os.readlink(Path("/proc") / str(pid) / "cwd"))
    except OSError:
        return None


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


def _validate_run_artifacts(*, run_dir: Path, expected_config: Mapping[str, object],
                            experiment_id: str, seed: int,
                            tuning_overrides: Mapping[str, object],
                            expected_run_id: str | None, launch_config_path: Path,
                            launch_output_root: Path) -> CompletedArtifact:
    """Strict local artifact validation; intentionally not a supervisor helper."""
    active_pids = _matching_live_runner_pids(
        config_path=launch_config_path, output_root=launch_output_root
    )
    if active_pids:
        raise ValueError(
            "active factorized_mixer.runner still matches the immutable launch "
            f"contract (PIDs: {', '.join(map(str, active_pids))})"
        )
    status = _read_json(run_dir / "status.json", "status")
    manifest = _read_json(run_dir / "manifest.json", "manifest")
    metrics = _read_json(run_dir / "metrics.json", "metrics")
    observed_config = _validated_config(run_dir / "config.json", "run config")
    checkpoint = run_dir / "best_peak_test_state_dict.pt"
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint is missing: {checkpoint}")
    if observed_config != dict(expected_config):
        raise ValueError("run config differs from its queued normalized config")
    if status.get("state") != "complete" or status.get("fresh_strict_replay_exact") is not True:
        raise ValueError("status does not prove strict completion")
    if metrics.get("fresh_strict_replay_exact") is not True:
        raise ValueError("metrics do not prove strict replay")
    if manifest.get("fresh_strict_replay_exact") is not True:
        raise ValueError("manifest does not prove strict replay")
    if manifest.get("experiment_id") != experiment_id or manifest.get("seed") != seed:
        raise ValueError("manifest experiment identity differs from the queued job")
    if expected_run_id is not None and manifest.get("run_id") != expected_run_id:
        raise ValueError("manifest run identity differs from the queued job")
    if dict(manifest.get("tuning_overrides", {})) != dict(tuning_overrides):
        raise ValueError("manifest tuning overrides differ from the queued job")
    checkpoint_sha = _sha256(checkpoint)
    if manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("manifest checkpoint SHA does not match the local checkpoint")
    if status.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("status checkpoint SHA does not match the local checkpoint")
    expected_features = str(expected_config.get("features", ""))
    if expected_features != RAWAUX_FILL53_FEATURES:
        raise ValueError("canonical final-tune config must use exact RawAux fill53 features")
    expected_feature_sha = _feature_artifact_sha(expected_features)
    if manifest.get("feature_artifact_sha256") != expected_feature_sha:
        raise ValueError("manifest feature artifact SHA does not match exact RawAux fill53 features")
    selected = metrics.get("selected")
    if not isinstance(selected, dict) or "weighted_f1" not in selected:
        raise ValueError("metrics are missing selected weighted_f1")
    weighted_f1 = float(selected["weighted_f1"])
    if not math.isfinite(weighted_f1):
        raise ValueError("selected weighted_f1 must be finite")
    return CompletedArtifact(run_dir, weighted_f1, manifest, metrics)


def validate_completed_job(job: QueuedJob) -> CompletedArtifact:
    """Validate a queue job using only its on-disk, strict-run artifacts."""
    expected_config = _canonical_job_config(job)
    queued_config = _validated_config(job.config_path, "queued config")
    if queued_config != expected_config:
        raise ValueError("queued config differs from canonical RawAux final-tune config")
    return _validate_run_artifacts(
        run_dir=job.output_root / "runs" / job.run_id,
        expected_config=expected_config,
        experiment_id=job.experiment_id,
        seed=job.seed,
        tuning_overrides=job.tuning_overrides,
        expected_run_id=job.run_id,
        launch_config_path=job.config_path,
        launch_output_root=job.output_root,
    )


def validate_round_completion(queue_root: Path, round_name: str) -> list[CompletedArtifact]:
    """Validate a fully completed screen before the next stage can advance."""
    jobs = build_round_jobs(queue_root, round_name)
    return [validate_completed_job(job) for job in jobs]


def _validate_baseline_seed(baseline: BaselineSeed) -> dict[str, object]:
    raw_config = _read_json(baseline.config_path, "Full baseline config")
    try:
        normalized = normalize_config(validate_ho_wo_tav_config(raw_config))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Full baseline seed {baseline.seed} is incompatible: {error}") from error
    if normalized.get("seed") != baseline.seed:
        raise ValueError(f"Full baseline config seed differs for {baseline.seed}")
    _validate_run_artifacts(
        run_dir=baseline.run_dir,
        expected_config=normalized,
        experiment_id="HO_WO_TAV",
        seed=baseline.seed,
        tuning_overrides={},
        expected_run_id=None,
        launch_config_path=baseline.config_path,
        launch_output_root=baseline.run_dir.parent.parent,
    )
    return normalized


def _canonical_job_config(job: QueuedJob) -> dict[str, object]:
    """Reconstruct the immutable config contract from queue identity, not disk."""
    if job.round_name in ROUND_PREFIXES:
        config = MixerRunConfig(
            experiment_id=job.experiment_id,
            seed=job.seed,
            tuning_overrides=job.tuning_overrides,
        )
    elif job.round_name == "R4":
        if job.baseline_config_path is None or job.baseline_run_dir is None:
            raise ValueError("R4 job lacks its validated paired baseline provenance")
        baseline_config = _validate_baseline_seed(BaselineSeed(
            seed=job.seed,
            config_path=job.baseline_config_path,
            run_dir=job.baseline_run_dir,
        ))
        values = dict(baseline_config)
        values.update({
            "experiment_id": job.experiment_id,
            "run_id": job.run_id,
            "tuning_overrides": dict(job.tuning_overrides),
        })
        config = MixerRunConfig(**values)
    else:
        raise ValueError(f"unknown final-tune round for canonical config: {job.round_name}")
    expected = normalize_config(config)
    if (expected.get("feature_protocol") != "rawaux_fill53"
            or expected.get("features") != RAWAUX_FILL53_FEATURES):
        raise ValueError("canonical final-tune config must use exact RawAux fill53 features")
    return expected


def _job_manifest_row(job: QueuedJob, config: MixerRunConfig) -> dict[str, object]:
    return {
        "experiment_id": job.experiment_id,
        "run_id": job.run_id,
        "seed": job.seed,
        "gpu_id": job.gpu_id,
        "config": str(job.config_path),
        "output_root": str(job.output_root),
        "log": str(job.log_path),
        "thread_caps": job.thread_caps,
        "tuning_overrides": dict(job.tuning_overrides),
        "config_sha256": config_sha256(config),
        **({"baseline_config_source": str(job.baseline_config_path)}
           if job.baseline_config_path is not None else {}),
        **({"baseline_run_dir": str(job.baseline_run_dir)}
           if job.baseline_run_dir is not None else {}),
    }


def job_environment(job: QueuedJob, inherited: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the execution environment with all CPU pools capped at one thread."""
    if job.gpu_id not in GPU_IDS:
        raise ValueError("final-tune jobs may only use GPUs 5, 6, and 7")
    environment = dict(os.environ if inherited is None else inherited)
    environment.update(job.thread_caps)
    environment["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
    return environment


def _write_prepared_round(jobs: Sequence[QueuedJob], configs: Mapping[str, MixerRunConfig],
                          extra_manifest: Mapping[str, object] | None = None) -> None:
    if not jobs:
        raise ValueError("cannot prepare an empty final-tune round")
    for job in jobs:
        _write_json(job.config_path, configs[job.run_id].to_dict())
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
    round_root = jobs[0].output_root
    payload = {
        "round": jobs[0].round_name,
        "shuffle_seed": SHUFFLE_SEED,
        "gpu_ids": list(GPU_IDS),
        "cpu_thread_caps": CPU_THREAD_CAPS,
        "jobs": [_job_manifest_row(job, configs[job.run_id]) for job in jobs],
    }
    if extra_manifest:
        payload.update(extra_manifest)
    _write_json(round_root / "queue_manifest.json", payload)


def prepare_round(queue_root: Path, round_name: str) -> list[QueuedJob]:
    """Write screen configs and its queue manifest without launching a run."""
    jobs = build_round_jobs(queue_root, round_name)
    _write_prepared_round(jobs, {
        job.run_id: MixerRunConfig(
            experiment_id=job.experiment_id,
            seed=job.seed,
            tuning_overrides=job.tuning_overrides,
        )
        for job in jobs
    })
    return jobs


def _screen_winners(queue_root: Path) -> dict[str, list[QueuedJob]]:
    counts = {"R1": 3, "R2": 2, "R3": 2}
    winners = {}
    for round_name, count in counts.items():
        try:
            jobs = build_round_jobs(queue_root, round_name)
            completed = [(job, validate_completed_job(job)) for job in jobs]
        except ValueError as error:
            raise ValueError(f"{round_name} is not strictly complete: {error}") from error
        completed.sort(key=lambda value: (-value[1].weighted_f1, value[0].experiment_id))
        winners[round_name] = [job for job, _ in completed[:count]]
    return winners


def _combined_experiment_id(component_ids: Sequence[str]) -> str:
    return "TUNE_R4_" + "__".join(name.removeprefix("TUNE_") for name in component_ids)


def prepare_r4(queue_root: Path, paired_midrange_baselines: Sequence[BaselineSeed]) -> list[QueuedJob]:
    """Materialize the 12 screened combinations across four verified Full seeds."""
    winners = _screen_winners(queue_root)
    baselines = list(paired_midrange_baselines)
    if len(baselines) != 4 or len({baseline.seed for baseline in baselines}) != 4:
        raise ValueError("R4 requires exactly four distinct paired mid-range baseline seeds")
    normalized_baselines = [(baseline, _validate_baseline_seed(baseline)) for baseline in baselines]
    round_root = Path(queue_root) / "R4"
    pending: list[tuple[QueuedJob, MixerRunConfig]] = []
    for r1, r2, r3 in itertools.product(winners["R1"], winners["R2"], winners["R3"]):
        components = (r1, r2, r3)
        overrides: dict[str, object] = {}
        for component in components:
            overlap = set(overrides).intersection(component.tuning_overrides)
            if overlap:
                raise ValueError(f"R4 component overrides overlap: {sorted(overlap)}")
            overrides.update(component.tuning_overrides)
        experiment_id = _combined_experiment_id([component.experiment_id for component in components])
        for baseline, normalized_config in normalized_baselines:
            run_id = f"{experiment_id}_seed{baseline.seed}"
            config_values = dict(normalized_config)
            config_values.update({
                "experiment_id": experiment_id,
                "run_id": run_id,
                "tuning_overrides": dict(overrides),
            })
            config = MixerRunConfig(**config_values)
            pending.append((QueuedJob(
                round_name="R4",
                experiment_id=experiment_id,
                run_id=run_id,
                seed=baseline.seed,
                gpu_id=-1,
                config_path=round_root / "configs" / f"{run_id}.json",
                output_root=round_root,
                log_path=round_root / "logs" / f"{run_id}.log",
                tuning_overrides=dict(overrides),
                baseline_config_path=baseline.config_path,
                baseline_run_dir=baseline.run_dir,
            ), config))
    if len(pending) != 48:
        raise RuntimeError("R4 must contain 12 screened combinations across four seeds")
    random.Random(SHUFFLE_SEED).shuffle(pending)
    jobs, configs = [], {}
    for index, (pending_job, config) in enumerate(pending):
        gpu_id = GPU_IDS[index % len(GPU_IDS)]
        job = QueuedJob(
            round_name=pending_job.round_name,
            experiment_id=pending_job.experiment_id,
            run_id=pending_job.run_id,
            seed=pending_job.seed,
            gpu_id=gpu_id,
            config_path=pending_job.config_path,
            output_root=pending_job.output_root,
            log_path=pending_job.log_path.with_name(
                f"{pending_job.run_id}_gpu{gpu_id}.log"),
            tuning_overrides=pending_job.tuning_overrides,
            baseline_config_path=pending_job.baseline_config_path,
            baseline_run_dir=pending_job.baseline_run_dir,
        )
        jobs.append(job)
        configs[job.run_id] = config
    partition_round_jobs(jobs)
    _write_prepared_round(jobs, configs, {
        "screen_winners": {
            round_name: [job.experiment_id for job in selected]
            for round_name, selected in winners.items()
        },
        "paired_midrange_baseline_seeds": [baseline.seed for baseline, _ in normalized_baselines],
    })
    return jobs
