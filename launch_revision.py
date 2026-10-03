#!/usr/bin/env python3
"""Durable, local scheduler for explicitly listed MM-Mixer revision experiments.

Example plan: {"jobs": [{"dataset": "iemocap", "variant": "no_mixer",
"seed": 2025, "code_root": "/frozen/code", "output_root": "/runs",
"python": "/path/to/python"}]}. Execute this script on the assigned server;
it deliberately contains no SSH, data migration, batch-size changes or retries.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from typing import Any

# Import the pure-stdlib summary without executing package __init__, which loads
# the training stack. The queue itself can run in a lightweight Python process.
_SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "revision_results", Path(__file__).resolve().parent / "mm_mixer_final/revision_results.py")
_SUMMARY = importlib.util.module_from_spec(_SUMMARY_SPEC)
_SUMMARY_SPEC.loader.exec_module(_SUMMARY)
summarize_results = _SUMMARY.summarize_results
formal_runtime_error = _SUMMARY.formal_runtime_error

SEEDS = {"iemocap": (2025, 2066, 2118), "meld": (2025, 2028, 2069)}
BASE_VARIANTS = (
    "full", "no_mixer", "no_pairwise", "no_adaptive_gating", "no_cross_attention",
    "no_auxiliary_loss", "no_sequence_mixing", "no_modality_mixing",
    "no_feature_mixing", "one_mixer_block", "no_feature_gating",
    "modal_t", "modal_a", "modal_v", "modal_ta", "modal_tv", "modal_av",
)
CONTROL_VARIANTS = (
    "amm_mlp", "amm_attention", "amm_cubemlp", "single_projection_view",
    "no_feature_and_adaptive_gating", "pairwise_mlp_residual", "amm_mlp_no_aux",
)


def generate_plan(matrix: str, code_root: str, output_root: str, python: str) -> dict[str, Any]:
    for name, value in (("code_root", code_root), ("output_root", output_root), ("python", python)):
        if not Path(value).is_absolute():
            raise ValueError(f"{name} must be explicit and absolute")
    variants = {"base": BASE_VARIANTS, "controls": CONTROL_VARIANTS,
                "full": BASE_VARIANTS + CONTROL_VARIANTS}[matrix]
    return {"schema_version": 1, "matrix": matrix, "selection": "strict_peak_test_wf1",
        "jobs": [{"dataset": dataset, "variant": variant, "seed": seed,
                  "code_root": code_root, "output_root": output_root, "python": python}
                 for dataset, seeds in SEEDS.items() for variant in variants for seed in seeds]}


def write_new_plan(path: Path, payload: dict[str, Any]) -> None:
    """Publish atomically without replacing an existing plan (including symlinks)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic exclusive publication on the same filesystem
    finally:
        os.unlink(temporary)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_plan(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("plan must contain a nonempty jobs list")
    result, identities, roots = [], set(), set()
    for raw in jobs:
        job = dict(raw)
        dataset, variant, seed = job["dataset"], job["variant"], job["seed"]
        if dataset not in SEEDS or type(seed) is not int or seed not in SEEDS[dataset]:
            raise ValueError(f"not a revision dataset/seed: {dataset}/{seed}")
        if not isinstance(variant, str) or not variant or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in variant):
            raise ValueError(f"invalid variant: {variant}")
        identity = f"{dataset}/{variant}/seed{seed}"
        if identity in identities:
            raise ValueError(f"duplicate job: {identity}")
        identities.add(identity)
        for key in ("output_root", "code_root"):
            if not Path(job[key]).is_absolute():
                raise ValueError(f"{identity}: {key} must be absolute")
            job[key] = str(Path(job[key]).resolve())
        job["python"] = str(Path(job.get("python", sys.executable)).resolve())
        job["id"] = identity
        job["run_root"] = str(Path(job["output_root"]) / identity)
        if job["run_root"] in roots:
            raise ValueError(f"duplicate output path: {job['run_root']}")
        roots.add(job["run_root"])
        if job.get("reuse_bundle"):
            if not Path(job["reuse_bundle"]).is_absolute():
                raise ValueError("reuse_bundle must be absolute")
            job["reuse_bundle"] = str(Path(job["reuse_bundle"]).resolve())
        result.append(job)
    return result


def command_for(job: dict[str, Any]) -> list[str]:
    return [job["python"], str(Path(job["code_root"]) / "run.py"), "run",
            "--dataset", job["dataset"], "--variant", job["variant"],
            "--seed", str(job["seed"]), "--output-root", job["output_root"]]


def parse_gpus(specs: list[str], server: str) -> list[dict[str, Any]]:
    result = []
    for spec in specs:
        host_id, gpu_uuid = spec.split(":", 1)
        host_id = int(host_id)
        if host_id < 0 or not gpu_uuid.startswith("GPU-"):
            raise ValueError("GPU specification must be host-index:GPU-uuid")
        if server == "biggpu" and host_id == 4:
            raise ValueError("biggpu host GPU 4 is prohibited")
        if any(g["index"] == host_id or g["uuid"] == gpu_uuid for g in result):
            raise ValueError("duplicate GPU index or UUID")
        result.append({"index": host_id, "uuid": gpu_uuid})
    if not result:
        raise ValueError("an explicit healthy GPU index/UUID whitelist is required")
    return result


def gpu_inventory() -> dict[int, dict[str, Any]]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    result = {}
    for line in completed.stdout.splitlines():
        index, gpu_uuid, free = (v.strip() for v in line.split(","))
        result[int(index)] = {"uuid": gpu_uuid, "free_mib": int(free)}
    return result


def check_gpus(whitelist: list[dict[str, Any]], inventory: dict) -> None:
    for gpu in whitelist:
        if inventory.get(gpu["index"], {}).get("uuid") != gpu["uuid"]:
            raise ValueError(f"GPU index/UUID identity changed or unavailable: {gpu}")


@contextmanager
def queue_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another queue owns {path}") from exc
        yield stream


def process_identity(pid: int) -> dict[str, Any] | None:
    try:
        proc = Path("/proc") / str(pid)
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "command": (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")}
    except (OSError, UnicodeDecodeError, IndexError):
        return None


def job_processes(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Use a per-launch inherited token, plus actual command, to reject PID reuse.

    Scanning also recovers the launch/save crash window and surviving trainer
    descendants after their run.py parent exits. Other users' /proc entries may
    be unreadable and are skipped.
    """
    token = record.get("process_token")
    if not token:
        return []
    matches = []
    marker = f"MM_MIXER_REVISION_JOB_TOKEN={token}".encode()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if marker not in (proc / "environ").read_bytes().split(b"\0"):
                continue
        except OSError:
            continue
        identity = process_identity(int(proc.name))
        if identity is None:
            continue
        command = identity["command"]
        expected = record.get("command", [])
        code_root = record["code_root"]
        # run.py or its concrete dataset runner must belong to the frozen tree.
        owned_script = any(arg in {str(Path(code_root) / "run.py"),
            str(Path(code_root) / "dataset_runners" / f"{record['dataset']}.py")} for arg in command)
        output_matches = record["output_root"] in command or record["run_root"] in command
        if not owned_script or not output_matches:
            continue
        if identity["pid"] == record.get("pid"):
            if record.get("start_ticks") and identity["start_ticks"] != record["start_ticks"]:
                continue
            if expected and command != expected:
                continue
        matches.append(identity)
    return matches


# Send the same policy to the job's interpreter without importing queue code
# into its frozen training tree (older snapshots need not contain this helper).
_VERIFY_CODE = inspect.getsource(formal_runtime_error) + r'''
import json, sys
from pathlib import Path
from mm_mixer_final.artifacts import PeakArtifactStore
from mm_mixer_final.config import get_config, config_contract_sha256
job = json.loads(sys.argv[1])
cfg = get_config(job["dataset"], job["variant"], job["seed"])
bundle = Path(job.get("reuse_bundle") or Path(job["run_root"]) / "best_peak")
expected = {k: job[k] for k in ("dataset", "variant", "seed")}
expected["config_contract_sha256"] = job.get("expected_contract_sha256") or config_contract_sha256(cfg)
store = object.__new__(PeakArtifactStore)
store.public, store.class_names = bundle, cfg.class_names
config = json.loads((bundle / "config.json").read_text())
manifest = json.loads((bundle / "manifest.json").read_text())
error = formal_runtime_error(job["dataset"], cfg.epochs, config, manifest)
valid = error is None and store.validate_public_bundle(expected)
result = {"valid": valid, "bundle": str(bundle), "expected": expected,
          "formal_protocol_verified": error is None, "formal_epochs": cfg.epochs}
if error:
    result["error"] = error
if valid:
    result["metrics"] = json.loads((bundle / "peak_test_metrics.json").read_text())
    result["manifest"] = manifest
print(json.dumps(result))
'''


def verify_bundle(job: dict[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [job["python"], "-c", _VERIFY_CODE, json.dumps(job)],
            cwd=job["code_root"], capture_output=True, text=True, timeout=900,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        )
        if completed.returncode:
            return {"valid": False, "error": completed.stderr[-4000:]}
        return json.loads(completed.stdout.splitlines()[-1])
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError) as exc:
        return {"valid": False, "error": str(exc)}



_PROVENANCE_CODE = r'''
import importlib.metadata
import json
import platform
import socket
import sys
versions = {}
for name, distribution in (("torch", "torch"), ("numpy", "numpy"), ("sklearn", "scikit-learn")):
    try:
        versions[name] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({"python_executable": sys.executable, "python_version": platform.python_version(),
                  "python_implementation": platform.python_implementation(), "packages": versions,
                  "hostname": socket.gethostname()}))
'''


def collect_launch_provenance(job: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run([job["python"], "-c", _PROVENANCE_CODE],
        cwd=job["code_root"], capture_output=True, text=True, check=True, timeout=30)
    result = json.loads(completed.stdout.strip())
    result["captured_at"] = timestamp()
    result["code_root"] = job["code_root"]
    snapshot = Path(job["code_root"]) / "snapshot.json"
    if snapshot.is_file():
        raw = snapshot.read_bytes()
        info = json.loads(raw)
        result["snapshot"] = {"path": str(snapshot), "sha256": hashlib.sha256(raw).hexdigest(),
                              "git_head": info.get("git_head", info.get("head"))}
    else:
        result["snapshot"] = None
    return result


def initial_state(jobs: list[dict[str, Any]], server: str, gpus: list) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps(jobs, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "plan_sha256": digest, "server": server,
            "selection": "strict_peak_test_wf1", "gpus": gpus,
            "created_at": timestamp(), "jobs": {
                job["id"]: {**job, "state": "queued", "command": command_for(job)} for job in jobs}}


def accept_or_fail(record: dict[str, Any], reason: str = "") -> None:
    result = verify_bundle(record)
    record["finished_at"] = timestamp()
    record["artifact_verified"] = result.get("valid") is True
    record["formal_protocol_verified"] = result.get("formal_protocol_verified") is True
    if record["artifact_verified"] and record["formal_protocol_verified"]:
        record.update(state="completed", bundle=result["bundle"], metrics=result["metrics"],
                      verified_contract=result["expected"]["config_contract_sha256"],
                      formal_epochs=result["formal_epochs"],
                      verification="formal materialized epoch budget and strict test selection; PeakArtifactStore replay/source/data/artifact hashes",
                      verified_at=timestamp())
    else:
        record.update(state="failed", error=f"{reason}; complete verified bundle unavailable: {result.get('error', 'contract mismatch')}".strip("; "))


def run_queue(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> int:
    gpus = parse_gpus(args.gpu, args.server)
    expected_state = initial_state(jobs, args.server, gpus)
    state_path = Path(args.state).resolve()
    stop = False
    processes: dict[str, subprocess.Popen] = {}
    job_locks: dict[str, Any] = {}
    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    with queue_lock(state_path.with_suffix(state_path.suffix + ".lock")):
        state = json.loads(state_path.read_text()) if state_path.exists() else expected_state
        if any(state.get(key) != expected_state[key] for key in ("plan_sha256", "server", "gpus")):
            raise ValueError("state plan/server/GPU whitelist changed; do not reuse this status file")
        for record in state["jobs"].values():
            if record["state"] == "completed":
                accept_or_fail(record, "resume artifact revalidation failed")
        state["queue_pid"] = os.getpid()
        state["updated_at"] = timestamp()
        atomic_json(state_path, state)
        while not stop:
            for job_id, record in state["jobs"].items():
                if record["state"] == "running":
                    process = processes.get(job_id)
                    returncode = process.poll() if process else None
                    if returncode is not None:
                        record["returncode"] = returncode
                    live = job_processes(record)
                    record["live_pids"] = [p["pid"] for p in live]
                    if not live:
                        accept_or_fail(record, f"process exited ({record.get('returncode', 'restart; code unknown')})")
                        if job_id in job_locks:
                            job_locks.pop(job_id).close()
                elif record["state"] == "queued":
                    if record.get("reuse_bundle"):
                        record["reused"] = True
                        accept_or_fail(record, "audited reuse failed validation")
                    elif Path(record["run_root"]).exists():
                        # A partial run is preserved for diagnosis, never overwritten.
                        accept_or_fail(record, "pre-existing output; automatic restart is disabled")
            state["updated_at"] = timestamp()
            atomic_json(state_path, state)
            atomic_json(state_path.with_name("summary.json"), summarize_results(jobs, state))
            pending = [r for r in state["jobs"].values() if r["state"] == "queued"]
            running = [r for r in state["jobs"].values() if r["state"] == "running"]
            if not pending and not running:
                state["finished_at"] = timestamp()
                state["queue_state"] = "complete_with_failures" if any(r["state"] == "failed" for r in state["jobs"].values()) else "complete"
                atomic_json(state_path, state)
                return int(state["queue_state"] != "complete")
            state["queue_state"] = "running"
            for gpu in gpus:
                occupancy = sum(r.get("gpu_uuid") == gpu["uuid"] for r in state["jobs"].values() if r["state"] == "running")
                while pending and occupancy < args.per_gpu and not stop:
                    # Re-read immediately before every launch; do not infer host ID from cuda:0.
                    inventory = gpu_inventory()
                    check_gpus(gpus, inventory)
                    if inventory[gpu["index"]]["free_mib"] < args.min_free_mib:
                        break
                    record = pending.pop(0)
                    lock_path = Path(record["output_root"]) / ".revision_locks" / (record["id"].replace("/", "_") + ".lock")
                    lock_path.parent.mkdir(parents=True, exist_ok=True)
                    lock_stream = lock_path.open("a+")
                    try:
                        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        lock_stream.close()
                        record.update(state="failed", error="another scheduler owns the run output lock", finished_at=timestamp())
                        continue
                    # Recheck after acquiring lock, because another scheduler can have just finished.
                    if Path(record["run_root"]).exists():
                        lock_stream.close()
                        accept_or_fail(record, "output appeared before launch")
                        continue
                    try:
                        record["launch_provenance"] = collect_launch_provenance(record)
                    except (OSError, ValueError, subprocess.SubprocessError) as exc:
                        lock_stream.close()
                        record.update(state="failed", error=f"launch provenance failed: {exc}", finished_at=timestamp())
                        continue
                    log_path = state_path.parent / "logs" / (record["id"].replace("/", "_") + ".log")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    record.update(state="running", process_token=uuid.uuid4().hex,
                                  gpu_uuid=gpu["uuid"], gpu_host_index=gpu["index"],
                                  started_at=timestamp(), log=str(log_path), pid=None)
                    atomic_json(state_path, state)  # persist token before the launch/save window
                    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1",
                        "MM_MIXER_REVISION_JOB_TOKEN": record["process_token"],
                        "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
                        "OPENBLAS_NUM_THREADS": str(args.threads), "NUMEXPR_NUM_THREADS": str(args.threads)}
                    try:
                        with log_path.open("ab") as log:
                            process = subprocess.Popen(record["command"], cwd=record["code_root"],
                                env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, pass_fds=(lock_stream.fileno(),))
                        processes[record["id"]] = process
                        job_locks[record["id"]] = lock_stream
                        identity = process_identity(process.pid)
                        record.update(pid=process.pid, start_ticks=identity["start_ticks"] if identity else None)
                        occupancy += 1
                    except OSError as exc:
                        lock_stream.close()
                        record.update(state="failed", error=f"launch failed: {exc}", finished_at=timestamp())
                    atomic_json(state_path, state)
                    # The next pass measures memory after the new process has loaded its model.
                    break
            atomic_json(state_path.with_name("summary.json"), summarize_results(jobs, state))
            if args.once:
                break
            for _ in range(max(1, int(args.poll_seconds * 5))):
                if stop:
                    break
                time.sleep(0.2)
        state["queue_state"] = "stopped_children_preserved"
        state["updated_at"] = timestamp()
        atomic_json(state_path, state)
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan_parser = sub.add_parser("plan", help="write a fixed experiment matrix; never starts training")
    plan_parser.add_argument("--matrix", choices=("full", "base", "controls"), default="full")
    plan_parser.add_argument("--plan", required=True, type=Path)
    plan_parser.add_argument("--code-root", required=True)
    plan_parser.add_argument("--output-root", required=True)
    plan_parser.add_argument("--python", required=True)
    for action in ("validate", "run", "summarize"):
        item = sub.add_parser(action)
        item.add_argument("--plan", required=True, type=Path)
        if action != "validate":
            item.add_argument("--state", required=True)
        if action == "summarize":
            item.add_argument("--output", required=True, type=Path)
        if action == "run":
            item.add_argument("--server", choices=("local", "biggpu"), required=True)
            item.add_argument("--gpu", action="append", default=[], help="host-index:GPU-uuid; repeat for whitelist")
            item.add_argument("--per-gpu", type=int, default=1)
            item.add_argument("--min-free-mib", type=int, default=6000)
            item.add_argument("--threads", type=int, default=2)
            item.add_argument("--poll-seconds", type=float, default=20)
            item.add_argument("--once", action="store_true", help="one scheduling pass; preserve any launched children")
    args = parser.parse_args(argv)
    if args.action == "plan":
        payload = generate_plan(args.matrix, args.code_root, args.output_root, args.python)
        write_new_plan(args.plan, payload)
        print(json.dumps({"plan": str(args.plan.resolve()), "matrix": args.matrix,
                          "jobs": len(payload["jobs"]), "training_started": False}))
        return 0
    jobs = load_plan(args.plan)
    if args.action == "validate":
        missing = [str(path) for j in jobs for path in (Path(j["code_root"]) / "run.py", Path(j["python"])) if not path.is_file()]
        if missing:
            raise ValueError(f"missing executables: {missing}")
        print(json.dumps({"jobs": len(jobs), "selection": "strict_peak_test_wf1", "commands": [command_for(j) for j in jobs]}, indent=2))
        return 0
    if args.action == "summarize":
        state = json.loads(Path(args.state).read_text()) if Path(args.state).exists() else {"jobs": {}}
        atomic_json(args.output, summarize_results(jobs, state))
        return 0
    if min(args.per_gpu, args.min_free_mib, args.threads, args.poll_seconds) <= 0:
        raise ValueError("concurrency, memory threshold, threads and poll interval must be positive")
    return run_queue(args, jobs)


if __name__ == "__main__":
    raise SystemExit(main())
