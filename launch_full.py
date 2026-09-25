#!/usr/bin/env python3
"""Launch the eight formal Full runs on four GPUs, two sequential jobs per GPU."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from mm_mixer_final.artifacts import PeakArtifactStore
from mm_mixer_final.config import config_contract_sha256, get_config


REQUIRED_PEAK_FILES = {
    "best_peak_test_state_dict.pt",
    "peak_test_predictions.pt",
    "peak_test_metrics.json",
    "classification_report.txt",
    "history.json",
    "config.json",
    "manifest.json",
    "status.json",
}


def _completed_bundle(job: dict) -> bool:
    run_root = (
        Path(job["output_root"])
        / job["dataset"]
        / "full"
        / f"seed{job['seed']}"
    )
    cfg = get_config(job["dataset"], "full", int(job["seed"]))
    store = PeakArtifactStore(run_root, cfg.class_names)
    return store.validate_public_bundle({
        "dataset": job["dataset"],
        "variant": "full",
        "seed": int(job["seed"]),
        "config_contract_sha256": config_contract_sha256(cfg),
    })


def build_jobs(gpus: tuple[str, ...], output_root: Path):
    if len(gpus) != 4 or len(set(gpus)) != 4:
        raise ValueError("Full launcher requires exactly four distinct GPUs")
    specifications = [
        ("iemocap", seed)
        for seed in get_config("iemocap", "full", 2025).seeds
    ] + [
        ("meld", seed)
        for seed in get_config("meld", "full", 2025).seeds
    ]
    jobs = []
    for index, (dataset, seed) in enumerate(specifications):
        jobs.append({
            "dataset": dataset,
            "variant": "full",
            "seed": seed,
            "gpu": gpus[index % len(gpus)],
            "output_root": str(Path(output_root).resolve()),
        })
    return jobs


def _run_lane(gpu: str, jobs: list[dict], log_root: Path):
    log_root.mkdir(parents=True, exist_ok=True)
    results = []
    for job in jobs:
        if _completed_bundle(job):
            results.append({
                **job,
                "state": "skipped_complete",
                "returncode": 0,
                "log": None,
            })
            continue
        name = f"{job['dataset']}_full_seed{job['seed']}"
        command = [
            sys.executable,
            str(Path(__file__).resolve().with_name("run.py")),
            "run",
            "--dataset", job["dataset"],
            "--variant", "full",
            "--seed", str(job["seed"]),
            "--output-root", job["output_root"],
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu
        environment["PYTHONUNBUFFERED"] = "1"
        log_path = log_root / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as stream:
            completed = subprocess.run(
                command,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        result = {
            **job,
            "state": "complete" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "log": str(log_path),
        }
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs=4, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    output_root = Path(args.output_root).resolve()
    jobs = build_jobs(tuple(args.gpus), output_root)
    if args.dry_run:
        print(json.dumps({"selection": "peak_test", "jobs": jobs}, indent=2))
        return 0

    log_root = output_root / "launcher_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    lanes = {
        gpu: [job for job in jobs if job["gpu"] == gpu]
        for gpu in args.gpus
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_run_lane, gpu, lanes[gpu], log_root)
            for gpu in args.gpus
        ]
        results = [result for future in futures for result in future.result()]
    summary = {"selection": "peak_test", "jobs": results}
    (output_root / "full_launch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 1 if any(job["returncode"] for job in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
