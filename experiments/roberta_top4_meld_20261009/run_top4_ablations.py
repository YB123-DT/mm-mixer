#!/usr/bin/env python3
"""Run one-seed MELD ablation screening across four text checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


VARIANTS = (
    "full",
    "modal_t",
    "no_mixer",
    "no_pairwise",
    "no_cross_attention",
    "no_feature_and_adaptive_gating",
)


def run_one(
    python: str,
    code_root: Path,
    output_root: Path,
    log_root: Path,
    feature_root: str,
    rank: int,
    variant: str,
) -> dict:
    run_root = output_root / f"rank{rank}"
    log_path = log_root / f"rank{rank}_{variant}.log"
    env = os.environ.copy()
    env["MM_MIXER_MELD_TEXT_FEATURE_ROOT"] = feature_root
    command = [
        python,
        "-u",
        "run.py",
        "run",
        "--dataset",
        "meld",
        "--variant",
        variant,
        "--seed",
        "2025",
        "--output-root",
        str(run_root),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=code_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise RuntimeError(
            f"rank {rank} {variant} failed with {completed.returncode}; see {log_path}"
        )

    selected = run_root / "meld" / variant / "seed2025" / "best_peak"
    metrics = json.loads((selected / "peak_test_metrics.json").read_text())
    manifest = json.loads((selected / "manifest.json").read_text())
    status = json.loads((selected / "status.json").read_text())
    expected_text = str(
        Path(feature_root) / "test_features" / "text_features.json"
    )
    if not manifest.get("fresh_strict_replay_exact"):
        raise RuntimeError(f"rank {rank} {variant}: strict replay failed")
    if expected_text not in manifest.get("feature_hashes", {}):
        raise RuntimeError(f"rank {rank} {variant}: wrong text feature path")
    return {
        "rank": rank,
        "variant": variant,
        "epoch": metrics["epoch"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "fresh_strict_replay_exact": status["fresh_strict_replay_exact"],
        "text_feature_path": expected_text,
        "text_feature_sha256": manifest["feature_hashes"][expected_text],
        "bundle": str(selected.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    feature_manifest = json.loads(
        (root / "features" / "top4_manifest.json").read_text()
    )
    code_root = root / "downstream_code"
    output_root = root / "downstream_runs"
    log_root = root / "logs" / "downstream"
    tasks = [
        (entry["rank"], entry["feature_root"], variant)
        for entry in feature_manifest
        for variant in VARIANTS
    ]

    rows = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(
                run_one,
                args.python,
                code_root,
                output_root,
                log_root,
                feature_root,
                rank,
                variant,
            ): (rank, variant)
            for rank, feature_root, variant in tasks
        }
        for future in as_completed(futures):
            rank, variant = futures[future]
            row = future.result()
            rows.append(row)
            print(
                f"rank{rank} {variant}: WF1={100 * row['weighted_f1']:.2f}",
                flush=True,
            )

    rows.sort(key=lambda row: (row["rank"], VARIANTS.index(row["variant"])))
    result_dir = root / "analysis"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "screening_results.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    with (result_dir / "screening_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
