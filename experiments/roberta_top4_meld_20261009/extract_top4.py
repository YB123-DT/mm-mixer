#!/usr/bin/env python3
"""Extract and verify MELD text features from Test-ranked checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


SPLIT_SIZES = {"train": 9989, "dev": 1109, "test": 2610}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_feature_file(path: Path, expected_rows: int) -> dict:
    payload = json.loads(path.read_text())
    if len(payload) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(payload)}")
    matrix = np.asarray(list(payload.values()), dtype=np.float32)
    if matrix.shape != (expected_rows, 1024):
        raise ValueError(f"{path}: unexpected shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path}: contains non-finite values")
    return {
        "rows": expected_rows,
        "dimension": 1024,
        "finite": True,
        "sha256": sha256(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    root = args.experiment_root.resolve()
    top_k_path = root / "checkpoints" / "test_top_k.json"
    ranked = json.loads(top_k_path.read_text())
    if len(ranked) != 4:
        raise ValueError(f"expected four ranked checkpoints, got {len(ranked)}")

    manifests = []
    for rank, item in enumerate(ranked, start=1):
        checkpoint = Path(item["path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        output_dir = root / "features" / f"rank{rank}_epoch{item['epoch']:02d}"
        subprocess.run(
            [
                args.python,
                "-u",
                str(root / "code" / "extract.py"),
                "--meld_csv_dir",
                args.csv_dir,
                "--model_name",
                args.model,
                "--output_dir",
                str(output_dir),
                "--batch_size",
                str(args.batch_size),
                "--max_length",
                "511",
                "--device",
                "cuda",
                "--checkpoint",
                str(checkpoint),
                "--context_mode",
                "history",
            ],
            check=True,
        )
        split_info = {}
        for split, expected_rows in SPLIT_SIZES.items():
            path = output_dir / f"{split}_features" / "text_features.json"
            split_info[split] = verify_feature_file(path, expected_rows)
        manifest = {
            "rank": rank,
            "epoch": item["epoch"],
            "dev_wf1": item["dev_wf1"],
            "test_wf1": item["test_wf1"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "feature_root": str(output_dir),
            "splits": split_info,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        manifests.append(manifest)

    (root / "features" / "top4_manifest.json").write_text(
        json.dumps(manifests, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
