#!/usr/bin/env python3
"""Merge and validate sharded MELD DenseFace feature exports."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from extract_meld import sha256
from model import DenseFace


def expected_keys(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    keys = [f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}" for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate utterance key in metadata CSV: {csv_path}")
    return keys


def load_progress(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record["utterance_id"])
            if key in records:
                raise ValueError(f"duplicate key {key!r} inside {path}:{line_number}")
            records[key] = record
    return records


def validate_vector(key: str, vector, source: Path) -> list[float]:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (DenseFace.feature_dim,):
        raise ValueError(f"{key!r} in {source} has shape {array.shape}, expected (342,)")
    if not np.isfinite(array).all():
        raise ValueError(f"{key!r} in {source} contains non-finite values")
    return array.astype(np.float32).tolist()


def validate_shard_roots(shard_roots: list[Path]) -> list[dict]:
    manifests = []
    for root in shard_roots:
        path = root / "extraction_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    counts = {int(manifest["shard"]["num_shards"]) for manifest in manifests}
    if len(counts) != 1:
        raise ValueError("feature shards disagree on num_shards")
    num_shards = counts.pop()
    indices = [int(manifest["shard"]["shard_index"]) for manifest in manifests]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate feature shard index")
    if set(indices) != set(range(num_shards)):
        raise ValueError(f"incomplete feature shards: got {sorted(indices)}, expected 0..{num_shards - 1}")
    return manifests


def merge_split(
    split: str,
    csv_path: Path,
    shard_roots: list[Path],
    output_root: Path,
) -> dict:
    ordered_expected = expected_keys(csv_path)
    expected = set(ordered_expected)
    merged_features: dict[str, list[float]] = {}
    merged_progress: dict[str, dict] = {}
    for shard_root in shard_roots:
        split_root = shard_root / f"{split}_features"
        features_path = split_root / "visual_features.json"
        progress_path = split_root / "denseface_progress.jsonl"
        features = json.loads(features_path.read_text(encoding="utf-8"))
        progress = load_progress(progress_path)
        if set(features) != set(progress):
            raise ValueError(
                f"feature/progress key mismatch in {split_root}: "
                f"feature_only={sorted(set(features) - set(progress))[:5]}, "
                f"progress_only={sorted(set(progress) - set(features))[:5]}"
            )
        duplicates = set(features) & set(merged_features)
        if duplicates:
            raise ValueError(f"duplicate keys across feature shards: {sorted(duplicates)[:5]}")
        for key, vector in features.items():
            checked = validate_vector(key, vector, features_path)
            progress_vector = validate_vector(key, progress[key]["feature"], progress_path)
            if checked != progress_vector:
                raise ValueError(f"feature/progress vector mismatch for {key!r} in {split_root}")
            merged_features[key] = checked
            merged_progress[key] = progress[key]

    unexpected = sorted(set(merged_features) - expected)
    missing = [key for key in ordered_expected if key not in merged_features]
    zero_vector = [0.0] * DenseFace.feature_dim
    for key in missing:
        merged_features[key] = zero_vector.copy()
        merged_progress[key] = {
            "utterance_id": key,
            "feature": zero_vector.copy(),
            "status": "missing_video_directory",
            "selected_track": None,
            "candidate_tracks": 0,
            "valid_frames": 0,
        }

    output_dir = output_root / f"{split}_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_features = {key: merged_features[key] for key in ordered_expected}
    (output_dir / "visual_features.json").write_text(
        json.dumps(ordered_features, separators=(",", ":")), encoding="utf-8"
    )
    with (output_dir / "denseface_progress.jsonl").open("w", encoding="utf-8") as handle:
        for key in ordered_expected:
            handle.write(json.dumps(merged_progress[key], separators=(",", ":")) + "\n")
    zero_keys = [
        key
        for key in ordered_expected
        if not np.any(np.asarray(merged_features[key], dtype=np.float32))
    ]
    status_counts: dict[str, int] = {}
    for key in ordered_expected:
        status = str(merged_progress[key].get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    statistics = {
        "split": split,
        "expected_utterances": len(ordered_expected),
        "merged_shard_utterances": len(ordered_expected) - len(missing),
        "ignored_non_metadata_keys": unexpected,
        "ignored_non_metadata_count": len(unexpected),
        "missing_keys": missing,
        "missing_count": len(missing),
        "zero_vector_keys": zero_keys,
        "zero_vector_count": len(zero_keys),
        "nonzero_count": len(ordered_expected) - len(zero_keys),
        "status_counts": dict(sorted(status_counts.items())),
        "metadata_csv": {"path": str(csv_path.resolve()), "sha256": sha256(csv_path)},
    }
    (output_dir / "extraction_stats.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = validate_shard_roots(args.shard_roots)
    summaries = {}
    for split in args.splits:
        summaries[split] = merge_split(
            split,
            args.metadata_root / f"{split}_sent_emo.csv",
            args.shard_roots,
            args.output_root,
        )
    merge_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_dim": DenseFace.feature_dim,
        "shards": [
            {
                "root": str(root.resolve()),
                "manifest_sha256": sha256(root / "extraction_manifest.json"),
                "shard": manifest["shard"],
            }
            for root, manifest in zip(args.shard_roots, manifests)
        ],
        "summaries": summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "merge_manifest.json").write_text(
        json.dumps(merge_manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
