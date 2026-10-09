#!/usr/bin/env python3
"""Merge disjoint MELD train face-statistics shards exactly by pixel counts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from extract_meld import sha256


def merge_statistics(paths: list[Path]) -> dict:
    if not paths:
        raise ValueError("at least one statistics shard is required")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    declared_num_shards = {int(payload["shard"]["num_shards"]) for payload in payloads}
    if len(declared_num_shards) != 1:
        raise ValueError("statistics shards disagree on num_shards")
    num_shards = declared_num_shards.pop()
    indices = [int(payload["shard"]["shard_index"]) for payload in payloads]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate statistics shard index")
    if set(indices) != set(range(num_shards)):
        raise ValueError(f"incomplete statistics shards: got {sorted(indices)}, expected 0..{num_shards - 1}")
    if any(payload.get("subset") for payload in payloads):
        raise ValueError("cannot produce formal statistics from --max-samples subset shards")

    pixel_count = sum(int(payload["pixel_count"]) for payload in payloads)
    pixel_sum = sum(float(payload["pixel_sum"]) for payload in payloads)
    pixel_square_sum = sum(float(payload["pixel_square_sum"]) for payload in payloads)
    if pixel_count <= 0:
        raise ValueError("merged statistics contain no pixels")
    mean = pixel_sum / pixel_count
    std = max(pixel_square_sum / pixel_count - mean * mean, 0.0) ** 0.5
    statuses: dict[str, int] = {}
    for payload in payloads:
        for status, count in payload["status_counts"].items():
            statuses[status] = statuses.get(status, 0) + int(count)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MELD train selected target-speaker face tracks",
        "pixel_pipeline": "decoded RGB -> grayscale -> bilinear resize 64x64",
        "mean": mean,
        "std": std,
        "pixel_sum": pixel_sum,
        "pixel_square_sum": pixel_square_sum,
        "population_std": True,
        "valid_frames": sum(int(payload["valid_frames"]) for payload in payloads),
        "pixel_count": pixel_count,
        "utterances_considered": sum(int(payload["utterances_considered"]) for payload in payloads),
        "status_counts": dict(sorted(statuses.items())),
        "subset": False,
        "merged_shards": [
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "shard_index": int(payload["shard"]["shard_index"]),
            }
            for path, payload in sorted(
                zip(paths, payloads), key=lambda item: int(item[1]["shard"]["shard_index"])
            )
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_statistics(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
