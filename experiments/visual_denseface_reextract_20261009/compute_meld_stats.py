#!/usr/bin/env python3
"""Compute reproducible grayscale statistics from selected MELD train faces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

from extract_meld import (
    LipSyncSelector,
    choose_track,
    load_selection_manifest,
    load_selector_plugin,
    read_rgb_frames,
    shard_items,
    sha256,
    validate_shard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-meld", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--selector-plugin", help="target-face selector as module:function")
    parser.add_argument("--lipsync-utils", type=Path)
    parser.add_argument("--lipsync-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-samples", type=int, help="bounded plumbing check; not valid final statistics")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    video_root = args.local_meld / "train_video"
    audio_root = args.local_meld / "train_audio"
    manifest = load_selection_manifest(args.selection_manifest, "train")
    selector = load_selector_plugin(args.selector_plugin)
    if args.lipsync_utils or args.lipsync_checkpoint:
        if not args.lipsync_utils or not args.lipsync_checkpoint:
            raise ValueError("--lipsync-utils and --lipsync-checkpoint must be supplied together")
        if selector is not None:
            raise ValueError("choose either --selector-plugin or the LipSyncNet adapter")
        selector = LipSyncSelector(
            args.lipsync_utils, args.lipsync_checkpoint, torch.device(args.device)
        )

    utterance_dirs = sorted(path for path in video_root.iterdir() if path.is_dir())
    utterance_dirs = shard_items(utterance_dirs, args.num_shards, args.shard_index)
    if args.max_samples is not None:
        utterance_dirs = utterance_dirs[: args.max_samples]
    pixel_count = 0
    value_sum = 0.0
    square_sum = 0.0
    valid_frames = 0
    statuses = Counter()
    for index, utterance_dir in enumerate(utterance_dirs, 1):
        utterance_id = utterance_dir.name
        candidates = sorted(utterance_dir.glob("face_*.mp4"))
        audio_path = audio_root / f"{utterance_id}.wav"
        selected, status = choose_track(
            utterance_id,
            utterance_dir,
            candidates,
            audio_path if audio_path.exists() else None,
            manifest,
            selector,
        )
        if selected is None:
            statuses[status] += 1
            continue
        frames = read_rgb_frames(selected)
        if not frames:
            statuses["decode_empty"] += 1
            continue
        statuses["ok"] += 1
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float64)
            # Statistics are computed on the exact 64x64 tensors used downstream.
            gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
            pixel_count += gray.size
            value_sum += float(gray.sum(dtype=np.float64))
            square_sum += float(np.square(gray).sum(dtype=np.float64))
            valid_frames += 1
        if index % 100 == 0:
            print(f"train: {index}/{len(utterance_dirs)}")
    if pixel_count == 0:
        raise RuntimeError("no selected valid MELD train face pixels; provide a selection method")
    mean = value_sum / pixel_count
    std = max(square_sum / pixel_count - mean * mean, 0.0) ** 0.5
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "MELD train selected target-speaker face tracks",
        "pixel_pipeline": "decoded RGB -> grayscale -> bilinear resize 64x64",
        "mean": mean,
        "std": std,
        "pixel_sum": value_sum,
        "pixel_square_sum": square_sum,
        "population_std": True,
        "valid_frames": valid_frames,
        "pixel_count": pixel_count,
        "utterances_considered": len(utterance_dirs),
        "status_counts": dict(sorted(statuses.items())),
        "subset": args.max_samples is not None,
        "shard": {"num_shards": args.num_shards, "shard_index": args.shard_index},
        "selection_manifest": (
            {"path": str(args.selection_manifest.resolve()), "sha256": sha256(args.selection_manifest)}
            if args.selection_manifest
            else None
        ),
        "selector_plugin": args.selector_plugin,
        "lipsync_checkpoint": (
            {"path": str(args.lipsync_checkpoint.resolve()), "sha256": sha256(args.lipsync_checkpoint)}
            if args.lipsync_checkpoint
            else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
