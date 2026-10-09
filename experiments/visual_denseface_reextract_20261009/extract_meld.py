#!/usr/bin/env python3
"""Extract utterance-level 342-D DenseFace features from MELD face tracks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import DenseFace, checkpoint_model  # noqa: E402


Selector = Callable[[str, Path, list[Path], Path | None], Path | str | None]


def validate_shard(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")


def shard_items(items: list, num_shards: int, shard_index: int) -> list:
    """Return a deterministic disjoint strided shard of an ordered list."""
    validate_shard(num_shards, shard_index)
    return items[shard_index::num_shards]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection_manifest(path: Path | None, split: str) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if split in payload and isinstance(payload[split], dict):
        payload = payload[split]
    if not isinstance(payload, dict):
        raise ValueError("selection manifest must be an object mapping utterance IDs to face files")
    return {str(key): str(value) for key, value in payload.items()}


def load_selector_plugin(specification: str | None) -> Selector | None:
    if not specification:
        return None
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("selector plugin must use module:function syntax")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"selector plugin is not callable: {specification}")
    return function


def read_rgb_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame is not None and frame.size:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


class LipSyncSelector:
    """Adapter for the existing MELD LipSyncNet speaker-track checkpoint."""

    def __init__(self, utils_path: Path, checkpoint: Path, device: torch.device, frames: int = 32):
        spec = importlib.util.spec_from_file_location("mm_mixer_meld_lipsync_utils", utils_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import LipSyncNet utilities from {utils_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.module = module
        self.device = device
        self.frames = frames
        self.model = module.LipSyncNet(fixed_num_frames=frames).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self, utterance_id: str, utterance_dir: Path, candidates: list[Path], audio_path: Path | None
    ) -> Path | None:
        if audio_path is None or not audio_path.exists():
            return None
        # Importing librosa is intentionally confined to this optional adapter.
        audio, sample_rate = self.module.librosa.load(str(audio_path), sr=16000)
        audio_tensor = self.module.compute_audio_features(
            audio, sample_rate, fixed_num_frames=self.frames
        ).unsqueeze(0).to(self.device)
        scores: dict[Path, float] = {}
        for candidate in candidates:
            frames = read_rgb_frames(candidate)
            if not frames:
                continue
            video_tensor = self.module.prepare_video_tensor(
                frames, fixed_num_frames=self.frames
            ).unsqueeze(0).to(self.device)
            video_features, audio_features = self.model(video_tensor, audio_tensor)
            scores[candidate] = float(F.pairwise_distance(video_features, audio_features).item())
        return min(scores, key=scores.get) if scores else None


def choose_track(
    utterance_id: str,
    utterance_dir: Path,
    candidates: list[Path],
    audio_path: Path | None,
    manifest: dict[str, str],
    selector: Selector | None,
) -> tuple[Path | None, str]:
    requested = manifest.get(utterance_id)
    if requested is not None:
        selected = utterance_dir / requested
        return (selected, "manifest") if selected in candidates else (None, "manifest_missing")
    if len(candidates) == 1:
        return candidates[0], "single_track"
    if not candidates:
        return None, "no_face_track"
    if selector is None:
        return None, "multiple_tracks_without_selector"
    selected = selector(utterance_id, utterance_dir, candidates, audio_path)
    if selected is None:
        return None, "selector_failed"
    selected = Path(selected)
    if not selected.is_absolute():
        selected = utterance_dir / selected
    return (selected, "selector") if selected in candidates else (None, "selector_invalid")


def preprocess_frames(frames: list[np.ndarray], mean: float, std: float) -> torch.Tensor:
    processed = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_LINEAR)
        processed.append(resized)
    array = np.asarray(processed, dtype=np.float32)[:, None, :, :]
    return torch.from_numpy((array - mean) / std)


@torch.inference_mode()
def average_track_features(
    model: DenseFace,
    track_path: Path,
    device: torch.device,
    batch_size: int,
    mean: float,
    std: float,
) -> tuple[np.ndarray | None, int]:
    frames = read_rgb_frames(track_path)
    if not frames:
        return None, 0
    tensors = preprocess_frames(frames, mean, std)
    chunks = []
    for start in range(0, len(tensors), batch_size):
        chunks.append(model.forward_features(tensors[start : start + batch_size].to(device)).cpu())
    features = torch.cat(chunks, dim=0)
    return features.mean(dim=0).numpy().astype(np.float32), len(frames)


def load_progress(path: Path) -> dict[str, dict]:
    records = {}
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records[record["utterance_id"]] = record
    return records


def extract_split(args: argparse.Namespace, model: DenseFace, split: str, selector: Selector | None) -> None:
    video_root = args.local_meld / f"{split}_video"
    audio_root = args.local_meld / f"{split}_audio"
    if not video_root.is_dir():
        raise FileNotFoundError(video_root)
    output_dir = args.output_root / f"{split}_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "denseface_progress.jsonl"
    completed = load_progress(progress_path) if args.resume else {}
    selection_manifest = load_selection_manifest(args.selection_manifest, split)
    utterance_dirs = sorted(path for path in video_root.iterdir() if path.is_dir())
    utterance_dirs = shard_items(utterance_dirs, args.num_shards, args.shard_index)
    if args.max_samples is not None:
        utterance_dirs = utterance_dirs[: args.max_samples]

    with progress_path.open("a" if args.resume else "w", encoding="utf-8") as progress:
        for index, utterance_dir in enumerate(utterance_dirs, 1):
            utterance_id = utterance_dir.name
            if utterance_id in completed:
                continue
            candidates = sorted(utterance_dir.glob("face_*.mp4"))
            audio_path = audio_root / f"{utterance_id}.wav"
            selected, selection_status = choose_track(
                utterance_id,
                utterance_dir,
                candidates,
                audio_path if audio_path.exists() else None,
                selection_manifest,
                selector,
            )
            feature = None
            valid_frames = 0
            status = selection_status
            if selected is not None:
                try:
                    feature, valid_frames = average_track_features(
                        model,
                        selected,
                        torch.device(args.device),
                        args.batch_size,
                        args.normalization_mean,
                        args.normalization_std,
                    )
                    status = "ok" if feature is not None else "decode_empty"
                except Exception as error:  # preserve an explicit per-utterance failure record
                    status = f"feature_error:{type(error).__name__}"
            if feature is None:
                feature = np.zeros(DenseFace.feature_dim, dtype=np.float32)
            record = {
                "utterance_id": utterance_id,
                "feature": feature.tolist(),
                "status": status,
                "selected_track": selected.name if selected is not None else None,
                "candidate_tracks": len(candidates),
                "valid_frames": valid_frames,
            }
            progress.write(json.dumps(record, separators=(",", ":")) + "\n")
            progress.flush()
            completed[utterance_id] = record
            if index % 100 == 0:
                print(f"{split}: {index}/{len(utterance_dirs)}")

    ordered_records = [completed[path.name] for path in utterance_dirs]
    features = {record["utterance_id"]: record["feature"] for record in ordered_records}
    (output_dir / "visual_features.json").write_text(
        json.dumps(features, separators=(",", ":")), encoding="utf-8"
    )
    all_counts = Counter(record["status"] for record in ordered_records)
    statistics = {
        "split": split,
        "utterances": len(ordered_records),
        "nonzero": sum(record["status"] == "ok" for record in ordered_records),
        "zero_filled": sum(record["status"] != "ok" for record in ordered_records),
        "status_counts": dict(sorted(all_counts.items())),
        "smoke_or_subset": args.max_samples is not None,
    }
    (output_dir / "extraction_stats.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-meld", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        required=True,
        help="JSON produced by compute_meld_stats.py from selected MELD train face frames",
    )
    parser.add_argument("--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test"))
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--selector-plugin", help="target-face selector as module:function")
    parser.add_argument("--lipsync-utils", type=Path)
    parser.add_argument("--lipsync-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, help="bounded smoke/subset extraction")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    normalization = json.loads(args.normalization_stats.read_text(encoding="utf-8"))
    args.normalization_mean = float(normalization["mean"])
    args.normalization_std = float(normalization["std"])
    if args.normalization_std <= 0:
        raise ValueError("normalization std must be positive")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = checkpoint_model(checkpoint, device).eval()
    selector = load_selector_plugin(args.selector_plugin)
    if args.lipsync_utils or args.lipsync_checkpoint:
        if not args.lipsync_utils or not args.lipsync_checkpoint:
            raise ValueError("--lipsync-utils and --lipsync-checkpoint must be supplied together")
        if selector is not None:
            raise ValueError("choose either --selector-plugin or the LipSyncNet adapter")
        selector = LipSyncSelector(args.lipsync_utils, args.lipsync_checkpoint, device)
    requested_output_root = args.output_root
    if args.num_shards > 1:
        args.output_root = requested_output_root / (
            f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}"
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        extract_split(args, model, split, selector)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "extractor": "locally retrained DenseNet-BC-100 FER+",
        "feature_dim": DenseFace.feature_dim,
        "frame_processing": "each decoded face-track frame -> grayscale -> 64x64 -> DenseFace 342-D",
        "utterance_pooling": "arithmetic mean over valid frames of the selected target-speaker track",
        "zero_policy": "explicit 342-D zero for missing selection, missing face, empty decode, or extraction error",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "normalization": {
            "source": str(args.normalization_stats.resolve()),
            "sha256": sha256(args.normalization_stats),
            "mean": args.normalization_mean,
            "std": args.normalization_std,
            "population": normalization,
        },
        "selection_manifest": str(args.selection_manifest.resolve()) if args.selection_manifest else None,
        "selector_plugin": args.selector_plugin,
        "lipsync_checkpoint": str(args.lipsync_checkpoint.resolve()) if args.lipsync_checkpoint else None,
        "splits": list(args.splits),
        "max_samples": args.max_samples,
        "shard": {"num_shards": args.num_shards, "shard_index": args.shard_index},
        "requested_output_root": str(requested_output_root.resolve()),
    }
    (args.output_root / "extraction_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
