#!/usr/bin/env python3
"""Evaluate one trained MELD model under test-time modality deletion."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


MODALITY_SETS = ("tav", "tv", "ta", "av", "t", "a", "v")
MODEL_MODALITIES = ("v", "a", "t")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    code_root = args.code_root.resolve()
    config_path = run / "config.json"
    checkpoint_path = run / "best_peak" / "best_peak_test_state_dict.pt"
    reference_path = run / "best_peak" / "peak_test_metrics.json"
    config = load_json(config_path)
    reference = load_json(reference_path)

    vendor = code_root / "vendor" / "meld"
    sys.path[:0] = [str(code_root), str(vendor)]
    model_module = importlib.import_module("model")

    fixed = config["fixed_params"]
    model = model_module.build_model(fixed["capacity_variant"], fixed["fusion_dropout"])
    model.disable_alignment()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    paths = config["feature_paths"]["meld"]["test"]
    features = {
        "t": load_json(Path(paths["text"])),
        "a": load_json(Path(paths["audio"])),
        "v": load_json(Path(paths["visual"])),
    }
    dims = config["embed_dims_full"]
    frame = pd.read_csv(args.csv_dir / "test_sent_emo.csv")
    classes = config["classes"]["meld"]
    class_to_index = {name: index for index, name in enumerate(classes)}
    labels = torch.tensor(
        [class_to_index[value] for value in frame["Emotion"]], dtype=torch.long
    )
    keys = [
        f"dia{dialogue}_utt{utterance}"
        for dialogue, utterance in zip(frame["Dialogue_ID"], frame["Utterance_ID"])
    ]
    tensors = {
        modality: torch.tensor(
            np.stack(
                [
                    np.asarray(
                        features[modality].get(key, np.zeros(dims[modality])),
                        dtype=np.float32,
                    )
                    for key in keys
                ]
            ),
            dtype=torch.float32,
        )
        for modality in MODEL_MODALITIES
    }

    results = {}
    with torch.inference_mode():
        for kept in MODALITY_SETS:
            logits = []
            for start in range(0, len(labels), args.batch_size):
                stop = min(start + args.batch_size, len(labels))
                batch = {
                    modality: (
                        tensors[modality][start:stop].to(device)
                        if modality in kept
                        else torch.zeros_like(tensors[modality][start:stop], device=device)
                    )
                    for modality in MODEL_MODALITIES
                }
                output = model(batch)
                logits.append(output[0] if isinstance(output, tuple) else output)
            logits = torch.cat(logits).cpu()
            predictions = logits.argmax(dim=1)
            y_true, y_pred = labels.numpy(), predictions.numpy()
            results[kept] = {
                "kept_modalities": list(kept),
                "deleted_modalities": [m for m in "tav" if m not in kept],
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "class_f1": {
                    name: float(score)
                    for name, score in zip(
                        classes,
                        f1_score(
                            y_true,
                            y_pred,
                            labels=list(range(len(classes))),
                            average=None,
                            zero_division=0,
                        ),
                    )
                },
            }

    tolerance = 1e-12
    replay_delta = abs(results["tav"]["weighted_f1"] - reference["weighted_f1"])
    if replay_delta > tolerance:
        raise RuntimeError(
            f"TAV replay mismatch: {results['tav']['weighted_f1']} vs "
            f"{reference['weighted_f1']} (delta={replay_delta})"
        )

    payload = {
        "protocol": "single-checkpoint test-time raw-input zero deletion",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config": str(config_path),
        "reference_peak_epoch": reference["epoch"],
        "reference_replay_delta": replay_delta,
        "sample_count": len(labels),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
