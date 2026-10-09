#!/usr/bin/env python3
"""Rebuild standard FER2013 CSV from a public Parquet mirror reproducibly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


SPLITS = (
    ("train", "Training", 28709),
    ("valid", "PublicTest", 3589),
    ("test", "PrivateTest", 3589),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_image(image_record: dict) -> np.ndarray:
    data = image_record.get("bytes")
    path = image_record.get("path")
    if data is not None:
        image = Image.open(io.BytesIO(data))
    elif path:
        image = Image.open(path)
    else:
        raise ValueError("Parquet image record has neither bytes nor path")
    pixels = np.asarray(image.convert("L"), dtype=np.uint8)
    if pixels.shape != (48, 48):
        raise ValueError(f"expected a 48x48 FER image, got {pixels.shape}")
    return pixels


def prepare(parquets: dict[str, Path], output: Path, manifest_path: Path) -> dict:
    for split, _, expected_count in SPLITS:
        count = pq.ParquetFile(parquets[split]).metadata.num_rows
        if count != expected_count:
            raise ValueError(f"{split} contains {count} rows, expected {expected_count}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    first_samples = {}
    counts = Counter()
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("emotion", "pixels", "Usage"))
        for split, usage, _ in SPLITS:
            row_index = 0
            parquet = pq.ParquetFile(parquets[split])
            for batch in parquet.iter_batches(batch_size=256, columns=("label", "image")):
                labels = batch.column("label").to_pylist()
                images = batch.column("image").to_pylist()
                for label, image_record in zip(labels, images):
                    label = int(label)
                    if not 0 <= label <= 6:
                        raise ValueError(f"invalid FER2013 label {label} in {split} row {row_index}")
                    pixels = decode_image(image_record).reshape(-1)
                    pixel_text = " ".join(map(str, pixels.tolist()))
                    writer.writerow((label, pixel_text, usage))
                    if row_index == 0:
                        first_samples[split] = {
                            "label": label,
                            "usage": usage,
                            "pixel_count": int(pixels.size),
                            "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                            "pixel_prefix": pixels[:16].tolist(),
                        }
                    row_index += 1
                    counts[usage] += 1
    os.replace(temporary, output)

    # Verify the serialized CSV, including split boundary rows and Usage counts.
    boundary_indices = {}
    offset = 0
    for split, usage, count in SPLITS:
        boundary_indices[offset] = (split, usage)
        offset += count
    verified_counts = Counter()
    verified_first = set()
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            usage = row["Usage"]
            verified_counts[usage] += 1
            if index in boundary_indices:
                split, expected_usage = boundary_indices[index]
                if usage != expected_usage:
                    raise ValueError(
                        f"serialized Usage mismatch at row {index}: {usage} != {expected_usage}"
                    )
                pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")
                sample = first_samples[split]
                if int(row["emotion"]) != sample["label"] or hashlib.sha256(
                    pixels.tobytes()
                ).hexdigest() != sample["pixel_sha256"]:
                    raise ValueError(f"serialized first sample mismatch for {split}")
                verified_first.add(split)
    expected_usage_counts = {usage: count for _, usage, count in SPLITS}
    if dict(verified_counts) != expected_usage_counts:
        raise ValueError(
            f"serialized Usage counts mismatch: {dict(verified_counts)} != {expected_usage_counts}"
        )
    if verified_first != {split for split, _, _ in SPLITS}:
        raise ValueError("not all split first samples were verified")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_source": "AutumnQiu/fer2013 public Hugging Face mirror (not an official image download)",
        "annotation_source": "Microsoft FER+ fer2013new.csv is supplied separately",
        "split_order": [split for split, _, _ in SPLITS],
        "usage_mapping": {split: usage for split, usage, _ in SPLITS},
        "counts": expected_usage_counts,
        "first_samples": first_samples,
        "inputs": {
            split: {"path": str(parquets[split].resolve()), "sha256": sha256(parquets[split])}
            for split, _, _ in SPLITS
        },
        "output": {"path": str(output.resolve()), "sha256": sha256(output)},
        "verification": {
            "row_count": sum(expected_usage_counts.values()),
            "usage_counts_match": True,
            "first_sample_per_split_matches": True,
            "pixels_per_image": 2304,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--valid-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(
        {"train": args.train_parquet, "valid": args.valid_parquet, "test": args.test_parquet},
        args.output,
        args.manifest,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
