"""FER2013/FER+ data loading with an explicit hard-label policy."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


EMOTIONS = (
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
)
ALL_VOTE_COLUMNS = EMOTIONS + ("unknown", "NF")
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "dev", "PrivateTest": "test"}


@dataclass(frozen=True)
class FerPlusRecord:
    image_index: int
    split: str
    label: int


def load_ferplus_records(
    fer2013_csv: str | Path, fer2013new_csv: str | Path
) -> tuple[list[FerPlusRecord], dict[str, int]]:
    """Align official FER+ votes to FER2013 rows and derive hard labels.

    The winning class is computed over all ten FER+ vote columns. Samples
    whose winner is ``unknown`` or ``NF`` (or whose votes are all zero) are
    excluded; the remaining winner is one of the eight emotion classes.
    """
    with Path(fer2013_csv).open(newline="", encoding="utf-8-sig") as handle:
        images = list(csv.DictReader(handle))
    with Path(fer2013new_csv).open(newline="", encoding="utf-8-sig") as handle:
        votes = list(csv.DictReader(handle))
    if len(images) != len(votes):
        raise ValueError(f"FER row mismatch: images={len(images)}, votes={len(votes)}")

    records: list[FerPlusRecord] = []
    stats = {"total": len(images), "kept": 0, "unknown": 0, "NF": 0, "zero_votes": 0}
    for index, (image_row, vote_row) in enumerate(zip(images, votes)):
        image_usage = image_row["Usage"].strip()
        vote_usage = vote_row["Usage"].strip()
        if image_usage != vote_usage:
            raise ValueError(f"Usage mismatch at row {index}: {image_usage!r} != {vote_usage!r}")
        split = USAGE_TO_SPLIT.get(image_usage)
        if split is None:
            raise ValueError(f"unknown FER Usage at row {index}: {image_usage!r}")
        counts = np.asarray([int(vote_row[name] or 0) for name in ALL_VOTE_COLUMNS])
        if int(counts.sum()) == 0:
            stats["zero_votes"] += 1
            continue
        label = int(counts.argmax())
        if label >= len(EMOTIONS):
            stats[ALL_VOTE_COLUMNS[label]] += 1
            continue
        records.append(FerPlusRecord(index, split, label))
        stats["kept"] += 1
    return records, stats


def compute_train_mean_std(
    fer2013_csv: str | Path, fer2013new_csv: str | Path
) -> tuple[float, float, int]:
    """Compute population grayscale statistics from kept FER+ train images."""
    with Path(fer2013_csv).open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    records, _ = load_ferplus_records(fer2013_csv, fer2013new_csv)
    train_indices = [record.image_index for record in records if record.split == "train"]
    total = 0
    value_sum = 0.0
    square_sum = 0.0
    batch: list[np.ndarray] = []

    def accumulate(items: list[np.ndarray]) -> tuple[int, float, float]:
        tensor = torch.from_numpy(np.stack(items).astype(np.float32))[:, None, :, :]
        resized = F.interpolate(tensor, size=(64, 64), mode="bilinear", align_corners=False)
        resized = resized.to(torch.float64)
        return resized.numel(), float(resized.sum()), float(resized.square().sum())

    for index in train_indices:
        pixels = np.fromstring(rows[index]["pixels"], dtype=np.float32, sep=" ")
        if pixels.size != 48 * 48:
            raise ValueError(f"row {index} contains {pixels.size} pixels")
        batch.append(pixels.reshape(48, 48))
        if len(batch) == 256:
            batch_total, batch_sum, batch_square_sum = accumulate(batch)
            total += batch_total
            value_sum += batch_sum
            square_sum += batch_square_sum
            batch.clear()
    if batch:
        batch_total, batch_sum, batch_square_sum = accumulate(batch)
        total += batch_total
        value_sum += batch_sum
        square_sum += batch_square_sum
    if not total:
        raise ValueError("no kept FER+ training pixels")
    mean = value_sum / total
    variance = max(square_sum / total - mean * mean, 0.0)
    return mean, variance**0.5, len(train_indices)


class FerPlusDataset(Dataset):
    def __init__(
        self,
        fer2013_csv: str | Path,
        fer2013new_csv: str | Path,
        split: str,
        augment: bool = False,
        mean: float = 131.0754,
        std: float = 47.858177,
    ) -> None:
        if split not in {"train", "dev", "test"}:
            raise ValueError(f"unsupported split: {split}")
        self.fer2013_csv = Path(fer2013_csv)
        with self.fer2013_csv.open(newline="", encoding="utf-8-sig") as handle:
            self.rows = list(csv.DictReader(handle))
        records, self.filter_stats = load_ferplus_records(fer2013_csv, fer2013new_csv)
        self.records = [record for record in records if record.split == split]
        self.split = split
        self.augment = augment
        self.mean = float(mean)
        self.std = float(std)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int]:
        record = self.records[item]
        pixels = np.fromstring(self.rows[record.image_index]["pixels"], dtype=np.float32, sep=" ")
        if pixels.size != 48 * 48:
            raise ValueError(f"row {record.image_index} contains {pixels.size} pixels")
        image = torch.from_numpy(pixels.reshape(1, 48, 48))
        image = F.interpolate(image.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)[0]
        if self.augment:
            image = F.pad(image, (4, 4, 4, 4), mode="reflect")
            top = int(torch.randint(0, 9, ()).item())
            left = int(torch.randint(0, 9, ()).item())
            image = image[:, top : top + 64, left : left + 64]
            if bool(torch.rand(()) < 0.5):
                image = image.flip(-1)
        image = (image - self.mean) / self.std
        return image, record.label
