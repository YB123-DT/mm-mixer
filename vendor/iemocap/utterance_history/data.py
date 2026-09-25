from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class UtteranceHistoryDataset(Dataset):
    """Add dialogue-local causal prefixes without changing base indexing."""

    def __init__(self, base: Dataset, history_modalities: Sequence[str] = ("a", "v")):
        self.base = base
        self.index = base.index
        self.history_modalities = tuple(history_modalities)
        if not set(self.history_modalities) <= {"a", "v"}:
            raise ValueError("history modalities must be a subset of {'a', 'v'}")

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        current, label = self.base[index]
        dialogue, turn = self.index[index]
        features = dict(current)
        for modality in self.history_modalities:
            array = self.base.z[f"{dialogue}__{modality}"][: turn + 1].copy()
            history = torch.as_tensor(array, dtype=torch.float32)
            # Use the exact current sample tensor, including train-time augmentation.
            history[-1] = current[modality]
            features[f"{modality}_history"] = history
        features["sample_key"] = f"{dialogue}__{turn}"
        features["history_keys"] = tuple(f"{dialogue}__{i}" for i in range(turn + 1))
        return features, label

    def collate_fn(self, items):
        features, labels = zip(*items)
        current_items = [({m: item[m] for m in ("t", "a", "v")}, label)
                         for item, label in zip(features, labels)]
        batch, labels_batch = self.base.collate_fn(current_items)
        lengths = torch.tensor([len(item["history_keys"]) for item in features], dtype=torch.long)
        width = int(lengths.max())
        batch["history_mask"] = torch.arange(width)[None, :] < lengths[:, None]
        for modality in self.history_modalities:
            batch[f"{modality}_history"] = pad_sequence(
                [item[f"{modality}_history"] for item in features], batch_first=True
            )
        return batch, labels_batch
