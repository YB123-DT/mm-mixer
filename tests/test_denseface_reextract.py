from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = ROOT / "experiments" / "visual_denseface_reextract_20261009"
sys.path.insert(0, str(MODULE_ROOT))

from ferplus import FerPlusDataset, compute_train_mean_std, load_ferplus_records  # noqa: E402
from model import DenseFace  # noqa: E402
from extract_meld import shard_items  # noqa: E402
from train import DEFAULT_EARLY_STOPPING_PATIENCE, checkpoint_selection_score  # noqa: E402
from merge_meld_features import merge_split  # noqa: E402
from merge_meld_stats import merge_statistics  # noqa: E402


def load_extract_module():
    spec = importlib.util.spec_from_file_location("denseface_extract_meld", MODULE_ROOT / "extract_meld.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fixture_csvs(directory: Path) -> tuple[Path, Path]:
    fer = directory / "fer2013.csv"
    votes = directory / "fer2013new.csv"
    pixels = " ".join(str(index % 256) for index in range(48 * 48))
    with fer.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(
            [
                {"emotion": 0, "pixels": pixels, "Usage": "Training"},
                {"emotion": 1, "pixels": pixels, "Usage": "PublicTest"},
                {"emotion": 2, "pixels": pixels, "Usage": "PrivateTest"},
            ]
        )
    columns = [
        "Usage", "Image name", "neutral", "happiness", "surprise", "sadness",
        "anger", "disgust", "fear", "contempt", "unknown", "NF",
    ]
    rows = [
        ["Training", "fer0000000.png", 0, 7, 0, 0, 0, 0, 0, 0, 1, 0],
        ["PublicTest", "fer0000001.png", 0, 0, 0, 0, 0, 0, 0, 0, 8, 2],
        ["PrivateTest", "fer0000002.png", 0, 0, 0, 0, 0, 0, 0, 0, 1, 9],
    ]
    with votes.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return fer, votes


class DenseFaceReextractTest(unittest.TestCase):
    def test_denseface_outputs_342_dimensional_features(self):
        model = DenseFace().eval()
        stage_shapes = {}
        hooks = [
            module.register_forward_hook(
                lambda _module, _inputs, output, name=name: stage_shapes.__setitem__(
                    name, tuple(output.shape)
                )
            )
            for name, module in (
                ("stem", model.stem),
                ("transition1", model.transition1),
                ("transition2", model.transition2),
                ("block3", model.block3),
            )
        ]
        with torch.inference_mode():
            feature_map = model.forward_feature_map(torch.zeros(2, 1, 64, 64))
            features = model.forward_features(torch.zeros(2, 1, 64, 64))
            logits = model(torch.zeros(2, 1, 64, 64))
        for hook in hooks:
            hook.remove()
        self.assertEqual(model.stem.stride, (2, 2))
        self.assertEqual(stage_shapes["stem"][-2:], (32, 32))
        self.assertEqual(stage_shapes["transition1"][-2:], (16, 16))
        self.assertEqual(stage_shapes["transition2"][-2:], (8, 8))
        self.assertEqual(stage_shapes["block3"], (2, 342, 8, 8))
        self.assertEqual(feature_map.shape, (2, 342, 8, 8))
        self.assertEqual(features.shape, (2, 342))
        self.assertEqual(logits.shape, (2, 8))

    def test_ferplus_hard_labels_filter_unknown_and_nf(self):
        with tempfile.TemporaryDirectory() as directory:
            fer, votes = write_fixture_csvs(Path(directory))
            records, stats = load_ferplus_records(fer, votes)
            self.assertEqual([(record.split, record.label) for record in records], [("train", 1)])
            self.assertEqual(stats["unknown"], 1)
            self.assertEqual(stats["NF"], 1)
            dataset = FerPlusDataset(fer, votes, "train")
            image, label = dataset[0]
            self.assertEqual(image.shape, (1, 64, 64))
            self.assertEqual(label, 1)
            mean, std, images = compute_train_mean_std(fer, votes)
            raw = torch.tensor(
                [index % 256 for index in range(48 * 48)], dtype=torch.float32
            ).reshape(1, 1, 48, 48)
            expected = torch.nn.functional.interpolate(
                raw, size=(64, 64), mode="bilinear", align_corners=False
            ).numpy()
            self.assertEqual(images, 1)
            self.assertTrue(np.isclose(mean, expected.mean()))
            self.assertTrue(np.isclose(std, expected.std()))

    def test_track_selection_is_explicit_for_ambiguous_or_manifest_cases(self):
        module = load_extract_module()
        with tempfile.TemporaryDirectory() as directory:
            utterance = Path(directory) / "dia1_utt0"
            utterance.mkdir()
            first = utterance / "face_1.mp4"
            second = utterance / "face_2.mp4"
            first.touch()
            second.touch()
            selected, status = module.choose_track(
                utterance.name, utterance, [first, second], None, {}, None
            )
            self.assertIsNone(selected)
            self.assertEqual(status, "multiple_tracks_without_selector")
            selected, status = module.choose_track(
                utterance.name,
                utterance,
                [first, second],
                None,
                {utterance.name: "face_2.mp4"},
                None,
            )
            self.assertEqual(selected, second)
            self.assertEqual(status, "manifest")

    def test_frame_preprocessing_is_grayscale_64_square(self):
        module = load_extract_module()
        frames = [
            np.zeros((12, 20, 3), dtype=np.uint8),
            np.full((8, 9, 3), 255, dtype=np.uint8),
        ]
        tensors = module.preprocess_frames(frames, mean=0.0, std=1.0)
        self.assertEqual(tensors.shape, (2, 1, 64, 64))
        self.assertEqual(tensors.dtype, torch.float32)

    def test_deterministic_shards_are_disjoint_and_complete(self):
        items = list(range(23))
        shards = [shard_items(items, 4, index) for index in range(4)]
        self.assertEqual(sorted(value for shard in shards for value in shard), items)
        for left in range(4):
            for right in range(left + 1, 4):
                self.assertFalse(set(shards[left]) & set(shards[right]))

    def test_checkpoint_selection_uses_macro_f1_and_default_patience(self):
        metrics = {"accuracy": 0.99, "macro_f1": 0.61, "weighted_f1": 0.95}
        self.assertEqual(checkpoint_selection_score(metrics), 0.61)
        self.assertEqual(DEFAULT_EARLY_STOPPING_PATIENCE, 8)

    def test_feature_merge_rejects_duplicate_keys_across_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "train_sent_emo.csv"
            with metadata.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["Dialogue_ID", "Utterance_ID"])
                writer.writeheader()
                writer.writerow({"Dialogue_ID": 1, "Utterance_ID": 2})
            vector = [0.0] * 342
            shard_roots = []
            for index in range(2):
                shard = root / f"shard{index}"
                split = shard / "train_features"
                split.mkdir(parents=True)
                (split / "visual_features.json").write_text(
                    json.dumps({"dia1_utt2": vector}), encoding="utf-8"
                )
                record = {"utterance_id": "dia1_utt2", "feature": vector, "status": "ok"}
                (split / "denseface_progress.jsonl").write_text(
                    json.dumps(record) + "\n", encoding="utf-8"
                )
                shard_roots.append(shard)
            with self.assertRaisesRegex(ValueError, "duplicate keys across feature shards"):
                merge_split("train", metadata, shard_roots, root / "merged")

    def test_statistics_merge_uses_global_sums_and_requires_all_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            components = [(2, 4.0, 10.0), (3, 12.0, 56.0)]
            for index, (count, value_sum, square_sum) in enumerate(components):
                path = Path(directory) / f"stats{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "shard": {"num_shards": 2, "shard_index": index},
                            "subset": False,
                            "pixel_count": count,
                            "pixel_sum": value_sum,
                            "pixel_square_sum": square_sum,
                            "valid_frames": 1,
                            "utterances_considered": 1,
                            "status_counts": {"ok": 1},
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            merged = merge_statistics(paths)
            self.assertEqual(merged["pixel_count"], 5)
            self.assertAlmostEqual(merged["mean"], 16.0 / 5.0)
            self.assertAlmostEqual(merged["std"], (66.0 / 5.0 - (16.0 / 5.0) ** 2) ** 0.5)
            with self.assertRaisesRegex(ValueError, "incomplete statistics shards"):
                merge_statistics(paths[:1])


if __name__ == "__main__":
    unittest.main()
