from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import measure_revision_efficiency as efficiency


def make_bundle(root, dataset="meld", variant="full"):
    config = {"dataset": dataset, "variant": variant, "seed": 2025,
              "batch_size": 32, "runtime_audit": {}}
    if dataset == "meld":
        config.update(classes={"meld": ["neutral"]}, feature_paths={"meld": {}},
                      fixed_params={"seed": 2025, "capacity_variant": "M4_PAIR",
                                    "aux_loss_weights": {"t": 1., "a": 1., "v": 1.}})
    if variant != "full":
        config["runtime_audit"]["revision_control"] = {"variant": variant}
    for name in efficiency.BUNDLE_ARTIFACTS:
        (root / name).write_bytes(b"fixture " + name.encode())
    (root / "config.json").write_text(json.dumps(config))
    hashes = {name: efficiency.sha256(root / name) for name in efficiency.BUNDLE_ARTIFACTS}
    manifest = {"dataset": dataset, "variant": variant, "seed": 2025,
                "fresh_strict_replay_exact": True,
                "config_sha256": efficiency.json_hash(config), "artifact_sha256": hashes}
    status = {"state": "complete", "fresh_strict_replay_exact": True,
              "checkpoint_sha256": hashes["best_peak_test_state_dict.pt"],
              "predictions_sha256": hashes["peak_test_predictions.pt"]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "status.json").write_text(json.dumps(status))
    return config, manifest, status


class EfficiencyProtocolTests(unittest.TestCase):
    def test_exact_targets_and_partial_batch_no_padding_or_duplicates(self):
        selected = efficiency.sample_indices(1623)
        self.assertEqual(selected, list(range(1000)))
        self.assertEqual([len(selected[i:i + 32]) for i in range(0, 1000, 32)], [32] * 31 + [8])
        self.assertEqual(efficiency.sample_indices(1623, 7), list(range(7)))
        for count in (0, -1, 1001):
            with self.assertRaises(ValueError):
                efficiency.sample_indices(1623, count)
        with self.assertRaises(ValueError):
            efficiency.sample_indices(999)

    def test_stats_are_per_whole_sweep_and_p90_is_interpolated(self):
        result = efficiency.summarize_times([4., 1., 3., 2.], 1000)
        self.assertEqual(result["median_seconds"], 2.5)
        self.assertAlmostEqual(result["p90_seconds"], 3.7)
        self.assertEqual(result["all_repeats_total_seconds"], 10.)
        self.assertEqual(result["median_utterances_per_second"], 400.)
        self.assertEqual(result["median_ms_per_utterance_amortized"], 2.5)
        for bad in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                efficiency.summarize_times(bad, 1000)

    def test_physical_bad_card_is_rejected_by_index_and_uuid(self):
        with patch("subprocess.check_output") as query:
            with self.assertRaises(ValueError):
                efficiency.select_gpu("4", "biggpu")
            query.assert_not_called()
        with patch("subprocess.check_output", return_value="4, GPU-bad, Tesla V100, 32510, 550\n"):
            with self.assertRaises(ValueError):
                efficiency.select_gpu("GPU-bad", "biggpu")
        with patch("subprocess.check_output", return_value="2, GPU-healthy, Tesla V100, 32510, 550\n"), patch.dict(os.environ):
            record = efficiency.select_gpu("2", "biggpu")
            self.assertEqual(record["physical_index"], 2)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-healthy")

    def test_other_process_blocks_formal_timing(self):
        with patch("subprocess.check_output", return_value="999999, python, 512\n"):
            with self.assertRaises(RuntimeError):
                efficiency.assert_idle_gpu("GPU-healthy")
        with patch("subprocess.check_output", return_value=f"{os.getpid()}, python, 512\n"):
            self.assertEqual(efficiency.assert_idle_gpu("GPU-healthy")["other_compute_processes"], [])

    def test_artifact_identity_and_integrity_before_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_bundle(root)
            checkpoint = root / "best_peak_test_state_dict.pt"
            efficiency.validate_artifact(root, "meld", "full", 2025)
            with self.assertRaises(ValueError):
                efficiency.validate_artifact(root, "meld", "amm_mlp", 2025)
            checkpoint.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                efficiency.validate_artifact(root, "meld", "full", 2025)

    def test_incomplete_and_unreplayed_bundles_fail_before_any_gpu_query(self):
        cases = [("status", "state", "peak_published"),
                 ("status", "fresh_strict_replay_exact", False),
                 ("manifest", "fresh_strict_replay_exact", False)]
        for document, key, value in cases:
            with self.subTest(document=document, key=key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest, status = make_bundle(root)
                changed = status if document == "status" else manifest
                changed[key] = value
                (root / f"{document}.json").write_text(json.dumps(changed))
                args = efficiency.parse_args(["--artifact-dir", str(root), "--dataset", "meld",
                    "--variant", "full", "--seed", "2025", "--gpu", "GPU-example", "--output", str(root / "out.json")])
                with patch("subprocess.check_output") as query, self.assertRaises(ValueError):
                    efficiency.benchmark(args)
                query.assert_not_called()

    def test_checksums_are_required_not_optional(self):
        for name in (*efficiency.BUNDLE_ARTIFACTS, "config_sha256", "status_checkpoint_sha256"):
            with self.subTest(missing=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest, status = make_bundle(root)
                if name == "config_sha256":
                    del manifest[name]
                elif name == "status_checkpoint_sha256":
                    del status["checkpoint_sha256"]
                else:
                    del manifest["artifact_sha256"][name]
                (root / "manifest.json").write_text(json.dumps(manifest))
                (root / "status.json").write_text(json.dumps(status))
                with self.assertRaises(ValueError):
                    efficiency.validate_artifact(root, "meld", "full", 2025)

    def test_missing_status_or_prediction_file_is_not_a_complete_bundle(self):
        for name in ("status.json", "peak_test_predictions.pt"):
            with self.subTest(missing=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_bundle(root)
                (root / name).unlink()
                with self.assertRaises(FileNotFoundError):
                    efficiency.validate_artifact(root, "meld", "full", 2025)

    def test_relabeling_manifest_cannot_override_config_variant(self):
        for dataset in ("iemocap", "meld"):
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest, _ = make_bundle(root, dataset, "amm_mlp")
                manifest["variant"] = "amm_attention"
                (root / "manifest.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    efficiency.validate_artifact(root, dataset, "amm_attention", 2025)

    def test_historical_meld_config_identity_is_checked_without_top_level_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, _ = make_bundle(root)
            for key in ("dataset", "variant", "seed"):
                del config[key]
            efficiency.validate_config_identity(config, "meld", "full", 2025)
            config["fixed_params"]["capacity_variant"] = "M4_PAIR_NO_MIXER"
            with self.assertRaises(ValueError):
                efficiency.validate_config_identity(config, "meld", "full", 2025)

    def test_changed_source_is_explicit_and_never_labeled_exact_reproduction(self):
        result = efficiency.compare_recorded_sources(
            {"/old/repo/vendor/model.py": "old", "/old/repo/missing.py": "lost"},
            {"vendor/model.py": "new"})
        self.assertFalse(result["all_recorded_sources_match"])
        self.assertEqual([row["status"] for row in result["entries"]], ["changed", "not_loaded_or_ambiguous"])
        self.assertFalse(efficiency.compare_recorded_sources({}, {})["all_recorded_sources_match"])

    def test_path_mapping_matches_complete_prefix_and_longest_first(self):
        mappings = [("/old", "/generic"), ("/old/data", "/specific")]
        self.assertEqual(efficiency.remap_path("/old/data/test.json", mappings), Path("/specific/test.json"))
        self.assertEqual(efficiency.remap_path("/older/test.json", mappings), Path("/older/test.json"))

    def test_missing_checkpoint_is_explicit_skip_not_completed_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "measurement.json"
            efficiency.main(["--artifact-dir", str(Path(directory) / "absent"),
                "--dataset", "meld", "--variant", "full", "--seed", "2025",
                "--gpu", "GPU-example", "--output", str(output), "--skip-missing"])
            self.assertEqual(json.loads(output.read_text())["status"], "skipped")
            self.assertNotIn("timing", json.loads(output.read_text()))


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed in this interpreter")
class EfficiencyTensorTests(unittest.TestCase):
    def test_default_tuple_preserved_and_aux_only_params_not_main_connected(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.main = torch.nn.Linear(3, 2)
                self.aux = torch.nn.Linear(3, 2)
            def forward(self, features):
                return self.main(features["t"]), {"t": self.aux(features["t"])}
        model = Toy().eval()
        inputs = {"t": torch.ones(4, 3)}
        result = efficiency.parameter_audit(model, inputs, torch)
        self.assertEqual(result["registered_trainable"], 16)
        self.assertEqual(result["main_logit_gradient_connected_trainable"], 8)
        self.assertEqual(result["main_logit_disconnected_trainable_names"], ["aux.weight", "aux.bias"])
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(tuple(efficiency.main_logits(model(inputs)).shape), (4, 2))

    def test_input_digest_changes_for_order_values_or_labels(self):
        batch = ({"t": torch.arange(6, dtype=torch.float32).reshape(2, 3)}, torch.tensor([1, 2]))
        baseline = efficiency.input_hash([batch])
        self.assertEqual(baseline, efficiency.input_hash([batch]))
        self.assertNotEqual(baseline, efficiency.input_hash([(batch[0], torch.tensor([2, 1]))]))
        self.assertNotEqual(baseline, efficiency.input_hash([({"t": batch[0]["t"].flip(0)}, batch[1])]))


if __name__ == "__main__":
    unittest.main()
