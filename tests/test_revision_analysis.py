from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import torch

from mm_mixer_final.artifacts import PeakArtifactStore, classification_metrics
from mm_mixer_final.audit import config_payload_sha256, sha256
from mm_mixer_final.config import config_contract_sha256, get_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_revision", ROOT / "analyze_revision.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class RevisionAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.jobs = []
        self.state = {"selection": "strict_peak_test_wf1", "jobs": {}}
        self.plan_path, self.state_path = self.root / "plan.json", self.root / "state.json"

    def make_run(self, variant="full", seed=2025, dataset="iemocap", errors=0, labels=None):
        cfg = get_config(dataset, variant, seed)
        identity = f"{dataset}/{variant}/seed{seed}"
        job = {"dataset": dataset, "variant": variant, "seed": seed,
               "output_root": str(self.root / "runs"), "code_root": str(ROOT)}
        self.jobs.append(job)
        labels = torch.arange(len(cfg.class_names)).repeat(2) if labels is None else labels
        predictions = labels.clone()
        predictions[:errors] = (predictions[:errors] + 1) % len(cfg.class_names)
        logits = torch.full((len(labels), len(cfg.class_names)), -2.)
        logits.scatter_(1, predictions[:, None], 2.)
        metrics = classification_metrics(logits, labels, cfg.class_names)
        if dataset == "iemocap":
            config = {"epochs": 100, "selection": "strict_peak_test_wf1"}
        else:
            config = {"num_epochs": 50, "fixed_params": {"selection_mode": "test"},
                      "runtime_audit": {"selection": "strict_peak_test_wf1"}}
        contract = config_contract_sha256(cfg)
        manifest = {"dataset": dataset, "variant": variant, "seed": seed,
                    "selection": "strict_peak_test_wf1", "fresh_strict_replay_exact": True,
                    "config_contract_sha256": contract, "config_sha256": config_payload_sha256(config)}
        store = PeakArtifactStore(self.root / "runs" / identity, cfg.class_names)
        store.publish_if_better({"epoch": 3, "state_dict": {"fixture": torch.ones(1)},
                                 "logits": logits, "labels": labels, "metrics": metrics,
                                 "history": {"epochs": [1, 2, 3]}, "config": config, "manifest": manifest})
        self.assertTrue(store.verify_saved_predictions(logits, labels))
        self.state["jobs"][identity] = {**job, "id": identity, "state": "completed",
            "artifact_verified": True, "formal_protocol_verified": True, "verified_contract": contract,
            "metrics": {"epoch": 3, **metrics}, "bundle": str(store.public)}
        return store.public, self.state["jobs"][identity]

    def analyze(self):
        self.plan_path.write_text(json.dumps({"jobs": self.jobs}))
        self.state_path.write_text(json.dumps(self.state))
        return analysis.analyze(self.plan_path, self.state_path)

    def rewrite_json_artifact(self, bundle, filename, edit):
        path = bundle / filename
        value = json.loads(path.read_text())
        edit(value)
        path.write_text(json.dumps(value))
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"][filename] = sha256(path)
        if filename == "config.json":
            manifest["config_sha256"] = config_payload_sha256(value)
        manifest_path.write_text(json.dumps(manifest))

    def test_three_seed_metrics_sample_sd_confusions_and_paired_deltas(self):
        for index, seed in enumerate((2025, 2066, 2118)):
            self.make_run("full", seed, errors=index)
            self.make_run("amm_mlp", seed, errors=index + 2)
        result = self.analyze()
        self.assertTrue(result["complete"])
        groups = {group["variant"]: group for group in result["groups"]}
        full, variant = groups["full"], groups["amm_mlp"]
        self.assertEqual(full["metrics"]["accuracy"]["n"], 3)
        self.assertAlmostEqual(full["metrics"]["accuracy"]["mean"], 11 / 12)
        self.assertAlmostEqual(full["metrics"]["accuracy"]["sample_std"], 1 / 12)
        self.assertAlmostEqual(variant["paired_delta_to_full"]["accuracy"]["mean"], -2 / 12)
        self.assertAlmostEqual(variant["paired_delta_to_full"]["accuracy"]["sample_std"], 0)
        for row in full["seeds"]:
            self.assertEqual(sum(map(sum, row["confusion_counts"])), 12)
            self.assertEqual(len(row["confusion_counts"]), 6)
        self.assertEqual(full["seeds"][0]["confusion_counts"], (2 * torch.eye(6, dtype=torch.int64)).tolist())
        self.assertAlmostEqual(sum(map(sum, full["confusion_counts"]["mean"])), 12)
        self.assertTrue(all(stats["n"] == 3 for stats in full["class_f1"].values()))
        markdown = analysis.render_markdown(result)
        self.assertIn("3/3 complete", markdown)
        self.assertIn("Macro-F1", markdown)
        self.assertIn("Same-seed differences", markdown)

    def test_failed_unverified_and_unplanned_seeds_are_explicit(self):
        _, first = self.make_run(seed=2025)
        _, second = self.make_run(seed=2066)
        second.update(state="failed", error="fixture failure")
        self.state["jobs"]["iemocap/full/seed2118"] = copy.deepcopy(first)
        result = self.analyze()
        group = result["groups"][0]
        self.assertFalse(group["complete_three_seeds"])
        self.assertEqual(group["n"], 1)
        self.assertEqual(group["metrics"]["accuracy"]["sample_std"], None)
        self.assertEqual(group["missing_or_failed_seeds"], [2066, 2118])
        self.assertEqual(group["seeds"][2]["state"], "not_planned")
        self.assertEqual(result["ignored_state_jobs"], ["iemocap/full/seed2118"])
        self.assertIn("INCOMPLETE", analysis.render_markdown(result))
        first["artifact_verified"] = False
        self.assertEqual(self.analyze()["groups"][0]["n"], 0)

    def test_changed_prediction_bytes_are_rejected_after_queue_verification(self):
        bundle, _ = self.make_run()
        with (bundle / "peak_test_predictions.pt").open("ab") as stream:
            stream.write(b"changed after verification")
        row = self.analyze()["groups"][0]["seeds"][0]
        self.assertFalse(row["included"])
        self.assertEqual(row["state"], "invalid_artifact")
        self.assertIn("artifact hash mismatch: peak_test_predictions.pt", row["error"])

    def test_recomputation_rejects_changed_metric_even_with_updated_hash(self):
        bundle, record = self.make_run(errors=2)
        self.rewrite_json_artifact(bundle, "peak_test_metrics.json", lambda value: value.update(macro_f1=.123))
        record["metrics"]["macro_f1"] = .123
        row = self.analyze()["groups"][0]["seeds"][0]
        self.assertFalse(row["included"])
        self.assertIn("recomputed metric mismatch: macro_f1", row["error"])

    def test_prediction_label_and_argmax_checks_precede_metric_use(self):
        bundle, _ = self.make_run()
        path = bundle / "peak_test_predictions.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["predictions"][0] = 1
        torch.save(payload, path)
        manifest_path = bundle / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifact_sha256"][path.name] = sha256(path)
        manifest_path.write_text(json.dumps(manifest))
        status_path = bundle / "status.json"
        status = json.loads(status_path.read_text())
        status["predictions_sha256"] = sha256(path)
        status_path.write_text(json.dumps(status))
        row = self.analyze()["groups"][0]["seeds"][0]
        self.assertIn("argmax", row["error"])
        self.assertFalse(row["included"])

    def test_formal_protocol_flag_and_saved_epoch_budget_are_required(self):
        bundle, record = self.make_run()
        record.pop("formal_protocol_verified")
        row = self.analyze()["groups"][0]["seeds"][0]
        self.assertIn("formal_protocol_verified", row["error"])
        record["formal_protocol_verified"] = True
        self.rewrite_json_artifact(bundle, "config.json", lambda value: value.update(epochs=1))
        row = self.analyze()["groups"][0]["seeds"][0]
        self.assertIn("formal epoch budget mismatch", row["error"])
        self.assertFalse(row["included"])

    def test_meld_selection_and_missing_classes_are_retained(self):
        bundle, _ = self.make_run(dataset="meld", labels=torch.tensor([0, 0, 1, 1]))
        result = self.analyze()
        group = result["groups"][0]
        self.assertEqual(group["n"], 1)
        self.assertEqual(group["class_f1"]["fear"], {"n": 0, "mean": None, "sample_std": None})
        self.assertIn("fear", group["seeds"][0]["classes_without_true_examples"])
        self.assertEqual(len(group["seeds"][0]["confusion_counts"]), 7)
        self.rewrite_json_artifact(bundle, "config.json", lambda value: value["fixed_params"].update(selection_mode="dev"))
        row = self.analyze()["groups"][0]["seeds"][0]
        self.assertIn("selection protocol mismatch", row["error"])

    def test_pairing_requires_same_seed_full_and_test_label_order(self):
        self.make_run("full", seed=2025)
        self.make_run("amm_mlp", seed=2066)
        self.make_run("amm_mlp", seed=2025, labels=torch.arange(6).repeat(2).flip(0))
        groups = {group["variant"]: group for group in self.analyze()["groups"]}
        group = groups["amm_mlp"]
        self.assertEqual(group["n"], 2)
        self.assertEqual(group["paired_delta_to_full"]["accuracy"]["n"], 0)
        self.assertIn("label order differs", group["seeds"][0]["pairing_error"])
        self.assertIn("same-seed verified Full unavailable", group["seeds"][1]["pairing_error"])

    def test_output_exclusive_creation_preserves_existing_file_and_cleans_reservation(self):
        self.make_run()
        result = self.analyze()
        output_json, output_md = self.root / "analysis.json", self.root / "analysis.md"
        output_md.write_text("preserve existing report")
        with self.assertRaises(FileExistsError):
            analysis.write_outputs(result, output_json, output_md)
        self.assertFalse(output_json.exists())
        self.assertEqual(output_md.read_text(), "preserve existing report")
        output_md.unlink()
        analysis.write_outputs(result, output_json, output_md)
        self.assertEqual(json.loads(output_json.read_text()), result)
        before = output_json.read_bytes()
        with self.assertRaises(FileExistsError):
            analysis.write_outputs(result, output_json, output_md)
        self.assertEqual(output_json.read_bytes(), before)

    def test_cli_writes_json_and_markdown_from_cpu_fixtures(self):
        self.make_run()
        self.analyze()
        output_json, output_md = self.root / "cli.json", self.root / "cli.md"
        self.assertEqual(analysis.main(["--plan", str(self.plan_path), "--state", str(self.state_path),
                                       "--output-json", str(output_json), "--output-md", str(output_md)]), 0)
        self.assertFalse(json.loads(output_json.read_text())["complete"])
        self.assertIn("1/3 INCOMPLETE", output_md.read_text())


if __name__ == "__main__":
    unittest.main()
