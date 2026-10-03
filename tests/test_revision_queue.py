from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

_SPEC = importlib.util.spec_from_file_location("revision_queue", Path(__file__).resolve().parents[1] / "launch_revision.py")
queue = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(queue)
summarize_results = queue.summarize_results


def job(tmp_path, variant="full", seed=2025, dataset="iemocap"):
    return {"dataset": dataset, "variant": variant, "seed": seed,
            "output_root": str(tmp_path / "outputs"), "code_root": str(tmp_path), "python": sys.executable}


def plan(tmp_path, jobs):
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": jobs}))
    return queue.load_plan(path)


def complete(score):
    return {"state": "completed", "artifact_verified": True, "formal_protocol_verified": True,
            "metrics": {"weighted_f1": score, "accuracy": score}}


def test_plan_rejects_duplicate_identity_even_if_outputs_differ(tmp_path):
    first, second = job(tmp_path), job(tmp_path)
    second["output_root"] += "_other"
    with pytest.raises(ValueError, match="duplicate job"):
        plan(tmp_path, [first, second])


def test_plan_uses_explicit_three_seeds_and_no_shell_command(tmp_path):
    with pytest.raises(ValueError, match="not a revision"):
        plan(tmp_path, [job(tmp_path, seed=2088)])
    normalized = plan(tmp_path, [job(tmp_path, variant="no_mixer")])[0]
    command = queue.command_for(normalized)
    assert command[-1] == str(tmp_path / "outputs")
    assert command[2:4] == ["run", "--dataset"]
    assert "--epochs" not in command
    assert normalized["run_root"] == str(tmp_path / "outputs/iemocap/no_mixer/seed2025")


def test_biggpu_gpu4_forbidden_and_uuid_must_match_host_index():
    with pytest.raises(ValueError, match="GPU 4 is prohibited"):
        queue.parse_gpus(["4:GPU-bad"], "biggpu")
    whitelist = queue.parse_gpus(["1:GPU-one", "2:GPU-two"], "biggpu")
    with pytest.raises(ValueError, match="identity changed"):
        queue.check_gpus(whitelist, {1: {"uuid": "GPU-two"}, 2: {"uuid": "GPU-one"}})
    assert queue.parse_gpus(["4:GPU-local"], "local")[0]["index"] == 4


def test_queue_lock_rejects_duplicate_scheduler(tmp_path):
    with queue.queue_lock(tmp_path / "status.lock"):
        with pytest.raises(RuntimeError, match="another queue"):
            with queue.queue_lock(tmp_path / "status.lock"):
                pytest.fail("duplicate lock acquired")


def test_summary_retains_failed_missing_and_unverified_seeds(tmp_path):
    jobs = plan(tmp_path, [job(tmp_path, seed=s) for s in (2025, 2066, 2118)])
    state = {"jobs": {
        jobs[0]["id"]: complete(0.70),
        jobs[1]["id"]: {"state": "failed", "error": "CUDA OOM"},
        jobs[2]["id"]: {"state": "completed", "metrics": {"weighted_f1": .99, "accuracy": .99}},
    }}
    result = summarize_results(jobs, state)
    group = result["groups"][0]
    assert result["complete"] is False
    assert group["missing_or_failed_seeds"] == [2066, 2118]
    assert group["metrics"]["weighted_f1"] == {"n": 1, "mean": .70, "sample_std": None}
    assert result["jobs"][1]["error"] == "CUDA OOM"
    # Even an omitted plan seed is explicit in the three-seed table.
    omitted = summarize_results(jobs[:1], {"jobs": {jobs[0]["id"]: complete(.70)}})
    assert omitted["groups"][0]["seeds"][1]["state"] == "missing"


def test_summary_sample_sd_and_deltas_are_paired_by_seed(tmp_path):
    jobs = plan(tmp_path, [job(tmp_path, variant=v, seed=s) for v in ("full", "no_mixer") for s in (2025, 2066, 2118)])
    records = {}
    for seed, full_score, variant_score in ((2025, .70, .68), (2066, .80, .79), (2118, .90, .87)):
        records[f"iemocap/full/seed{seed}"] = complete(full_score)
        records[f"iemocap/no_mixer/seed{seed}"] = complete(variant_score)
    result = summarize_results(jobs, {"jobs": records})
    full, variant = result["groups"]
    assert result["complete"] is True
    assert full["metrics"]["weighted_f1"]["sample_std"] == pytest.approx(.1)
    deltas = variant["paired_delta_to_full"]["weighted_f1"]
    assert deltas["mean"] == pytest.approx(-.02)
    assert deltas["sample_std"] == pytest.approx(.01)


def test_partial_output_is_preserved_and_never_restarted(tmp_path, monkeypatch):
    jobs = plan(tmp_path, [job(tmp_path)])
    run_root = Path(jobs[0]["run_root"])
    run_root.mkdir(parents=True)
    unfinished = run_root / "history.partial"
    unfinished.write_text("keep this diagnostic evidence")
    monkeypatch.setattr(queue, "verify_bundle", lambda _: {"valid": False})
    monkeypatch.setattr(queue, "gpu_inventory", lambda: pytest.fail("no GPU launch should occur"))
    args = type("Args", (), dict(gpu=["1:GPU-one"], server="biggpu", state=str(tmp_path / "status.json"), per_gpu=2,
                                min_free_mib=6000, threads=1, poll_seconds=.2, once=False))()
    assert queue.run_queue(args, jobs) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["jobs"][jobs[0]["id"]]["state"] == "failed"
    assert "pre-existing output" in state["jobs"][jobs[0]["id"]]["error"]
    assert unfinished.read_text() == "keep this diagnostic evidence"


def test_pid_identity_does_not_accept_unrelated_running_process(tmp_path):
    identity = queue.process_identity(os.getpid())
    assert identity["pid"] == os.getpid()
    record = {**job(tmp_path), "run_root": str(tmp_path / "run"),
              "pid": os.getpid(), "start_ticks": identity["start_ticks"],
              "command": identity["command"], "process_token": "nonexistent-test-token"}
    assert queue.job_processes(record) == []


def test_exit_zero_without_verified_artifacts_is_failure(tmp_path, monkeypatch):
    record = {**job(tmp_path), "returncode": 0}
    monkeypatch.setattr(queue, "verify_bundle", lambda _: {"valid": False})
    queue.accept_or_fail(record, "process exited (0)")
    assert record["state"] == "failed"
    assert record["artifact_verified"] is False


def test_restart_adopts_live_child_without_duplicate_launch(tmp_path, monkeypatch):
    # A tiny fake runner tests supervision only; it performs no training/GPU work.
    (tmp_path / "run.py").write_text("import time\ntime.sleep(1)\n")
    jobs = plan(tmp_path, [job(tmp_path)])
    monkeypatch.setattr(queue, "gpu_inventory", lambda: {1: {"uuid": "GPU-one", "free_mib": 10000}})
    args = type("Args", (), dict(gpu=["1:GPU-one"], server="biggpu", state=str(tmp_path / "status.json"), per_gpu=2,
                                min_free_mib=6000, threads=1, poll_seconds=.2, once=True))()
    assert queue.run_queue(args, jobs) == 0
    state = json.loads((tmp_path / "status.json").read_text())
    first_pid = state["jobs"][jobs[0]["id"]]["pid"]
    provenance = state["jobs"][jobs[0]["id"]]["launch_provenance"]
    assert provenance["python_executable"] and provenance["hostname"]
    assert set(provenance["packages"]) == {"torch", "numpy", "sklearn"}
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("duplicate process launched"))
    assert queue.run_queue(args, jobs) == 0
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["jobs"][jobs[0]["id"]]["state"] == "running"
    assert first_pid in state["jobs"][jobs[0]["id"]]["live_pids"]
    os.waitpid(first_pid, 0)
    monkeypatch.setattr(queue, "verify_bundle", lambda _: {"valid": False})
    assert queue.run_queue(args, jobs) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["jobs"][jobs[0]["id"]]["state"] == "failed"


@pytest.mark.parametrize("matrix,count,variants", [
    ("full", 144, queue.BASE_VARIANTS + queue.CONTROL_VARIANTS),
    ("base", 102, queue.BASE_VARIANTS),
    ("controls", 42, queue.CONTROL_VARIANTS),
])
def test_generated_fixed_matrices_are_complete_and_do_not_launch(tmp_path, monkeypatch, matrix, count, variants):
    target = tmp_path / f"{matrix}.json"
    monkeypatch.setattr(queue, "run_queue", lambda *a: pytest.fail("plan started training"))
    assert queue.main(["plan", "--matrix", matrix, "--plan", str(target),
                       "--code-root", str(tmp_path / "frozen"), "--output-root", str(tmp_path / "runs"),
                       "--python", sys.executable]) == 0
    jobs = queue.load_plan(target)
    assert len(jobs) == count
    assert {j["variant"] for j in jobs} == set(variants)
    for dataset, seeds in queue.SEEDS.items():
        for variant in variants:
            assert {j["seed"] for j in jobs if j["dataset"] == dataset and j["variant"] == variant} == set(seeds)
    assert not (tmp_path / "runs").exists()


def test_plan_publication_never_overwrites_even_dangling_symlinks(tmp_path):
    target = tmp_path / "plan.json"
    queue.write_new_plan(target, {"original": True})
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        queue.write_new_plan(target, {"replacement": True})
    assert target.read_bytes() == original
    target.unlink()
    target.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        queue.write_new_plan(target, {})
    assert target.is_symlink()
    assert not (tmp_path / "absent").exists()


def formal_config(dataset, epochs):
    if dataset == "iemocap":
        return {"epochs": epochs, "selection": "strict_peak_test_wf1", "legacy_training_config": {"epochs": epochs}}
    return {"num_epochs": epochs, "fixed_params": {"selection_mode": "test"},
            "runtime_audit": {"selection": "strict_peak_test_wf1"}}


@pytest.mark.parametrize("dataset,epochs", [("iemocap", 100), ("meld", 50)])
def test_formal_budget_rejects_smoke_even_when_contract_hash_matches(dataset, epochs):
    manifest = {"selection": "strict_peak_test_wf1", "config_contract_sha256": "same-formal-contract"}
    assert queue.formal_runtime_error(dataset, epochs, formal_config(dataset, 1), manifest)
    assert queue.formal_runtime_error(dataset, epochs, formal_config(dataset, epochs), manifest) is None
    config = formal_config(dataset, epochs)
    # Early stopping/early peak does not change the requested formal budget.
    config["history"] = [{"epoch": 1}, {"epoch": 2}]
    assert queue.formal_runtime_error(dataset, epochs, config, manifest) is None
    manifest["selection"] = "validation_wf1"
    assert queue.formal_runtime_error(dataset, epochs, config, manifest)


def test_runtime_selection_and_conflicting_nested_budget_are_rejected():
    manifest = {"selection": "strict_peak_test_wf1"}
    iemocap = formal_config("iemocap", 100)
    iemocap["legacy_training_config"]["epochs"] = 1
    assert "legacy" in queue.formal_runtime_error("iemocap", 100, iemocap, manifest)
    iemocap = formal_config("iemocap", 100)
    iemocap["selection"] = "validation_wf1"
    assert queue.formal_runtime_error("iemocap", 100, iemocap, manifest)
    meld = formal_config("meld", 50)
    meld["fixed_params"]["selection_mode"] = "dev"
    assert queue.formal_runtime_error("meld", 50, meld, manifest)
    meld = formal_config("meld", 50)
    meld["runtime_audit"]["selection"] = "validation_wf1"
    assert queue.formal_runtime_error("meld", 50, meld, manifest)


def test_summary_excludes_old_artifact_only_verification(tmp_path):
    jobs = plan(tmp_path, [job(tmp_path)])
    record = complete(.9)
    record.pop("formal_protocol_verified")
    result = summarize_results(jobs, {"jobs": {jobs[0]["id"]: record}})
    assert result["groups"][0]["metrics"]["weighted_f1"]["n"] == 0
    assert result["jobs"][0]["formal_protocol_verified"] is False


def test_launch_provenance_uses_job_interpreter_and_contains_no_environment_secrets(tmp_path, monkeypatch):
    import hashlib
    import platform
    import socket
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps({"git_head": "frozen-head", "private_extra_field": "do not copy"}))
    monkeypatch.setenv("VERY_SECRET_TEST_CREDENTIAL", "credential-must-not-be-recorded")
    result = queue.collect_launch_provenance(job(tmp_path))
    assert result["python_version"] == platform.python_version()
    assert Path(result["python_executable"]).resolve() == Path(sys.executable).resolve()
    assert result["hostname"] == socket.gethostname()
    assert set(result["packages"]) == {"torch", "numpy", "sklearn"}
    assert result["snapshot"]["sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert result["snapshot"]["git_head"] == "frozen-head"
    assert "credential-must-not-be-recorded" not in json.dumps(result)
    assert "private_extra_field" not in json.dumps(result)


@pytest.mark.parametrize("dataset", ["iemocap", "meld"])
def test_real_peak_artifact_smoke_is_not_accepted_as_formal(tmp_path, dataset):
    # Optional interpreter override lets the stdlib queue/pytest environment
    # exercise real torch artifacts without adding packages to either env.
    python = os.environ.get("MM_MIXER_TEST_TORCH_PYTHON")
    if not python:
        if importlib.util.find_spec("torch") is None:
            pytest.skip("set MM_MIXER_TEST_TORCH_PYTHON for real artifact integration")
        python = sys.executable
    repo = Path(__file__).resolve().parents[1]
    raw = job(tmp_path, dataset=dataset)
    raw.update(python=python, code_root=str(repo))
    normalized = plan(tmp_path, [raw])[0]
    fixture = r'''
import json, sys
from pathlib import Path
import torch
from mm_mixer_final.artifacts import PeakArtifactStore, classification_metrics
from mm_mixer_final.config import get_config, config_contract_sha256
from mm_mixer_final.audit import config_payload_sha256, sha256
job = json.loads(sys.argv[1])
cfg = get_config(job["dataset"], job["variant"], job["seed"])
root = Path(job["run_root"])
root.mkdir(parents=True)
feature = root / "tiny-feature.txt"
feature.write_text("fixture data version")
config = json.loads(sys.argv[2])
manifest = {k: job[k] for k in ("dataset", "variant", "seed")}
manifest.update(selection="strict_peak_test_wf1", fresh_strict_replay_exact=True,
    config_contract_sha256=config_contract_sha256(cfg), config_sha256=config_payload_sha256(config),
    source_hashes={str(Path(job["code_root"]) / "run.py"): sha256(Path(job["code_root"]) / "run.py")},
    feature_hashes={str(feature): sha256(feature)})
logits = torch.zeros(3, len(cfg.class_names))
logits[range(3), range(3)] = 2
labels = torch.tensor([0, 1, 2])
store = PeakArtifactStore(root, cfg.class_names)
store.publish_if_better(dict(epoch=1, state_dict={"weight": torch.ones(1)}, logits=logits,
    labels=labels, metrics=classification_metrics(logits, labels, cfg.class_names),
    history=[{"epoch": 1}], config=config, manifest=manifest))
assert store.verify_saved_predictions(logits, labels)
expected = {k: manifest[k] for k in ("dataset", "variant", "seed", "config_contract_sha256")}
assert store.validate_public_bundle(expected), "fixture must reproduce the prior acceptance boundary"
'''
    queue.subprocess.run([python, "-c", fixture, json.dumps(normalized), json.dumps(formal_config(dataset, 1))],
                         cwd=repo, check=True, capture_output=True, text=True, timeout=60)
    result = queue.verify_bundle(normalized)
    assert result["valid"] is False
    assert result["formal_protocol_verified"] is False
    assert "epoch budget" in result["error"]

    # A genuine formal-budget bundle may peak/stop at epoch one. The budget
    # guard must accept it rather than requiring 100/50 history entries.
    full_job = {**raw, "output_root": str(tmp_path / "formal_outputs")}
    full_job = plan(tmp_path, [full_job])[0]
    epochs = 100 if dataset == "iemocap" else 50
    queue.subprocess.run([python, "-c", fixture, json.dumps(full_job), json.dumps(formal_config(dataset, epochs))],
                         cwd=repo, check=True, capture_output=True, text=True, timeout=60)
    result = queue.verify_bundle(full_job)
    assert result["valid"] is True
    assert result["formal_protocol_verified"] is True
    assert result["formal_epochs"] == epochs
