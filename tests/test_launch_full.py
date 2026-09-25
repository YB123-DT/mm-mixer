from __future__ import annotations

import json
from pathlib import Path

import launch_full
from launch_full import build_jobs


def test_four_gpu_full_launcher_assigns_all_eight_jobs_round_robin(tmp_path):
    jobs = build_jobs(("1", "2", "5", "6"), tmp_path)
    assert len(jobs) == 8
    assert {job["dataset"] for job in jobs} == {"iemocap", "meld"}
    assert [job["gpu"] for job in jobs] == ["1", "2", "5", "6"] * 2
    assert all(job["variant"] == "full" for job in jobs)


def test_four_gpu_full_launcher_uses_formal_seeds(tmp_path):
    jobs = build_jobs(("1", "2", "5", "6"), tmp_path)
    seeds = {
        dataset: {job["seed"] for job in jobs if job["dataset"] == dataset}
        for dataset in ("iemocap", "meld")
    }
    assert seeds["iemocap"] == {2025, 2066, 2088, 2118}
    assert seeds["meld"] == {2025, 2028, 2069, 2101}


def test_lane_continues_after_first_job_failure(monkeypatch, tmp_path):
    jobs = build_jobs(("1", "2", "5", "6"), tmp_path)
    lane = [jobs[0], jobs[4]]
    returncodes = iter((7, 0))

    class Completed:
        def __init__(self, returncode):
            self.returncode = returncode

    monkeypatch.setattr(
        launch_full.subprocess,
        "run",
        lambda *args, **kwargs: Completed(next(returncodes)),
    )
    results = launch_full._run_lane("1", lane, tmp_path / "logs")
    assert [result["returncode"] for result in results] == [7, 0]
    assert len(results) == 2


def test_completed_peak_bundle_is_resumed_without_execution(monkeypatch, tmp_path):
    job = build_jobs(("1", "2", "5", "6"), tmp_path)[0]
    peak = (
        tmp_path
        / job["dataset"]
        / "full"
        / f"seed{job['seed']}"
        / "best_peak"
    )
    peak.mkdir(parents=True)
    for name in launch_full.REQUIRED_PEAK_FILES:
        (peak / name).write_bytes(b"x")
    (peak / "status.json").write_text(json.dumps({
        "state": "complete", "fresh_strict_replay_exact": True
    }))
    (peak / "manifest.json").write_text(json.dumps({
        "fresh_strict_replay_exact": True
    }))

    def forbidden(*args, **kwargs):
        raise AssertionError("completed job must not execute")

    monkeypatch.setattr(launch_full.subprocess, "run", forbidden)
    monkeypatch.setattr(
        launch_full.PeakArtifactStore,
        "validate_public_bundle",
        lambda self, expected: True,
    )
    result = launch_full._run_lane("1", [job], tmp_path / "logs")
    assert result[0]["state"] == "skipped_complete"
    assert result[0]["returncode"] == 0
