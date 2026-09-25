from __future__ import annotations

import json

from mm_mixer_final.cli import main
from mm_mixer_final.adapters import materialize_command
from mm_mixer_final.config import get_config


def test_dry_run_prints_audited_command(capsys, tmp_path):
    rc = main([
        "run", "--dataset", "iemocap", "--variant", "full",
        "--seed", "2025", "--output-root", str(tmp_path), "--dry-run",
    ])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["dataset"] == "iemocap"
    assert result["selection"] == "peak_test"
    assert result["variant"] == "full"


def test_matrix_has_four_formal_seeds_and_six_variants(capsys, tmp_path):
    rc = main([
        "matrix", "--dataset", "meld", "--output-root", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["runs"]) == 24


def test_full_commands_use_final_dataset_runners(tmp_path):
    for dataset in ("iemocap", "meld"):
        command = materialize_command(
            get_config(dataset, "full", 2025), tmp_path, epochs=1
        )
        assert f"dataset_runners/{dataset}.py" in command[1]


def test_meld_output_prefix_is_not_encoded_into_results_name(tmp_path):
    command = materialize_command(get_config("meld", "full", 2025), tmp_path)
    assert all("checkpoint_prefix" not in item for item in command)
