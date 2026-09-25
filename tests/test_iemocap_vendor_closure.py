from __future__ import annotations

from pathlib import Path


def test_active_iemocap_loader_is_vendored():
    root = Path(__file__).resolve().parents[1]
    runner = (root / "vendor/iemocap/utterance_history/runner.py").read_text()
    formal = (root / "vendor/iemocap/ablation48/formal_runner.py").read_text()
    assert 'DATASET_SOURCE = ROOT / "data/fill53_dataset.py"' in runner
    assert 'FILL = ROOT / "data"' in formal
    assert (root / "vendor/iemocap/data/fill53_dataset.py").is_file()


def test_iemocap_manifest_hashes_vendored_loader():
    root = Path(__file__).resolve().parents[1]
    final_runner = (root / "dataset_runners/iemocap.py").read_text()
    assert 'SOURCE / "data/fill53_dataset.py"' in final_runner


def test_meld_fresh_replay_requires_constructor_factory():
    root = Path(__file__).resolve().parents[1]
    trainer = (root / "vendor/meld/train_erc.py").read_text()
    assert 'globals().get("FINAL_FRESH_MODEL_FACTORY")' in trainer
    assert "fresh_model = copy.deepcopy(fusion_model)" not in trainer
