from mm_mixer_final.config import get_config


def test_meld_text_feature_root_override_changes_only_text(monkeypatch, tmp_path):
    monkeypatch.delenv("MM_MIXER_MELD_TEXT_FEATURE_ROOT", raising=False)
    baseline = get_config("meld", "full", 2025)
    monkeypatch.setenv("MM_MIXER_MELD_TEXT_FEATURE_ROOT", str(tmp_path))
    overridden = get_config("meld", "full", 2025)

    for split in ("train", "dev", "test"):
        assert overridden.feature_paths[split]["t"] == str(
            tmp_path / f"{split}_features" / "text_features.json"
        )
        assert overridden.feature_paths[split]["a"] == baseline.feature_paths[split]["a"]
        assert overridden.feature_paths[split]["v"] == baseline.feature_paths[split]["v"]
