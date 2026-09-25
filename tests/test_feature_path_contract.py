from pathlib import Path

from dataset_runners.iemocap import materialize_legacy_config
from dataset_runners.meld import materialize_config as materialize_meld_config
from mm_mixer_final.config import get_config


CSS_FEATURES = (
    "/data2/yb/multimodalERC/IEMOCAP/"
    "Model_rawaux_textpeak_cssv_v1/artifacts/iemocap_textpeak_audio_cssv.npz"
)


def test_iemocap_declared_and_materialized_features_are_canonical_css():
    cfg = get_config("iemocap", "no_mixer", 2025)
    actual = materialize_legacy_config(cfg, epochs=100, variant="no_mixer")
    assert cfg.feature_paths["packed"] == CSS_FEATURES
    assert actual.features == CSS_FEATURES
    assert Path(actual.features).is_file()


def test_meld_declared_features_match_materialized_runtime_config(tmp_path):
    cfg = get_config("meld", "no_mixer", 2025)
    actual = materialize_meld_config(tmp_path, epochs=50, seed=2025, variant="no_mixer")
    for modality, path in cfg.feature_paths["train"].items():
        assert actual["runtime_audit"]["feature_paths"]["train"][modality] == path
