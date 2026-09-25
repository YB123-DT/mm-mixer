from __future__ import annotations

import pytest

from mm_mixer_final.config import DATASETS, VARIANTS, get_config


def test_iemocap_full_contract_is_frozen():
    cfg = get_config("iemocap", "full", 2025)
    assert cfg.seeds == (2025, 2066, 2088, 2118)
    assert cfg.input_dims == {"t": 1024, "a": 1024, "v": 342}
    assert cfg.mixer == {"blocks": 2, "tokens": 6, "dim": 256, "ffn": 1536}
    assert cfg.learning_rates == {
        "projection": 3e-5,
        "pairwise_cross": 1.2e-4,
        "mixer": 6e-5,
        "cross_attention": 6e-5,
        "remaining": 6e-5,
        "classifier": 1.2e-4,
    }
    assert cfg.loss["kind"] == "iemocap_batch_focal_detached"
    assert cfg.loss["weights"] == {
        "main": 0.45,
        "t": 0.33,
        "a": 0.11,
        "v": 0.11,
    }


def test_meld_full_contract_is_frozen():
    cfg = get_config("meld", "full", 2025)
    base = 2.0832826726482106e-5
    assert cfg.seeds == (2025, 2028, 2069, 2101)
    assert cfg.input_dims == {"t": 1024, "a": 1024, "v": 342}
    assert cfg.mixer == {"blocks": 2, "tokens": 6, "dim": 256, "ffn": 1536}
    assert cfg.learning_rates == {
        "projection": base * 0.5,
        "mixer": base,
        "remaining": base,
        "classifier": base * 2.0,
    }
    assert cfg.loss["kind"] == "meld_per_sample_focal_no_detach"
    assert cfg.loss["main_class_weight"] is False
    assert cfg.loss["aux_class_weight"] is True
    assert cfg.loss["task_weighting"] == "fixed_unit"
    assert cfg.loss["fixed_task_weights"] == {
        "main": 1.0, "t": 1.0, "a": 1.0, "v": 1.0
    }


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_each_ablation_changes_exactly_one_switch(dataset, variant):
    full = get_config(dataset, "full", 2025)
    candidate = get_config(dataset, variant, 2025)
    changed = {
        key
        for key, value in full.switches.items()
        if candidate.switches[key] != value
    }
    assert changed == (set() if variant == "full" else {variant})


def test_rejects_nonformal_seed():
    with pytest.raises(ValueError, match="formal seed"):
        get_config("meld", "full", 999)
