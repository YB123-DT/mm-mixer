import torch


def test_mixer_forward_backward_and_grouped_shape():
    from factorized_mixer.model import build_factorized_mixer_model
    model = build_factorized_mixer_model("M1", 0.).train()
    features = {"t": torch.randn(3, 1024), "a": torch.randn(3, 1024), "v": torch.randn(3, 342)}
    labels = torch.tensor([0, 1, 2])
    output = model(features, labels=labels)
    assert torch.isfinite(output[0]).all()
    output[0].sum().backward()
    assert model.transformer_encoder._forward_context.grouped is not None


def test_mixer_keeps_encoder_capacity_close_to_transformer():
    from factorized_mixer.model import build_factorized_mixer_model
    from structured_cross.model import build_structured_cross_model
    mixer = sum(p.numel() for p in build_factorized_mixer_model("M1", 0.).transformer_encoder.parameters())
    transformer = sum(p.numel() for p in build_structured_cross_model("X3", 0.).transformer_encoder.parameters())
    assert abs(mixer - transformer) / transformer < .03


def test_all_mixer_variants_build():
    from factorized_mixer.model import build_factorized_mixer_model
    for name in ("M1", "M2", "M3", "M4", "M1_LR05", "M1_LR2", "M4_LR05", "M4_LR2", "M1_LR05_D0_H1024", "M4_LR2_D2_H1536"):
        assert build_factorized_mixer_model(name, 0.).capacity_variant.endswith(name)


def test_learning_rate_variants_only_change_optimizer_multiplier():
    from factorized_mixer.model import build_factorized_mixer_model
    assert build_factorized_mixer_model("M1_LR05", 0.).mixer_lr_multiplier == .5
    assert build_factorized_mixer_model("M1_LR2", 0.).mixer_lr_multiplier == 2.


def test_hybrid_variants_build_and_run():
    from factorized_mixer.model import build_factorized_mixer_model
    for name in ("H1", "H2"):
        model = build_factorized_mixer_model(name, 0.).train()
        output = model({"t": torch.randn(2, 1024), "a": torch.randn(2, 1024),
                        "v": torch.randn(2, 342)}, labels=torch.tensor([0, 1]))
        assert torch.isfinite(output[0]).all()


def test_mixer_scale_tuning_only_updates_existing_residual_buffers():
    from factorized_mixer.model import build_factorized_mixer_model

    model = build_factorized_mixer_model("TUNE_R4_MS_M01", 0.)
    blocks = model.transformer_encoder.blocks

    assert blocks
    assert all(torch.all(block.alpha_sub == .5) for block in blocks)
    assert all(torch.all(block.alpha_mod == 1.) for block in blocks)
    assert all(torch.all(block.alpha_ffn == 1.) for block in blocks)
    assert not hasattr(model.transformer_encoder, "post_refinement")


def test_mixer_scale_tuning_rejects_nonpositive_values():
    import pytest
    from factorized_mixer.model import (
        _apply_final_tune_overrides,
        build_factorized_mixer_model,
    )

    model = build_factorized_mixer_model("HO_WO_TAV", 0.)
    with pytest.raises(ValueError, match="positive finite"):
        _apply_final_tune_overrides(model, {"mixer_sub_scale": 0.})


def test_final_no_ca_removes_cross_modal_attention_only():
    from torch import nn
    from factorized_mixer.model import build_factorized_mixer_model

    model = build_factorized_mixer_model("FINAL_F10_NO_CA", 0.)

    assert all(isinstance(module, nn.Sequential) for module in model.cross_attn.values())
    assert all(
        module[0].__class__.__name__ == "IdentityAttention"
        for module in model.cross_attn.values()
    )
    assert model.transformer_encoder.blocks
    assert model.feature_integrator.residual_scale == 1.


def test_multitask_loss_can_keep_explicit_effective_weights():
    from peer_residual.runner import _formal_module

    loss_type = _formal_module()._load_base_module().MultitaskFusionLoss
    loss = loss_type(
        main_weight=.45,
        aux_weights={"t": .33, "a": .11, "v": .11},
        normalize_aux_weights=False,
    )

    assert loss.main_weight == .45
    assert loss.aux_weights == {"t": .33, "a": .11, "v": .11}


def test_effective_training_manifest_records_baseline_sha_and_params(tmp_path):
    import json
    from factorized_mixer.runner import (
        MixerRunConfig,
        effective_training_manifest,
    )

    source = json.loads(
        open(
            "/data2/yb/multimodalERC/IEMOCAP/Model_rawaux_clean/"
            "configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
        ).read()
    )
    source["fixed_params"]["lr_fusion"] = 2.7e-5
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(source))
    config = MixerRunConfig(
        experiment_id="HO_WO_TAV", baseline_config=str(baseline),
    )

    audit = effective_training_manifest(config)

    assert len(audit["baseline_config_sha256"]) == 64
    assert audit["effective_training_params"]["lr_fusion"] == 2.7e-5
    assert audit["scheduler_mode"] == "legacy"
    assert audit["gradient_accumulation"] == 1
