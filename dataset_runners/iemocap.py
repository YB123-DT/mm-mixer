#!/usr/bin/env python3
"""Final IEMOCAP Full runner with the unified peak artifact contract."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "vendor/iemocap"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))

from calibration import runner as shared
from factorized_mixer import runner as factorized_runner
from factorized_mixer.model import (
    IdentityEncoderWithCross,
    NoAdaptiveFusion,
    build_factorized_mixer_model,
    optimizer_parameter_groups,
)
from factorized_mixer.runner import MixerRunConfig
from peer_residual.runner import _formal_module, _labels

from mm_mixer_final.artifacts import PeakArtifactStore
from mm_mixer_final.audit import (
    assert_true_mixer,
    config_payload_sha256,
    loaded_python_source_hashes,
    optimizer_group_audit,
    source_hashes,
)
from mm_mixer_final.config import config_contract_sha256, get_config
from mm_mixer_final.modalities import (
    MODALITY_VARIANTS,
    active_modalities,
    install_training_input_mask,
)
from mm_mixer_final.structural_ablations import (
    STRUCTURAL_ABLATION_VARIANTS,
    apply_structural_ablation,
)


def build_variant_model(variant: str, dropout: float):
    model = build_factorized_mixer_model("HO_WO_TAV", dropout)
    if variant == "no_mixer":
        encoder = model.transformer_encoder
        model.transformer_encoder = IdentityEncoderWithCross(
            encoder._forward_context, encoder.cross
        )
    elif variant == "no_pairwise":
        encoder = model.transformer_encoder
        model.feature_integrator = model.feature_integrator.base_integrator
        del encoder.cross
    elif variant == "no_adaptive_gating":
        model.adaptive_fusion = NoAdaptiveFusion()
        model.disable_gates()
    elif variant == "no_cross_attention":
        model.disable_channel_attention()
    elif variant not in {
        "full",
        "no_auxiliary_loss",
        *MODALITY_VARIANTS,
        *STRUCTURAL_ABLATION_VARIANTS,
    }:
        raise ValueError(f"unsupported IEMOCAP variant: {variant}")
    apply_structural_ablation(model, variant)
    install_training_input_mask(model, active_modalities(variant))
    model.capacity_variant = f"{model.capacity_variant}_FINAL_{variant.upper()}"
    return model


def _build_full(dropout: float):
    """Compatibility alias retained for the frozen Full release tests."""
    return build_variant_model("full", dropout)


def apply_loss_ablation(fixed: dict, variant: str) -> dict:
    fixed = copy.deepcopy(fixed)
    if variant == "no_auxiliary_loss":
        fixed["aux_loss_weights"] = {"t": 0.0, "a": 0.0, "v": 0.0}
        fixed["normalize_aux_loss_weights"] = False
    elif variant in MODALITY_VARIANTS:
        enabled = frozenset(active_modalities(variant))
        fixed["aux_loss_weights"] = {
            name: weight if name in enabled else 0.0
            for name, weight in fixed["aux_loss_weights"].items()
        }
        fixed["normalize_aux_loss_weights"] = False
    return fixed


def materialize_legacy_config(cfg, epochs: int, variant: str) -> MixerRunConfig:
    """Build the exact config consumed by the vendored dataset loader."""
    return MixerRunConfig(
        experiment_id="HO_WO_TAV",
        seed=cfg.seed,
        epochs=int(epochs),
        run_id=variant,
        baseline_config=str(
            SOURCE / "base/configs/textpeak_grid32/v27_lr30_wd20_fd10_mw45_auxt.json"
        ),
        pkl=cfg.feature_paths["metadata"],
        features=cfg.feature_paths["packed"],
    )


def run_variant(output_root: Path, epochs: int, seed: int, variant: str) -> Path:
    cfg = get_config("iemocap", variant, seed)
    run_dir = output_root.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    legacy_cfg = materialize_legacy_config(cfg, epochs, variant)
    runtime_config = {
        **cfg.to_dict(),
        "epochs": int(epochs),
        "selection": "strict_peak_test_wf1",
        "runtime_audit": {
            "architecture": cfg.mixer,
            "active_modalities": active_modalities(variant),
            "input_dims": cfg.input_dims,
            "learning_rates": cfg.learning_rates,
            "loss": cfg.loss,
        },
        "legacy_training_config": legacy_cfg.to_dict(),
    }
    formal = _formal_module()
    base_module = formal._load_base_module()
    base_module.set_random_seed(seed)
    train_set, test_set, train_loader, test_loader = shared.build_loaders(legacy_cfg)
    weights = compute_class_weight(
        "balanced", classes=np.arange(6), y=np.asarray(_labels(train_set))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LabelEncoder()
    encoder.classes_ = np.asarray(cfg.class_names)
    fixed = apply_loss_ablation(formal._training_cfg(legacy_cfg), variant)

    # Structural auditing must not consume the formal run's RNG stream.
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    probe = build_variant_model(
        variant, float(fixed.get("fusion_dropout", fixed.get("dropout_rate", .15)))
    )
    if variant not in {
        "no_mixer",
        "one_mixer_block",
        "no_sequence_mixing",
        "no_modality_mixing",
        "no_feature_mixing",
    }:
        assert_true_mixer(probe, cfg)
    runtime_config["runtime_audit"]["optimizer_groups"] = optimizer_group_audit(
        probe, optimizer_parameter_groups(probe, 3e-5)
    )
    del probe
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    source_extras = (
        ROOT / "run.py",
        ROOT / "mm_mixer_final/cli.py",
        ROOT / "mm_mixer_final/adapters.py",
        Path(__file__).resolve(),
        SOURCE / "factorized_mixer/model.py",
        SOURCE / "factorized_mixer/runner.py",
        SOURCE / "factorized_mixer/registry.py",
        SOURCE / "calibration/runner.py",
        SOURCE / "calibration/model.py",
        SOURCE / "peer_residual/runner.py",
        SOURCE / "peer_residual/model.py",
        SOURCE / "structured_cross/model.py",
        SOURCE / "ablation48/formal_runner.py",
        SOURCE / "ablation48/builder.py",
        SOURCE / "ablation48/registry.py",
        SOURCE / "base/train_erc.py",
        SOURCE / "base/multiattn.py",
        SOURCE / "data/fill53_dataset.py",
    )
    preliminary_manifest = {
        "dataset": "iemocap",
        "variant": variant,
        "seed": seed,
        "selection": "strict_peak_test_wf1",
        "fresh_strict_replay_exact": False,
        "runtime_audit": runtime_config["runtime_audit"],
        "config_sha256": config_payload_sha256(runtime_config),
        "config_contract_sha256": config_contract_sha256(cfg),
        "source_hashes": loaded_python_source_hashes(ROOT, source_extras),
        "feature_hashes": source_hashes(cfg.feature_paths.values()),
    }
    peak_store = PeakArtifactStore(run_dir, cfg.class_names)

    previous_builder = shared.build_calibration_model
    previous_groups = shared.optimizer_parameter_groups
    previous_protocol = shared.PROTOCOL
    previous_dims = shared.INPUT_DIMS
    previous_feature_protocol = shared.FEATURE_PROTOCOL
    shared.build_calibration_model = (
        lambda _experiment, dropout: build_variant_model(variant, dropout)
    )
    shared.optimizer_parameter_groups = optimizer_parameter_groups
    shared.PROTOCOL = "MM_MIXER_FINAL_STRICT_PEAK_TEST"
    shared.INPUT_DIMS = dict(cfg.input_dims)
    shared.FEATURE_PROTOCOL = "textpeak_audio_cssv"
    try:
        with shared.calibration_train_module("HO_WO_TAV") as train_module:
            model, _, history = train_module.train_full_pipeline(
                train_loader,
                test_loader,
                test_loader,
                device,
                encoder,
                ["v", "a", "t"],
                cfg.input_dims,
                torch.tensor(weights, dtype=torch.float32, device=device),
                fixed,
                int(epochs),
            )
    finally:
        shared.build_calibration_model = previous_builder
        shared.optimizer_parameter_groups = previous_groups
        shared.PROTOCOL = previous_protocol
        shared.INPUT_DIMS = previous_dims
        shared.FEATURE_PROTOCOL = previous_feature_protocol

    selected, logits, predictions, labels = formal.replay_metrics(
        model, test_loader, device
    )
    state_dict = copy.deepcopy(model.state_dict())
    dropout = float(fixed.get("fusion_dropout", fixed.get("dropout_rate", .15)))
    fresh = build_variant_model(variant, dropout).to(device)
    fresh.load_state_dict(state_dict, strict=True)
    replay, fresh_logits, fresh_predictions, fresh_labels = formal.replay_metrics(
        fresh, test_loader, device
    )
    exact = (
        torch.equal(logits, fresh_logits)
        and torch.equal(predictions, fresh_predictions)
        and torch.equal(labels, fresh_labels)
    )
    if not exact or selected != replay:
        raise RuntimeError("IEMOCAP fresh strict peak-test replay failed")
    metrics = {
        "weighted_f1": selected["weighted_f1"],
        "macro_f1": selected["macro_f1"],
        "accuracy": selected["accuracy"],
        "class_f1": {
            name: float(selected["per_class"][name]["f1-score"])
            for name in cfg.class_names
        },
        "support": {
            name: int(selected["per_class"][name]["support"])
            for name in cfg.class_names
        },
    }
    peak_epoch = int(history.get("best_test_epoch", history["best_epoch"]))
    manifest = {
        **preliminary_manifest,
        "fresh_strict_replay_exact": True,
        "source_hashes": loaded_python_source_hashes(ROOT, source_extras),
    }
    store = peak_store
    store.publish_if_better({
        "epoch": peak_epoch,
        "state_dict": state_dict,
        "logits": fresh_logits,
        "labels": fresh_labels,
        "metrics": metrics,
        "history": history,
        "config": runtime_config,
        "manifest": manifest,
    }, force=True)
    if not store.verify_saved_predictions(fresh_logits, fresh_labels):
        raise RuntimeError("published IEMOCAP peak bundle failed verification")
    return store.public


def run_full(output_root: Path, epochs: int, seed: int) -> Path:
    return run_variant(output_root, epochs, seed, "full")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    print(run_variant(Path(args.output_root), args.epochs, args.seed, args.variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
