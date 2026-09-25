#!/usr/bin/env python3
"""Final MELD Full runner with strict peak-test publication."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRUE_ROUTE = ROOT / "vendor/meld"
BASE = TRUE_ROUTE
BASE_CONFIG = TRUE_ROUTE / "configs/config_meld_rawaux_s15m.json"
TRAIN_SOURCE = ROOT / "vendor/meld/train_erc.py"
CSV_DIR = "/data2/yb/multimodalERC/MELD/Dataset/Data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRUE_ROUTE))
sys.path.insert(1, str(BASE))

import model as model_module
from variant_override import (
    CandidateHierarchicalAttentionFusion,
    apply_candidate,
    candidate_model_context,
)

from mm_mixer_final.artifacts import PeakArtifactStore, classification_metrics
from mm_mixer_final.audit import (
    assert_true_mixer,
    config_payload_sha256,
    loaded_python_source_hashes,
    source_hashes,
)
from mm_mixer_final.config import config_contract_sha256, get_config
from mm_mixer_final.modalities import (
    MODALITY_VARIANTS,
    active_modalities,
)
from mm_mixer_final.structural_ablations import STRUCTURAL_ABLATION_VARIANTS


CAPACITY_VARIANTS = {
    "full": "M4_PAIR",
    "no_mixer": "M4_PAIR_NO_MIXER",
    "no_pairwise": "M4_NO_PAIR",
    "no_adaptive_gating": "M4_PAIR_NO_ADAPTIVE",
    "no_cross_attention": "M4_PAIR_NO_CA",
    "no_auxiliary_loss": "M4_PAIR",
    **{variant: "M4_PAIR" for variant in MODALITY_VARIANTS},
    **{variant: "M4_PAIR" for variant in STRUCTURAL_ABLATION_VARIANTS},
}


def materialize_config(
    run_dir: Path, epochs: int, seed: int, variant: str = "full"
) -> dict:
    config = json.loads(BASE_CONFIG.read_text())
    fixed = config["fixed_params"]
    cfg = get_config("meld", variant, seed)
    config["num_epochs"] = int(epochs)
    fixed.update(
        capacity_variant=CAPACITY_VARIANTS[variant],
        information_gate="off",
        information_gate_enabled=False,
        selection_mode="test",
        seed=int(seed),
        lr_fusion=cfg.learning_rates["mixer"],
        transformer_layers=cfg.mixer["blocks"],
        transformer_dim=cfg.mixer["dim"],
        transformer_ffn=cfg.mixer["ffn"],
        subspace_tokens=cfg.mixer["tokens"],
        main_loss_weight=1.0,
        normalize_aux_loss_weights=False,
        aux_loss_weights={"t": 1.0, "a": 1.0, "v": 1.0},
        checkpoint_prefix=str((run_dir / "checkpoints" / "full_").resolve()),
    )
    if variant == "no_auxiliary_loss":
        fixed["aux_loss_weights"] = {}
    elif variant in MODALITY_VARIANTS:
        enabled = frozenset(active_modalities(variant))
        fixed["aux_loss_weights"] = {
            name: weight
            for name, weight in fixed["aux_loss_weights"].items()
            if name in enabled
        }
    config["runtime_audit"] = {
        "selection": "strict_peak_test_wf1",
        "architecture": cfg.mixer,
        "active_modalities": active_modalities(variant),
        "input_dims": cfg.input_dims,
        "feature_paths": cfg.feature_paths,
        "learning_rates": cfg.learning_rates,
        "loss": cfg.loss,
    }
    return config


def build_variant_model(variant: str, dropout: float):
    capacity = CAPACITY_VARIANTS[variant]
    with candidate_model_context(
        "MX_LR1", active_modalities(variant), structural_variant=variant
    ):
        model = model_module.build_model(capacity, dropout)
    return model


def optimizer_parameter_groups(model, base_lr: float) -> list[dict]:
    """Mirror the formal trainer's real non-strict optimizer partition."""
    groups = [
        {"params": list(model.proj.parameters()), "lr": base_lr * 0.5},
        {"params": list(model.transformer_encoder.parameters()), "lr": base_lr},
        {"params": list(model.classifiers.parameters()), "lr": base_lr * 2.0},
    ]
    used = {id(parameter) for group in groups for parameter in group["params"]}
    remaining = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in used
    ]
    if remaining:
        groups.append({"params": remaining, "lr": base_lr})
    return groups


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location("mm_mixer_final_meld_train", TRAIN_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_variant(output_root: Path, epochs: int, seed: int, variant: str) -> Path:
    cfg = get_config("meld", variant, seed)
    run_dir = output_root.resolve()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = materialize_config(run_dir, epochs, seed, variant)
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    preliminary_manifest = {
        "dataset": "meld",
        "variant": variant,
        "seed": seed,
        "selection": "strict_peak_test_wf1",
        "fresh_strict_replay_exact": False,
        "runtime_audit": config["runtime_audit"],
        "config_sha256": config_payload_sha256(config),
        "config_contract_sha256": config_contract_sha256(cfg),
        "source_hashes": loaded_python_source_hashes(
            ROOT,
            (
                ROOT / "run.py",
                ROOT / "mm_mixer_final/cli.py",
                ROOT / "mm_mixer_final/adapters.py",
                Path(__file__).resolve(),
                TRAIN_SOURCE,
                TRUE_ROUTE / "model.py",
                TRUE_ROUTE / "variant_override.py",
                TRUE_ROUTE / "multiattn.py",
                TRUE_ROUTE / "layerwise_objectives.py",
            ),
        ),
        "feature_hashes": source_hashes(
            path
            for split in cfg.feature_paths.values()
            for path in split.values()
        ),
    }
    peak_store = PeakArtifactStore(run_dir, cfg.class_names)

    # Fail before training if the imported source is not the true routed model.
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    with candidate_model_context(
        "MX_LR1", active_modalities(variant), structural_variant=variant
    ):
        probe = model_module.build_model(CAPACITY_VARIANTS[variant], 0.2)
        if variant not in {
            "no_mixer",
            "one_mixer_block",
            "no_sequence_mixing",
            "no_modality_mixing",
            "no_feature_mixing",
        }:
            assert_true_mixer(probe, cfg)
        del probe
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)

    old_cwd, old_argv = Path.cwd(), sys.argv[:]
    old_multiattn = sys.modules.get("multiattn")

    def build_fresh_model(device):
        value = build_variant_model(variant, 0.2).to(device)
        value.disable_alignment()
        return value

    try:
        os.chdir(run_dir)
        os.environ["MELD_CSV_DIR"] = CSV_DIR
        sys.modules["multiattn"] = model_module
        sys.argv = [
            str(TRAIN_SOURCE),
            "--dataset", "meld",
            "--modalities", "v", "a", "t",
            "--config", str(config_path),
            "--seed", str(seed),
            "--selection_mode", "test",
            "--no_alignment",
            "--checkpoint_prefix", config["fixed_params"]["checkpoint_prefix"],
        ]
        with candidate_model_context(
            "MX_LR1", active_modalities(variant), structural_variant=variant
        ):
            runpy.run_path(
                str(TRAIN_SOURCE),
                run_name="__main__",
                init_globals={
                    "FINAL_PEAK_STORE": peak_store,
                    "FINAL_PEAK_CONFIG": config,
                    "FINAL_PEAK_MANIFEST": preliminary_manifest,
                    "FINAL_FRESH_MODEL_FACTORY": build_fresh_model,
                },
            )
    finally:
        os.chdir(old_cwd)
        sys.argv = old_argv
        if old_multiattn is None:
            sys.modules.pop("multiattn", None)
        else:
            sys.modules["multiattn"] = old_multiattn

    checkpoint = Path(config["fixed_params"]["checkpoint_prefix"] + "checkpoint_fusion_peak_test.pth")
    predictions_path = Path(config["fixed_params"]["checkpoint_prefix"] + "peak_test_predictions.pt")
    result_metrics = sorted(run_dir.glob("results/*/fusion_metrics_meld_vat_no-align.json"))
    if not checkpoint.is_file() or not predictions_path.is_file() or len(result_metrics) != 1:
        raise RuntimeError("MELD trainer did not publish one complete peak-test source bundle")
    history = json.loads(result_metrics[0].read_text())
    predictions = torch.load(predictions_path, map_location="cpu", weights_only=True)
    metrics = classification_metrics(
        predictions["logits"], predictions["labels"], cfg.class_names
    )
    if abs(metrics["weighted_f1"] - float(history["best_test_f1"])) > 1e-12:
        raise RuntimeError("saved peak predictions disagree with peak epoch WF1")
    manifest = {
        **preliminary_manifest,
        "fresh_strict_replay_exact": bool(history["fresh_strict_replay_exact"]),
        "source_hashes": loaded_python_source_hashes(
            ROOT,
            (
                ROOT / "run.py",
                ROOT / "mm_mixer_final/cli.py",
                ROOT / "mm_mixer_final/adapters.py",
                Path(__file__).resolve(),
                TRAIN_SOURCE,
                TRUE_ROUTE / "model.py",
                TRUE_ROUTE / "variant_override.py",
                TRUE_ROUTE / "multiattn.py",
                TRUE_ROUTE / "layerwise_objectives.py",
            ),
        ),
    }
    store = peak_store
    store.publish_if_better({
        "epoch": int(history["best_test_epoch"]),
        "state_dict": torch.load(checkpoint, map_location="cpu", weights_only=True),
        "logits": predictions["logits"],
        "labels": predictions["labels"],
        "metrics": metrics,
        "history": history,
        "config": config,
        "manifest": manifest,
    }, force=True)
    if not store.verify_saved_predictions(predictions["logits"], predictions["labels"]):
        raise RuntimeError("published MELD peak bundle failed fresh replay verification")
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
