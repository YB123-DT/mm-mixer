from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


def _load_vendor_loss():
    vendor = Path(__file__).resolve().parents[1] / "vendor/meld"
    sys.path.insert(0, str(vendor))
    spec = importlib.util.spec_from_file_location("final_meld_multiattn", vendor / "multiattn.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.MultitaskFusionLoss


def test_explicit_unit_tasks_exactly_equal_historical_zero_log_vars():
    loss_type = _load_vendor_loss()
    common = {
        "main_weight": 1.0,
        "aux_weights": {"t": 1.0, "a": 1.0, "v": 1.0},
        "normalize_aux_weights": False,
        "poly_alpha": 1.9508055462649292,
        "poly_gamma": 1.402459950741192,
        "focal_reweight_gamma": 2.5,
        "loss_type": "poly",
        "num_classes": 7,
    }
    torch.manual_seed(2025)
    labels = torch.tensor([0, 2, 6, 1])
    main_a = torch.randn(4, 7, requires_grad=True)
    aux_a = {
        name: torch.randn(4, 7, requires_grad=True) for name in ("t", "a", "v")
    }
    main_b = main_a.detach().clone().requires_grad_(True)
    aux_b = {
        name: value.detach().clone().requires_grad_(True)
        for name, value in aux_a.items()
    }
    explicit = loss_type(**common)
    # Frozen mathematical reference for the removed historical branch with
    # log_vars={main,t,a,v}=0: exp(0)*L + 0 for every task.
    def historical_zero_log_vars(main_logits, aux_logits):
        ce_per_sample = torch.nn.functional.cross_entropy(
            main_logits, labels, reduction="none"
        )
        pt = torch.exp(-ce_per_sample)
        main_per_sample = (
            ce_per_sample
            + common["poly_alpha"] * (1 - pt) ** (common["poly_gamma"] + 1)
        )
        old_main = (
            main_per_sample * (1 - pt) ** common["focal_reweight_gamma"]
        ).mean()
        return old_main + sum(
            explicit.ce_loss(value, labels) for value in aux_logits.values()
        )

    old_value = historical_zero_log_vars(main_a, aux_a)
    new_value = explicit(main_b, aux_b, labels)
    assert torch.equal(old_value, new_value)
    old_value.backward()
    new_value.backward()
    assert torch.equal(main_a.grad, main_b.grad)
    for name in aux_a:
        assert torch.equal(aux_a[name].grad, aux_b[name].grad)
