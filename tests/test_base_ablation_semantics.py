"""CPU checks for the eleven base ablations, isolated by dataset vendor.

Set MM_MIXER_SEMANTICS_ROOT to a frozen snapshot to audit exactly the code
scheduled for training. The test file itself can remain in the working tree.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(os.environ.get("MM_MIXER_SEMANTICS_ROOT", Path(__file__).resolve().parents[1])).resolve()

PROBE = r'''
import importlib
import json
import pathlib
import sys
import torch
from torch import nn

torch.set_num_threads(1)
root, dataset = pathlib.Path(sys.argv[1]), sys.argv[2]
sys.path.insert(0, str(root))
runner = importlib.import_module('dataset_runners.' + dataset)
assert pathlib.Path(runner.__file__).resolve().is_relative_to(root)
from mm_mixer_final.config import get_config
from mm_mixer_final.modalities import MODALITY_VARIANTS, active_modalities

def build(variant):
    torch.manual_seed(197)
    model = runner.build_variant_model(variant, 0.0)
    model.disable_alignment()
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    return model

def count(module):
    return sum(p.numel() for p in module.parameters())

def inputs():
    return {m: torch.randn(3, width, requires_grad=True) for m, width in (('v', 342), ('a', 1024), ('t', 1024))}

def logits(model, features):
    result = model(features)
    value = result[0] if isinstance(result, tuple) else result
    assert value.shape == (3, 6 if dataset == 'iemocap' else 7)
    assert torch.isfinite(value).all()
    return value

def backward(model):
    values = logits(model, inputs())
    torch.nn.functional.cross_entropy(values, torch.tensor([0, 1, 2])).backward()

def has_live_grad(module):
    parameters = list(module.parameters())
    assert parameters
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
    assert any(p.grad.abs().sum().item() > 0 for p in parameters)

full = build('full')
full_parameters = count(full)
full_fg_parameters = count(full.feature_selectors)
assert full_fg_parameters > 0
assert len(full.transformer_encoder.blocks) == 2
block_parameters = count(full.transformer_encoder.blocks[1])
report = {'dataset': dataset, 'root': str(root), 'full_parameters': full_parameters, 'checks': []}

fg = build('no_feature_gating')
assert all(isinstance(module, nn.Identity) for module in fg.feature_selectors.values())
assert count(fg.feature_selectors) == 0
assert full_parameters - count(fg) == full_fg_parameters
assert count(fg.adaptive_fusion) == count(full.adaptive_fusion) > 0
backward(fg)
has_live_grad(fg.proj)
report['checks'].append({'variant': 'no_feature_gating', 'removed_parameters': full_fg_parameters})
del fg

for variant, disabled_flag, branch in (
    ('no_sequence_mixing', 'use_sub', 'sub'),
    ('no_modality_mixing', 'use_mod', 'route'),
    ('no_feature_mixing', 'use_ffn', 'ffn'),
):
    model = build(variant)
    assert count(model) == full_parameters, 'axis ablations bypass rather than delete parameter slots'
    for block in model.transformer_encoder.blocks:
        flags = {name: getattr(block, name) for name in ('use_sub', 'use_mod', 'use_ffn')}
        assert flags[disabled_flag] is False
        assert all(value is True for name, value in flags.items() if name != disabled_flag)
    backward(model)
    for block in model.transformer_encoder.blocks:
        branches = {'sub': getattr(block, 'subspace_mlp', getattr(block, 'sub', None)), 'route': block.route, 'ffn': block.ffn}
        for name, module in branches.items():
            if name == branch:
                assert all(p.grad is None for p in module.parameters()), (variant, name, 'disabled branch received gradient')
            else:
                has_live_grad(module)
    report['checks'].append({'variant': variant, 'disabled_branch_gradient': None, 'other_branches_live': True})
    del model

single = build('one_mixer_block')
assert len(single.transformer_encoder.blocks) == 1
assert full_parameters - count(single) == block_parameters
assert not any('blocks.1.' in name for name, _ in single.named_parameters())
backward(single)
has_live_grad(single.transformer_encoder.blocks[0].ffn)
report['checks'].append({'variant': 'one_mixer_block', 'removed_parameters': block_parameters})
del single

if dataset == 'iemocap':
    cfg = get_config(dataset, 'full', 2025)
    legacy = runner.materialize_legacy_config(cfg, cfg.epochs, 'full')
    original_fixed = runner._formal_module()._training_cfg(legacy)
    full_fixed = runner.apply_loss_ablation(original_fixed, 'full')
    def fixed_for(variant):
        return runner.apply_loss_ablation(original_fixed, variant)
else:
    full_fixed = runner.materialize_config(root / 'unused_test_output', 50, 2025, 'full')['fixed_params']
    def fixed_for(variant):
        return runner.materialize_config(root / 'unused_test_output', 50, 2025, variant)['fixed_params']

for variant in MODALITY_VARIANTS:
    model = build(variant)
    enabled = set(active_modalities(variant))
    assert set(model.active_input_modalities) == enabled
    fixed = fixed_for(variant)
    assert fixed['main_loss_weight'] == full_fixed['main_loss_weight']
    assert fixed['normalize_aux_loss_weights'] is False
    for modality in ('t', 'a', 'v'):
        expected = full_fixed['aux_loss_weights'][modality] if modality in enabled else 0.0
        assert fixed['aux_loss_weights'].get(modality, 0.0) == expected
    for training in (False, True):
        model.train(training)
        features = inputs()
        changed = {name: value if name in enabled else torch.randn_like(value) * 1000 + 400 for name, value in features.items()}
        original = {name: value.detach().clone() for name, value in features.items()}
        # Identical RNG makes this compare input masking, even if a vendor
        # contains a fixed-rate dropout beyond the constructor's dropout value.
        torch.manual_seed(811)
        first = logits(model, features)
        torch.manual_seed(811)
        second = logits(model, changed)
        assert torch.equal(first, second), (variant, training, 'unavailable raw feature changes logits')
        assert all(torch.equal(features[name], value) for name, value in original.items()), 'mask mutated caller input'
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(first, torch.tensor([0, 1, 2])).backward()
        for name, value in features.items():
            if name in enabled:
                assert value.grad is not None and torch.isfinite(value.grad).all()
                assert value.grad.abs().sum().item() > 0, (variant, name, 'active input disconnected')
            else:
                assert value.grad is None or torch.equal(value.grad, torch.zeros_like(value.grad)), (variant, name, 'masked input receives gradient')
    report['checks'].append({'variant': variant, 'active': sorted(enabled), 'raw_input_invariance': ['train', 'eval'], 'auxiliary_weights': fixed['aux_loss_weights']})
    del model

assert len(report['checks']) == 11
print(json.dumps(report, sort_keys=True))
'''


class BaseAblationSemanticsTest(unittest.TestCase):
    def test_both_dataset_implementations(self):
        for dataset in ("iemocap", "meld"):
            with self.subTest(dataset=dataset):
                completed = subprocess.run(
                    [sys.executable, "-c", PROBE, str(ROOT), dataset], cwd=ROOT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                print(completed.stdout.strip(), flush=True)


if __name__ == "__main__":
    unittest.main()
