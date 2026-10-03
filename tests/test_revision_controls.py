from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch
from torch import nn

from mm_mixer_final.revision_controls import (
    CubeMLPStyleBlock,
    JointResidualMLPBlock,
    OrdinaryResidualMLP,
    REVISION_CONTROL_VARIANTS,
    pairwise_live_parameter_count,
    parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]

# Dataset vendors use a shared top-level module name. Separate interpreters keep
# a successful test from accidentally exercising the other dataset's imports.
DATASET_CHECK = r'''
import importlib
import json
import sys
import torch
from torch import nn
from mm_mixer_final.revision_controls import (
    REVISION_CONTROL_VARIANTS, apply_revision_control,
    revision_control_metadata, parameter_count,
)

torch.set_num_threads(1)
dataset = sys.argv[1]
runner = importlib.import_module('dataset_runners.' + dataset)
torch.manual_seed(71)
full = runner.build_variant_model('full', .2).eval()
full_inputs = {name: torch.randn(3, width) for name, width in (('v', 342), ('a', 1024), ('t', 1024))}
with torch.no_grad():
    before_result = full(full_inputs)
    before_logits = before_result[0] if isinstance(before_result, tuple) else before_result
before_state = {k: v.clone() for k, v in full.state_dict().items()}
before_rng = torch.get_rng_state().clone()
assert apply_revision_control(full, 'full') is full
assert revision_control_metadata(full) == {}
assert torch.equal(before_rng, torch.get_rng_state())
assert all(torch.equal(v, full.state_dict()[k]) for k, v in before_state.items())
with torch.no_grad():
    after_result = full(full_inputs)
    after_logits = after_result[0] if isinstance(after_result, tuple) else after_result
assert torch.equal(before_logits, after_logits)

mlp_initial_state = None
for variant in REVISION_CONTROL_VARIANTS:
    torch.manual_seed(71)
    model = runner.build_variant_model('full', .2).eval()
    encoder, pool, integrator = model.transformer_encoder, model.pool, model.feature_integrator
    cross = encoder.cross
    context = getattr(encoder, '_forward_context', getattr(encoder, '_channel', None))
    split = encoder.split
    apply_revision_control(model, variant)
    assert model.transformer_encoder is encoder
    assert model.pool is pool and model.feature_integrator is integrator
    assert getattr(encoder, '_forward_context', getattr(encoder, '_channel', None)) is context
    if variant != 'pairwise_mlp_residual':
        assert encoder.cross is cross
    if variant in ('amm_attention', 'amm_cubemlp'):
        assert encoder.split is split

    meta = revision_control_metadata(model)
    json.dumps(meta)
    assert meta['variant'] == variant
    assert meta['model_parameters_after'] == parameter_count(model)
    if variant == 'amm_mlp':
        mlp_initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    if variant == 'amm_mlp_no_aux':
        assert mlp_initial_state.keys() == model.state_dict().keys()
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in mlp_initial_state.items())
    if variant in ('amm_mlp', 'amm_attention', 'amm_cubemlp', 'amm_mlp_no_aux'):
        assert len(encoder.blocks) == 2
        assert abs(meta['target_parameters'] - meta['replacement_parameters']) / meta['target_parameters'] < .001
    if variant == 'single_projection_view':
        assert encoder.token_count == 1 and encoder.split.out_features == 256
        for block in encoder.blocks:
            sub = getattr(block, 'subspace_mlp', getattr(block, 'sub', None))
            assert sub[0].in_features == 1 and sub[0].out_features == 2
            assert sub[-1].out_features == 1
    if variant == 'no_feature_and_adaptive_gating':
        assert all(isinstance(layer, nn.Identity) for layer in model.feature_selectors.values())
        assert parameter_count(model.adaptive_fusion) == 0
        assert all(parameter_count(gate) == 0 for gate in model.gates.values())

    groups = runner.optimizer_parameter_groups(model, 3e-5)
    ids = [id(p) for group in groups for p in group['params']]
    assert len(ids) == len(set(ids)), 'duplicate optimizer parameter'
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)

    inputs = {name: torch.randn(3, width) for name, width in (('v', 342), ('a', 1024), ('t', 1024))}
    if variant == 'pairwise_mlp_residual':
        with torch.no_grad():
            zero = encoder.cross(torch.randn(3, 3, 256))
            assert torch.equal(zero, torch.zeros_like(zero))
    if variant.startswith('amm_'):
        watched = [(name, p) for name, p in encoder.named_parameters() if not name.startswith('cross.')]
    elif variant == 'pairwise_mlp_residual':
        watched = list(encoder.cross.named_parameters())
    else:
        watched = []
    initial = {name: p.detach().clone() for name, p in watched}
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        result = model(inputs)
        logits = result[0] if isinstance(result, tuple) else result
        assert logits.shape == (3, 6 if dataset == 'iemocap' else 7)
        assert torch.isfinite(logits).all()
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
        for name, parameter in watched:
            assert parameter.grad is not None, (variant, name)
            assert torch.isfinite(parameter.grad).all(), (variant, name)
        optimizer.step()
    for name, parameter in watched:
        assert not torch.equal(initial[name], parameter), ('parameter never updated', variant, name)

    # The ordinary residual consumes post-encoder branches and is added at the
    # identical post-pooling/integrator position, including after nonzero updates.
    if variant == 'pairwise_mlp_residual':
        with torch.no_grad():
            grouped = encoder(torch.randn(3, 3, 256))
            residual = encoder.cross(grouped)
            pooled = torch.randn(3, 768)
            expected = integrator.base_integrator(pooled) + residual
            actual = integrator(pooled)
            assert torch.allclose(expected, actual, atol=1e-6)
    print(dataset, variant, meta['model_parameters_after'], flush=True)
'''


class RevisionControlTest(unittest.TestCase):
    def test_dataset_models_forward_backward_and_budget(self):
        for dataset in ("iemocap", "meld"):
            with self.subTest(dataset=dataset):
                result = subprocess.run(
                    [sys.executable, "-c", DATASET_CHECK, dataset], cwd=ROOT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                    capture_output=True, text=True, timeout=180,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(len(result.stdout.strip().splitlines()), len(REVISION_CONTROL_VARIANTS))

    def test_cube_axis_operations_follow_paper_residual_then_axis_norm(self):
        block = CubeMLPStyleBlock(tokens=4, dim=8, hidden=12)
        inputs = torch.randn(2, 3, 4, 8)
        sequence = inputs.transpose(-1, -2)
        expected = block.norm_sub(sequence + block.sub(sequence)).transpose(-1, -2)
        modality = expected.permute(0, 2, 3, 1)
        expected = block.norm_mod(modality + block.route(modality)).permute(0, 3, 1, 2)
        expected = block.norm_ffn(expected + block.ffn(expected))
        torch.testing.assert_close(block(inputs), expected)

    def test_joint_mlp_mixes_modalities_and_preserves_shape(self):
        block = JointResidualMLPBlock(dim=4, hidden=7)
        inputs = torch.randn(2, 3, 1, 4, requires_grad=True)
        outputs = block(inputs)
        self.assertEqual(outputs.shape, inputs.shape)
        outputs[:, 0].sum().backward()
        self.assertGreater(inputs.grad[:, 1:].abs().sum().item(), 0)

    def test_ordinary_residual_zero_init_and_live_budget(self):
        class PairwiseFixture(nn.Module):
            def __init__(self):
                super().__init__()
                self.dim, self.rank = 256, 32
                self.pair_projections = nn.ModuleList(nn.Linear(256, 32) for _ in range(3))
                self.triple_projections = nn.ModuleList(nn.Linear(256, 32) for _ in range(3))
                self.output = nn.Linear(128, 256)
                self.active_interactions = frozenset(('av', 'tv', 'ta'))

        original = PairwiseFixture()
        self.assertEqual(parameter_count(original), 82368)
        self.assertEqual(pairwise_live_parameter_count(original), 49504)
        replacement = OrdinaryResidualMLP(dim=256, hidden=48)
        self.assertEqual(parameter_count(replacement), 49456)
        inputs = torch.randn(3, 3, 256)
        torch.testing.assert_close(replacement(inputs), torch.zeros(3, 256))
        self.assertEqual(set(replacement.decomposed(inputs)), {'mlp'})


if __name__ == '__main__':
    unittest.main()
