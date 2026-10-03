"""Exercise revision controls through the production construction boundaries.

The two vendored datasets share top-level import names, so every dataset check
uses a separate interpreter. No feature files, CUDA devices, or training runs
are needed. Full equivalence also uses a separate interpreter for the frozen
pre-integration source tree, when that local snapshot is available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BASE_SNAPSHOT = ROOT / "outputs/revision_20261003/snapshots/base_v1"


BUILD_CHECK = r'''
import copy
import importlib
from pathlib import Path
import sys
import torch
from mm_mixer_final.audit import assert_true_mixer, loaded_python_source_hashes
from mm_mixer_final.config import RUN_VARIANTS, get_config
from mm_mixer_final.revision_controls import (
    REVISION_CONTROL_VARIANTS, apply_revision_control, revision_control_metadata,
)

torch.set_num_threads(1)
dataset = sys.argv[1]
runner = importlib.import_module('dataset_runners.' + dataset)
inputs = {name: torch.randn(2, width, generator=torch.Generator().manual_seed(width))
          for name, width in (('v', 342), ('a', 1024), ('t', 1024))}
for variant in REVISION_CONTROL_VARIANTS:
    assert variant in RUN_VARIANTS
    cfg = get_config(dataset, variant, 2025)
    torch.manual_seed(73)
    direct = runner.build_variant_model(variant, .2).eval()
    direct_rng = torch.get_rng_state().clone()
    torch.manual_seed(73)
    expected = apply_revision_control(runner.build_variant_model('full', .2), variant).eval()
    assert torch.equal(direct_rng, torch.get_rng_state()), (dataset, variant, 'constructor RNG')
    assert direct.state_dict().keys() == expected.state_dict().keys(), variant
    assert all(torch.equal(value, expected.state_dict()[key])
               for key, value in direct.state_dict().items()), (dataset, variant, 'wrong constructed control')
    assert revision_control_metadata(direct)['variant'] == variant
    if variant in ('single_projection_view', 'pairwise_mlp_residual', 'no_feature_and_adaptive_gating'):
        assert_true_mixer(direct, cfg)
    with torch.no_grad():
        first = direct(inputs)
        first = first[0] if isinstance(first, tuple) else first
    restored = runner.build_variant_model(variant, .2).eval()
    restored.load_state_dict(direct.state_dict(), strict=True)
    with torch.no_grad():
        replay = restored(inputs)
        replay = replay[0] if isinstance(replay, tuple) else replay
    assert torch.equal(first, replay), (dataset, variant, 'fresh strict replay')
    assert first.shape == (2, 6 if dataset == 'iemocap' else 7)
    assert torch.isfinite(first).all()
    del direct, expected, restored

hashes = loaded_python_source_hashes(Path.cwd())
assert str((Path.cwd() / 'mm_mixer_final/revision_controls.py').resolve()) in hashes
if dataset == 'iemocap':
    full_cfg = get_config(dataset, 'full', 2025)
    legacy = runner.materialize_legacy_config(full_cfg, 100, 'full')
    fixed = runner._formal_module()._training_cfg(legacy)
    original = copy.deepcopy(fixed)
    with_aux = runner.apply_loss_ablation(fixed, 'amm_mlp')
    without_aux = runner.apply_loss_ablation(fixed, 'amm_mlp_no_aux')
    existing = runner.apply_loss_ablation(fixed, 'no_auxiliary_loss')
    assert fixed == original and with_aux == fixed
    assert without_aux == existing
    assert without_aux['aux_loss_weights'] == {'t': 0., 'a': 0., 'v': 0.}
    assert without_aux['normalize_aux_loss_weights'] is False
    assert without_aux['main_loss_weight'] == fixed['main_loss_weight'] == .45
else:
    full = runner.materialize_config(Path('/tmp/mm_mixer_integration'), 50, 2025, 'full')
    for variant in ('amm_mlp_no_aux', 'no_auxiliary_loss'):
        no_aux = runner.materialize_config(Path('/tmp/mm_mixer_integration'), 50, 2025, variant)
        assert no_aux['fixed_params']['aux_loss_weights'] == {}
        assert no_aux['fixed_params']['normalize_aux_loss_weights'] is False
        assert no_aux['fixed_params']['main_loss_weight'] == full['fixed_params']['main_loss_weight'] == 1.
print('OK', dataset)
'''


MELD_TRAINER_CHECK = r'''
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import LabelEncoder
from dataset_runners import meld as runner
from mm_mixer_final.config import get_config
from mm_mixer_final.revision_controls import REVISION_CONTROL_VARIANTS, revision_control_metadata

torch.set_num_threads(1)
class ReachedTraining(Exception):
    pass

class UnusedGraphEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

encoder = LabelEncoder()
encoder.classes_ = np.asarray(get_config('meld', 'full', 2025).class_names)
for variant in REVISION_CONTROL_VARIANTS:
    config = runner.materialize_config(Path('/tmp/mm_mixer_integration'), 50, 2025, variant)
    fixed = config['fixed_params']
    assert fixed['skip_pretrain'] and not fixed.get('use_context_encoders', False)
    # Exactly the module alias and construction context used by run_variant.
    with runner.candidate_model_context('MX_LR1', structural_variant=variant):
        sys.modules['multiattn'] = runner.model_module
        spec = importlib.util.spec_from_file_location('revision_actual_meld_trainer', runner.TRAIN_SOURCE)
        trainer = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = trainer
        spec.loader.exec_module(trainer)
        # Graph encoders are constructed but unused by the frozen raw-feature
        # protocol. Avoid allocating those unrelated graphs in this CPU test.
        trainer.TextGraphEncoder = trainer.AudioGraphEncoder = trainer.VisualGraphEncoder = UnusedGraphEncoder
        reached = []
        def capture(model, train_loader, val_loader, test_loader, criterion, optimizer, *args, **kwargs):
            metadata = revision_control_metadata(model)
            assert metadata['variant'] == variant, ('actual trainer constructed Full', variant)
            optimized = [id(p) for group in optimizer.param_groups for p in group['params']]
            assert len(optimized) == len(set(optimized)), 'duplicate optimized parameter'
            assert set(optimized) == {id(p) for p in model.parameters() if p.requires_grad}
            aux = {name: torch.randn(2, 7, requires_grad=True) for name in ('t', 'a', 'v')}
            logits = torch.randn(2, 7, requires_grad=True)
            loss = criterion(logits, aux, torch.tensor([0, 4]))
            assert torch.isfinite(loss)
            loss.backward()
            assert logits.grad is not None and logits.grad.abs().sum() > 0
            if variant == 'amm_mlp_no_aux':
                assert criterion.aux_weights == {}
                assert all(value.grad is None for value in aux.values())
            else:
                assert criterion.aux_weights == {'t': 1., 'a': 1., 'v': 1.}
                assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in aux.values())
            reached.append(True)
            raise ReachedTraining
        trainer.train_fusion_model = capture
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train_full_pipeline(
                    [None], [None], [None], torch.device('cpu'), encoder,
                    ['v', 'a', 't'], {'v': 342, 'a': 1024, 't': 1024},
                    torch.ones(7), [1] * 7, fixed, 50,
                )
        except ReachedTraining:
            pass
        assert reached == [True], (variant, 'did not reach the actual training boundary')
print('OK actual MELD trainer')
'''


FULL_FINGERPRINT = r'''
import hashlib
import importlib
import json
import sys
import torch
from mm_mixer_final.config import config_contract_sha256, get_config

torch.set_num_threads(1)
runner = importlib.import_module('dataset_runners.' + sys.argv[1])
def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
torch.manual_seed(2025)
model = runner.build_variant_model('full', .2)
result = {'contract': config_contract_sha256(get_config(sys.argv[1], 'full', 2025)),
          'state': {key: tensor_hash(value) for key, value in model.state_dict().items()},
          'rng_after_build': tensor_hash(torch.get_rng_state())}
inputs = {name: torch.randn(3, width) for name, width in (('v', 342), ('a', 1024), ('t', 1024))}
for mode in ('eval', 'train'):
    model.train(mode == 'train')
    with torch.no_grad():
        output = model(inputs)
        output = output[0] if isinstance(output, tuple) else output
    result[mode + '_logits'] = tensor_hash(output)
    result[mode + '_rng'] = tensor_hash(torch.get_rng_state())
print(json.dumps(result, sort_keys=True))
'''


class RevisionIntegrationTest(unittest.TestCase):
    def run_isolated(self, source, *arguments, root=ROOT):
        environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root)}
        completed = subprocess.run(
            [sys.executable, "-c", source, *arguments], cwd=root, env=environment,
            text=True, capture_output=True, timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return completed.stdout

    def test_all_revision_identifiers_reach_dataset_builders_and_replay(self):
        for dataset in ("iemocap", "meld"):
            with self.subTest(dataset=dataset):
                self.assertIn("OK " + dataset, self.run_isolated(BUILD_CHECK, dataset))

    def test_actual_meld_training_constructor_optimizer_and_no_auxiliary_loss(self):
        self.assertIn("OK actual MELD trainer", self.run_isolated(MELD_TRAINER_CHECK))

    @unittest.skipUnless(BASE_SNAPSHOT.is_dir(), "local pre-integration base_v1 snapshot not present")
    def test_full_config_state_logits_and_rng_unchanged_from_frozen_base(self):
        for dataset in ("iemocap", "meld"):
            with self.subTest(dataset=dataset):
                old = json.loads(self.run_isolated(FULL_FINGERPRINT, dataset, root=BASE_SNAPSHOT))
                current = json.loads(self.run_isolated(FULL_FINGERPRINT, dataset))
                self.assertEqual(old, current)


if __name__ == "__main__":
    unittest.main()
