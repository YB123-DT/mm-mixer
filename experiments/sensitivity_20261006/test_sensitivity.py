#!/usr/bin/env python3
"""Shape/gradient/optimizer checks and original S=6 numerical regression."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

BASE = '1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab'
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = REPO / 'outputs/sensitivity_20261006/code'
PROBE = r'''
import hashlib, importlib, json, sys
import torch
from mm_mixer_final.audit import assert_true_mixer
from mm_mixer_final.config import get_config

torch.set_num_threads(1)
dataset, variant = sys.argv[1:]
runner = importlib.import_module('dataset_runners.' + dataset)
torch.manual_seed(2025)
model = runner.build_variant_model(variant, .2).eval()
cfg = get_config(dataset, variant, 2025)
assert_true_mixer(model, cfg)
encoder = model.transformer_encoder
state_hash = hashlib.sha256()
for name, value in model.state_dict().items():
    state_hash.update(name.encode())
    state_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
inputs = {name: torch.randn(2, width) for name, width in (('v',342),('a',1024),('t',1024))}
result = model(inputs)
logits = result[0] if isinstance(result, tuple) else result
assert logits.shape == (2, len(cfg.class_names))
assert torch.isfinite(logits).all()
output_hash = hashlib.sha256(logits.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
optimizer = torch.optim.AdamW(runner.optimizer_parameter_groups(model, 3e-5))
tracked = [encoder.split.weight]
for block in encoder.blocks:
    sub = block.subspace_mlp if hasattr(block, 'subspace_mlp') else block.sub
    tracked += [sub[0].weight, sub[2].weight, block.route.weight, block.ffn[0].weight]
registered = {id(p) for group in optimizer.param_groups for p in group['params']}
assert all(id(p) in registered for p in tracked)
before = [p.detach().clone() for p in tracked]
logits.square().mean().backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in tracked)
optimizer.step()
assert all(not torch.equal(old, p.detach()) for old,p in zip(before,tracked))
print(json.dumps({'dataset': dataset, 'variant': variant, 'tokens':encoder.token_count,
  'subspace_hidden':2 * encoder.token_count, 'blocks':len(encoder.blocks),
  'feature_hidden':encoder.blocks[0].ffn[0].out_features,
  'parameters':sum(p.numel() for p in model.parameters()),
  'initial_state_sha256':state_hash.hexdigest(), 'logits_sha256':output_hash,
  'forward_backward_optimizer_update':'passed'}))
'''


def probe(root, dataset, variant):
    env = dict(os.environ, PYTHONPATH=str(root), CUDA_VISIBLE_DEVICES='')
    value = subprocess.check_output([sys.executable, '-c', PROBE, dataset, variant], cwd=root, env=env, text=True)
    return json.loads(value.splitlines()[-1])


def main():
    results=[]
    with tempfile.TemporaryDirectory(prefix='original-', dir=HERE) as original:
        raw = subprocess.check_output(['git', '-C', str(REPO), 'archive', BASE])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            archive.extractall(original)
        for dataset in ('iemocap','meld'):
            old = probe(original, dataset, 'full')
            full = probe(ROOT, dataset, 'full')
            assert old == full, (dataset, old, full)
            full['original_commit_bitwise_regression']='passed'
            results.append(full)
            for s in (2,4,8):
                result = probe(ROOT, dataset, f'projection_views_{s}')
                assert result['tokens'] == s
                results.append(result)
    (HERE / 'verification.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
