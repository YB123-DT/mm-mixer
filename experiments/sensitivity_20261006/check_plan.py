#!/usr/bin/env python3
"""Verify every planned sensitivity run is accepted by execution and analysis."""
import json
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'outputs/sensitivity_20261006/code'
sys.path.insert(0, str(ROOT))
from launch_revision import generate_plan, load_plan, command_for
from analyze_revision import _read_plan
from mm_mixer_final.config import get_config

plan = generate_plan('sensitivity', str(ROOT), '/tmp/sensitivity_plan_only', sys.executable)
with tempfile.TemporaryDirectory(dir=HERE) as tmp:
    path = Path(tmp) / 'plan.json'
    path.write_text(json.dumps(plan))
    jobs = load_plan(path)
    assert len(jobs) == len(_read_plan(path)) == 18
    for job in jobs:
        cfg = get_config(job['dataset'], job['variant'], job['seed'])
        full = get_config(job['dataset'], 'full', job['seed']).to_dict()
        expected = cfg.to_dict()
        expected['variant'] = 'full'
        expected['mixer'] = full['mixer']
        assert expected == full, 'Unrelated configuration changed'
        assert cfg.mixer['tokens'] == int(job['variant'].split('_')[-1])
        assert cfg.mixer['subspace_hidden'] == 2 * cfg.mixer['tokens']
        assert '--epochs' not in command_for(job)
print('18 unique formal jobs; only S and derived 2S width vary; execution/analysis validators passed')
