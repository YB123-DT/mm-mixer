"""Reuse the original FLOPs counter with explicit no_pairwise identity checks."""
import hashlib
import json
import runpy
import sys
from pathlib import Path

root = Path('/data2/yb/multimodalERC/MM_Mixer_Revision_20261003')
dataset = sys.argv[1]
assert dataset in ('iemocap', 'meld')
sys.path.insert(0, str(root / 'code/revision_1b8b1ff'))
import measure_revision_efficiency as efficiency


def validate_no_pairwise(config, ds, variant, seed):
    assert variant == 'no_pairwise' and ds == dataset and seed == 2025
    assert not config.get('runtime_audit', {}).get('revision_control')
    if ds == 'iemocap':
        assert (config['dataset'], config['variant'], config['seed']) == (ds, variant, seed)
    else:
        fixed = config['fixed_params']
        assert fixed['seed'] == seed and fixed['capacity_variant'] == 'M4_NO_PAIR'
        assert fixed['no_alignment'] is True
        assert 'meld' in config['classes'] and 'meld' in config['feature_paths']


# Only replace the replacement-control-specific identity validator. All manifest,
# artifact, feature, strict weight loading and numerical checks remain intact.
efficiency.validate_config_identity = validate_no_pairwise
state = json.loads((root / 'pipeline/combined_state.json').read_text())
job = state['jobs'][dataset + '/no_pairwise/seed2025']
out = root / 'flops_no_epirc_20261009' / (dataset + '.json')
sys.argv = ['measure_revision_flops.py', '--code-root', str(root / 'code/revision_1b8b1ff'),
            '--artifact-dir', job['bundle'], '--dataset', dataset, '--variant', 'no_pairwise',
            '--output', str(out)]
runpy.run_path(str(root / 'automation/measure_revision_flops.py'), run_name='__main__')
report = json.loads(out.read_text())
report['identity_adapter_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
out.write_text(json.dumps(report, indent=2) + '\n')
