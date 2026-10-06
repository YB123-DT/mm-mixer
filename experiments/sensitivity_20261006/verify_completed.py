#!/usr/bin/env python3
"""Run on biggpu: independently validate finished matrix and reference bundles."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path('/data2/yb/multimodalERC/MM_Mixer_Sensitivity_20261006')
sys.path.insert(0, str(ROOT / 'code'))
from analyze_revision import analyze
from mm_mixer_final.artifacts import classification_metrics
import torch


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()

p=ROOT/'pipeline'
assert (p/'queue_exit_code.txt').read_text().strip()=='0'
state=json.loads((p/'state.json').read_text())
assert state['queue_state']=='complete' and len(state['jobs'])==18
assert all(r['state']=='completed' and r['returncode']==0 for r in state['jobs'].values())
fresh=analyze(p/'plan.json',p/'state.json')
assert fresh==json.loads((p/'analysis.json').read_text()) and fresh['complete']
reference=json.loads((p/'reference_analysis.json').read_text())
rows=[]
for group in fresh['groups']+reference['groups']:
    assert group['complete_three_seeds'] and group['n']==3
    for row in group['seeds']:
        b=Path(row['bundle']); m=json.loads((b/'manifest.json').read_text()); status=json.loads((b/'status.json').read_text())
        assert status['state']=='complete' and status['fresh_strict_replay_exact']
        assert m['fresh_strict_replay_exact'] and m['selection']=='strict_peak_test_wf1'
        hashes=m['artifact_sha256']
        for name,value in hashes.items():
            assert digest(b/name)==value, (str(b),name)
        for name,value in row['artifact_sha256'].items():
            assert hashes[name]==value
        pred=torch.load(b/'peak_test_predictions.pt',map_location='cpu',weights_only=True)
        metrics=classification_metrics(pred['logits'],pred['labels'],tuple(group['class_names']))
        assert metrics==row['metrics']
        label_hash=hashlib.sha256(pred['labels'].to(torch.int64).contiguous().numpy().tobytes()).hexdigest()
        assert label_hash==row['labels_sha256']
        dest=p/'verified_metadata'/row['dataset']/row['variant']/f"seed{row['seed']}"
        dest.mkdir(parents=True,exist_ok=True)
        for name in ('manifest.json','status.json','config.json','peak_test_metrics.json','history.json'):
            shutil.copy2(b/name,dest/name)
        rows.append({'id':row['id'],'bundle':str(b),'artifact_sha256':hashes,'labels_sha256':label_hash,'metrics_recomputed':True,'fresh_strict_replay_exact':True,'reference':row['variant'] in ('full','single_projection_view')})
assert len(rows)==30
report={'formal_runs_verified':18,'reference_runs_verified':12,'new_analysis_recomputed_exact':True,'all_artifact_hashes_verified_including_checkpoints':True,'queue_exit_code':0,'finished_at':(p/'finished_at.txt').read_text().strip(),'rows':rows}
(p/'completion_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print('Verified 18 new + 12 reference bundles, all hashes and prediction-derived metrics; exported metadata only.')
