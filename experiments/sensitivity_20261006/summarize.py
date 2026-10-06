#!/usr/bin/env python3
"""Summarize all seeds, including original S=1 and S=6 reference runs."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

REPO=Path(__file__).resolve().parents[2]
ROOT=REPO/'results/sensitivity_20261006'
new=json.loads((ROOT/'analysis.json').read_text())
ref=json.loads((ROOT/'reference_analysis.json').read_text())
proof=json.loads((ROOT/'completion_verification.json').read_text())
assert new['complete'] and proof['formal_runs_verified']==18 and proof['reference_runs_verified']==12
allgroups=new['groups']+ref['groups']
groups=[]
for dataset in ('iemocap','meld'):
    full=next(g for g in allgroups if g['dataset']==dataset and g['variant']=='full')
    fullseeds={r['seed']:r for r in full['seeds']}
    for g in sorted((g for g in allgroups if g['dataset']==dataset),key=lambda g: {'full':6,'single_projection_view':1}.get(g['variant'],int(g['variant'].split('_')[-1]) if g['variant'].startswith('projection_views_') else 0)):
        assert g['n']==3 and g['complete_three_seeds']
        s={'full':6,'single_projection_view':1}.get(g['variant'])
        if s is None:s=int(g['variant'].split('_')[-1])
        pairs=[]
        for r in g['seeds']:
            f=fullseeds[r['seed']]
            assert r['included'] and r['labels_sha256']==f['labels_sha256']
            meta=ROOT/'verified_metadata'/dataset/g['variant']/f"seed{r['seed']}"
            manifest=json.loads((meta/'manifest.json').read_text())
            for filename in ('config.json','peak_test_metrics.json','history.json'):
                assert hashlib.sha256((meta/filename).read_bytes()).hexdigest()==manifest['artifact_sha256'][filename]
            pairs.append({k:r['metrics'][k]-f['metrics'][k] for k in ('accuracy','weighted_f1','macro_f1')})
        groups.append({**g,'projection_views':s,'source':'reference_20261003' if s in (1,6) else 'sensitivity_20261006',
          'paired_delta_to_full':{k:{'n':3,'mean':statistics.mean(r[k] for r in pairs),'sample_std':statistics.stdev(r[k] for r in pairs)} for k in ('accuracy','weighted_f1','macro_f1')}})
result={'complete':True,'new_runs':18,'reference_runs':12,'selection':'strict_peak_test_wf1','metric_scale':'fraction','standard_deviation':'sample, ddof=1','groups':groups,
 'limitations':['S=1 is the existing post-construction single_projection_view control; S=2/4/8 change S in the constructor. S=6 remains bitwise equivalent to original Full.','Changing S also changes learned projection and S-axis MLP parameter counts; this is not a parameter-matched comparison.','These are three seeds and test-peak selection; no significance claim or independent validation-based tuning claim is made.']}
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
with (ROOT/'comparison.csv').open('w',newline='') as f:
    fields=['dataset','S','n','accuracy_mean_pct','accuracy_sample_sd_pct','weighted_f1_mean_pct','weighted_f1_sample_sd_pct','weighted_f1_delta_to_S6_pp','source'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for g in groups:
        a=g['metrics']['accuracy'];wf=g['metrics']['weighted_f1']
        w.writerow(dict(dataset=g['dataset'],S=g['projection_views'],n=g['n'],accuracy_mean_pct=100*a['mean'],accuracy_sample_sd_pct=100*a['sample_std'],weighted_f1_mean_pct=100*wf['mean'],weighted_f1_sample_sd_pct=100*wf['sample_std'],weighted_f1_delta_to_S6_pp=100*g['paired_delta_to_full']['weighted_f1']['mean'],source=g['source']))
lines=['# Projection-view sensitivity results','', '**Complete: 18/18 new formal runs.** Queue exit code 0; finished 2026-10-06T12:30:18Z. Twelve existing S=1/S=6 reference runs were independently rechecked. All checkpoint/other artifact hashes match; prediction-derived metrics match; every bundle records exact fresh checkpoint replay. No failed, omitted or selected seeds.','', 'Values below are percentages, mean ± sample standard deviation over three seeds (ddof=1). IEMOCAP: 2025, 2066, 2118; MELD: 2025, 2028, 2069. Existing user-authorized `strict_peak_test_wf1` selection is retained.','', '| S | IEMOCAP Acc | IEMOCAP WF1 | MELD Acc | MELD WF1 |','| --- | --- | --- | --- | --- |']
for s in (1,2,4,6,8):
    cells=[]
    for d in ('iemocap','meld'):
        g=next(g for g in groups if g['dataset']==d and g['projection_views']==s)
        for metric in ('accuracy','weighted_f1'):
            m=g['metrics'][metric];cells.append(f"{100*m['mean']:.2f} ± {100*m['sample_std']:.2f}")
    lines.append('| '+str(s)+' | '+' | '.join(cells)+' |')
lines+=['','## Interpretation','', 'S=6 has the highest mean WF1 among these five settings on both datasets. IEMOCAP S=2/4/8 trails S=6 by 0.62/0.38/0.25 percentage points; S=1 trails by 0.44. The trend is not monotonic: S=2 is below S=1, and S=8 is below S=6.','', 'On MELD, S=2 is effectively tied in mean WF1 with S=6 (67.84 vs 67.85); S=4 and S=8 trail by 0.09 and 0.16 points. The full S range spans only about 0.16 WF1 points. These results support retaining S=6 as a reasonable setting, not a claim that more views consistently help or that S=6 is statistically superior.','', 'S=1 and S=6 are verified existing reference experiments, not newly rerun. S=1 uses the prior post-construction control implementation; S=2/4/8 use constructor-level S. S=6 is bitwise unchanged from the original Full implementation. S-axis hidden width always follows 2S (2/4/8/12/16), so parameter counts vary.','', '## Evidence and artifacts','', '- `analysis.json` / `analysis.md`: original frozen analyzer output for all 18 new runs, with every seed, class F1 and confusion counts.','- `comparison.json` / `comparison.csv`: five-S comparison including all reference seeds and paired differences to S=6.','- `reference_analysis.json`: original verified S=1/S=6 reference subset.','- `completion_verification.json`: independent verification of all 30 bundles, including checkpoint hashes and recomputed metrics.','- `verified_metadata/`: per-run configurations, manifests, histories, metrics and statuses; no weights or raw dataset files downloaded.','- `state.json`, `summary.json`, `queue_exit_code.txt`, `finished_at.txt`: final scheduler evidence.','', 'Server: biggpu; physical GPU 7 UUID `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`. Frozen snapshot SHA256 `689bd97ff0a57fbd5fde0770f0b12c33e235934c7587d1dceee5347219020c67`. Remote new checkpoints remain under `/data2/yb/multimodalERC/MM_Mixer_Sensitivity_20261006/runs/`.','']
(ROOT/'README.md').write_text('\n'.join(lines))
print('\n'.join(lines[6:13]))
