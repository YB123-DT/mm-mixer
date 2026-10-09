"""Pair each verified capacity run with its fixed original Full seed."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); ref=json.loads((root/'references.json').read_text()); new=json.loads((root/'analysis.json').read_text())
baseline={(g['dataset'],s['seed']):s for g in ref['groups'] for s in g['seeds']}
rows=[]
for g in new['groups']:
 pairs=[]
 for s in g['seeds']:
  if not (s['included'] and s['artifact_verified'] and s['formal_protocol_verified']):continue
  b=baseline[(g['dataset'],s['seed'])]
  assert s['labels_sha256']==b['labels_sha256']
  pairs.append({'seed':s['seed'],'wf1':s['metrics']['weighted_f1']*100,'full_wf1':b['metrics']['weighted_f1']*100,'delta_pp':(s['metrics']['weighted_f1']-b['metrics']['weighted_f1'])*100})
 rows.append({'dataset':g['dataset'],'variant':g['variant'],'n':len(pairs),'complete_three_seeds':len(pairs)==3,'paired_seeds':pairs,'mean_delta_pp':statistics.mean(p['delta_pp'] for p in pairs) if len(pairs)==3 else None})
(root/'comparison.json').write_text(json.dumps({'complete':new['complete'],'selection':'strict_peak_test_wf1','groups':rows},indent=2)+'\n')
print(json.dumps(rows,indent=2))
