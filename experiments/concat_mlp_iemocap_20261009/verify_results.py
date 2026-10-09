"""Recompute final metrics from the saved prediction artifact, without training."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]/'results/concat_mlp_iemocap_20261009'
r=json.loads((root/'result.json').read_text());rows=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
assert (root/'exit_code.txt').read_text().strip()=='0'
assert r['status']=='completed' and not r['smoke_only'] and r['fresh_strict_replay_exact']
assert r['epochs_completed']==len(rows) and (len(rows)==100 or r['early_stopped'])
assert [x['epoch'] for x in rows]==list(range(1,len(rows)+1))
assert r['selected_epoch']==max(rows,key=lambda x:x['test']['weighted_f1'])['epoch']
p=np.load(root/'predictions.npz');y=p['labels'];pred=p['predictions']
assert len(y)==1623 and np.array_equal(pred,p['logits'].argmax(-1))
assert hashlib.sha256((root/'predictions.npz').read_bytes()).hexdigest()==r['predictions_sha256']
counts=np.zeros((6,6),dtype=np.int64)
np.add.at(counts,(y,pred),1)
support=counts.sum(1);denom=support+counts.sum(0)
f1=np.divide(2*np.diag(counts),denom,out=np.zeros(6),where=denom!=0)
actual={'accuracy':100*np.mean(y==pred),'weighted_f1':100*np.sum(f1*support)/len(y),'class_f1':(100*f1).tolist()}
for k,v in actual.items():assert np.allclose(v,r['test'][k],atol=1e-12,rtol=0),k
full={'weighted_f1':71.92653797281977,'accuracy':71.78065311152187,'parameters':6358082}
summary={'status':'completed','dataset':'iemocap','seed':2025,'pure_concat_mlp':r,'full_same_seed':full,'delta_wf1_pp':actual['weighted_f1']-full['weighted_f1'],'delta_acc_pp':actual['accuracy']-full['accuracy'],'limitations':['one seed only','test-WF1-selected checkpoint, same existing diagnostic protocol','Full includes auxiliary loss; this plainMLP has none']}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
