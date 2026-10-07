import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_text());runs=[];missing=[]
for job in plan['jobs']:
 path=Path(job['output']);result=path/'result.json'
 if not result.exists():missing.append(job['id']);continue
 r=json.loads(result.read_text());assert r['status']=='completed' and not r['smoke_only'];assert r['epochs_completed']==30 and r['checkpoint_predictions_verified'];assert r['seed']==job['seed'] and r['dataset']==job['dataset']
 with np.load(path/'predictions.npz') as z:y,pred=z['y_true'],z['y_pred']
 assert len(y)==(1623 if job['dataset']=='iemocap' else 2610)
 assert hashlib.sha256((path/'predictions.npz').read_bytes()).hexdigest()==r['predictions_sha256']
 metrics={'accuracy':float(accuracy_score(y,pred)*100),'weighted_f1':float(f1_score(y,pred,average='weighted')*100),'class_f1':(f1_score(y,pred,labels=range(len(r['class_names'])),average=None,zero_division=0)*100).tolist()}
 for key in metrics: assert np.allclose(metrics[key],r['test'][key],rtol=0,atol=1e-10)
 runs.append(r)
groups=[]
for ds in ['iemocap','meld']:
 items=[r for r in runs if r['dataset']==ds]
 if len(items)!=3:continue
 g={'dataset':ds,'seeds':[r['seed'] for r in items],'class_names':items[0]['class_names'],'parameters':items[0]['parameters'],'n':3}
 for key in ['accuracy','weighted_f1','class_f1']:
  values=np.asarray([r['test'][key] for r in items]);g[key]={'mean':values.mean(axis=0).tolist(),'sample_std':values.std(axis=0,ddof=1).tolist()}
 groups.append(g)
a.output.write_text(json.dumps({'complete':len(runs)==6,'missing':missing,'groups':groups,'runs':runs},indent=2)+'\n')
if missing:raise SystemExit('Missing formal runs: '+','.join(missing))
