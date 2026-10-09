import os,json,hashlib,csv,itertools
from pathlib import Path
import numpy as np
base=Path('/data2/yb/multimodalERC/MELD'); out={}; hs={}; labels={}
for split in ['train','dev','test']:
 rows=list(csv.DictReader(open('/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw/'+split+'_sent_emo.csv')))
 keys={f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}":r for r in rows};labels[split]=keys;out[split]={}
 for mod,path in [('t',base/'Model/features_roberta_large_ft'/f'{split}_features/text_features.json'),('a',base/'Dataset/Data'/f'{split}_features/audio_features.json'),('v',base/'Model/features_denseface'/f'{split}_features/visual_features.json')]:
  blob=path.read_bytes();d=json.loads(blob);ks=list(d);arr=np.asarray([d[k] for k in ks],dtype=np.float32)
  hashes={}
  for k,a in zip(ks,arr): hashes.setdefault(hashlib.sha256(a.tobytes()).hexdigest(),[]).append(k)
  hs[split,mod]=hashes
  out[split][mod]={'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'shape':list(arr.shape),'missing_keys':sorted(set(keys)-set(ks)),'extra_keys':sorted(set(ks)-set(keys)),'nonfinite':int((~np.isfinite(arr)).sum()),'zero_rows':int((np.abs(arr).sum(axis=1)==0).sum()),'duplicate_vector_excess':len(ks)-len(hashes),'norm_mean':float(np.linalg.norm(arr,axis=1).mean())}
out['cross_split_exact_vectors']={}
for a,b in itertools.combinations(['train','dev','test'],2):
 for m in 'tav':
  shared=hs[a,m].keys()&hs[b,m].keys(); pairs=[]
  for h in shared:
   for ka in hs[a,m][h]:
    for kb in hs[b,m][h]:
     ra=labels[a].get(ka,{});rb=labels[b].get(kb,{})
     pairs.append({'a':ka,'b':kb,'same_utterance':ra.get('Utterance')==rb.get('Utterance'),'same_label':ra.get('Emotion')==rb.get('Emotion')})
  out['cross_split_exact_vectors'][a+'_'+b+'_'+m]={'unique_shared_vectors':len(shared),'pairs':pairs[:30],'pair_count':len(pairs)}
Path('results/pretraining_audit_20261009/meld_features_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
