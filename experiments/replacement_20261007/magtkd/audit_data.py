import argparse,hashlib,json,pickle
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
for ds in ['IEMOCAP','MELD']:
 split_vids=[]
 for split in ['train','dev','test']:
  path=a.data/ds/f'first_stage_{split}_features.pkl'
  d=pickle.loads(path.read_bytes());vids=d['vids']; labels=[int(y) for k in vids for y in d['labels'][k]]
  rec={'dataset':ds,'split':split,'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dialogues':len(vids),'utterances':len(labels),'label_counts':np.bincount(labels,minlength=6 if ds=='IEMOCAP' else 7).tolist(),'features':{}}
  split_vids.append(set(vids))
  for name in ['text','audio','video','audio_kd','video_kd']:
   dims=set()
   for k in vids:
    ar=np.asarray(d[name][k]);assert ar.shape[0]==len(d['labels'][k]);assert np.isfinite(ar).all();dims.add(tuple(ar.shape[1:]))
   assert dims=={(768,)}
   rec['features'][name]={'dimensions':[list(x) for x in dims],'finite':True}
  assert len(labels)==(1623 if ds=='IEMOCAP' else 2610) if split=='test' else True
  records.append(rec)
 # Dialogue ids can restart across splits in MELD; do not assume global uniqueness.
 print(ds,json.dumps([x for x in records if x['dataset']==ds]),flush=True)
a.output.write_text(json.dumps(records,indent=2)+'\n')
