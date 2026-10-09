import pickle,json,csv,hashlib,warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore');p=pickle.load(open('/home/yangbin/HRM_2/meld_multimodal_features.pkl','rb'),encoding='latin1')
v={s:json.load(open(f'/data2/yb/multimodalERC/MELD/Model/features_denseface/{s}_features/visual_features.json')) for s in ['train','dev','test']}
c={s:{f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}":r for r in csv.DictReader(open(f'/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw/{s}_sent_emo.csv'))} for s in v}
out={};direct={}
for s in v:
 matched=0; nonzero=0;missing=0
 for k,x in v[s].items():
  dia,utt=map(int,k.replace('dia','').replace('_utt',' ').split()); ix=list(p[0].get(dia,[]));
  if utt in ix:
   z=p[8][dia][ix.index(utt)];matched+=int(np.array_equal(np.asarray(x,dtype=np.float32),np.asarray(z,dtype=np.float32)));nonzero+=int(any(x))
  else:missing+=1
 direct[s]={'rows':len(v[s]),'matches_pkl_global_dia_using_splitlocal_id':matched,'missing_pkl_utt':missing}
out['direct_global_id_comparison']=direct
for a,b in [('train','dev'),('train','test'),('dev','test')]:
 ks=sorted(v[a].keys()&v[b].keys()); eq=[k for k in ks if v[a][k]==v[b][k] and any(v[a][k])];diff=[k for k in eq if c[a][k]['Utterance']!=c[b][k]['Utterance']];examples=[]
 for k in diff[:3]:
  dia,utt=map(int,k.replace('dia','').replace('_utt',' ').split());ix=list(p[0].get(dia,[])); i=ix.index(utt) if utt in ix else None
  examples.append({'key':k,'text_a':c[a][k]['Utterance'],'text_b':c[b][k]['Utterance'],'label_a':c[a][k]['Emotion'],'label_b':c[b][k]['Emotion'],'pkl_global_same_id_sentence':p[9][dia][i] if i is not None else None,'vector_sha256':hashlib.sha256(np.asarray(v[a][k],dtype=np.float32).tobytes()).hexdigest()})
 out[a+'_'+b]={'shared_keys':len(ks),'equal_nonzero_vectors':len(eq),'equal_nonzero_but_different_text':len(diff),'examples':examples}
Path('results/pretraining_audit_20261009/meld_visual_collision.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
