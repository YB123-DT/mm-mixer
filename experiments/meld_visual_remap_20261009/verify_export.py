"""Independent check of exported vectors, metadata and preserved old inputs."""
import csv,hashlib,json,pickle,warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings('ignore')
root=Path(__file__).resolve().parents[2]
r=root/'results/meld_visual_remap_20261009'
m=json.loads((r/'manifest.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for p,h in {**m['source_hashes'],**m['output_sha256']}.items():assert sha(p)==h,p
assert sha(r/'mapping.csv')==m['mapping_sha256']
with open('/data2/yb/paper/datasets/meld_multimodal_features.pkl','rb') as f:p=pickle.load(f,encoding='latin1')
records=list(csv.DictReader((r/'mapping.csv').open()))
assert len(records)==13707 and len({(x['split'],x['key']) for x in records})==13707
old_audit=json.loads((root/'results/pretraining_audit_20261009/meld_features_probe.json').read_text())
seen={};stats={}
for split,n in [('train',9988),('dev',1109),('test',2610)]:
 data=json.loads((root/f'outputs/meld_visual_remap_20261009/{split}_features/visual_features.json').read_text())
 rows=list(csv.DictReader(open(f'/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw/{split}_sent_emo.csv')))
 meta={f"dia{x['Dialogue_ID']}_utt{x['Utterance_ID']}":x for x in rows}
 oldpath=old_audit[split]['v']['path'];assert sha(oldpath)==old_audit[split]['v']['sha256']
 old=json.loads(Path(oldpath).read_text());assert len(data)==n
 relevant=[x for x in records if x['split']==split];assert set(data)=={x['key'] for x in relevant}
 changed=0
 for x in relevant:
  d,i=int(x['source_dialogue']),int(x['source_index']);key=x['key'];row=meta[key]
  v=np.array(data[key],dtype=np.float32);source=np.array(p[8][d][i],dtype=np.float32)
  assert np.array_equal(v,source) and v.shape==(342,) and np.isfinite(v).all()
  assert ' '.join(row['Utterance'].lower().split())==' '.join(p[9][d][i].lower().split())
  assert int(row['Utterance_ID'])==int(p[0][d][i])
  assert row['Emotion']==m['source_label_map'][str(p[2][d][i])]
  target=(split,row['Dialogue_ID']);assert d not in seen or seen[d]==target;seen[d]=target
  assert d in p[11 if split=='test' else 10]
  changed+=int(not np.array_equal(v,np.asarray(old[key],dtype=np.float32)))
 stats[split]={'verified':n,'changed_vectors_vs_old':changed,'old_features_hash_unchanged':True}
assert len(seen)==1432
assert 'dia556_utt6' not in json.loads((root/'outputs/meld_visual_remap_20261009/train_features/visual_features.json').read_text())
assert not m['training_ready']
result={'status':'passed','all_exported_vectors_verified':13707,'unique_source_dialogues':len(seen),'splits':stats,'quarantined':'train/dia556_utt6','training_ready':False}
(r/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
