import argparse,importlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--dataset',choices=['iemocap','meld'],required=True);a=p.parse_args()
os.chdir(a.source/a.dataset.upper());sys.path.insert(0,str(Path.cwd()));d=importlib.import_module('dataloader')
sets={'train':d.IEMOCAPRobertaDataset(True),'test':d.IEMOCAPRobertaDataset(False)} if a.dataset=='iemocap' else {s:d.MELDRobertaDataset(s) for s in ['train','valid','test']}
out={}
for split,data in sets.items():
 rows=[data[i] for i in range(len(data))];out[split]={'dialogues':len(rows),'utterances':sum(len(r[6]) for r in rows),'first_feature_widths_in_return_order':[int(x.shape[-1]) for x in rows[0][:5]],'dialogue_ids':data.keys}
print(json.dumps(out))
