import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,torch,importlib.util,hashlib
from pathlib import Path
from transformers import RobertaModel,RobertaTokenizer
import pandas as pd,numpy as np
torch.set_num_threads(4)
b=Path('/data2/yb/multimodalERC/MELD/Model');spec=importlib.util.spec_from_file_location('ex',b/'extract_roberta_features.py');ex=importlib.util.module_from_spec(spec);spec.loader.exec_module(ex)
c=torch.load(b/'results/roberta_large_ft/best_roberta_text_only.pt',map_location='cpu',weights_only=False)
tok=RobertaTokenizer.from_pretrained('roberta-large',local_files_only=True);tok.add_special_tokens({'additional_special_tokens':[f'<s{i}>' for i in range(1,10)]})
m=RobertaModel.from_pretrained('roberta-large',local_files_only=True);m.resize_token_embeddings(c['tokenizer_len']);st={k[len('roberta.'):]:v for k,v in c['model'].items() if k.startswith('roberta.')};missing=m.load_state_dict(st,strict=True);m.eval();out={'checkpoint_args':c['args'],'checkpoint_pooling':c['pooling'],'strict_encoder_load':True,'tokenizer_truncation_side':tok.truncation_side,'samples':[],'length_counts':{}}
for s in ['train','dev','test']:
 df=pd.read_csv('/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw/'+s+'_sent_emo.csv');features=json.load(open(b/f'features_roberta_large_ft/{s}_features/text_features.json'));count=0;maxlen=0
 for did,dd in df.groupby('Dialogue_ID',sort=False):
  for uid in dd.Utterance_ID:
   txt=ex.build_meld_context_input(dd,uid);n=len(tok.tokenize(txt))+2;count+=n>511;maxlen=max(n,maxlen)
 out['length_counts'][s]={'over_511':count,'total':len(df),'max_length':maxlen}
 for ix in [0,1,5]:
  row=df.iloc[ix];txt=ex.build_meld_context_input(df[df.Dialogue_ID==row.Dialogue_ID],row.Utterance_ID);enc=tok([txt],padding=True,truncation=True,max_length=511,return_tensors='pt')
  with torch.no_grad(): z=m(**enc).last_hidden_state[0,enc.attention_mask.sum()-1].numpy()
  saved=np.asarray(features[f'dia{row.Dialogue_ID}_utt{row.Utterance_ID}']);out['samples'].append({'split':s,'row':ix,'max_abs_diff':float(np.max(np.abs(saved-z))),'cosine':float(np.dot(saved,z)/(np.linalg.norm(saved)*np.linalg.norm(z)))})
Path('results/pretraining_audit_20261009/meld_text_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
