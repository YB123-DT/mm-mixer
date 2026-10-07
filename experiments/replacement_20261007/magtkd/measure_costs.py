#!/usr/bin/env python3
"""Saved MAGTKD stage-two full-test matrix/convolution FLOPs; 2 per MAC.
CPU-only adapter keeps CUDA branches but maps Tensor.cuda to CPU. Elementwise,
normalization, nonlinear and scatter aggregation are excluded, as in prior tables.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse, ast, builtins, hashlib, importlib, json, random, sys, traceback
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def mm(a,b,**kw):return 2*(a._nnz() if a.layout==torch.sparse_coo else a.numel())*b.shape[-1]
mm._get_raw=True
def addmm(c,a,b,**kw):return mm(a,b)
addmm._get_raw=True
class Count(FlopCounterMode):
 def __init__(self):
  super().__init__(display=False,custom_mapping={torch.ops.aten.mm:mm,torch.ops.aten.addmm:addmm,torch.ops.aten._sparse_addmm:addmm});self.ops=Counter()
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  self.ops[str(func._overloadpacket)]+=1
  return super().__torch_dispatch__(func,types,args,kwargs)
def seed():
 torch.manual_seed(1234);np.random.seed(1234);random.seed(1234)
def load_original(path,excluded=()):
 tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module in excluded)]
 ns={'__name__':'_magtkd_original','__file__':str(path)};exec(compile(tree,str(path),'exec'),ns);return ns

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',choices=['iemocap','meld'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 run=a.root/'runs'/f'{a.dataset}_seed2025';res=json.loads((run/'result.json').read_text())
 assert res['status']=='completed' and not res['smoke_only'] and res['epochs_completed']==30 and res['checkpoint_predictions_verified']
 checkpoint=run/'test_peak.pt';assert sha(checkpoint)==res['checkpoint_sha256']
 torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
 torch.Tensor.cuda=lambda self,*args,**kwargs:self.cpu();torch.nn.Module.cuda=lambda self,*args,**kwargs:self.cpu()
 with Count() as c:torch.mm(torch.ones(2,3),torch.ones(3,4))
 assert c.get_total_flops()==48
 result={'model':'MAGTKD','dataset':a.dataset,'status':'started','stage':'stage2_only','seed':2025,'device':'cpu','batch_size_dialogues':16,'counting_convention':'2 FLOPs per explicit matrix/convolution MAC. Excludes stage1/feature extraction, elementwise/norm/nonlinear/scatter, backward and losses. Includes complete released stage2 forward.','records':[],'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),'training_result_sha256':sha(run/'result.json'),'torch':torch.__version__,'counter_script_sha256':sha(Path(__file__))}
 try:
  saved=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=saved['config'];folder=a.root/'code/MAGTKD'/a.dataset.upper();ns=load_original(folder/'model.py',('transformers',));dsns=load_original(folder/'dataset.py')
  for source in [folder/'model.py',folder/'dataset.py']:assert sha(source)==cfg['source_sha256'][str(source)]
  result['source_sha256']={str(source):sha(source) for source in [folder/'model.py',folder/'dataset.py']}
  model=ns['Transformer_Based_Model'](SimpleNamespace(**cfg['args']));model.load_state_dict(saved['model'],strict=True);model.cpu().eval()
  result['parameters']=sum(p.numel() for p in model.parameters());assert result['parameters']==res['parameters'];result['configuration']=cfg
  feature=Path(cfg['data']['test']['path']);assert sha(feature)==cfg['data']['test']['sha256'];result['feature']={'path':str(feature),'sha256':sha(feature)}
  cls=dsns['IEMOCAP_Dataset' if a.dataset=='iemocap' else 'MELD_MM_Dataset'];ds=cls(feature);loader=torch.utils.data.DataLoader(ds,batch_size=16,shuffle=False,num_workers=0,collate_fn=ds.collate_fn)
  total=valid=0;preds=[];labels=[]
  for idx,data in enumerate(loader):
   text,aud,vid,akd,vkd,q,u,y=data[:-1];lengths=[int((row==1).nonzero()[-1,0])+1 for row in u];inp=(text,akd,vid if a.dataset=='iemocap' else vkd,u,q,lengths)
   seed()
   with torch.no_grad():ref=tuple(x.detach().clone() for x in model(*inp))
   seed()
   with torch.enable_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False),Count() as c:out=model(*inp)
   for old,new in zip(ref,out):torch.testing.assert_close(old,new.detach(),rtol=1e-4,atol=1e-5)
   opaque=[k for k in c.ops if any(s in k for s in ['mkldnn_rnn','_thnn_fused','_cudnn_rnn','_scaled_dot_product','native_multi_head_attention']) or k in ('aten.gru','aten.lstm','aten.rnn_tanh','aten.rnn_relu')]
   if opaque:raise RuntimeError('Opaque operators '+str(opaque))
   n=sum(lengths);f=c.get_total_flops();total+=f;valid+=n
   preds.extend(ref[0][u.bool()].argmax(-1).tolist());labels.extend(y[u.bool()].tolist())
   rec={'batch':idx,'valid_utterances':n,'padded_positions':u.numel(),'lengths':lengths,'flops':f,'max_abs_forward_difference':max(float((o-x.detach()).abs().max()) for o,x in zip(ref,out)),'counted_operators':{str(k):int(v) for k,v in c.get_flop_counts()['Global'].items()},'observed_operator_calls':dict(c.ops)}
   result['records'].append(rec);print(a.dataset,idx,n,f,flush=True);del out
  assert valid==res['samples']==(1623 if a.dataset=='iemocap' else 2610)
  with np.load(run/'predictions.npz') as pr:
   assert np.array_equal(labels,pr['y_true']);matches=np.array_equal(preds,pr['y_pred'])
  result.update(status='completed',valid_utterances=valid,total_flops=total,mflops_per_utterance=total/valid/1e6,cpu_predictions_match_saved=bool(matches),cpu_accuracy=float(np.mean(np.array(preds)==np.array(labels))*100))
 except Exception:result.update(status='blocked',error=traceback.format_exc());print(result['error'],flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ['status','parameters','mflops_per_utterance']}),flush=True)
 if result['status']!='completed':sys.exit(1)
if __name__=='__main__':main()
