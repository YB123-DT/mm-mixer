#!/usr/bin/env python3
"""Saved ECERC/ConFilMER full-test matrix/convolution FLOPs; 2 per MAC.
CPU-only adapter keeps CUDA branches but maps Tensor.cuda to CPU. Elementwise,
normalization, nonlinear and scatter aggregation are excluded, as in prior tables.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse, builtins, hashlib, importlib, json, random, sys, traceback
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
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--model',choices=['ecerc','confilmer'],required=True);p.add_argument('--dataset',choices=['iemocap','meld'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 batch=16 if a.model=='confilmer' else 32
 torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
 torch.Tensor.cuda=lambda self,*args,**kwargs:self.cpu()
 torch.nn.Module.cuda=lambda self,*args,**kwargs:self.cpu()
 with Count() as c:torch.mm(torch.ones(2,3),torch.ones(3,4))
 assert c.get_total_flops()==48
 opened=set();orig_open=builtins.open
 def tracking(f,*args,**kwargs):
  if isinstance(f,(str,Path)) and Path(f).suffix in ['.pkl','.pickle']:opened.add(str(Path(f).resolve()))
  return orig_open(f,*args,**kwargs)
 builtins.open=tracking
 result={'status':'started','model':a.model,'dataset':a.dataset,'device':'cpu','torch':torch.__version__,'batch_size_dialogues':batch,'seed':2025,'counting_convention':__doc__,'records':[]}
 try:
  plan=json.loads((a.root/'plans/recovery_12_runs.json').read_text());job=next(j for j in plan['jobs'] if j['id']==f'{a.model}_{a.dataset}_seed2025');run=Path(job['output'])
  cp=run/('test_peak.pt' if a.model=='ecerc' else 'best.pt');saved=torch.load(cp,map_location='cpu',weights_only=False);cfg=SimpleNamespace(**saved['config']);iemo=a.dataset=='iemocap'
  result['checkpoint']={'path':str(cp),'sha256':sha(cp),'epoch':saved['epoch']};result['configuration']=saved['config']
  repo=a.root/('code/ECERC/'+a.dataset.upper() if a.model=='ecerc' else 'source/confilmer');os.chdir(repo);sys.path.insert(0,str(repo))
  train=importlib.import_module('train' if a.model=='ecerc' else 'train_our')
  if a.model=='ecerc':
   opts=SimpleNamespace(feature_type='multi',cls_type='emotion')
   model=train.ECERC(opts,d_t=1024,d_a=1582 if iemo else 300,d_v=342,base_layer=1,input_size=1024+(1582 if iemo else 300)+342,hidden_size=128,n_speakers=2 if iemo else 9,n_classes=6 if iemo else 7,cuda_flag=True)
   loader=train.get_IEMOCAP_bert_loaders(batch_size=32,valid_rate=.1)[2] if iemo else train.get_MELD_bert_loaders(None,batch_size=32)[2]
  else:
   model=train.Model(cfg.base_model,1024,512 if iemo else 1024,150,100,100,100,512,n_speakers=2 if iemo else 9,max_seq_len=200,window_past=cfg.windowp,window_future=cfg.windowf,n_classes=6 if iemo else 7,listener_state=cfg.active_listener,context_attention=cfg.attention,dropout=cfg.dropout,nodal_attention=cfg.nodal_attention,no_cuda=cfg.no_cuda,graph_type=cfg.graph_type,use_topic=cfg.use_topic,alpha=cfg.alpha,multiheads=cfg.multiheads,graph_construct=cfg.graph_construct,use_GCN=cfg.use_gcn,use_residue=cfg.use_residue,D_m_v=342,D_m_a=1582 if iemo else 300,modals=cfg.modals,att_type=cfg.mm_fusion_mthd,av_using_lstm=cfg.av_using_lstm,Deep_GCN_nlayers=cfg.Deep_GCN_nlayers,dataset=cfg.Dataset,use_speaker=cfg.use_speaker,use_modal=cfg.use_modal,norm=cfg.norm,num_L=cfg.num_L,num_K=cfg.num_K)
   loader=getattr(train,'get_'+a.dataset.upper()+'_loaders')(batch_size=batch,valid=0.,num_workers=0)[2]
  model.load_state_dict(saved['model'],strict=True);model.cpu().eval();result['registered_parameters']=sum(p.numel() for p in model.parameters());result['source_sha256']={str(f):sha(f) for f in repo.glob('*.py')}
  total=valid=0
  for idx,data in enumerate(loader):
   if a.model=='ecerc':
    e,s,aud,v,q,u,y=data[:-1];lengths=[int((row==1).nonzero()[-1,0])+1 for row in u];inp=(torch.cat([e,aud,v],dim=-1),s,q,u,lengths)
   else:
    t1,t2,t3,t4,v,aud,q,u,y,sentence=data[:-1];lengths=[int((row==1).nonzero()[-1,0])+1 for row in u];inp=([t1,t2,t3,t4],q,u,lengths,sentence,torch.nn.Identity(),aud,v,saved['epoch']-1)
   seed()
   with torch.no_grad():
    ref=model(*inp);ref=(ref if a.model=='ecerc' else ref[0]).detach().clone()
   seed()
   with torch.enable_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False),Count() as c:out=model(*inp)
   pred=(out if a.model=='ecerc' else out[0]).detach();torch.testing.assert_close(ref,pred,rtol=1e-4,atol=1e-5)
   opaque=[k for k in c.ops if any(s in k for s in ['mkldnn_rnn','_thnn_fused','_cudnn_rnn','_scaled_dot_product','native_multi_head_attention']) or k in ('aten.gru','aten.lstm','aten.rnn_tanh','aten.rnn_relu')]
   if opaque:raise RuntimeError('Opaque operators: '+str(opaque))
   n=sum(lengths);flops=c.get_total_flops();total+=flops;valid+=n
   result['records'].append({'batch':idx,'valid_utterances':n,'padded_positions':u.numel(),'lengths':lengths,'flops':flops,'max_abs_logit_difference':float((ref-pred).abs().max()),'counted_operators':{str(k):int(v) for k,v in c.get_flop_counts()['Global'].items()},'observed_operator_calls':dict(c.ops)})
   print(a.model,a.dataset,idx,n,flops,flush=True);del out
  assert valid==(1623 if iemo else 2610),valid
  result.update(status='completed',total_flops=total,valid_utterances=valid,flops_per_utterance=total/valid,mflops_per_utterance=total/valid/1e6)
 except Exception:
  result.update(status='blocked',error=traceback.format_exc());print(result['error'],flush=True)
 builtins.open=orig_open
 result['feature_source_sha256']={f:sha(f) for f in opened};result['counter_script_sha256']=sha(Path(__file__))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k in ['status','registered_parameters','mflops_per_utterance']}),flush=True)
 if result['status']!='completed':sys.exit(1)
if __name__=='__main__':main()
