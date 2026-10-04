#!/usr/bin/env python3
"""Count saved graph baseline forwards on CPU; two FLOPs per matrix MAC.
CPU adapter maps hardcoded Tensor.cuda calls to CPU without changing math.
Elementwise/norm/nonlinear operations excluded. Hypergraph/highConv weighted aggregation MAC equivalents are supplementary
only: 2 FLOPs per edge-feature, excluded from primary matrix total.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse, hashlib, importlib, json, sys, traceback, builtins
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import torch
from torch.utils.flop_counter import FlopCounterMode

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''): h.update(b)
 return h.hexdigest()

def matrix_cost(a,b):
 if a.layout not in (torch.sparse_coo,torch.strided):raise ValueError('Unsupported matrix layout: '+str(a.layout))
 return 2*(a._nnz() if a.layout==torch.sparse_coo else a.numel())*b.shape[-1]

def mm_rule(a,b,**kw): return matrix_cost(a,b)
mm_rule._get_raw=True

def addmm_rule(base,a,b,**kw): return matrix_cost(a,b)
addmm_rule._get_raw=True

class Count(FlopCounterMode):
 def __init__(self):
  super().__init__(display=False,custom_mapping={torch.ops.aten.mm:mm_rule,torch.ops.aten.addmm:addmm_rule,torch.ops.aten._sparse_addmm:addmm_rule})
  self.ops=Counter()
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  self.ops[str(func._overloadpacket)]+=1
  return super().__torch_dispatch__(func,types,args,kwargs)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['mmgcn','mmdfn','m3net'],required=True);ap.add_argument('--dataset',choices=['iemocap','meld'],required=True);ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
 if args.output.exists():raise FileExistsError(args.output)
 torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
 # Sparse rule regression: three stored coefficients, four output channels.
 sparse=torch.sparse_coo_tensor(torch.tensor([[0,1,1],[0,1,2]]),torch.tensor([1.,2.,3.]),(2,3));dense=torch.randn(3,4)
 with Count() as toy:toy_out=torch.sparse.mm(sparse,dense)
 assert toy.get_total_flops()==24
 torch.testing.assert_close(toy_out,sparse.to_dense()@dense)
 with Count() as toy:torch.mm(torch.ones(2,3),torch.ones(3,4))
 assert toy.get_total_flops()==48
 torch.Tensor.cuda=lambda self,*a,**kw:self.cpu()
 torch.nn.Module.cuda=lambda self,*a,**kw:self.cpu()
 root=args.root; repo=root/({'mmgcn':'MMGCN','mmdfn':'MM-DFN/code','m3net':'M3NET'}[args.model]);sys.path.insert(0,str(repo));os.chdir(repo)
 opened_data=set(); original_open=builtins.open
 def track_open(file,*a,**kw):
  if isinstance(file,(str,Path)) and Path(file).suffix in ('.pkl','.pickle','.npy','.npz'):opened_data.add(str(Path(file).resolve()))
  return original_open(file,*a,**kw)
 builtins.open=track_open
 result={'model':args.model,'dataset':args.dataset,'status':'started','environment':{'torch':torch.__version__,'python':sys.version,'server':'biggpu','device':'cpu','cpu_threads':1,'forward_parity_seed':1234},'records':[],'counting_convention':'2 FLOPs per explicit matrix MAC; nonlinear/elementwise/scatter ops excluded. Sparse message MAC equivalents reported separately, NOT added to total_flops.','cpu_adapter':'Tensor.cuda/Module.cuda mapped to cpu; MKLDNN disabled to expose recurrent matmuls'}
 args.output.parent.mkdir(parents=True,exist_ok=True)
 try:
  cp=root/'results'/args.model/args.dataset/'model.pt';result['checkpoint']={'path':str(cp),'sha256':sha(cp)}
  result['source_sha256']={str(p):sha(p) for p in repo.glob('*.py')}
  saved=torch.load(cp,map_location='cpu')
  metrics=json.loads((cp.parent/'metrics.json').read_text());result['configuration']=metrics
  if args.model=='mmgcn':
   cfg=SimpleNamespace(**saved['args']);model=importlib.import_module('export_checkpoint').build_model(cfg);model.load_state_dict(saved['state_dict'],strict=True)
   dl=importlib.import_module('dataloader');dataset=dl.IEMOCAPDataset(train=False) if args.dataset=='iemocap' else dl.MELDDataset('MELD_features/MELD_features_raw1.pkl',train=False)
   loader=torch.utils.data.DataLoader(dataset,batch_size=32,collate_fn=dataset.collate_fn)
  else:
   model=saved['model'];train=importlib.import_module('run_train_erc' if args.model=='mmdfn' else 'train');fn=getattr(train,'get_'+args.dataset.upper()+'_loaders')
   opts={'batch_size':32,'num_workers':0}
   if args.model=='mmdfn':opts.update(data_path=metrics['command_args']['data_dir'],valid_rate=0.)
   else:opts['valid']=0.
   loader=fn(**opts)[2]
  model.cpu().eval();result['registered_parameters']=sum(p.numel() for p in model.parameters())
  for mod in model.modules():
   if hasattr(mod,'no_cuda'):mod.no_cuda=True
  aggregation=[0]
  for mod in model.modules():
   if mod.__class__.__name__ in ('HypergraphConv','highConv'):
    orig=mod.message
    def message(*a,_orig=orig,**kw):
     out=_orig(*a,**kw);aggregation[0]+=2*out.numel();return out
    mod.message=message
  total=valid=0;matrix_total=0;aggregation_total=0
  for idx,data in enumerate(loader):
   tensors=data[:-1];umask=tensors[-2];lengths=[int((u==1).nonzero()[-1,0])+1 for u in umask]
   if args.model=='m3net':
    t1,t2,t3,t4,v,a,q,u,y=tensors;inp=([t1,t2,t3,t4],q,u,lengths,a,v,int(saved['epoch']))
   else:
    t,v,a,q,u,y=tensors;inp=(t,q,u,lengths,a,v)+( (False,) if args.model=='mmdfn' else ())
   # Fixed RNG also covers any accidental stochastic graph construction in eval.
   torch.manual_seed(1234)
   with torch.no_grad():ref=model(*inp)[0].detach().clone()
   aggregation[0]=0;torch.manual_seed(1234)
   with torch.enable_grad(),torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False),Count() as counter:out=model(*inp)
   logits=out[0].detach();torch.testing.assert_close(logits,ref,rtol=1e-4,atol=1e-5)
   opaque=[k for k in counter.ops if any(s in k for s in ['mkldnn_rnn','_thnn_fused','_cudnn_rnn','_scaled_dot_product','native_multi_head_attention'])]
   if opaque:raise ValueError('Opaque operators: '+str(opaque))
   count=counter.get_total_flops();n=sum(lengths);total+=count;valid+=n;matrix_total+=counter.get_total_flops();aggregation_total+=aggregation[0]
   input_hash=hashlib.sha256()
   for tensor in tensors:input_hash.update(tensor.contiguous().numpy().tobytes())
   rec={'sample_ids':list(data[-1]),'input_tensor_sha256':input_hash.hexdigest(),'matrix_convolution_flops':counter.get_total_flops(),'batch':idx,'valid_utterances':n,'padded_positions':umask.numel(),'dialogue_lengths':lengths,'total_flops':count,'supplementary_sparse_message_mac_flops':aggregation[0],'counted_operators':{str(k):int(v) for k,v in counter.get_flop_counts()['Global'].items()},'observed_operator_calls':dict(counter.ops),'max_abs_logit_difference':float((ref-logits).abs().max())}
   result['records'].append(rec);print(args.model,args.dataset,idx,n,count,flush=True);del out
  result.update(status='completed',matrix_convolution_flops=matrix_total,supplementary_sparse_message_mac_flops=aggregation_total,total_flops=total,valid_utterances=valid,flops_per_utterance=total/valid)
 except Exception:
  result.update(status='blocked',error=traceback.format_exc());print(result['error'],flush=True)
 builtins.open=original_open
 result['feature_source_sha256']={p:sha(p) for p in opened_data}
 result['counter_script_sha256']=sha(Path(__file__))
 args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
