import json,hashlib,importlib,sys,subprocess,os
from pathlib import Path
PROBE=r'''
import torch,sys,importlib,json,hashlib
from mm_mixer_final.config import get_config
from mm_mixer_final.audit import assert_true_mixer
torch.set_num_threads(1)
dataset,variant=sys.argv[1:]; r=importlib.import_module('dataset_runners.'+dataset)
torch.manual_seed(2025); full=r.build_variant_model('full',.2)
torch.manual_seed(2025); model=r.build_variant_model(variant,.2).eval(); cfg=get_config(dataset,variant,2025)
if variant=='no_mca_no_amm':
 assert cfg.switches['no_mixer'] and cfg.switches['no_cross_attention']
 assert not hasattr(model.transformer_encoder,'blocks')
 assert hasattr(model.transformer_encoder,'cross')
 assert all('IdentityAttention' in str(x) for x in model.cross_attn.values())
else:
 assert_true_mixer(model,cfg)
 for name,p in model.state_dict().items():
  if '.ffn.' not in name: assert torch.equal(p,full.state_dict()[name]),name
inputs={n:torch.randn(2,w) for n,w in [('v',342),('a',1024),('t',1024)]}
result=model(inputs); out=result[0] if isinstance(result,tuple) else result
assert out.shape==(2,len(cfg.class_names)) and torch.isfinite(out).all()
h=hashlib.sha256()
for n,p in model.state_dict().items():h.update(n.encode()); h.update(p.cpu().numpy().tobytes())
report={'dataset':dataset,'variant':variant,'parameters':sum(p.numel() for p in model.parameters()),'state_hash':h.hexdigest(),'output_hash':hashlib.sha256(out.detach().numpy().tobytes()).hexdigest(),'switches':cfg.switches,'mixer':cfg.mixer}
groups=r.optimizer_parameter_groups(model,3e-5) if dataset=='iemocap' else r.optimizer_parameter_groups(model,3e-5)
opt=torch.optim.AdamW(groups)
tracked=[p for n,p in model.named_parameters() if ('.ffn.' in n if variant!='no_mca_no_amm' else 'transformer_encoder.cross.' in n)]
assert tracked
ids={id(p) for g in groups for p in g['params']}; assert all(id(p) in ids for p in tracked)
before=[p.detach().clone() for p in tracked]; out.square().mean().backward()
active=[(a,p) for a,p in zip(before,tracked) if p.grad is not None and p.grad.abs().sum()>0]
assert active and all(torch.isfinite(p.grad).all() for a,p in active)
if variant!='no_mca_no_amm': assert len(active)==len(tracked)
opt.step(); assert all(not torch.equal(a,p) for a,p in active)
report['tracked_count']=len(tracked); report['active_updated_count']=len(active)
report['gradient_optimizer_update']='passed'; print(json.dumps(report))
'''
if __name__=='__main__':
 root=Path(sys.argv[1]); original=Path(sys.argv[2]); rows=[]
 for ds in ('iemocap','meld'):
  def probe(path,v):
   env=dict(os.environ,PYTHONPATH=str(path),CUDA_VISIBLE_DEVICES=''); out=subprocess.check_output([sys.executable,'-c',PROBE,ds,v],cwd=path,env=env,text=True);return json.loads(out.splitlines()[-1])
  a=probe(original,'full'); b=probe(root,'full'); assert all(a[k]==b[k] for k in ('parameters','state_hash','output_hash','mixer')), (a,b); b['original_bitexact']=True; rows.append(b)
  for v in ('no_mca_no_amm','ffn_width_256','ffn_width_512','ffn_width_768'): rows.append(probe(root,v))
 print(json.dumps(rows,indent=2))
