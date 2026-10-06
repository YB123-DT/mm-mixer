import sys,importlib.util,importlib,json,tempfile
from pathlib import Path
import torch
ROOT=Path('/data2/yb/multimodalERC/MM_Mixer_AuxWeights_20261006/code')
sys.path.insert(0,str(ROOT))
from mm_mixer_final.aux_weights import weights
from mm_mixer_final.config import get_config
dataset=sys.argv[1]
runner=importlib.import_module('dataset_runners.'+dataset)
path='vendor/iemocap/base/multiattn.py' if dataset=='iemocap' else 'vendor/meld/multiattn.py'
def load(root):
 spec=importlib.util.spec_from_file_location('checked_loss',root/path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.MultitaskFusionLoss
cls=load(ROOT);oldcls=load(ROOT.parent/'original')
torch.manual_seed(2025)
x=torch.randn(4,6,requires_grad=True);aux={m:torch.randn(4,6,requires_grad=True) for m in ('t','a','v')};y=torch.tensor([0,1,2,3])
extras={'use_uncertainty':False} if dataset=='iemocap' else {}
def criterion(w,old=False):return (oldcls if old else cls)(main_weight=w['main'],aux_weights={m:w[m] for m in aux},normalize_aux_weights=False,**extras)
basew=weights(dataset,'full');base=criterion(basew)(x,aux,y);old=criterion(basew,True)(x,aux,y);assert torch.equal(base,old)
main=criterion(basew)(x,{},y)
results=[]
for variant in (('aux_half','aux_double','aux_equal') if dataset=='iemocap' else ('aux_half','aux_double')):
 cfg=get_config(dataset,variant,2025)
 if dataset=='iemocap':
  fixed=runner.apply_loss_ablation(runner._formal_module()._training_cfg(runner.materialize_legacy_config(cfg,1,variant)),variant)
 else:
  fixed=runner.materialize_config(ROOT.parent/'smoke',1,2025,variant)['fixed_params']
 w=weights(dataset,variant)
 assert fixed['main_loss_weight']==w['main'];assert fixed['aux_loss_weights']=={m:w[m] for m in aux};assert fixed['normalize_aux_loss_weights'] is False
 loss=criterion(w)(x,aux,y)
 expected=main.clone()
 for m in aux:
  single=criterion(basew)(x,{m:aux[m]},y)-main
  expected=expected+single*(w[m]/basew[m])
 torch.testing.assert_close(loss,expected)
 grads=torch.autograd.grad(loss,[x,*aux.values()],retain_graph=True)
 assert all(g.isfinite().all() and g.abs().sum()>0 for g in grads)
 results.append({'variant':variant,'weights':w,'loss':loss.item(),'expected':expected.item(),'gradients':'finite_nonzero','effective_configuration':'verified'})
(ROOT.parent/'checks'/('loss_'+dataset+'.json')).write_text(json.dumps({'original_full_loss_bitwise_equal':True,'checks':results},indent=2)+'\n')
