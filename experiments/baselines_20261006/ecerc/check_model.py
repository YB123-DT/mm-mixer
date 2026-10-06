"""Synthetic GPU forward/backward and future perturbation audit of unmodified ECERC."""
import argparse,importlib,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--dataset',choices=['iemocap','meld'],required=True);a=p.parse_args()
sys.path.insert(0,str(a.source/a.dataset.upper()));ECERC=importlib.import_module('model').ECERC
torch.manual_seed(2025);i=a.dataset=='iemocap';da=1582 if i else 300;classes=6 if i else 7
m=ECERC(argparse.Namespace(),d_t=1024,d_a=da,d_v=342,base_layer=1,input_size=1366+da,hidden_size=128,n_speakers=2 if i else 9,n_classes=classes,cuda_flag=True).cuda()
x=torch.randn(5,2,1366+da,device='cuda');s=torch.randn(5,2,1024,device='cuda');q=torch.zeros(5,2,2 if i else 9,device='cuda');q[:,:,0]=1;q[1::2,:,0]=0;q[1::2,:,1]=1;u=torch.ones(2,5,device='cuda')
m.train();z=m(x,s,q,u,[5,5]);loss=-z[:,0].mean();loss.backward();finite=all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
m.eval()
with torch.no_grad():
 before=m(x,s,q,u,[5,5]);x2=x.clone();s2=s.clone();x2[3:]+=5;s2[3:]-=5;after=m(x2,s2,q,u,[5,5])
 # Flat outputs are dialogue-major.
 delta=(before[[0,1,2,5,6,7]]-after[[0,1,2,5,6,7]]).abs().max().item()
print(json.dumps({'dataset':a.dataset,'finite_backward':finite,'output_shape':list(z.shape),'future_perturbation_max_past_delta':delta,'causal_at_1e-6':delta<1e-6,'peak_cuda_bytes':torch.cuda.max_memory_allocated()}))
