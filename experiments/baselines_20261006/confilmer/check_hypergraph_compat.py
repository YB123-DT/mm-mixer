"""Run from patched source cwd: tests scatter and reversed-edge compatibility."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from HypergraphConv import HypergraphConv
x=torch.randn(5,4,requires_grad=True)
ei=torch.tensor([[0,1,2,2,3,4],[0,0,0,1,1,1]])
m=HypergraphConv(4,4)
y=m(x,ei)
h=torch.zeros(5,2);h[ei[0],ei[1]]=1
ref=torch.nn.functional.leaky_relu((h/h.sum(1,keepdim=True))@(h.T/h.sum(0).unsqueeze(1))@x+m.bias)
assert torch.allclose(y,ref,atol=1e-6), (y-ref).abs().max()
y.sum().backward();assert x.grad.isfinite().all()
print('dense hypergraph parity passed; max error', (y-ref).abs().max().item())
