from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LinearTokenizer(nn.Module):
    def __init__(self):
        super().__init__(); self.split = nn.Linear(256, 512)
    def forward(self, x):
        return self.split(x).reshape(x.shape[0], 3, 4, 128)


class IndependentMLPTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = nn.ModuleList([nn.Sequential(nn.Linear(256,256), nn.GELU(), nn.Linear(256,128)) for _ in range(12)])
    def forward(self, x):
        return torch.stack([self.slots[m*4+k](x[:,m]) for m in range(3) for k in range(4)], 1).reshape(x.shape[0],3,4,128)


class SharedProjectorTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList([nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,256)) for _ in range(3)])
        self.projector = nn.Linear(256,512)
        for adapter in self.adapters:
            nn.init.zeros_(adapter[-1].weight); nn.init.zeros_(adapter[-1].bias)
    def forward(self, x):
        adapted = torch.stack([x[:,m] + self.adapters[m](x[:,m]) for m in range(3)], 1)
        return self.projector(adapted).reshape(x.shape[0],3,4,128)


class LatentQueryTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.modality_ids = nn.Parameter(torch.randn(3,128)*.02)
        self.slot_ids = nn.Parameter(torch.randn(4,128)*.02)
        self.keys = nn.Linear(256,128); self.values = nn.Linear(256,128)
        self.last_attention = None
    def forward(self, x):
        b=x.shape[0]; q=(self.modality_ids[:,None]+self.slot_ids[None]).reshape(1,12,128).expand(b,-1,-1)
        weights=torch.softmax(torch.matmul(q,self.keys(x).transpose(-1,-2))/128**.5,-1)
        self.last_attention=weights
        return torch.matmul(weights,self.values(x)).reshape(b,3,4,128)


class CompetitiveTokenizer(nn.Module):
    def __init__(self):
        super().__init__(); self.base=LinearTokenizer(); self.score=nn.Linear(128,1); self.last_weights=None
    def forward(self,x):
        z=self.base(x); self.last_weights=torch.softmax(self.score(z).squeeze(-1),-1)*4
        return z*self.last_weights.unsqueeze(-1)


class SoftImportanceMixer(nn.Module):
    def __init__(self): super().__init__(); self.score=nn.Linear(128,1); self.last_weights=None
    def forward(self,z,**_):
        self.last_weights=torch.softmax(self.score(z).squeeze(-1),-1)*4
        return z*self.last_weights.unsqueeze(-1)


class IdentityMixer(nn.Module):
    def forward(self, z, **_): return z


class GumbelTop2Mixer(nn.Module):
    def __init__(self): super().__init__(); self.score=nn.Linear(128,1); self.last_hard_weights=None
    @staticmethod
    def temperature(epoch): return 1.-.5*min(max(epoch-1,0),29)/29
    def forward(self,z,epoch=1,**_):
        logits=self.score(z).squeeze(-1)
        if self.training:
            # One shared Gumbel draw owns both the soft surrogate and hard top-2.
            u=torch.rand_like(logits).clamp_(1e-8,1-1e-8)
            sampled=(logits-torch.log(-torch.log(u)))/self.temperature(epoch)
            soft=torch.softmax(sampled,-1); idx=sampled.topk(2,-1).indices
        else:
            soft=torch.softmax(logits/self.temperature(epoch),-1); idx=logits.topk(2,-1).indices
        hard=torch.zeros_like(logits).scatter_(-1,idx,1.)
        self.last_hard_weights=hard
        weights=hard-soft.detach()+soft if self.training else hard
        return z*weights.unsqueeze(-1)


class TokenFusionMixer(nn.Module):
    def __init__(self): super().__init__(); self.score=nn.Linear(128,1); self.last_replaced_slots=None; self.last_donors=None
    def forward(self,z,**_):
        scores=self.score(z).squeeze(-1); low=scores.argmin(-1); out=z.clone(); donors=torch.zeros_like(z)
        for m in range(3):
            others=[j for j in range(3) if j!=m]
            donor_weights=torch.softmax(scores[:,others],1)
            donor=(z[:,others]*donor_weights[...,None]).sum(1)
            donors[:,m]=donor
            out[torch.arange(z.shape[0]),m,low[:,m]]=donor[torch.arange(z.shape[0]),low[:,m]]
        self.last_replaced_slots=low; self.last_donors=donors
        return out


class BottleneckMixer(nn.Module):
    def __init__(self):
        super().__init__(); self.is_complete_mixer=True; self.cross_modal_enabled=True; self.within=nn.MultiheadAttention(128,8,batch_first=True); self.bottlenecks=nn.Parameter(torch.randn(4,128)*.02)
        self.b_read=nn.MultiheadAttention(128,8,batch_first=True); self.s_read=nn.MultiheadAttention(128,8,batch_first=True)
    def forward(self,z,**_):
        b=z.shape[0]; local=torch.stack([self.within(z[:,m],z[:,m],z[:,m],need_weights=False)[0] for m in range(3)],1)
        if not self.cross_modal_enabled:
            return local
        flat=local.reshape(b,12,128); bott=self.bottlenecks[None].expand(b,-1,-1)
        bott=self.b_read(bott,flat,flat,need_weights=False)[0]
        return self.s_read(flat,bott,bott,need_weights=False)[0].reshape(b,3,4,128)


class IdentityInfo(nn.Module):
    def forward(self,x,context): return x


class PostCrossFiLM(nn.Module):
    def __init__(self):
        super().__init__(); self.affine=nn.Linear(512,512); nn.init.zeros_(self.affine.weight); nn.init.zeros_(self.affine.bias)
    def forward(self,x,context):
        c=context[:,None].expand(-1,x.shape[1],-1); gamma,beta=self.affine(torch.cat([x,c],-1)).chunk(2,-1)
        return (1+gamma)*x+beta


class PostCrossMAG(nn.Module):
    def __init__(self,max_scale=.1):
        super().__init__(); self.max_scale=max_scale; self.delta=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,256)); nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
    def forward(self,x,context):
        c=context[:,None].expand(-1,x.shape[1],-1); return x+self.max_scale*torch.tanh(self.delta(torch.cat([x,c],-1)))


class OriginalGateBlock(nn.Module):
    def __init__(self,gates,stage): super().__init__(); self.gates=gates; self.stage=stage; self.modalities=tuple(gates.keys()); self.original_gate_calls=0
    def forward(self,x,context):
        self.original_gate_calls=0; outputs=[]
        for i,m in enumerate(self.modalities): outputs.append(self.gates[m](x[:,i],context)); self.original_gate_calls+=1
        return torch.stack(outputs,1)


class FamilyEncoder(nn.Module):
    def __init__(self, tokenizer, mixer, dropout=.1, regulation=None, regulation_stage=None, context_provider=None):
        super().__init__(); self.tokenizer=tokenizer; self.mixer=mixer; self.regulation=regulation; self.regulation_stage=regulation_stage; self.context_provider=context_provider
        self.transformer_is_identity=bool(getattr(mixer,"is_complete_mixer",False))
        if self.transformer_is_identity:
            self.transformer=nn.Identity()
        else:
            layer=nn.TransformerEncoderLayer(128,8,512,dropout,activation="gelu",batch_first=True,norm_first=True)
            self.transformer=nn.TransformerEncoder(layer,1)
        self.output_adapter=nn.Linear(128,256); self.epoch=1
        # compatibility capture aliases
        self.split=getattr(tokenizer,"split",tokenizer)
    def forward(self,x):
        context=self.context_provider() if self.context_provider else None
        if self.regulation is not None and self.regulation_stage=="post_cross": x=self.regulation(x,context)
        z=self.tokenizer(x); z=self.mixer(z,epoch=self.epoch); enc=self.transformer(z.reshape(z.shape[0],12,128))
        h=self.output_adapter(enc.reshape(z.shape[0],3,4,128).mean(2))
        if self.regulation is not None and self.regulation_stage=="post_aggregation": h=self.regulation(h,context)
        return h
