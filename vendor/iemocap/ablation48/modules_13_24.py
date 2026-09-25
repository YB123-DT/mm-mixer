from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .modules_01_12 import LinearTokenizer


def attention_allow_matrix(topology: str, device=None) -> torch.Tensor:
    index = torch.arange(12, device=device)
    modality, slot = index.div(4, rounding_mode="floor"), index.remainder(4)
    if topology == "full":
        return torch.ones(12, 12, dtype=torch.bool, device=device)
    if topology == "cross_modal_only":
        # Cross-modal communication plus the token's own diagonal. Other slots
        # from the same modality are the only forbidden keys.
        return (modality[:, None] != modality[None, :]) | torch.eye(12, dtype=torch.bool, device=device)
    if topology == "same_slot_cross_modal":
        return slot[:, None] == slot[None, :]
    if topology == "within_modality":
        return modality[:, None] == modality[None, :]
    raise ValueError(f"unknown topology: {topology}")


class _AttentionFFNBlock(nn.Module):
    def __init__(self, dim=128, heads=8, ffn_dim=512, dropout=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim), nn.Dropout(dropout))

    def branches(self, x, attn_mask=None):
        a = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                      attn_mask=attn_mask, need_weights=False)[0]
        return a, self.ffn(self.norm2(x))


class ReZeroBlock(_AttentionFFNBlock):
    def __init__(self, dim=128, heads=8, ffn_dim=512, dropout=0):
        super().__init__(dim, heads, ffn_dim, dropout)
        self.alpha_attn = nn.Parameter(torch.zeros(())); self.alpha_ffn = nn.Parameter(torch.zeros(()))

    def forward(self, x, attn_mask=None):
        a = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                      attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.alpha_attn * a
        return x + self.alpha_ffn * self.ffn(self.norm2(x))


class LayerScaleBlock(_AttentionFFNBlock):
    def __init__(self, dim=128, heads=8, ffn_dim=512, dropout=0, eps=1e-5):
        super().__init__(dim, heads, ffn_dim, dropout)
        self.gamma_attn = nn.Parameter(torch.full((dim,), eps))
        self.gamma_ffn = nn.Parameter(torch.full((dim,), eps))

    def forward(self, x, attn_mask=None):
        a = self.attn(self.norm1(x), self.norm1(x), self.norm1(x),
                      attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.gamma_attn * a
        return x + self.gamma_ffn * self.ffn(self.norm2(x))


class ReAttentionBlock(nn.Module):
    def __init__(self, dim=128, heads=8, ffn_dim=512, dropout=0):
        super().__init__(); self.heads=heads; self.head_dim=dim//heads
        self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim)
        self.qkv=nn.Linear(dim,dim*3); self.out=nn.Linear(dim,dim)
        self.head_mix=nn.Parameter(torch.eye(heads)); self.drop=nn.Dropout(dropout)
        self.ffn=nn.Sequential(nn.Linear(dim,ffn_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ffn_dim,dim),nn.Dropout(dropout))
        self.last_attention=None

    def forward(self,x,attn_mask=None):
        b,n,d=x.shape; q,k,v=self.qkv(self.norm1(x)).chunk(3,-1)
        def heads(t): return t.reshape(b,n,self.heads,self.head_dim).transpose(1,2)
        q,k,v=heads(q),heads(k),heads(v)
        scores=q@k.transpose(-1,-2)/math.sqrt(self.head_dim)
        if attn_mask is not None: scores=scores.masked_fill(attn_mask[None,None],torch.finfo(scores.dtype).min)
        weights=torch.softmax(scores,-1)
        weights=torch.einsum("hg,bgij->bhij",self.head_mix,weights)
        weights=weights/(weights.sum(-1,keepdim=True)+1e-8)
        self.last_attention=weights
        context=(self.drop(weights)@v).transpose(1,2).reshape(b,n,d)
        x=x+self.out(context)
        return x+self.ffn(self.norm2(x))


class SwiGLUPreNormBlock(nn.Module):
    def __init__(self, dim=128, heads=8, ffn_dim=512, dropout=0):
        super().__init__(); self.norm1=nn.LayerNorm(dim); self.norm2=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.value=nn.Linear(dim,ffn_dim); self.gate=nn.Linear(dim,ffn_dim); self.out=nn.Linear(ffn_dim,dim)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,attn_mask=None):
        n=self.norm1(x); x=x+self.attn(n,n,n,attn_mask=attn_mask,need_weights=False)[0]
        n=self.norm2(x); return x+self.drop(self.out(self.value(n)*F.silu(self.gate(n))))


class TopologyBlock(_AttentionFFNBlock):
    def forward(self,x,attn_mask=None):
        a=self.attn(self.norm1(x),self.norm1(x),self.norm1(x),attn_mask=attn_mask,need_weights=False)[0]
        x=x+a; return x+self.ffn(self.norm2(x))


class TopologyEncoder(nn.Module):
    def __init__(self, topology: str, dropout=0):
        super().__init__(); self.topology=topology
        names=("same_slot_cross_modal","within_modality") if topology=="cross_then_within" else (topology,)
        self.blocks=nn.ModuleList([TopologyBlock(dropout=dropout) for _ in names])
        self.allow_matrices=tuple(attention_allow_matrix(name) for name in names)
    def forward(self,x):
        for block,allow in zip(self.blocks,self.allow_matrices): x=block(x,attn_mask=~allow.to(x.device))
        return x


class RelationTokenizer(nn.Module):
    relation_names=("TA","TV","AV")
    def __init__(self, modality_order=("v","a","t")):
        super().__init__()
        self.modality_order=tuple(modality_order)
        index={name:i for i,name in enumerate(self.modality_order)}
        if set(index)!={"v","a","t"}: raise ValueError("relation tokenizer requires exactly v/a/t")
        self.relation_indices=((index["t"],index["a"]),(index["t"],index["v"]),(index["a"],index["v"]))
        self.projectors=nn.ModuleList([nn.Linear(256,128) for _ in self.relation_indices])
    def forward(self,z):
        return torch.stack([proj(torch.cat((z[:,a],z[:,b]),-1)) for proj,(a,b) in zip(self.projectors,self.relation_indices)],1)


class MeanProjectAggregator(nn.Module):
    def __init__(self): super().__init__(); self.output_adapter=nn.Linear(128,256)
    def forward(self,z,coarse): return self.output_adapter(z.mean(2))


class WeightedSumAggregator(nn.Module):
    def __init__(self):
        super().__init__(); self.score=nn.Linear(128,1); self.output_adapter=nn.Linear(128,256); self.last_weights=None
    def forward(self,z,coarse):
        self.last_weights=torch.softmax(self.score(z).squeeze(-1),-1)
        return self.output_adapter((z*self.last_weights[...,None]).sum(2))


class PMAAggregator(nn.Module):
    def __init__(self):
        super().__init__(); self.queries=nn.Parameter(torch.randn(3,1,128)*.02)
        self.attn=nn.MultiheadAttention(128,8,batch_first=True); self.output_adapter=nn.Linear(128,256)
        self.last_attention=None
    def pooled(self,z):
        outputs=[]; weights=[]
        for m in range(3):
            q=self.queries[m:m+1].expand(z.shape[0],-1,-1)
            out,w=self.attn(q,z[:,m],z[:,m],need_weights=True,average_attn_weights=False)
            outputs.append(out[:,0]); weights.append(w.mean(1))
        self.last_attention=torch.stack(weights,1)
        return torch.stack(outputs,1)
    def forward(self,z,coarse): return self.output_adapter(self.pooled(z))


class ResidualConcatAggregator(nn.Module):
    def __init__(self): super().__init__(); self.delta=nn.Linear(512,256); self.alpha=nn.Parameter(torch.zeros(()))
    def forward(self,z,coarse): return coarse+self.alpha*self.delta(z.flatten(2))


class ResidualPMAAggregator(PMAAggregator):
    def __init__(self): super().__init__(); self.alpha=nn.Parameter(torch.zeros(()))
    def forward(self,z,coarse): return coarse+self.alpha*self.output_adapter(self.pooled(z))


class Family1324Encoder(nn.Module):
    def __init__(self, experiment_id: str, dropout=0, modality_order=("v","a","t")):
        super().__init__(); self.experiment_id=experiment_id; self.tokenizer=LinearTokenizer(); self.num_layers=1
        self.relation_tokenizer=None
        if experiment_id=="13": self.blocks=nn.ModuleList([ReZeroBlock(dropout=dropout) for _ in range(2)]); self.num_layers=2
        elif experiment_id=="14": self.blocks=nn.ModuleList([LayerScaleBlock(dropout=dropout) for _ in range(2)]); self.num_layers=2
        elif experiment_id=="15": self.blocks=nn.ModuleList([ReAttentionBlock(dropout=dropout) for _ in range(2)]); self.num_layers=2
        elif experiment_id=="16": self.blocks=nn.ModuleList([SwiGLUPreNormBlock(dropout=dropout) for _ in range(2)]); self.num_layers=2
        elif experiment_id in {"17","18","19"}:
            topology={"17":"cross_modal_only","18":"same_slot_cross_modal","19":"cross_then_within"}[experiment_id]
            self.topology_encoder=TopologyEncoder(topology,dropout)
        else:
            self.blocks=nn.ModuleList([TopologyBlock(dropout=dropout)])
        if experiment_id=="20": self.relation_tokenizer=RelationTokenizer(modality_order)
        if experiment_id=="21": self.aggregator=WeightedSumAggregator()
        elif experiment_id=="22": self.aggregator=PMAAggregator()
        elif experiment_id=="23": self.aggregator=ResidualConcatAggregator()
        elif experiment_id=="24": self.aggregator=ResidualPMAAggregator()
        else: self.aggregator=MeanProjectAggregator()
        self.split=self.tokenizer.split

    def forward(self,x):
        z=self.tokenizer(x)
        if self.relation_tokenizer is not None: z=self.relation_tokenizer(z)
        flat=z.reshape(z.shape[0],12,128)
        if hasattr(self,"topology_encoder"): encoded=self.topology_encoder(flat)
        else:
            encoded=flat
            for block in self.blocks: encoded=block(encoded)
        grouped=encoded.reshape(z.shape[0],3,4,128)
        # Relation groups are TA/TV/AV evidence views and remain three readout groups.
        return self.aggregator(grouped,x)
