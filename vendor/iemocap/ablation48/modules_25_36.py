from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


CANONICAL_EMOTIONS = ("happiness", "sadness", "neutral", "anger", "excited", "frustration")


class ForwardViewContext:
    """Per-model ephemeral cache; deepcopy preserves aliases but rebinds ownership."""
    def __init__(self): self.modalities=None


class ResidualSwiGLUIntegrator(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0):
        super().__init__()
        self.norm=nn.LayerNorm(768); self.value=nn.Linear(768,hidden_dim); self.gate=nn.Linear(768,hidden_dim)
        self.out=nn.Linear(hidden_dim,256); self.skip=nn.Linear(768,256); self.final_norm=nn.LayerNorm(256)
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        n=self.norm(x)
        return self.final_norm(self.skip(x)+self.drop(self.out(self.value(n)*F.silu(self.gate(n)))))


class GlobalPMA(nn.Module):
    def __init__(self, dropout=0):
        super().__init__(); self.query=nn.Parameter(torch.randn(1,1,256)*.02)
        self.attn=nn.MultiheadAttention(256,8,dropout=dropout,batch_first=True); self.last_attention=None
    def forward(self,_query,key,value):
        q=self.query.expand(key.shape[0],-1,-1)
        out,w=self.attn(q,key,value,need_weights=True,average_attn_weights=False)
        self.last_attention=w
        return out,w


class ResidualMLP256(nn.Module):
    def __init__(self, dropout=0):
        super().__init__(); self.norm=nn.LayerNorm(256); self.fc1=nn.Linear(256,512); self.fc2=nn.Linear(512,256); self.drop=nn.Dropout(dropout)
    def forward(self,x): return x+self.drop(self.fc2(F.gelu(self.fc1(self.norm(x)))))


class PassThroughPool(nn.Module):
    def forward(self,_query,key,value):
        weights=key.new_ones(key.shape[0],1,key.shape[1])/key.shape[1]
        return value,weights


class HierarchicalPoolEncoder(nn.Module):
    def __init__(self,dropout=0):
        super().__init__(); self.split=nn.Linear(256,512)
        layer=nn.TransformerEncoderLayer(128,8,512,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,1)
        self.modality_queries=nn.Parameter(torch.randn(3,1,128)*.02)
        self.modality_attn=nn.MultiheadAttention(128,8,dropout=dropout,batch_first=True)
        self.modality_out=nn.Linear(128,256)
        self.global_query=nn.Parameter(torch.randn(1,1,256)*.02)
        self.global_attn=nn.MultiheadAttention(256,8,dropout=dropout,batch_first=True)
        self.last_modality_tokens=None; self.last_global=None
    def forward(self,x):
        b=x.shape[0]; z=self.split(x).reshape(b,12,128); z=self.transformer(z).reshape(b,3,4,128)
        mods=[]
        for m in range(3):
            q=self.modality_queries[m:m+1].expand(b,-1,-1)
            mods.append(self.modality_attn(q,z[:,m],z[:,m],need_weights=False)[0][:,0])
        mods=self.modality_out(torch.stack(mods,1)); self.last_modality_tokens=mods
        q=self.global_query.expand(b,-1,-1); out=self.global_attn(q,mods,mods,need_weights=False)[0]
        self.last_global=out[:,0]
        return out


class RawEmotionTokenEncoder(nn.Module):
    def __init__(self,dropout=0):
        super().__init__(); self.split=nn.Linear(256,512)
        layer=nn.TransformerEncoderLayer(128,8,512,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,1); self.output_adapter=nn.Linear(128,256)
    def forward(self,x):
        z=self.split(x).reshape(x.shape[0],12,128)
        return self.output_adapter(self.transformer(z))


class EmotionQueryPool(nn.Module):
    emotion_names=CANONICAL_EMOTIONS
    def __init__(self,dropout=0):
        super().__init__(); self.queries=nn.Parameter(torch.randn(1,6,256)*.02)
        self.attn=nn.MultiheadAttention(256,8,dropout=dropout,batch_first=True); self.last_attention=None
    def forward(self,_query,key,value):
        q=self.queries.expand(key.shape[0],-1,-1)
        out,w=self.attn(q,key,value,need_weights=True,average_attn_weights=False); self.last_attention=w
        return out,w


class EmotionTokenIdentity(nn.Module):
    def forward(self,x): return x.reshape(x.shape[0],6,256)


class ClassQueryScalars(nn.Module):
    def __init__(self): super().__init__(); self.heads=nn.ModuleList([nn.Linear(256,1) for _ in range(6)])
    def forward(self,x): return torch.cat([head(x[:,i]) for i,head in enumerate(self.heads)],-1)


class SharedPrivateTokenizer(nn.Module):
    def __init__(self,n_shared,n_private):
        super().__init__(); assert n_shared+n_private==4
        self.n_shared=n_shared; self.n_private=n_private
        self.shared_projectors=nn.ModuleList([nn.Linear(256,128) for _ in range(n_shared)])
        self.private_projectors=nn.ModuleList([nn.Linear(256,128) for _ in range(3*n_private)])
        self.last_shared=None; self.last_private=None
    def forward(self,x):
        shared=torch.stack([torch.stack([proj(x[:,m]) for proj in self.shared_projectors],1) for m in range(3)],1)
        private=torch.stack([torch.stack([self.private_projectors[m*self.n_private+k](x[:,m]) for k in range(self.n_private)],1) for m in range(3)],1)
        self.last_shared=shared; self.last_private=private
        return torch.cat((shared,private),2)
    def decorrelation_loss(self):
        if self.last_shared is None: raise RuntimeError("forward must precede decorrelation loss")
        s=self.last_shared.mean(2).reshape(-1,128); p=self.last_private.mean(2).reshape(-1,128)
        s=s-s.mean(0,keepdim=True); p=p-p.mean(0,keepdim=True)
        cross=s.transpose(0,1)@p/max(1,s.shape[0]-1)
        # Squared Frobenius norm of the empirical shared/private cross-covariance.
        return cross.square().sum()


class SharedPrivateEncoder(nn.Module):
    def __init__(self,n_shared,n_private,dropout=0,use_decorrelation=False):
        super().__init__(); self.tokenizer=SharedPrivateTokenizer(n_shared,n_private); self.split=self.tokenizer.shared_projectors[0]
        layer=nn.TransformerEncoderLayer(128,8,512,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,1); self.output_adapter=nn.Linear(128,256)
        self.use_decorrelation=use_decorrelation; self.last_tokens=None
    def forward(self,x):
        z=self.tokenizer(x); enc=self.transformer(z.reshape(x.shape[0],12,128)).reshape(x.shape[0],3,4,128)
        self.last_tokens=enc
        return self.output_adapter(enc.mean(2))
    def experiment_losses(self):
        if not self.use_decorrelation: return {}
        raw=self.tokenizer.decorrelation_loss()
        return {"decorrelation_raw":raw,"decorrelation_weighted":raw*.01}


class ObservableSubspaceEncoder(nn.Module):
    def __init__(self,dropout=0):
        super().__init__(); self.split=nn.Linear(256,512)
        layer=nn.TransformerEncoderLayer(128,8,512,dropout,activation="gelu",batch_first=True,norm_first=True)
        self.transformer=nn.TransformerEncoder(layer,1); self.output_adapter=nn.Linear(128,256)
        self.last_tokens=None; self.view_context=ForwardViewContext()
    @property
    def last_output(self): return self.view_context.modalities
    @last_output.setter
    def last_output(self,value): self.view_context.modalities=value
    def forward(self,x):
        z=self.split(x).reshape(x.shape[0],12,128); self.last_tokens=self.transformer(z).reshape(x.shape[0],3,4,128)
        self.last_output=self.output_adapter(self.last_tokens.mean(2)); return self.last_output


class ViewHead(nn.Module):
    def __init__(self,encoder,view_name):
        super().__init__(); self.view_context=encoder.view_context; self.view_name=view_name
        self.head=nn.Linear(256,6); self.last_view=None
    def view(self,global_feature):
        modal=self.view_context.modalities
        return global_feature if self.view_name=="global" else modal[:,2] if self.view_name=="text" else modal[:,:2].mean(1)
    def forward(self,x): self.last_view=self.view(x); return self.head(self.last_view)


class RelationHead(nn.Module):
    INDEX={"TA":(2,1),"TV":(2,0),"AV":(1,0)}
    def __init__(self,encoder,relation_name):
        super().__init__(); self.view_context=encoder.view_context; self.relation_name=relation_name
        self.indices=self.INDEX[relation_name]
        self.builder=nn.Sequential(nn.Linear(1024,512),nn.GELU(),nn.Linear(512,256))
        self.head=nn.Linear(256,6); self.last_relation=None
    def forward(self,_global):
        modal=self.view_context.modalities; a,b=self.indices
        left,right=modal[:,a],modal[:,b]
        relation=torch.cat((left,right,left*right,(left-right).abs()),-1)
        self.last_relation=self.builder(relation)
        return self.head(self.last_relation)


class RoutedViewClassifier(nn.Module):
    view_names=("global","text","mean_av")
    def __init__(self,encoder):
        super().__init__(); self.view_context=encoder.view_context
        self.heads=nn.ModuleList([nn.Linear(256,6) for _ in range(3)]); self.router=nn.Linear(256,3); self.last_weights=None; self.last_views=None
    def forward(self,global_feature):
        modal=self.view_context.modalities
        views=(global_feature,modal[:,2],modal[:,:2].mean(1)); self.last_views=views
        self.last_weights=torch.softmax(self.router(global_feature),-1)
        logits=torch.stack([h(v) for h,v in zip(self.heads,views)],1)
        return (logits*self.last_weights[...,None]).sum(1)


class ResidualNonlinearClassifier(nn.Module):
    def __init__(self):
        super().__init__(); self.norm=nn.LayerNorm(256); self.fc1=nn.Linear(256,512); self.fc2=nn.Linear(512,256); self.out=nn.Linear(256,6)
    def forward(self,x): return self.out(x+self.fc2(F.gelu(self.fc1(self.norm(x)))))
