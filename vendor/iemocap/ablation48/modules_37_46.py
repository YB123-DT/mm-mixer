"""Isolated implementations for ablations 37--46."""
from __future__ import annotations

import math
from dataclasses import dataclass
import pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .e14 import SubspaceTokenEncoder, ThreeTokenWideEncoder


def modality_dropout_mask(batch, modalities, probability, device):
    keep=torch.rand(batch,modalities,device=device)>=probability
    empty=~keep.any(1)
    if empty.any():
        chosen=torch.randint(modalities,(int(empty.sum()),),device=device)
        keep[empty]=False; keep[empty,chosen]=True
    return keep


def token_dropout_mask(batch, modalities, tokens, probability, device):
    keep=torch.rand(batch,modalities,tokens,device=device)>=probability
    empty=~keep.any(2)
    if empty.any():
        b,m=empty.nonzero(as_tuple=True)
        chosen=torch.randint(tokens,(len(b),),device=device)
        keep[b,m,chosen]=True
    return keep


class _ModalityDropState:
    def __init__(self, modalities, probability):
        self.modalities=tuple(modalities); self.probability=probability; self.mask=None; self.clean={}
    def reset(self): self.mask=None; self.clean={}
    def apply(self, modality, value, training):
        self.clean[modality]=value
        if not training: return value
        if self.mask is None:
            self.mask=modality_dropout_mask(value.size(0),len(self.modalities),self.probability,value.device)
        return value*self.mask[:,self.modalities.index(modality)].unsqueeze(-1).to(value.dtype)


class DropSelector(nn.Module):
    def __init__(self, inner, modality, state): super().__init__(); self.inner=inner; self.modality=modality; self.state=state
    def forward(self,x): return self.state.apply(self.modality,self.inner(x),self.training)


class CleanAuxClassifier(nn.Module):
    def __init__(self,inner,modality,state): super().__init__(); self.inner=inner; self.modality=modality; self.state=state
    def forward(self,_x): return self.inner(self.state.clean[self.modality])


def install_modality_dropout(model, probability):
    state=_ModalityDropState(model.modalities,probability)
    model.register_forward_pre_hook(lambda *_: state.reset())
    for m in model.modalities:
        def selector_hook(_module,_inputs,output,modality=m):
            return state.apply(modality,output,_module.training)
        def aux_pre_hook(_module,_inputs,modality=m):
            return (state.clean[modality],)
        model.feature_selectors[m].register_forward_hook(selector_hook)
        model.mod_classifiers[m].register_forward_pre_hook(aux_pre_hook)
    object.__setattr__(model,"modality_dropout_state",state)


class TokenDropSubspaceEncoder(SubspaceTokenEncoder):
    def __init__(self,dropout,probability=.1): super().__init__(dropout); self.token_dropout_probability=probability; self.mixer=nn.Identity()
    def forward(self,x):
        b,m,_=x.shape
        z=self.split(x).reshape(b,m,4,128)
        if self.training:
            keep=token_dropout_mask(b,m,4,self.token_dropout_probability,x.device)
            z=z*keep.unsqueeze(-1).to(z.dtype)
        encoded=self.transformer(z.reshape(b,m*4,128)).reshape(b,m,4,128).mean(2)
        return self.output_adapter(encoded)


class IdentityMixerSubspaceEncoder(SubspaceTokenEncoder):
    """Experiment 46: expansion and unchanged aggregation, no token interaction."""
    def __init__(self,dropout): super().__init__(dropout); self.mixer=nn.Identity()
    def forward(self,x):
        b,m,_=x.shape
        z=self.mixer(self.split(x).reshape(b,m*4,128))
        return self.output_adapter(z.reshape(b,m,4,128).mean(2))


class QualityAttentionState:
    def __init__(self,model,lambda_quality=.1):
        self.model=model; self.modalities=tuple(model.modalities); self.lambda_quality=lambda_quality
        self.clean={}; self.last_quality=None
    def reset(self): self.clean={}; self.last_quality=None
    def quality(self):
        if self.last_quality is None:
            with torch.no_grad():
                logits=torch.stack([self.model.mod_classifiers[m](self.clean[m]) for m in self.modalities],1)
                p=logits.softmax(-1); entropy=-(p*(p.clamp_min(1e-8).log())).sum(-1)
                quality=-entropy; quality=quality-quality.mean(1,keepdim=True)
            self.last_quality=quality.detach()
        return self.last_quality


class CaptureSelector(nn.Module):
    def __init__(self,inner,modality,state): super().__init__(); self.inner=inner; self.modality=modality; self.state=state
    def forward(self,x):
        if self.modality==self.state.modalities[0]: self.state.reset()
        y=self.inner(x); self.state.clean[self.modality]=y; return y


class QualityBiasedAttention(nn.Module):
    def __init__(self,inner,q_mod,state): super().__init__(); self.inner=inner; self.q_mod=q_mod; self.state=state
    def forward(self,query,key,value,**kwargs):
        quality=self.state.quality(); key_mods=[m for m in self.state.modalities if m!=self.q_mod]
        bias=torch.stack([quality[:,self.state.modalities.index(m)] for m in key_mods],1)*self.state.lambda_quality
        b=query.size(0); heads=self.inner.num_heads; qlen=query.size(1)
        kwargs["attn_mask"]=bias[:,None,:].expand(b,qlen,len(key_mods)).repeat_interleave(heads,0)
        return self.inner(query,key,value,**kwargs)
    @property
    def num_heads(self): return self.inner.num_heads


def install_quality_bias(model,lambda_quality=.1,copy_safe=False):
    state=QualityAttentionState(model,lambda_quality)
    if copy_safe:
        for m in model.modalities:
            model.feature_selectors[m]=CaptureSelector(model.feature_selectors[m],m,state)
            model.cross_attn[m]=QualityBiasedAttention(model.cross_attn[m],m,state)
        object.__setattr__(model,"quality_attention_state",state)
        model.quality_bias_installation="copy_safe_wrappers"
        return
    model.register_forward_pre_hook(lambda *_: state.reset())
    for m in model.modalities:
        def capture(_module,_inputs,output,modality=m): state.clean[modality]=output
        model.feature_selectors[m].register_forward_hook(capture)
    for m in model.modalities:
        def add_bias(_module,args,kwargs,q_mod=m):
            quality=state.quality(); key_mods=[name for name in state.modalities if name!=q_mod]
            query=kwargs["query"]; b,qlen=query.shape[:2]; heads=_module.num_heads
            bias=torch.stack([quality[:,state.modalities.index(name)] for name in key_mods],1)*state.lambda_quality
            kwargs["attn_mask"]=bias[:,None,:].expand(b,qlen,len(key_mods)).repeat_interleave(heads,0)
            return args,kwargs
        model.cross_attn[m].register_forward_pre_hook(add_bias,with_kwargs=True)
    object.__setattr__(model,"quality_attention_state",state)


class SinusoidalPosition(nn.Module):
    def __init__(self,dim): super().__init__(); self.dim=dim
    def forward(self,x):
        pos=torch.arange(x.size(1),device=x.device,dtype=x.dtype)[:,None]
        scale=torch.exp(torch.arange(0,self.dim,2,device=x.device,dtype=x.dtype)*(-math.log(10000.)/self.dim))
        enc=torch.zeros(x.size(1),self.dim,device=x.device,dtype=x.dtype)
        enc[:,0::2]=torch.sin(pos*scale); enc[:,1::2]=torch.cos(pos*scale[:enc[:,1::2].shape[1]])
        return x+enc[None]


class DialogueTemporalEncoder(nn.Module):
    def __init__(self,dim=256,heads=8,ffn_dim=1024,dropout=.1,causal=True):
        super().__init__(); self.causal=causal; self.position=SinusoidalPosition(dim)
        layer=nn.TransformerEncoderLayer(dim,heads,ffn_dim,dropout,"gelu",batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,1); self.norm=nn.LayerNorm(dim); self.alpha=nn.Parameter(torch.zeros(()))
    def delta(self,x,valid):
        mask=torch.triu(torch.ones(x.size(1),x.size(1),dtype=torch.bool,device=x.device),1) if self.causal else None
        return self.norm(self.encoder(self.position(x),mask=mask,src_key_padding_mask=~valid.bool()))*valid.unsqueeze(-1).to(x.dtype)
    def forward(self,x,valid): return (x+self.alpha*self.delta(x,valid))*valid.unsqueeze(-1).to(x.dtype)


class DialogueTemporalWrapper(nn.Module):
    def __init__(self,base,modalities,causal=True,microbatch=32):
        super().__init__(); self.base=base; self.fusion_microbatch=microbatch; self.temporal_causal=causal
        self.temporal=nn.ModuleDict({m:DialogueTemporalEncoder(causal=causal) for m in modalities})
    def __getattr__(self,name):
        try: return super().__getattr__(name)
        except AttributeError: return getattr(self.base,name)
    @staticmethod
    def _inject(delta,alpha): return lambda _m,_i,output:output+alpha*delta
    def forward(self,features,return_attention=False,labels=None):
        if return_attention: raise NotImplementedError
        valid=features["graph_mask"].bool(); target=features.get("target_mask",valid).bool()
        flat={m:features[m][target] for m in ("t","a","v")}; replacements={}
        # Exact DC0: do not even execute the temporal kernels in evaluation.
        # Besides saving work, this prevents backend workspace/algorithm changes
        # from perturbing a later RawAux GEMM by a few ulps.
        zero_init_eval=(not self.training and all(module.alpha.detach().item()==0.0 for module in self.temporal.values()))
        if not zero_init_eval:
            for m,module in self.temporal.items(): replacements[m]=(module.delta(self.base.proj[m](features[m]),valid)[target],module.alpha)
        outputs=[]
        for start in range(0,len(flat["t"]),self.fusion_microbatch):
            stop=min(start+self.fusion_microbatch,len(flat["t"])); handles=[]
            try:
                for m,(delta,alpha) in replacements.items(): handles.append(self.base.proj[m].register_forward_hook(self._inject(delta[start:stop],alpha)))
                outputs.append(self.base({m:v[start:stop] for m,v in flat.items()},labels=None if labels is None else labels[start:stop]))
            finally:
                for h in handles: h.remove()
        return self._concat(outputs)
    @staticmethod
    def _concat(outputs):
        if len(outputs)==1:return outputs[0]
        result=[]
        for i in range(len(outputs[0])):
            values=[o[i] for o in outputs]
            result.append({k:torch.cat([v[k] for v in values]) for k in values[0]} if isinstance(values[0],dict) else torch.cat(values))
        return tuple(result)


class TurnChunkLoader:
    def __init__(self,dataset,chunk_size=32,shuffle=False): self.dataset=dataset; self.chunk_size=chunk_size; self.shuffle=shuffle
    def __len__(self): return math.ceil(self.dataset.turn_count/self.chunk_size)
    def __iter__(self):
        items=[self.dataset[i] for i in range(len(self.dataset))]
        order=np.random.permutation(len(items)).tolist() if self.shuffle else list(range(len(items)))
        refs=[(d,t) for d in order for t in range(len(items[d][1]))]
        for start in range(0,len(refs),self.chunk_size):
            chunk=refs[start:start+self.chunk_size]; ds=list(dict.fromkeys(d for d,_ in chunk)); local={d:i for i,d in enumerate(ds)}
            selected=[items[d] for d in ds]; features={m:pad_sequence([v[0][m] for v in selected],batch_first=True) for m in ("t","a","v")}
            lengths=torch.tensor([len(v[1]) for v in selected]); width=int(lengths.max())
            features["graph_mask"]=torch.arange(width)[None]<lengths[:,None]; features["target_mask"]=torch.zeros(len(ds),width,dtype=torch.bool)
            for d,t in chunk: features["target_mask"][local[d],t]=True
            yield features,torch.tensor([int(items[d][1][t]) for d,t in chunk])


def augment_fill53_utterance_inplace(features):
    """Bit-for-bit random-call sequence of Fill53Dataset.__getitem__."""
    t,a,v=features["t"],features["a"],features["v"]
    mask=np.random.rand(t.numel())>.8; t[torch.from_numpy(mask)]=0
    t += torch.tensor(np.random.normal(0,.1,t.shape),dtype=torch.float32)
    if np.random.rand()<.3:
        ix=np.random.choice(t.numel(),int(.1*t.numel()),replace=False)
        arr=t.numpy(); values=arr[ix].copy(); np.random.shuffle(values); arr[ix]=values
    if np.random.rand()<.4: t *= float(np.random.uniform(.8,1.2))
    a += torch.tensor(np.random.normal(0,.05,a.shape),dtype=torch.float32)
    v *= float(np.random.uniform(.95,1.05))
    return features


class DialogueFeatureDataset(Dataset):
    """Dialogue-preserving view of the pinned fill53 feature archive."""
    LABEL_NAMES=("happiness","sadness","neutral","anger","excited","frustration")
    def __init__(self,pkl_path,feature_path,split,is_training=False):
        with Path(pkl_path).open("rb") as stream: objects=pickle.load(stream,encoding="latin1")
        if split not in {"train","test"}: raise ValueError("split must be train or test")
        self.labels=objects[2]; self.dialogues=list(objects[10] if split=="train" else objects[11])
        self.features=np.load(feature_path,mmap_mode="r")
        self.turn_count=sum(len(self.labels[d]) for d in self.dialogues)
        self.is_training=bool(is_training)
    def __len__(self): return len(self.dialogues)
    def __getitem__(self,index):
        dialogue=self.dialogues[index]
        features={m:torch.from_numpy(np.asarray(self.features[f"{dialogue}__{m}"],dtype=np.float32).copy()) for m in ("t","a","v")}
        if self.is_training:
            for turn in range(len(features["t"])):
                augment_fill53_utterance_inplace({m:features[m][turn] for m in ("t","a","v")})
        labels=torch.tensor(self.labels[dialogue],dtype=torch.long)
        if any(len(features[m])!=len(labels) for m in features): raise ValueError(f"unaligned dialogue {dialogue}")
        return {"dialogue_id":dialogue,**features},labels


def build_dialogue_loaders(pkl_path,feature_path,chunk_size=32):
    """Explicit factory used by the formal runner; no process-global protocol."""
    train=DialogueFeatureDataset(pkl_path,feature_path,"train",is_training=True)
    test=DialogueFeatureDataset(pkl_path,feature_path,"test",is_training=False)
    if (len(train),len(test),train.turn_count,test.turn_count)!=(120,31,5810,1623):
        raise RuntimeError("pinned dialogue/turn counts changed")
    return TurnChunkLoader(train,chunk_size,True),TurnChunkLoader(test,chunk_size,False)
