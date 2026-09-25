"""Utterance loader for the controlled 53-turn completion experiment."""
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class Fill53Dataset(Dataset):
    LABEL_NAMES=("happiness","sadness","neutral","anger","excited","frustration")
    def __init__(self,pkl_path,features_npz,split,is_training=False):
        o=pickle.load(open(pkl_path,"rb"),encoding="latin1")
        self.ids,self.labels=o[0],o[2]; self.dialogues=list(o[10] if split=="train" else o[11])
        self.z=np.load(features_npz,mmap_mode="r"); self.index=[(d,i) for d in self.dialogues for i in range(len(self.labels[d]))]
        self.is_training=is_training
    def __len__(self): return len(self.index)
    def __getitem__(self,n):
        d,i=self.index[n]; f={m:torch.tensor(self.z[f"{d}__{m}"][i].copy(),dtype=torch.float32) for m in ("t","a","v")}
        if self.is_training:
            t,a,v=f["t"],f["a"],f["v"]
            mask=np.random.rand(t.numel())>.8; t[torch.from_numpy(mask)]=0
            t += torch.tensor(np.random.normal(0,.1,t.shape),dtype=torch.float32)
            if np.random.rand()<.3:
                ix=np.random.choice(t.numel(),int(.1*t.numel()),replace=False); arr=t.numpy(); values=arr[ix].copy(); np.random.shuffle(values); arr[ix]=values
            if np.random.rand()<.4: t *= float(np.random.uniform(.8,1.2))
            a += torch.tensor(np.random.normal(0,.05,a.shape),dtype=torch.float32); v *= float(np.random.uniform(.95,1.05))
        return f,torch.tensor(self.labels[d][i],dtype=torch.long)
    @staticmethod
    def collate_fn(items):
        fs,ys=zip(*items); return {m:torch.stack([x[m] for x in fs]) for m in ("t","a","v")},torch.stack(ys)
