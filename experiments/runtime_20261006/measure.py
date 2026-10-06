#!/usr/bin/env python3
"""Direct timing of exactly 1000 valid utterances, never full-test extrapolation."""
import argparse
import builtins
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from adapters import load


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def select(raw,fmt,n):
    selected=[]; records=[];remaining=n
    for row in raw:
        length=len(row['labels']) if fmt=='ada2i' else len(row[0])
        count=min(remaining,length)
        if fmt=='ada2i':
            out={k:(v[:count] if k!='uid' else v) for k,v in row.items()}
            uid=row['uid']
        else:
            out=[]
            for x in row[:-1]:
                assert len(x)==length,(type(x),len(x),length)
                out.append(x[:count])
            out.append(row[-1]);uid=row[-1]
        selected.append(out)
        records.append(dict(id=str(uid),original_length=length,selected_length=count))
        remaining-=count
        if remaining==0:break
    assert remaining==0,(n,remaining)
    return selected,records


class Batch(list):pass


def move(x,device):
    if torch.is_tensor(x):return x.to(device)
    if isinstance(x,dict):return {k:(v if k=='length' else move(v,device)) for k,v in x.items()}
    if isinstance(x,tuple):return tuple(move(v,device) for v in x)
    if isinstance(x,list):return [move(v,device) for v in x]
    return x


def gpu_processes():
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True)
    return [r.strip() for r in rows.splitlines() if r.strip()]


def verify_output(name, data, out):
    def finite(value):
        if torch.is_tensor(value): assert bool(torch.isfinite(value).all()), 'Nonfinite output'
        elif isinstance(value,dict):
            for item in value.values(): finite(item)
        elif isinstance(value,(tuple,list)):
            for item in value: finite(item)
    finite(out)
    logits=out if name=='ecerc' else out[3] if name in ('sdt','css') else out[0]
    if name=='ada2i': labels=data['label_tensor']; n=int(labels.numel())
    else:
        u=data[-4] if name=='confilmer' else data[-3]
        y=data[-3] if name=='confilmer' else data[-2]
        labels=torch.cat([y[j,:n] for j,n in enumerate(data.lengths)])
        assert labels.numel()==int(u.sum())==sum(data.lengths)
        n=int(labels.numel())
        if name=='dialoguernn':
            assert logits.shape[:2]==(u.shape[1],u.shape[0])
            logits=torch.cat([logits[:length,j] for j,length in enumerate(data.lengths)])
        elif name in ('sdt','css'):
            assert logits.shape[:2]==u.shape
            logits=torch.cat([logits[j,:length] for j,length in enumerate(data.lengths)])
    assert logits.ndim==2 and logits.shape[0]==n,(name,logits.shape,n)
    assert logits.shape[1] in (6,7)
    assert int(labels.min())>=0 and int(labels.max())<logits.shape[1]
    return {'valid_predictions':n,'logit_shape':list(logits.shape),'all_tensor_outputs_finite':True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',choices=['iemocap','meld'],required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-smoke',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    device='cpu' if a.cpu_smoke or a.prepare_only else 'cuda'
    if device=='cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4','Must use audited host GPU 2 UUID only'
    torch.set_num_threads(1);random.seed(7);np.random.seed(7);torch.manual_seed(7)
    if device=='cpu':
        torch.Tensor.cuda=lambda self,*x,**kw:self.cpu()
        torch.nn.Module.cuda=lambda self,*x,**kw:self.cpu()
    opened=set();orig=builtins.open
    def tracking(f,*x,**kw):
        if isinstance(f,(str,Path)) and Path(f).suffix in ('.pkl','.pickle','.npy','.npz'):opened.add(str(Path(f).resolve()))
        return orig(f,*x,**kw)
    builtins.open=tracking
    result=dict(status='started',model=a.model,dataset=a.dataset,device=device,cpu_threads=1,python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),script_sha256={f.name:sha(f) for f in Path(__file__).parent.glob('*.py')})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try:
        model,raw,collate,provenance,forward=load(a.model,a.dataset,device)
        selected,records=select(raw,provenance['raw_format'],1000)
        targets=[]
        for row,record in zip(selected,records):
            labels=row['labels'] if provenance['raw_format']=='ada2i' else row[-3] if a.model=='confilmer' else row[-2]
            for index,label in enumerate(labels):targets.append([record['id'],index,int(label)])
        assert len(targets)==1000
        result['target_ids_labels_sha256']=hashlib.sha256(json.dumps(targets,separators=(',',':')).encode()).hexdigest()
        batches=[];bs=provenance['batch_size_dialogues'];batch_records=[]
        for offset in range(0,len(selected),bs):
            rows=selected[offset:offset+bs];data=collate(rows)
            lens=[x['selected_length'] for x in records[offset:offset+bs]]
            if provenance['raw_format']=='tuple':
                umask=data[-4] if a.model=='confilmer' else data[-3]
                assert [int(x.sum()) for x in umask]==lens
                data=Batch(move(data,device));data.lengths=lens
                padded=int(umask.numel())
            else:
                assert data['length'].tolist()==lens
                data=move(data,device);padded=int(data['tensor']['t'].shape[0]*data['tensor']['t'].shape[1])
            batches.append(data);batch_records.append(dict(lengths=lens,valid_utterances=sum(lens),padded_positions=padded))
        assert sum(sum(r['lengths']) for r in batch_records)==1000
        result.update(provenance=provenance,dialogues=records,batches=batch_records,valid_utterances=1000,registered_parameters=sum(p.numel() for p in model.parameters()),protocol='Original test dialogue order; concatenate whole dialogues then retain only final dialogue prefix needed for exactly 1000. Re-collate all fields/masks and construct graphs in original model. No features re-extracted; upstream contextual embeddings unchanged. GPU resident inputs; default eval forward incl. executed diagnostics; no feature extraction/H2D; 20 warmup batches cycling the selected workload, 10 timed complete passes; synchronize before/after each pass.')
        if a.prepare_only:
            result['status']='prepared_only'
        else:
            model.to(device).eval()
            with torch.inference_mode():
                if a.cpu_smoke:
                    # Verify genuine collate/forward on first full dialogue only, separate from1000 preparation.
                    data=collate(selected[:1]);lens=[records[0]['selected_length']]
                    if provenance['raw_format']=='tuple':data=Batch(data);data.lengths=lens
                    out=forward(data)
                    result['smoke_validation']=verify_output(a.model,data,out)
                    result['status']='cpu_smoke_passed';result['smoke_dialogue_utterances']=lens[0]
                else:
                    result['gpu_name']=torch.cuda.get_device_name();result['visible_gpu_uuid']=os.environ['CUDA_VISIBLE_DEVICES']
                    proc_before=gpu_processes();foreign=[r for r in proc_before if r.startswith(os.environ['CUDA_VISIBLE_DEVICES']) and int(r.split(',')[1])!=os.getpid()]
                    if foreign:raise RuntimeError('Competing GPU processes before timing: '+repr(foreign))
                    result['validation']=[verify_output(a.model,data,forward(data)) for data in batches]
                    assert sum(v['valid_predictions'] for v in result['validation'])==1000
                    for index in range(20):forward(batches[index%len(batches)])
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    times=[]
                    for _ in range(10):
                        torch.cuda.synchronize();start=time.perf_counter()
                        for data in batches:forward(data)
                        torch.cuda.synchronize();times.append(time.perf_counter()-start)
                    proc_after=gpu_processes();foreign=[r for r in proc_after if r.startswith(os.environ['CUDA_VISIBLE_DEVICES']) and int(r.split(',')[1])!=os.getpid()]
                    if foreign:raise RuntimeError('Competing GPU processes after timing: '+repr(foreign))
                    result.update(status='completed',seconds_per_1000_utterances=times,mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),std_seconds=statistics.stdev(times),warmup_batches=20,repeats=10,peak_allocated_bytes=torch.cuda.max_memory_allocated(),processes_before=proc_before,processes_after=proc_after)
        cp=Path(provenance['checkpoint']);result['checkpoint_sha256']=sha(cp) if cp.is_file() else None
        repo=Path(provenance['repo']);result['source_sha256']={str(f.relative_to(repo)):sha(f) for f in repo.rglob('*.py') if '.git' not in f.parts}
    except Exception:
        result.update(status='failed',error=traceback.format_exc());print(result['error'],flush=True)
    finally:
        builtins.open=orig
        result['feature_sha256']={f:sha(f) for f in opened}
        a.output.write_text(json.dumps(result,indent=2,default=str)+'\n')
        print(json.dumps({k:result[k] for k in ('status','model','dataset')}),flush=True)
    if result['status']=='failed':sys.exit(1)

if __name__=='__main__':main()
