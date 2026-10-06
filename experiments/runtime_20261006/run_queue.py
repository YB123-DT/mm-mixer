#!/usr/bin/env python3
"""Persistent sequential queue; never uses GPU4 or shared occupied GPU."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data2/yb/multimodalERC/MM_Mixer_Runtime_20261006')
UUID='GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4'
PYTHON='/data2/yb/reproduction_envs/s0/bin/python'
JOBS=[(m,d) for m in ('dialoguernn','ada2i','mmgcn','mmdfn','m3net','sdt','css','ecerc','confilmer') for d in ('iemocap','meld')]


def gpu_state():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    row=next(r for r in lines if UUID in r)
    idx,uid,util,mem=[v.strip() for v in row.split(',')]
    assert idx=='2' and uid==UUID
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).splitlines()
    busy=[r for r in processes if r.startswith(UUID)]
    return dict(index=2,uuid=UUID,utilization=int(util),memory_used_mib=int(mem),processes=busy,idle=not busy and int(util)<=2 and int(mem)<500)


def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
    lock=(ROOT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'status':'waiting_for_idle_gpu','gpu_uuid':UUID,'server':'biggpu','pid':os.getpid(),'started_at':time.time(),'jobs':{f'{m}_{d}':{'status':'pending'} for m,d in JOBS}}
    def save():
        state['updated_at']=time.time();tmp=ROOT/'state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(ROOT/'state.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR='/data2/yb/tmp/mmrt')
    for model,dataset in JOBS:
        key=f'{model}_{dataset}';out=ROOT/'results'/f'{key}.json'
        if out.exists():
            saved=json.loads(out.read_text());state['jobs'][key]={'status':saved['status'],'output':str(out),'reused_existing_artifact':True};save();continue
        smoke=ROOT/'smoke'/f'{key}.json'
        while not smoke.exists():
            state['status']='waiting_for_smoke';state['next_job']=key;save();time.sleep(30)
        check=json.loads(smoke.read_text())
        if check.get('status')!='cpu_smoke_passed':
            state['jobs'][key]={'status':'blocked_by_smoke','output':str(smoke)};save();continue
        stable=0
        while stable<3:
            try:
                gpu=gpu_state();state['gpu_last_observation']=gpu;stable=stable+1 if gpu['idle'] else 0
                state['status']='waiting_for_idle_gpu';state['next_job']=key;save()
            except Exception as error:
                state['last_poll_error']=str(error);save();stable=0
            if stable<3:time.sleep(30)
        state['status']='running';state['jobs'][key]={'status':'running','started_at':time.time()};save()
        cmd=[PYTHON,str(ROOT/'code/measure.py'),'--model',model,'--dataset',dataset,'--output',str(out)]
        with (ROOT/'logs'/f'{key}.log').open('w') as log:
            proc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=json.loads(out.read_text()) if out.exists() else {'status':'failed','error':'No expected JSON'}
        state['jobs'][key].update(status=result['status'],exit_code=proc.returncode,finished_at=time.time(),output=str(out),command=cmd)
        save()
        # Failed jobs remain failed, never silently retried; independent models continue.
    state['status']='completed' if all(v['status']=='completed' for v in state['jobs'].values()) else 'finished_with_failures';state['completed_at']=time.time();save()

if __name__=='__main__':main()
