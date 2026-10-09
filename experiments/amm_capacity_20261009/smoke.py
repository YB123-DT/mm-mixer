import os,subprocess,json,time,concurrent.futures
from pathlib import Path
ROOT=Path('/data2/yb/multimodalERC/MM_Mixer_AMM_CAPACITY_20261009')
PY='/data2/yb/reproduction_envs/s0/bin/python'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES='GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',TMPDIR='/data2/yb/tmp/amm_capacity')
def run(job):
 ds,v=job; out=ROOT/'smoke'/ds/v; out.mkdir(parents=True,exist_ok=True)
 cmd=[PY,str(ROOT/'code/dataset_runners'/f'{ds}.py'),'--variant',v,'--seed','2025','--epochs','1','--output-root',str(out)]
 if not (out/'best_peak/status.json').exists():
  with (out/'stdout.log').open('w') as f:subprocess.run(cmd,cwd=ROOT/'code',env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
 status_path=out/'best_peak/status.json'
 status=json.loads(status_path.read_text()); assert status['state']=='complete' and status['fresh_strict_replay_exact'],status
 return dict(dataset=ds,variant=v,status=status,smoke_only=True,bundle=str(status_path.parent.resolve()))
if __name__=='__main__':
 jobs=[(ds,v) for v in ('ffn_width_3072','ffn_width_6144') for ds in ('iemocap','meld')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(run,jobs))
 (ROOT/'pipeline/smoke_verification.json').write_text(json.dumps(rows,indent=2)+'\n')
 print('All 4 one-epoch real-data smoke runs and fresh strict replay passed')
