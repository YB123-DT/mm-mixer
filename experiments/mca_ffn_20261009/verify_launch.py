import json,os,datetime,subprocess
from pathlib import Path
root=Path('/data2/yb/multimodalERC/MM_Mixer_MCA_FFN_20261009');p=root/'pipeline';s=json.loads((p/'state.json').read_text());assert len(s['jobs'])==24
active=[v for v in s['jobs'].values() if v['state']=='running'];assert len(active)==2
rows=[]
for v in active:
 assert v['gpu_host_index']==1 and v['gpu_uuid']=='GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049';os.kill(v['pid'],0)
 configs=list(Path(v['run_root']).rglob('config.json'))
 text=Path(v['log']).read_text();epochs=[line for line in text.splitlines() if 'Epoch ' in line]
 rows.append({k:v[k] for k in ('id','pid','gpu_host_index','gpu_uuid','log','run_root','started_at','command') }|{'published_configs':[str(c) for c in configs],'configuration_note':'IEMOCAP publishes canonical config with final peak bundle; fixed formal command audited before launch','epoch_log_tail':epochs[-2:]})
report={'status':'running','verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'formal_jobs':24,'running':2,'queued':22,'smoke_runs_passed':8,'cpu_probes_passed':10,'tmux_session':'mca_ffn_24runs','queue_pid':s['queue_pid'],'queue_process_alive':True,'first_jobs':rows,'gpu_state':subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True).strip(),'estimated_completion':'Not estimated from loading/one-epoch smoke; formal epoch timing not yet sufficient.'}
os.kill(s['queue_pid'],0)
(p/'launch_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
