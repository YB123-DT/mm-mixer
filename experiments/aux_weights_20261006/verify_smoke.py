import json,hashlib
from pathlib import Path
ROOT=Path('/data2/yb/multimodalERC/MM_Mixer_AuxWeights_20261006')
items=[]
for dataset,variant in [('iemocap','aux_equal'),('meld','aux_double')]:
 p=ROOT/'smoke'/dataset/variant/'seed2025'/'best_peak'
 status=json.loads((p/'status.json').read_text())
 assert status['state']=='complete' and status['fresh_strict_replay_exact']
 cfg=json.loads((p/'config.json').read_text())
 items.append(dict(dataset=dataset,variant=variant,status=status,config=cfg,smoke_only=True,server='biggpu',gpu_uuid='GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e'))
(ROOT/'checks/smoke_verification.json').write_text(json.dumps(items,indent=2)+'\n')
print('Both smoke runs completed with fresh strict checkpoint replay.')
