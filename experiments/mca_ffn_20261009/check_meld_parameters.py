import sys,json,torch
from pathlib import Path
root=Path('/data2/yb/multimodalERC/MM_Mixer_MCA_FFN_20261009');sys.path.insert(0,str(root/'code'))
import dataset_runners.meld as r
p=Path('/data2/yb/multimodalERC/MM_Mixer_Revision_20261003/runs/revision_1b8b1ff/meld/full/seed2025/best_peak/best_peak_test_state_dict.pt')
a=r.build_variant_model('full',.2).state_dict(); b=torch.load(p,map_location='cpu',weights_only=True)
if 'model_state_dict' in b:b=b['model_state_dict']
print(type(b),list(b)[:3])
report={'helper':sum(x.numel() for x in a.values()),'checkpoint':sum(x.numel() for x in b.values()),'helper_only':{k:list(v.shape) for k,v in a.items() if k not in b},'checkpoint_only':{k:list(v.shape) for k,v in b.items() if k not in a},'different_shapes':{k:[list(a[k].shape),list(b[k].shape)] for k in a.keys()&b.keys() if a[k].shape!=b[k].shape}}
print(json.dumps(report,indent=2)); (root/'pipeline/meld_parameter_reconciliation.json').write_text(json.dumps(report,indent=2)+'\n')
