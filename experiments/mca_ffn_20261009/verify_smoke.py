import hashlib,json,sys,torch
from pathlib import Path
ROOT=Path('/data2/yb/multimodalERC/MM_Mixer_MCA_FFN_20261009')
rows=json.loads((ROOT/'pipeline/smoke_verification.json').read_text()); assert len(rows)==8
expected={'iemocap':{'no_mca_no_amm':3593964,'ffn_width_256':5044802,'ffn_width_512':5307458,'ffn_width_768':5570114},'meld':{'no_mca_no_amm':2947952,'ffn_width_256':4398790,'ffn_width_512':4661446,'ffn_width_768':4924102}}
for r in rows:
 b=Path(r['bundle']); m=json.loads((b/'manifest.json').read_text());c=json.loads((b/'config.json').read_text())
 assert m['fresh_strict_replay_exact']
 for name,h in m['artifact_sha256'].items():assert hashlib.sha256((b/name).read_bytes()).hexdigest()==h,(b,name)
 state=torch.load(b/'best_peak_test_state_dict.pt',map_location='cpu',weights_only=True)
 buffers=sum(t.numel() for k,t in state.items() if r['dataset']=='iemocap' and k.startswith('transformer_encoder.blocks.') and k.rsplit('.',1)[-1] in ('alpha_sub','alpha_mod','alpha_ffn'))
 n=sum(t.numel() for t in state.values())-buffers;assert n==expected[r['dataset']][r['variant']],(r,n)
 r['registered_parameters']=n;r['buffer_elements']=buffers;r['artifact_hashes_verified']=True
 if r['dataset']=='meld':assert c['fixed_params']['no_alignment'] is True
 if r['variant'].startswith('ffn_width_'):
  hidden=int(r['variant'].split('_')[-1]); assert all(state[f'transformer_encoder.blocks.{i}.ffn.0.weight'].shape==(hidden,256) for i in (0,1))
 else:
  assert not any(n.startswith('transformer_encoder.blocks.') or n.startswith('cross_attn.') for n in state)
  assert any(n.startswith('transformer_encoder.cross.') for n in state)
manifest=json.loads((ROOT/'code/snapshot.json').read_text())
for name,h in manifest['file_sha256'].items():assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==h,name
(ROOT/'pipeline/smoke_verification.json').write_text(json.dumps(rows,indent=2)+'\n')
print('8 smoke checkpoints: hashes, strict replay, actual shapes, actual counts and complete frozen source manifest verified')
