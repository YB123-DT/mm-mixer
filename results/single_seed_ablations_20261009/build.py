import csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sources=['results/revision_20261003/analysis.json','results/sensitivity_20261006/analysis.json',str((OUT/'sources/mca_ffn.json').relative_to(ROOT)),str((OUT/'sources/amm_capacity.json').relative_to(ROOT))]
rows=[];pending=[];hashes={}
for src in sources:
 p=ROOT/src; hashes[src]=hashlib.sha256(p.read_bytes()).hexdigest()
 for g in json.loads(p.read_text())['groups']:
  for s in g['seeds']:
   if not s['included']:
    pending.append(dict(dataset=g['dataset'],variant=g['variant'],seed=s['seed'],state=s['state'],source=src));continue
   assert s['artifact_verified'] and s['formal_protocol_verified']
   m=s['metrics']; c=np.array(s['confusion_counts']);support=c.sum(1);denom=support+c.sum(0)
   f=np.divide(2*c.diagonal(),denom,out=np.zeros(len(c)),where=denom!=0)
   assert np.isclose((f*support).sum()/support.sum(),m['weighted_f1'],atol=1e-12,rtol=0)
   assert np.isclose(c.trace()/c.sum(),m['accuracy'],atol=1e-12,rtol=0)
   assert np.allclose(f,[m['class_f1'][k] for k in s['class_names']],atol=1e-12,rtol=0)
   rows.append(dict(dataset=g['dataset'],variant=g['variant'],seed=s['seed'],accuracy=m['accuracy']*100,weighted_f1=m['weighted_f1']*100,class_f1={k:v*100 for k,v in m['class_f1'].items()},epoch=s['epoch'],source=src))
for src in ['results/concat_mlp_iemocap_20261009/result.json','results/concat_mlp_20261009/meld_seed2025/result.json']:
 p=ROOT/src;a=json.loads(p.read_text());hashes[src]=hashlib.sha256(p.read_bytes()).hexdigest()
 assert a['status']=='completed' and a['fresh_strict_replay_exact'] and not a['smoke_only']
 rows.append(dict(dataset=a['dataset'],variant='concat_mlp',seed=a['seed'],accuracy=a['test']['accuracy'],weighted_f1=a['test']['weighted_f1'],class_f1=dict(zip(a['class_names'],a['test']['class_f1'])),epoch=a['selected_epoch'],source=src))
assert len(rows)==len({(r['dataset'],r['variant'],r['seed']) for r in rows})
names={'full':'Full（S=6，FFN=1536）','no_feature_gating':'去 Feature Gating','no_adaptive_gating':'去 AG','no_feature_and_adaptive_gating':'去 Feature Gating 和 AG','no_cross_attention':'去 MCA','no_mixer':'去 AMM','no_pairwise':'去 EPIRC','no_auxiliary_loss':'去辅助损失','no_sequence_mixing':'去视角轴 mixing','no_modality_mixing':'去模态轴 mixing','no_feature_mixing':'去特征轴 mixing','one_mixer_block':'AMM 仅1层','amm_mlp':'AMM→近容量 MLP','amm_attention':'AMM→近容量 Attention','amm_cubemlp':'AMM→CubeMLP 对照','amm_mlp_no_aux':'AMM→MLP，去辅助损失','pairwise_mlp_residual':'EPIRC→近容量 MLP residual','single_projection_view':'S=1','projection_views_2':'S=2','projection_views_4':'S=4','projection_views_8':'S=8','no_mca_no_amm':'同时去 MCA 和 AMM','concat_mlp':'仅原始特征 Concat＋MLP'}
for k in ['t','a','v','ta','tv','av']:names['modal_'+k]='仅 '+k.upper()
for w in [256,512,768,3072,6144]:names['ffn_width_'+str(w)]='FFN宽度 '+str(w)
order=list(names); rows.sort(key=lambda r:(r['dataset'],order.index(r['variant']),r['seed']))
index={(r['dataset'],r['variant'],r['seed']):r for r in rows}
ps={(r['dataset'],r['variant'],r['seed']):r for r in pending}
lines=['# MM-Mixer 所有消融：逐种子分数','', '所有数值为百分制。每行/每格对应一个实际种子，不计算跨种子均值，不挑最好种子。沿用现有 test-WF1 选检查点协议。常规消融保持其他设置；纯 Concat＋MLP 同时移除所有融合模块及辅助头，仅有 seed2025。','',f'本次汇总：{len(rows)} 个已完成并验证的运行；{len(pending)} 个未完成任务。每类指标为 F1；ACC、WF1 单独列出。','']
classes={'iemocap':['happiness','sadness','neutral','anger','excited','frustration'],'meld':['neutral','surprise','fear','sadness','joy','disgust','anger']}
for d,seeds in [('iemocap',[2025,2066,2118]),('meld',[2025,2028,2069])]:
 lines += ['## '+d.upper()+'：总分展开','','每格为 **WF1 / ACC**。','','| 实验 | '+' | '.join(map(str,seeds))+' |','|---|'+'---:|'*len(seeds)]
 for v in order:
  cells=[]
  for seed in seeds:
   k=(d,v,seed)
   if k in index:r=index[k];cells.append(f"{r['weighted_f1']:.2f} / {r['accuracy']:.2f}")
   elif k in ps:cells.append({'running':'训练中','queued':'排队中'}.get(ps[k]['state'],ps[k]['state']))
   else:cells.append('未运行')
  lines.append('| '+names[v]+' | '+' | '.join(cells)+' |')
 lines+=['','## '+d.upper()+'：每类 F1 展开','']
 for seed in seeds:
  lines+=['### Seed '+str(seed),'','| 实验 | '+' | '.join(classes[d])+ ' | ACC | WF1 |','|---|'+'---:|'*(len(classes[d])+2)]
  for r in rows:
   if r['dataset']==d and r['seed']==seed:
    vals=[r['class_f1'][k] for k in classes[d]]+[r['accuracy'],r['weighted_f1']]
    lines.append('| '+names[r['variant']]+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
  lines+=['']
lines+=['## 溯源','','CSV/JSON 保留原始 variant、种子、选中轮次与来源。常规消融和敏感性/宽度对照的 ACC、WF1、每类 F1 已从逐种子混淆矩阵重新核对。Concat＋MLP 引用已完成严格重载和预测重算的独立报告。','']
for s,h in hashes.items():lines.append(f'- `{s}`；SHA256 `{h}`')
(OUT/'ALL_SINGLE_SEED_RESULTS.md').write_text('\n'.join(lines)+'\n')
(OUT/'results.json').write_text(json.dumps({'completed':rows,'pending':pending,'source_sha256':hashes},indent=2)+'\n')
fields=['dataset','variant','seed','accuracy','weighted_f1','epoch']+list(dict.fromkeys(classes['iemocap']+classes['meld']))+['source']
with (OUT/'results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
 for r in rows:w.writerow({**{k:r[k] for k in fields if k in r},**r['class_f1']})
print('Verified completed runs:',len(rows),'pending:',len(pending))
