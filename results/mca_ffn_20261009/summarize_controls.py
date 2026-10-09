"""Merge verified analysis groups; never present partial seeds as final results."""
import argparse,json,statistics
from pathlib import Path

def stats(xs):
    return {'n':len(xs),'mean':statistics.mean(xs),'sample_std':statistics.stdev(xs) if len(xs)>1 else None}

def summarize(ref,new):
    gs={(g['dataset'],g['variant']):g for g in ref['groups']}
    for g in new['groups']:
        key=(g['dataset'],g['variant'])
        if key in gs: raise ValueError('Unexpected overlap '+str(key))
        gs[key]=g
    wanted=['full','no_cross_attention','no_mixer','no_mca_no_amm','ffn_width_256','ffn_width_512','ffn_width_768']
    out={'complete':True,'missing':[],'groups':[],'conditional_effects':[],'width_deltas':[],'selection':'strict_peak_test_wf1','difference_unit':'percentage_points'}
    for ds in ['iemocap','meld']:
        for v in wanted:
            g=gs.get((ds,v))
            if not g or not g.get('complete_three_seeds') or g.get('n')!=3:
                out['complete']=False;out['missing'].append(ds+'/'+v);continue
            out['groups'].append(g)
        def seeds(v):
            g=gs.get((ds,v))
            if not g or not g.get('complete_three_seeds'):return None
            return {s['seed']:s['metrics']['weighted_f1']*100 for s in g['seeds'] if s['included'] and s['artifact_verified'] and s['formal_protocol_verified']}
        f,a,m,b=[seeds(v) for v in ['full','no_cross_attention','no_mixer','no_mca_no_amm']]
        if all(x is not None and len(x)==3 for x in [f,a,m,b]):
            assert f.keys()==a.keys()==m.keys()==b.keys()
            on=[f[k]-m[k] for k in f];off=[a[k]-b[k] for k in f]
            out['conditional_effects'].append({'dataset':ds,'amm_gain_with_mca':stats(on),'amm_gain_without_mca':stats(off),'interaction_with_minus_without':stats([x-y for x,y in zip(on,off)])})
        for w in [256,512,768]:
            q=seeds('ffn_width_'+str(w))
            if q is not None and f is not None and len(q)==3:
                assert q.keys()==f.keys()
                out['width_deltas'].append({'dataset':ds,'ffn_width':w,'delta_wf1_to_1536':stats([q[k]-f[k] for k in f])})
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--new-analysis',type=Path,required=True);p.add_argument('--references',type=Path,default=Path(__file__).with_name('references.json'));p.add_argument('--output',type=Path,default=Path(__file__).with_name('comparison.json'));a=p.parse_args()
    result=summarize(json.loads(a.references.read_text()),json.loads(a.new_analysis.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'complete':result['complete'],'missing':result['missing']}))
