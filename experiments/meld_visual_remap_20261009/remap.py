"""Strict, split-aware MELD visual remapping. Unresolved rows are quarantined."""
import argparse
import csv
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np

LABELS = {0:'neutral', 1:'surprise', 2:'fear', 3:'sadness', 4:'joy', 5:'disgust', 6:'anger'}

def norm(text):
    return ' '.join(text.lower().split())

def signature(ids, sentences):
    if len(ids) != len(sentences) or len(set(ids)) != len(ids):
        raise ValueError('Invalid or duplicate utterance IDs')
    return tuple((int(i), norm(t)) for i,t in zip(ids,sentences))

def unique_match(sig, index, allowed):
    candidates = set(index.get(sig, [])) & set(allowed)
    if len(candidates) != 1:
        raise ValueError(f'Expected one dialogue, got {sorted(candidates)}')
    return next(iter(candidates))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def vector_sha(x):
    return hashlib.sha256(np.asarray(x,dtype='<f4').tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',default='/data2/yb/paper/datasets/meld_multimodal_features.pkl')
    ap.add_argument('--csv-root',default='/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw')
    ap.add_argument('--output',default='outputs/meld_visual_remap_20261009')
    ap.add_argument('--report',default='results/meld_visual_remap_20261009')
    args=ap.parse_args(); out=Path(args.output); report=Path(args.report)
    out.mkdir(parents=True,exist_ok=True); report.mkdir(parents=True,exist_ok=True)
    targets=[out/s/'visual_features.json' for s in ['train_features','dev_features','test_features']]
    if any(t.exists() for t in targets): raise FileExistsError('Refusing to overwrite exported features')
    with open(args.source,'rb') as f: p=pickle.load(f,encoding='latin1')
    assert not set(p[10]) & set(p[11])
    index=defaultdict(list)
    for d in p[0]: index[signature(p[0][d],p[9][d])].append(d)
    records=[]; quarantine=[]; used=set(); stats={}; exports={}; speaker_bad=[]
    source_paths=[Path(args.source),Path(__file__)]
    for split in ['train','dev','test']:
        csvpath=Path(args.csv_root)/f'{split}_sent_emo.csv'; source_paths.append(csvpath)
        dialogs=defaultdict(list)
        with csvpath.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f): dialogs[int(row['Dialogue_ID'])].append(row)
        features={}; nzero=0
        for dia, rows in dialogs.items():
            ids=[int(r['Utterance_ID']) for r in rows]; texts=[r['Utterance'] for r in rows]
            sig=signature(ids,texts); allowed=p[11 if split=='test' else 10]
            omit=set(); policy='exact_ordered_dialogue'
            try: src=unique_match(sig,index,allowed)
            except ValueError:
                # Audited exception: six exact ordered anchors establish this dialogue,
                # but neither sentence nor speaker supports the last row's vector.
                if (split,dia,ids)!=('train',556,list(range(7))): raise
                if norm(texts[6])!=norm('My son? Pretty serious. Oh hey Katie! What uh, what are you doing here?'): raise
                candidates=[d for d in allowed if len(p[0][d])==7 and signature(p[0][d][:6],p[9][d][:6])==sig[:6] and list(p[0][d])==ids]
                if len(candidates)!=1: raise ValueError('Ambiguous exception anchors')
                src=candidates[0]
                assert norm(p[9][src][6])=='what do i do?'
                omit={6}; policy='six_exact_anchors_last_row_quarantined'
            if src in used: raise ValueError(f'Source dialogue reused: {src}')
            used.add(src)
            # Speaker IDs are dialogue-local: validate their equivalence relation,
            # never use emotion labels to select the mapping.
            csv_to_src={}; src_to_csv={}
            for i,r in enumerate(rows):
                if i in omit: continue
                speaker=int(np.argmax(p[1][src][i])); name=r['Speaker']
                if csv_to_src.get(name,speaker)!=speaker or src_to_csv.get(speaker,name)!=name:
                    speaker_bad.append([split,dia,ids[i],src])
                csv_to_src[name]=speaker; src_to_csv[speaker]=name
            for i,r in enumerate(rows):
                key=f'dia{dia}_utt{ids[i]}'
                rec={'split':split,'key':key,'source_dialogue':int(src),'source_index':i,'source_utterance_id':int(p[0][src][i]),'dialogue_mapping':policy}
                if i in omit:
                    rec.update(status='quarantined',reason='source_sentence_and_speaker_mismatch',csv_sentence=r['Utterance'],source_sentence=p[9][src][i])
                    quarantine.append(rec); continue
                assert norm(r['Utterance'])==norm(p[9][src][i])
                assert ids[i]==int(p[0][src][i])
                label_ok=r['Emotion'].lower()==LABELS[int(p[2][src][i])]
                if not label_ok: raise ValueError(f'Independent label validation failed: {split}/{key}')
                v=np.asarray(p[8][src][i]); assert v.shape==(342,) and np.isfinite(v).all()
                if key in features: raise ValueError('Duplicate split-local key')
                features[key]=v.tolist(); nzero+=int(not np.any(v))
                assert np.array_equal(np.asarray(features[key],dtype=v.dtype),v)
                rec.update(status='verified',sentence_match=True,label_match=True,source_vector_sha256=vector_sha(v),all_zero=not bool(np.any(v)))
                records.append(rec)
        exports[split]=features
        stats[split]={'csv_rows':sum(map(len,dialogs.values())),'dialogues':len(dialogs),'exported_rows':len(features),'quarantined_rows':sum(q['split']==split for q in quarantine),'source_zero_vectors':nzero,'dimensions':342,'nonfinite':0}
    if speaker_bad: raise ValueError(f'Speaker consistency failed: {speaker_bad[:20]}')
    assert len(used)==len(p[0])==1432
    hashes={}
    for split,features in exports.items():
        target=out/f'{split}_features'/'visual_features.json';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(features,separators=(',',':')))
        replay=json.loads(target.read_text())
        for rec in records:
            if rec['split']==split: assert vector_sha(replay[rec['key']])==rec['source_vector_sha256']
        hashes[str(target.resolve())]=sha(target)
    keys=['split','key','source_dialogue','source_index','source_utterance_id','dialogue_mapping','status','sentence_match','label_match','source_vector_sha256','all_zero']
    with (report/'mapping.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    (report/'quarantine.json').write_text(json.dumps(quarantine,indent=2,ensure_ascii=False))
    manifest={'status':'partial_train_quarantine_complete_dev_test','training_ready':False,'warning':'Do not feed incomplete train JSON into a loader that silently zero-fills missing rows. Resolve quarantined row or explicitly approve exclusion/new missing-visual policy first.','stats':stats,'source_label_map':LABELS,'source_hashes':{str(x.resolve()):sha(x) for x in source_paths},'output_sha256':hashes,'mapping_sha256':sha(report/'mapping.csv'),'matching_uses_emotion_labels':False,'speaker_partition_validation':True,'one_to_one_source_dialogues':len(used),'json_reload_exact_vector_hashes':len(records),'fresh_zero_fills':0}
    (report/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
