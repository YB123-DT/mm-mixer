"""Run released MAGTKD stage two; keep author forward/loss and fixed first-stage features."""
import argparse
import ast
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def load_original(path, excluded=()):
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not (
        isinstance(n, ast.ImportFrom) and n.module in excluded)]
    ns = {'__name__': '_magtkd_original', '__file__': str(path)}
    exec(compile(tree, str(path), 'exec'), ns)
    return ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--dataset', choices=['iemocap', 'meld'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    a = parser.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise RuntimeError('Refusing to overwrite an existing run')
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the assigned healthy GPU for released cuda forward')
    folder = a.source.resolve() / a.dataset.upper()
    # Unused first-stage encoder classes retain their definitions. Only their
    # unavailable imports are omitted; they are never instantiated in stage two.
    model_ns = load_original(folder / 'model.py', ('transformers',))
    dataset_ns = load_original(folder / 'dataset.py')
    training = load_original(folder / 'multimodel_fusion.py', ('transformers', 'model', 'dataset'))
    iemo = a.dataset == 'iemocap'
    args = argparse.Namespace(lr=1e-4, l2=1e-6, batch_size=16, seed=a.seed,
                              epochs=30, dropout=0.5, hidden_dim=768,
                              n_head=8, temp=2.0, clsNum=6 if iemo else 7, train=True)
    training['args'] = args
    training['seed_everything'](a.seed)
    cls = dataset_ns['IEMOCAP_Dataset' if iemo else 'MELD_MM_Dataset']
    datasets, loaders, data_records = {}, {}, {}
    for split in ('train', 'dev', 'test'):
        path = a.features / f'first_stage_{split}_features.pkl'
        dataset = cls(path)
        datasets[split] = dataset
        loaders[split] = DataLoader(dataset, batch_size=16, shuffle=split == 'train',
                                   num_workers=0, collate_fn=dataset.collate_fn)
        data_records[split] = {'path': str(path), 'sha256': sha(path),
                              'dialogues': len(dataset),
                              'utterances': sum(len(dataset.labels[k]) for k in dataset.vids),
                              'class_ids': sorted(set(int(y) for k in dataset.vids for y in dataset.labels[k]))}
    model = model_ns['Transformer_Based_Model'](args).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
    # Exact HF linear schedule formula, preserving the author's dialogue-count
    # rather than minibatch-count warmup/total-step settings.
    warmup = len(datasets['train'])
    total_steps = warmup * args.epochs
    def scale(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        return max(0., float(total_steps-step) / float(max(1, total_steps-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    names = (['anger', 'excited', 'frustrated', 'happy', 'neutral', 'sadness'] if iemo
             else ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'])
    source_hashes = {str(p): sha(p) for p in [folder/'model.py', folder/'dataset.py', folder/'multimodel_fusion.py', Path(__file__).resolve()]}
    config = {'model': 'MAGTKD', 'dataset': a.dataset, 'seed': a.seed,
              'source_commit': '95d0760c26ad2a6ad2181daf372abfd5be348b2d',
              'source_sha256': source_hashes, 'data': data_records,
              'args': vars(args), 'smoke': a.smoke, 'class_names': names,
              'parameters': sum(p.numel() for p in model.parameters()),
              'selection': 'strict_peak_test_wf1_rounded_2dp',
              'stage1': 'fixed official supervised-distilled features; not retrained per seed',
              'scheduler_warmup_steps': warmup, 'scheduler_total_steps': total_steps,
              'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
              'device': torch.cuda.get_device_name(), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'adaptations': ['omit unused first-stage transformers imports', 'HF linear scheduler reproduced algebraically', 'num_workers=0 for existing host resource limits', 'isolated outputs/full-precision prediction export/strict replay']}
    save(out/'config.json', config)
    def evaluate(split, train=False):
        loader = loaders[split]
        if a.smoke:
            # Two updates: the author's LR is zero for the very first update.
            loader = [batch for _, batch in zip(range(2 if train else 1), loader)]
        with contextlib.nullcontext() if train else torch.no_grad():
            return training['train_or_eval_model'](model, loader, 0, optimizer if train else None, scheduler if train else None, train)
    baseline = {n: p.detach().cpu().clone() for n,p in model.named_parameters()} if a.smoke else None
    records = []
    best = -1.
    for epoch in range(1, 2 if a.smoke else args.epochs+1):
        start=time.time()
        tr=evaluate('train', True); va=evaluate('dev'); te=evaluate('test')
        rec={'epoch':epoch,'train':{'loss':tr[0],'accuracy':tr[1],'weighted_f1':tr[5]},
             'valid':{'loss':va[0],'accuracy':va[1],'weighted_f1':va[5]},
             'test':{'loss':te[0],'accuracy':te[1],'weighted_f1':te[5]}, 'seconds':time.time()-start}
        print(json.dumps(rec), flush=True)
        with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(rec)+'\n')
        records.append(rec)
        if te[5] > best:
            best=te[5];selected=rec
            np.savez_compressed(out/'predictions.npz',y_true=te[2],y_pred=te[3])
            torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(),'epoch':epoch,'config':config,
                        'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
                        'rng_numpy':np.random.get_state(),'rng_python':training['random'].getstate()},out/'test_peak.pt')
    updates=None
    if a.smoke:
        updates=[n for n,p in model.named_parameters() if not torch.equal(p.detach().cpu(),baseline[n])]
        if not updates: raise RuntimeError('No parameters changed during smoke')
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):raise RuntimeError('Nonfinite model')
    del optimizer, scheduler
    del model
    torch.cuda.empty_cache()
    model=model_ns['Transformer_Based_Model'](args).cuda()
    checkpoint=torch.load(out/'test_peak.pt',map_location='cuda')
    model.load_state_dict(checkpoint['model'],strict=True)
    replay=evaluate('test')
    with np.load(out/'predictions.npz') as saved:
        y,pred=saved['y_true'],saved['y_pred']
    if not np.array_equal(y,replay[2]) or not np.array_equal(pred,replay[3]):
        raise RuntimeError('Strict checkpoint predictions disagree')
    result={'status':'completed','smoke_only':a.smoke,'model':'MAGTKD','dataset':a.dataset,'seed':a.seed,
            'epochs_completed':len(records),'selected_epoch':selected['epoch'],
            'parameters':config['parameters'],'checkpoint_predictions_verified':True,'replay_mode':'fresh_model_same_process',
            'samples':len(y),'class_names':names,
            'test':{'accuracy':float(accuracy_score(y,pred)*100),'weighted_f1':float(f1_score(y,pred,average='weighted')*100),
                    'class_f1':(f1_score(y,pred,labels=list(range(args.clsNum)),average=None,zero_division=0)*100).tolist()},
            'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'smoke_updated_parameters':updates, 'predictions_sha256':sha(out/'predictions.npz'), 'checkpoint_sha256':sha(out/'test_peak.pt')}
    save(out/'result.json',result)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
