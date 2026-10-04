#!/usr/bin/env python3
"""Count actual saved DialogueRNN/Ada2I test forward matrix/convolution FLOPs."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['dialoguernn', 'ada2i'], required=True)
    p.add_argument('--dataset', choices=['iemocap', 'meld'], required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from measure_revision_flops import count_forward
    torch.set_num_threads(1)
    # Disable MKLDNN so CPU recurrent gates expose their matrix products.
    torch.backends.mkldnn.enabled = False
    root = args.root.resolve()
    repo = root / ('DialogueRNN' if args.model == 'dialoguernn' else 'Ada2I')
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.model == 'dialoguernn':
        from benchmark_checkpoint import make_model
        from dataloader import IEMOCAPDataset, MELDDataset
        checkpoint = root / 'results/dialoguernn' / args.dataset / 'model.pt'
        saved = torch.load(checkpoint, map_location='cpu')
        model = make_model(args.dataset, saved['causal_history'])
        model.load_state_dict(saved['state_dict'], strict=True)
        name = 'IEMOCAP' if args.dataset == 'iemocap' else 'MELD'
        feature = repo / f'DialogueRNN_features/{name}_features/{name}_features_raw.pkl'
        ds = IEMOCAPDataset(path=str(feature), train=False) if args.dataset == 'iemocap' else MELDDataset(path=str(feature), n_classes=7, train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=ds.collate_fn, num_workers=0)
        def prepare(data):
            tensors = data[:-1]
            if args.dataset == 'iemocap':
                text, visual, audio, qmask, umask, labels = tensors
                x = text
            else:
                text, audio, qmask, umask, labels = tensors
                x = torch.cat((text, audio), dim=-1)
            return (x, qmask, umask), {'att2': True}, int(umask.sum()), int(umask.numel()), {'x': list(x.shape), 'qmask': list(qmask.shape)}, list(data[-1])
        cfg = {'causal_history': saved['causal_history'], 'input_modalities': 'text only' if args.dataset == 'iemocap' else 'text+audio'}
    else:
        from dataloader import Dataloader, load_iemocap, load_meld
        from model import Ada2I
        from utils import set_seed
        checkpoint = repo / f'checkpoint/{args.dataset}_causal_s13_best.pt'
        saved = torch.load(checkpoint, map_location='cpu')
        cfg = dict(vars(saved['args']))
        model_args = saved['args']
        model_args.device = 'cpu'
        model_args.dataset = args.dataset
        model_args.batch_size = 32
        model_args.causal_history = True
        set_seed(int(model_args.seed))
        ds = load_iemocap() if args.dataset == 'iemocap' else load_meld()
        loader = Dataloader(ds['test'], model_args)
        # Explicit finite generator: Dataloader has __getitem__ but may not raise IndexError.
        loader = [loader[i] for i in range(len(loader))]
        model = Ada2I(model_args)
        model.load_state_dict(saved['state_dict'], strict=True)
        feature = repo / f'data/{args.dataset}/{args.dataset}.pkl'
        def prepare(data):
            shape = list(data['tensor']['t'].shape)
            return (data,), {}, int(data['label_tensor'].numel()), shape[0] * shape[1], {m: list(x.shape) for m, x in data['tensor'].items()}, data['length'].tolist()
    model.eval()
    records = []
    operations = Counter()
    uncounted = Counter()
    for idx, data in enumerate(loader):
        inputs, kwargs, valid, padded, shapes, ids = prepare(data)
        with torch.no_grad():
            ref = model(*inputs, **kwargs)[0].detach().clone()
        output, result = count_forward(model, inputs, kwargs)
        actual = output[0].detach()
        torch.testing.assert_close(actual, ref, rtol=1e-4, atol=1e-5, equal_nan=False)
        diff = float((actual-ref).abs().max())
        operations.update(result['counted_operators'])
        uncounted.update(result['uncounted_operator_calls'])
        records.append({'batch_index': idx, 'valid_utterances': valid, 'padded_utterance_slots': padded, 'input_shapes': shapes, 'dialogue_ids_or_lengths': ids, 'max_abs_logit_difference': diff, **result})
        print(json.dumps({'batch': idx, 'valid': valid, 'flops': result['matrix_convolution_flops'], 'max_diff': diff}), flush=True)
        del output, ref, actual
    total = sum(r['matrix_convolution_flops'] for r in records)
    valid = sum(r['valid_utterances'] for r in records)
    sources = [f for f in repo.rglob('*.py') if '.git' not in f.parts]
    result = dict(status='completed', model=args.model, dataset=args.dataset, device='cpu', torch_version=torch.__version__, total_flops=total, valid_utterances=valid, flops_per_utterance=total/valid, registered_parameters=sum(p.numel() for p in model.parameters()), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad), batch_size_dialogues=32, padded_utterance_slots=sum(r['padded_utterance_slots'] for r in records), convention='2 FLOPs per MAC; matrix multiplication and convolution only; excludes bias, elementwise, activations, norms, softmax, feature extraction, backward', counting_mode='grad-enabled math SDPA; MKLDNN disabled; no backward; every batch compared with no_grad forward', forward_scope='Same default model forward as historical benchmark; including diagnostic and auxiliary computations executed', padding_note='Actual padded batch operations counted; divided by valid test utterances; not single-utterance FLOPs', checkpoint={'path': str(checkpoint), 'sha256': sha(checkpoint)}, features={'path': str(feature), 'sha256': sha(feature)}, config=cfg, source_hashes={str(f.relative_to(repo)): sha(f) for f in sources}, script_sha256=sha(Path(__file__)), counted_operators=dict(operations), uncounted_operator_calls=dict(uncounted), records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps({k: result[k] for k in ['model','dataset','total_flops','valid_utterances','flops_per_utterance','registered_parameters']}), flush=True)

if __name__ == '__main__':
    main()
