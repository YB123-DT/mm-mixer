#!/usr/bin/env python3
"""Count historical CSS/SDT test forwards on CPU, 2 FLOPs/MAC."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['css', 'sdt'], required=True)
    p.add_argument('--dataset', choices=['iemocap', 'meld'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(7)
    from torch.utils.data import DataLoader
    from measure_revision_flops import count_forward
    root = Path('/data2/yb/paper/tsne_baselines_20260728')
    if a.model == 'sdt':
        repo = root / 'SDT'
        checkpoint = root / 'results' / 'sdt' / a.dataset / 'model.pt'
        config = dict(temp=1 if a.dataset == 'iemocap' else 8, hidden_dim=1024,
                      n_head=8, dropout=0.5, causal_context=True)
        evidence = root / 'results' / 'sdt' / a.dataset / 'metrics.json'
    else:
        run = (Path('/data2/yb/reproduction_workspace/runs/CSS/causal_iemocap_10seeds/seed61080')
               if a.dataset == 'iemocap' else
               Path('/data2/yb/reproduction_workspace/runs/paper_meld_seed_queue_20260724/css_meld_causal/seed10073'))
        repo = run / 'code' if a.dataset == 'iemocap' else Path('/data2/yb/paper/CSS_causal_meld')
        evidence = run / 'train.log'
        # Parse the logged Namespace safely (literal keyword values only).
        import ast
        node = ast.parse(evidence.read_text().splitlines()[0], mode='eval').body
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Namespace'
        logged = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
        config = {k: logged[k] for k in ['temp', 'hidden_dim', 'n_head', 'dropout', 'causal_context', 'rank', 'order']}
        historical = json.loads((root / 'results' / 'css' / a.dataset / 'metrics.json').read_text())
        checkpoint = Path(historical['checkpoint'])
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from dataloader import IEMOCAPDataset, MELDDataset
    from model import Transformer_Based_Model
    ds = IEMOCAPDataset(train=False) if a.dataset == 'iemocap' else MELDDataset('data/meld_multimodal_features.pkl', train=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=ds.collate_fn, num_workers=0)
    args = [a.dataset.upper(), config['temp'], 1024, 342, 1582 if a.dataset == 'iemocap' else 300, config['n_head']]
    if a.model == 'css':
        args.extend([config['rank'], config['order']])
    model = Transformer_Based_Model(*args, n_classes=6 if a.dataset == 'iemocap' else 7,
                                   hidden_dim=config['hidden_dim'], n_speakers=2 if a.dataset == 'iemocap' else 9,
                                   dropout=config['dropout'], causal_context=True).eval()
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(saved['model_state_dict'] if a.model == 'css' else saved, strict=True)
        mode = 'trained_checkpoint'
    else:
        assert a.model == 'css'
        mode = 'architecture_reconstructed_missing_checkpoint'
        expected = 50452563 if a.dataset == 'iemocap' else 53398634
        assert sum(x.numel() for x in model.parameters()) == expected
    # Original forwards construct integer padding-speaker indices with .cuda().
    # Restrict CPU remapping to those one-dimensional integer constants only.
    def cpu_padding_cuda(tensor, *args, **kwargs):
        assert tensor.device.type == 'cpu' and tensor.ndim == 1 and tensor.dtype == torch.int32
        assert tensor.numel() == 0 or bool((tensor == (2 if a.dataset == 'iemocap' else 9)).all())
        return tensor
    records = []
    for i, data in enumerate(loader):
        text, visual, audio, qmask, umask, labels = data[:-1]
        qmask = qmask.permute(1, 0, 2)
        lengths = [int(mask.sum()) for mask in umask]
        inputs = (text, visual, audio, umask, qmask, lengths)
        with patch.object(torch.Tensor, 'cuda', cpu_padding_cuda):
            with torch.no_grad():
                reference = model(*inputs)[3].clone()
            output, result = count_forward(model, inputs)
        torch.testing.assert_close(output[3], reference, rtol=1e-4, atol=1e-5)
        record = dict(batch_index=i, input_shapes=[list(v.shape) for v in inputs[:5]],
                      dialogue_ids=[str(v) for v in data[-1]], lengths=lengths,
                      valid_utterances=int(umask.sum()), padded_slots=umask.numel(),
                      max_abs_logit_difference=float((output[3].detach() - reference).abs().max()), **result)
        records.append(record)
        del output, reference
        print(json.dumps(dict(model=a.model, dataset=a.dataset, batch=i, flops=result['matrix_convolution_flops'])), flush=True)
    total = sum(r['matrix_convolution_flops'] for r in records)
    n = sum(r['valid_utterances'] for r in records)
    assert n == (1623 if a.dataset == 'iemocap' else 2610)
    result = dict(status='completed', model=a.model.upper(), dataset=a.dataset, device='cpu', torch_version=torch.__version__,
                  total_flops=total, valid_utterances=n, flops_per_utterance=total/n,
                  registered_parameters=sum(v.numel() for v in model.parameters()), records=records,
                  batch_size_dialogues=32, padding_included=True, configuration=config, weight_source=mode,
                  cpu_adaptation='Only integer 1-D padding-speaker .cuda() construction remapped to CPU; arithmetic unchanged',
                  parity_scope='Counted grad-enabled vs no_grad CPU outputs; no GPU parity claim',
                  convention='2 FLOPs/MAC; matrix multiplication and convolution only; includes padded positions and all default forward heads',
                  exclusions=['feature extraction', 'backward', 'elementwise operations', 'normalization', 'softmax', 'activations', 'bias additions'],
                  checkpoint=dict(path=str(checkpoint), exists=checkpoint.is_file(), sha256=sha(checkpoint) if checkpoint.is_file() else None),
                  sources={str(f): sha(f) for f in [repo/'model.py', repo/'dataloader.py', evidence, repo/'data'/f'{a.dataset}_multimodal_features.pkl', Path(__file__), Path(__file__).with_name('measure_revision_flops.py')]})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['model', 'dataset', 'total_flops', 'valid_utterances', 'flops_per_utterance', 'registered_parameters', 'weight_source']}), flush=True)


if __name__ == '__main__':
    main()
