#!/usr/bin/env python3
"""CPU forward operation counts for frozen MM-Mixer checkpoints.

Convention: 2 FLOPs per multiply-accumulate, counting matrix products and
convolutions (including attention projections/products). Activations, norms,
softmax, elementwise operations, data movement and feature extractors are
excluded. This is an operation count, never a latency measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def count_forward(model, args=(), kwargs=None):
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    class ObservedCounter(FlopCounterMode):
        def __init__(self):
            super().__init__(display=False)
            self.observed = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.observed[str(func._overloadpacket)] += 1
            return super().__torch_dispatch__(func, types, args, kwargs)

    if model.training:
        raise ValueError('Model must be in eval mode')
    # Grad-enabled forward prevents the opaque native MHA fast path in torch
    # 2.2. No backward pass or parameter update is performed. Math SDPA exposes
    # QK^T and AV to the dispatcher, avoiding uncounted fused CPU attention.
    with torch.enable_grad(), torch.backends.cuda.sdp_kernel(
        enable_flash=False, enable_math=True, enable_mem_efficient=False
    ), ObservedCounter() as counter:
        output = model(*args, **(kwargs or {}))
    opaque = [name for name in counter.observed if any(word in name for word in (
        'native_multi_head_attention', '_transformer_encoder_layer_fwd',
        '_scaled_dot_product', 'aten.gru', 'aten.lstm', '_cudnn_rnn',
        '_thnn_fused', 'mkldnn_rnn_layer',
    ))]
    if opaque:
        raise ValueError(f'Opaque compute operators require an explicit counting rule: {opaque}')
    counts = counter.get_flop_counts()
    global_counts = {str(k): int(v) for k, v in counts['Global'].items()}
    return output, {
        'matrix_convolution_flops': int(counter.get_total_flops()),
        'counted_operators': global_counts,
        'observed_operator_calls': dict(counter.observed),
        'uncounted_operator_calls': {k: v for k, v in counter.observed.items()
                                     if k not in global_counts},
    }


def self_test():
    import torch
    torch.manual_seed(7)
    linear = torch.nn.Linear(3, 4).eval()
    _, result = count_forward(linear, (torch.randn(2, 3),))
    assert result['matrix_convolution_flops'] == 2 * 2 * 3 * 4
    mha = torch.nn.MultiheadAttention(8, 2, batch_first=True).eval()
    q, kv = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
    output, result = count_forward(mha, (q, kv, kv))
    # Q/K/V/output projections plus QK^T and AV, biases excluded.
    expected = 2 * 2 * ((3 + 5 + 5 + 3) * 8 * 8 + 2 * 3 * 5 * 8)
    assert result['matrix_convolution_flops'] == expected, result
    with torch.no_grad():
        reference = mha(q, kv, kv)[0]
    torch.testing.assert_close(output[0], reference)
    print('Self-test passed: exact Linear and cross-attention counts; output parity.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path)
    parser.add_argument('--artifact-dir', type=Path)
    parser.add_argument('--dataset', choices=('iemocap', 'meld'))
    parser.add_argument('--variant', default='full')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    if args.self_test:
        return self_test()
    if not all((args.code_root, args.artifact_dir, args.dataset, args.output)):
        parser.error('--code-root, --artifact-dir, --dataset and --output are required')
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.code_root.resolve()
    sys.path.insert(0, str(root))
    efficiency = importlib.import_module('measure_revision_efficiency')
    config, manifest, checkpoint, hashes = efficiency.validate_artifact(
        args.artifact_dir.resolve(), args.dataset, args.variant, args.seed)
    runner = importlib.import_module(f'dataset_runners.{args.dataset}')
    model = runner.build_variant_model(args.variant, .2)
    if args.dataset == 'meld':
        model.disable_alignment()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    dataset, collate, ids, sources = efficiency.build_test_dataset(args.dataset, runner, config, [])
    feature_hashes = {}
    for key, (original, actual) in sources.items():
        digest = efficiency.sha256(actual)
        expected = manifest.get('feature_hashes', {}).get(original)
        if expected is not None and digest != expected:
            raise ValueError(f'Feature hash mismatch: {original}')
        feature_hashes[key] = {'path': str(actual), 'sha256': digest,
                               'manifest_hash_verified': expected is not None}
    records = []
    for batch_size in (1, 32):
        features, labels = collate([dataset[i] for i in range(batch_size)])
        with torch.no_grad():
            reference = efficiency.main_logits(model(features)).detach().clone()
        output, count = count_forward(model, (features,))
        logits = efficiency.main_logits(output).detach()
        torch.testing.assert_close(logits, reference, rtol=1e-4, atol=1e-5)
        records.append({'batch_size': batch_size,
                        'input_shapes': {k: list(v.shape) for k, v in features.items()},
                        'sample_ids': ids[:batch_size],
                        'input_tensor_sha256': efficiency.input_hash([(features, labels)]),
                        'matrix_convolution_flops_per_utterance': count['matrix_convolution_flops'] / batch_size,
                        'max_abs_logit_difference': float((logits - reference).abs().max()),
                        **count})
        del output
    if records[0]['matrix_convolution_flops_per_utterance'] != records[1]['matrix_convolution_flops_per_utterance']:
        raise ValueError('Per-utterance count changes with batch size; inspect batch-dependent computation')
    result = {'status': 'completed', 'dataset': args.dataset, 'variant': args.variant,
              'seed': args.seed, 'device': 'cpu', 'torch_version': torch.__version__,
              'convention': '2 FLOPs per MAC; matrix multiplication and convolution only',
              'excluded': ['pretrained feature extractors', 'backward', 'bias addition',
                           'elementwise operations', 'normalization', 'activations', 'softmax', 'data movement'],
              'forward_scope': 'Unmodified default model eval output, including auxiliary heads and diagnostic calculations executed by forward',
              'counting_mode': 'grad-enabled forward with math SDPA; no backward; checked against no_grad logits',
              'registered_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'artifact_dir': str(args.artifact_dir.resolve()), 'artifact_sha256': hashes,
              'feature_sources': feature_hashes, 'records': records,
              'source_provenance': efficiency.source_provenance(),
              'counter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'dataset': args.dataset, 'variant': args.variant,
                      'flops_per_utterance': records[0]['matrix_convolution_flops_per_utterance'],
                      'parameters': result['trainable_parameters']}))


if __name__ == '__main__':
    main()
