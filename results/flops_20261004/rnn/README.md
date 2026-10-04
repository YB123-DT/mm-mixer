# DialogueRNN and Ada2I checkpoint FLOPs

CPU operation counts on `biggpu`, 2026-10-04. Source/checkpoint/features are from `/data2/yb/paper/tsne_baselines_20260728`, matching the historical timing adapters. Every checkpoint loaded strictly. No training or GPU use.

| Model | Dataset | MFLOPs/valid utterance | Registered parameters | Valid utterances | Padded slots |
|---|---|---:|---:|---:|---:|
| DialogueRNN | IEMOCAP | 42.784997 | 8,880,406 | 1,623 | 2,821 |
| DialogueRNN | MELD | 53.966395 | 2,666,607 | 2,610 | 6,232 |
| Ada2I | IEMOCAP | 124.894905 | 35,670,850 | 1,623 | 2,821 |
| Ada2I | MELD | 87.942997 | 18,564,043 | 2,610 | 6,232 |

## Scope

Two FLOPs per MAC, matrix multiplication and convolution only. Excludes biases, elementwise ops, activations, norms, softmax, pre-extracted feature generation and backward. Forward matches the original benchmark including executed diagnostics and auxiliary computations. Counts use batch size 32 **dialogues**, all test batches, then divide total FLOPs by valid utterances. Padding and causal dialogue attention are therefore included; these are not batch-size-independent single-utterance costs.

DialogueRNN's historical IEMOCAP configuration consumes **text only**, and MELD consumes **text and audio**. Neither is a three-modality matched-input baseline. Registered parameters can include inactive branches; this is not an active-parameter count. Ada2I uses all three modalities and executes extra modality score calculations in the benchmark's `model(batch)` call.

The saved checkpoints, complete Python source hashes, feature hashes, configuration and per-batch operations/shapes are in the four JSON files. `uncounted_operator_calls` exposes operations outside the stated scope. No opaque attention/recurrent kernels or uncounted matrix/linear/convolution operations were observed.

## Validation and reproduction

All test batches were compared against the same loaded model's `torch.no_grad()` forward. DialogueRNN max difference is 0; Ada2I max difference is approximately 3.82e-6. Grad-enabled math SDPA avoids opaque fused attention; CPU MKLDNN is disabled. No backward is run.

An additional GRUCell analytic check used batch 3, input width 5, hidden width 7: `2 * 3 * 3 * 7 * (5+7) = 1512 FLOPs`. The shared counter returned exactly 1512, all in `aten.addmm`, confirming recurrent gate matrices are counted rather than omitted.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
/data2/yb/reproduction_envs/s0/bin/python measure_baseline_rnn_flops.py \
  --root /data2/yb/paper/tsne_baselines_20260728 \
  --model dialoguernn --dataset iemocap \
  --output /absolute/new/path/dialoguernn_iemocap.json
```

Repeat with `ada2i` and `meld`. Output paths must not already exist. Requires the existing `s0` environment and adjacent `measure_revision_flops.py`. Persistent execution was `tmux flops_rnn_20261004`; all four jobs completed and produced their expected JSON files.
