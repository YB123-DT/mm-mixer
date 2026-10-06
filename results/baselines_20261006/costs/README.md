# Recent baseline downstream computational cost

Measured on biggpu CPU using the seed-2025 saved test-peak checkpoints, loaded strictly. Full test sets: IEMOCAP 1,623 and MELD 2,610 valid utterances. Two FLOPs per matrix/convolution multiply-add; padding is included in batch cost and the sum is divided by valid utterances. Feature extraction, backward, elementwise operations, normalization, nonlinearities, softmax and scatter aggregation are excluded, matching the existing explicit-matrix FLOPs table. Parameters include all registered downstream parameters, including branches not exercised by evaluation.

ECERC uses batches of 32 dialogues. ConFilMER uses its released/training batch size of 16 on both datasets: IEMOCAP cannot run batch 32 because the learned hyperedge weight vector has a fixed length of 1,000 and the first test batch exceeds that number. We did not extend checkpoint parameters. Thus batch/padding settings differ; these numbers describe the released systems under the profiled configurations, not a controlled architecture-only comparison.

CPU adapter preserves ECERC's `cuda_flag=True` mask construction and maps `.cuda()` calls to CPU; it does not change the mathematical operations. MKLDNN is disabled so GRU computations decompose into counted matrix operators. The original and instrumented forwards agree within rtol=1e-4, atol=1e-5 on every batch. RNGs are reset identically before each forward, and opaque recurrent/attention operators are checked. The recorded operator lists contain the actual dispatched operations.

`counter_batch32.py` and `counter_batch16.py` are immutable snapshots of the exact executed counters; their SHA-256 values appear in each result. The maintained entry point is `experiments/baselines_20261006/measure_costs.py` (additional explicit rejection of opaque `aten.gru/lstm` added after measurement; none occurred in the measured results). JSON files record source, checkpoint and feature hashes, checkpoint configuration, batch lengths, counted operators and forward parity. `selftest.json` records analytic bidirectional two-layer GRU and sparse-matrix regression checks.

Reproduce on biggpu, substituting model `ecerc`/`confilmer` and dataset `iemocap`/`meld`:

```sh
/data2/yb/reproduction_envs/s0/bin/python measure_costs.py \
  --root /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006 \
  --model ecerc --dataset iemocap --output /new/output.json
```

The script refuses to overwrite an existing output. No training or GPU allocation is involved.

| Model | Dataset | Parameters | MFLOPs / valid utterance |
|---|---|---:|---:|
| ecerc | iemocap | 5,975,310 | 24.076100 |
| ecerc | meld | 5,810,187 | 30.306251 |
| confilmer | iemocap | 8,137,288 | 34.043402 |
| confilmer | meld | 14,131,273 | 50.320264 |
