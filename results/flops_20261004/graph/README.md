# Graph-baseline FLOPs audit

Scope: the saved historical MMGCN, MM-DFN and M3NET checkpoints under
`biggpu:/data2/yb/paper/tsne_baselines_20260728/results`. Full test split,
batch size 32 **dialogues**, normalize by actual non-padding utterances.
This counts padded recurrent work and dense graph operations actually executed;
these are not fixed-shape single-utterance FLOPs.

The primary `total_flops` counts explicit matrix/convolution operations at
2 FLOPs/MAC, including sparse matrix products at `2 * nnz * output_width`.
Bias, normalization, activation, softmax, elementwise and scatter operations
are excluded, as in the MM-Mixer matrix/convolution count. M3NET implements
weighted message aggregation through elementwise multiplication and scatter.
Its `supplementary_sparse_message_mac_flops` gives a **separate** sparse-MAC
interpretation of HypergraphConv/highConv aggregation: two operations per
message feature (one weight multiplication and accumulation). It is not added
to the primary total and excludes computing/normalizing those coefficients.

The source audit found no NumPy/SciPy matrix products in the active graph
forward implementations. NumPy supplies constants and indices; SciPy imports
in these modules are not used for graph matrix computation. MMGCN's
`model_mm.py` performs dense `D.mm(adj).mm(D)` normalization of its multimodal
adjacency. Dense products include cross-dialogue zero regions and may dominate
the measured FLOPs; this is retained as executed, not replaced by a hypothetical
sparse/diagonal optimized implementation.

Execution uses CPU, one thread and no visible GPU. Process-local `.cuda()`
mapping preserves tensor arithmetic while accommodating old hardcoded device
moves; `no_cuda` flags are set true. MKLDNN is disabled to expose recurrent
matrix products. Model weights are loaded from saved checkpoints, no training.
Every batch compares counted grad-enabled forward logits against a seeded
no-grad CPU forward. This checks instrumentation parity; it is not a GPU/CPU
numerical-equivalence test. Hashes preserve checkpoint, Python sources, data
files opened by loaders, input tensors, and counter source. Sample IDs,
dialogue lengths, padded positions, operator counts and differences are saved.

The embedded counter regression checks a dense 2x3 by 3x4 product (48 FLOPs)
and sparse 2x3 product with three stored coefficients by 3x4 (24 FLOPs), with
sparse/dense numerical equality. Sparse dispatch exposes only `_sparse_addmm`
in this test, so it is not counted a second time as a nested dense product.
Other sparse layouts are rejected instead of estimated using dense sizes.

Reproduction (one model/dataset per process, output must not exist):

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /data2/yb/reproduction_envs/s0/bin/python \
  measure_baseline_graph_flops.py \
  --root /data2/yb/paper/tsne_baselines_20260728 \
  --model mmgcn --dataset iemocap --output /new/path/mmgcn_iemocap.json
```

Final run: tmux `flops_graph_verified`, remote directory
`/data2/yb/paper/tsne_baselines_20260728/flops_graph_verified`.

Final validation: all six runs completed, 1623 IEMOCAP / 2610 MELD valid
utterances; every counted logit matched the no-grad reference exactly (maximum
absolute difference 0). Every primary total equals the sum of the recorded
operator counts. M3NET's observed matrix operations are `aten.addmm`; its
message paths expose `aten.mul` and `aten.scatter_add_`. No opaque torch_sparse,
torch_scatter matrix kernel or fused recurrent operation appeared in these
forwards. Source-defined GraphConvolution/spmm code is not exercised by these
saved M3NET configurations.
