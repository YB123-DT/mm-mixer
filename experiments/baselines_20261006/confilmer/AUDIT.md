# ConFilMER reproduction audit — 2026-10-06

- Official repository: https://github.com/G22-web/ConFilMER
- Pinned commit: `4ac4c27a4355ab2e5a9003e0359eebc3e6cda6a1`.
- Paper: **Enhanced Multimodal Emotion Recognition in Conversations via Contextual Filtering and Multi-Frequency Graph Propagation**, ICASSP 2025; repository README gives author citation.
- Server: **biggpu only**. GPU 4 forbidden. Smoke used physical GPU 5 UUID `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`.
- Environment: `/data2/yb/reproduction_envs/s0/bin/python`, torch 2.2.2+cu121 / PyG 2.7.0. No dependency installation.
- Upstream checkout local: `/data2/yb/paper/baseline_sources_20261006/ConFilMER` (untouched).
- Patched deployed source: `/data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/source/confilmer`.

## Data and comparison limits

Existing author-format files under `/data2/yb/paper/tsne_baselines_20260728/M3NET/{IEMOCAP,MELD}_features` are symlinked read-only. IEMOCAP: 120 train /31 test dialogues, 5810/1623 utterances; 4×1024 RoBERTa, 1582 IS10 audio, 342 DenseFace. MELD:1152 train /280 test dialogues, 11098/2610 utterances;4×1024 RoBERTa,300 audio,342 DenseFace. MELD training IDs combine usual training and development utterances. Alignment of feature/label utterance lengths was checked for all supplied train/test dialogues.

This is **original-context / official-feature reproduction**, not a unified history-only experiment. The code uses bidirectional GRU, full-dialogue graph propagation, and context-filter similarities across flattened batch nodes. Batch composition can therefore affect outputs. Do not silently replace inputs, masks, pooling or normalize this result into a causal leaderboard. Original data IDs/order are preserved (MELD upstream IDs are a set; run.sh fixes PYTHONHASHSEED).

## Minimal compatibility changes

`prepare_source.py` deterministically generates patches from the pinned checkout:
1. Removes unused CLIP load/imports, ipdb, utils and obsolete unused topk import. CLIP argument stays an Identity placeholder; official forward never reads it.
2. Replaces unavailable torch_scatter.scatter_add with existing PyG scatter(reduce='sum'), same index-add semantics.
3. Hypergraph node→edge→node propagation uses reversed edge_index for second source_to_target pass instead of mutating MessagePassing.flow. PyG2.7 generated propagate assumes construction-time flow; upstream dynamic mutation fails shape checks. Reversal preserves edge→node messages.
4. Removes upstream CUDA_VISIBLE_DEVICES=0 override; requires scheduler device. Adds seed/output/smoke controls and machine-readable metrics/checkpoints.
5. Makes seed_everything default resolve selected seed instead of Python definition-time hardcoded1475. Preserves upstream epoch-level reseeding behavior.
6. Uses existing train_our.py for both datasets; README's MELD train_one.py does not exist.
7. Records Accuracy/MacroF1 at the exact maximum test-WF1 checkpoint rather than independently choosing max Accuracy. Test-peak selection is preserved and explicitly labeled, consistent with user instruction; not validation-selected.

## Runs and outputs

`run.sh IEMOCAP SEED OUTPUT` keeps README settings:80epochs, batch16, lr1e-4,dropout.5, num_L5,num_K4. MELD:15epochs,batch16,lr1e-4,dropout.4,num_L3,num_K3,modal embedding. Seeds I2025/2066/2118; M2025/2028/2069. GPU is supplied externally.

Each run produces config.json,best.pt(model+optimizer+epoch),best.json,predictions.npz(y_true/y_pred),result.json,completed.json; caller captures stdout log. best.pt is a best-model training-state snapshot, **not exact resumable checkpoint** (RNG/dataloader state absent). No formal training launched by this preparatory agent.

IEMOCAP full batch16 train/backward+test smoke passed after PyG compatibility patch; one train/test batch is not a performance result. Smoke artifacts at root/smoke/confilmer_iemocap_v2. MELD smoke status reported separately once verified.

## Final verification

Both dataset train/backward/test smoke runs passed at batch16. Final MELD smoke verified strict best.pt reload gives identical predictions, full-precision metrics, smoke_completed status, and per-epoch timing/peak-memory JSONL. MELD smoke peak allocated586821632B/reserved717225984B (one batch only; full-run peaks may exceed). CPU dense incidence-matrix reference verifies patched non-attention hypergraph aggregation with maximum difference2.98e-8 and finite gradients. Artifact hashes in source_sha256.json. Source frozen after this verification.

The original checkpoint selection compares WF1 rounded to2decimals; this is preserved and recorded in result.json. Reported test metrics are recomputed from selected predictions at full precision. Outputs with config.json/result.json/best.pt/completed.json already present are rejected to prevent overwrite; launch.json and console.log do not block execution.
