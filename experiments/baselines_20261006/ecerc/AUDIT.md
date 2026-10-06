# ECERC ACL 2025 reproduction audit

- Official paper: https://aclanthology.org/2025.acl-long.102/
- Official repository: https://github.com/TAN-OpenLab/ECERC
- Pinned commit: `339fcaa68262f3fc64d860ebe6c6f79696e25818`
- Upstream README declares MIT licensing; upstream git contains no standalone LICENSE file.
- Local clean source: `/data2/yb/paper/baseline_sources_20261006/ECERC`.
- Run server: **biggpu**, GPU 4 forbidden. Interpreter `/data2/yb/reproduction_envs/s0/bin/python` (torch 2.2.2+cu121).
- Remote root: `/data2/yb/multimodalERC/MM_Mixer_Baselines_20261006`.

## Data and model fidelity

Use original author-provided preprocessing from the Google Drive folder linked in the official README. `download_data.py` records IDs, byte counts, and SHA256 in data/manifest.json. ECERC requires **two distinct 1024-dimensional text streams**, emotion and semantic RoBERTa, plus IEMOCAP 1582-dimensional audio / 342-dimensional visual and MELD 300 / 342 dimensions. Copying our existing text feature into both streams would not reproduce this model. Official data were obtainable; no feature substitution is planned. Pretrained extractors differ from MM-Mixer, so this is an original-system baseline, not a same-feature architecture control.

IEMOCAP keeps upstream first 10% of train dialogue indices as validation; MELD uses released train/dev/test lists. The wrapper preserves source model, loss, optimizer, hidden size, dropout, batching and early stopping. Original IEMOCAP uses class-weighted loss and MELD unweighted loss. Epoch limits 200/40, batch sizes 64/32 dialogues, Adam learning rates 1e-4/1e-5, weight decay 2e-4, joint validation-F1 and validation-loss patience 50/20 remain unchanged. Three seeds: IEMOCAP 2025/2066/2118; MELD 2025/2028/2069.

Original code selects by validation WF1. User-requested output selects strict test WF1 peak, while recording the independently validation-selected result. Scores are kept at upstream two-decimal precision for epoch choice; metrics recomputed from exported predictions retain full precision. This change is explicit and must not be called validation-selected in the paper. No repeated hyperparameter tuning is introduced.

## Source issues and instrumentation

1. Original CPU forward path does not define `mask`/`imask`; use the assigned healthy GPU. No model patch applied.
2. Original MELD training helper returns an empty list instead of its already-computed labels/predictions. Wrapper changes only that function return in memory, recording original/patched source hashes. It does not change input order or computations.
3. Evaluation runs under no_grad; original model.eval behavior preserved.
4. Export isolated best-test and best-valid state dictionaries/optimizer checkpoints, predictions and per-epoch metrics. Reload the selected state dictionary and require exact prediction equality before completion.
5. Original MELD dataloader yields its named `videoVisual` before `videoAudio`, and training consumes them as audio/vision; preserve original behavior and audit actual released array widths rather than silently swapping.

## Context audit

The source constructs lower triangular attention masks, but uses finite `-1e9` before softmax. Speaker-restricted attention rows with no eligible prior key consequently become uniform over all positions, including future turns. A synthetic five-turn/two-dialogue GPU test changed only future input features: early logits changed by **0.1274669170** on IEMOCAP. Forward/backward were finite, output shape [10,6], peak allocated memory 70,060,544 bytes for this tiny check. **The unmodified original cannot be represented as strictly history-only.** Keep its result separate or mark its actual context behavior; do not silently repair architecture to claim an official reproduction. `check_model.py` reproduces the audit.

## Commands

Remote data lives in `data/ecerc`, linked from `code/ECERC/data`. Source is `code/ECERC`, wrappers in `code/ecerc`.

```bash
CUDA_VISIBLE_DEVICES=3 /data2/yb/reproduction_envs/s0/bin/python \
 /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/code/ecerc/run.py \
 --source /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/code/ECERC \
 --dataset iemocap --seed 2025 \
 --output /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/runs/ecerc_iemocap_seed2025
```

Use `--smoke` for one train/validation/test batch, including checkpoint reload validation. Formal GPU placement is owned by the shared scheduler and must be checked immediately before launch.

## Completed checks (2026-10-06)

Both datasets passed real-data one-batch train/dev/test smoke runs and exact selected-checkpoint prediction replay on biggpu GPU 3 (`GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`). Results are `iemocap_smoke_result.json` and `meld_smoke_result.json`; their low one-step scores are **not formal benchmark results**. No formal training was launched by this audit.

- IEMOCAP released data: 120 training dialogues / 5810 utterances; 31 test dialogues / 1623 utterances. Training retains the official first 12 dialogues for dev.
- MELD: train 1038 dialogues / 9989 utterances, dev 114 / 1109, test 280 / 2610.
- Actual MELD returned feature widths are [1024, 1024, **342, 300**, 9], while its model splits concatenated non-text features at **300, 342**. This confirms an upstream modality slicing inconsistency, which is deliberately retained for faithful released-code reproduction. A corrected implementation would be a separate variant, not silently substituted.
- MELD synthetic future-perturbation max earlier-output change: 0.1578385830; finite backward, [10,7] output.
- Data hashes in `data_manifest.json`; original Python source hashes in `source_manifest.json`; full dialogue ID lists in the dataset audit JSON files.
