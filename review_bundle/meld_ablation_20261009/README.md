# MELD module-ablation review bundle

This directory is a compact, code-grounded bundle for investigating why most
MM-Mixer ablations on MELD are tied with, or occasionally slightly better
than, the Full model. It contains source code and compact metrics only. It
does not contain MELD data, extracted feature JSONs, model weights, or full
per-sample predictions.

## Start here

Read [`REVIEW_PROMPT.md`](REVIEW_PROMPT.md), then inspect these implementation
files in repository order:

1. [`../../vendor/meld/multiattn.py`](../../vendor/meld/multiattn.py): base
   feature projection, Feature Gating, Adaptive Gating, MCA, pooling/fusion,
   auxiliary heads and training-facing model API.
2. [`../../vendor/meld/model.py`](../../vendor/meld/model.py): AMM and EPIRC
   installation used by the reported Full model.
3. [`../../vendor/meld/variant_override.py`](../../vendor/meld/variant_override.py):
   runtime variant routing.
4. [`../../mm_mixer_final/structural_ablations.py`](../../mm_mixer_final/structural_ablations.py),
   [`../../mm_mixer_final/revision_controls.py`](../../mm_mixer_final/revision_controls.py),
   and [`../../mm_mixer_final/modalities.py`](../../mm_mixer_final/modalities.py):
   structural controls and modality-only variants.
5. [`../../dataset_runners/meld.py`](../../dataset_runners/meld.py) and
   [`../../vendor/meld/train_erc.py`](../../vendor/meld/train_erc.py): actual
   run materialization, optimizer/loss wiring, checkpoint publication and
   evaluation.
6. [`../../mm_mixer_final/config.py`](../../mm_mixer_final/config.py): active
   dimensions, feature paths, seeds, learning rates and loss semantics.

The model sources above are the repository's runnable implementation. The
formal results referenced below were produced from isolated snapshots of this
route; later local experimental edits are not required to understand the
reported Full and base ablations.

## Feature extraction code

`feature_extraction/` contains copies of the available MELD upstream scripts:

- `train_roberta_text_only_meld.py`: supervised RoBERTa-large text-only
  fine-tuning using MELD emotion labels.
- `extract_roberta_features.py`: exporter linked by sample replay to the
  active 1024-dimensional text feature JSONs.
- `extract_frozen_roberta.py`: the new task-agnostic frozen RoBERTa diagnostic.
- `legacy_multimodal_feature_extract.py`: available historical text/audio/
  visual extractor. Its audio function uses
  `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` and temporal mean
  pooling to 1024 dimensions.
- `video_audio_extract.py`: MP4-to-16-kHz mono WAV preprocessing.
- `run_feature_extract_split.py`: historical split wrapper.

Important provenance boundary: the active audio JSONs match the expected
1024-dimensional format and show no large cross-split exact-collision pattern,
but no generation manifest was recovered that cryptographically links them to
one particular execution of the historical audio function. The text linkage
is stronger: nine replayed samples match the active stored vectors at cosine
approximately one, with maximum absolute error below `1e-4`.

## Compact result files

- [`meld_seed_results.csv`](meld_seed_results.csv): per-seed MELD Full,
  modality, module, axis and matched-control results with classwise F1.
- [`../../results/revision_20261003/analysis.md`](../../results/revision_20261003/analysis.md):
  three-seed aggregate table and paired deltas.
- [`../../results/concat_mlp_20261009/README.md`](../../results/concat_mlp_20261009/README.md):
  raw three-modality concat + two-layer MLP control.
- [`../../results/pretraining_audit_20261009/meld_audit.md`](../../results/pretraining_audit_20261009/meld_audit.md):
  upstream text/audio/visual audit.
- [`../../results/meld_visual_remap_20261009/README.md`](../../results/meld_visual_remap_20261009/README.md):
  corrected visual-duplication diagnosis.

All existing revision results use `strict_peak_test_wf1`: each variant keeps
the epoch with the highest test weighted F1. This is a diagnostic protocol and
is itself a likely source of compressed differences between variants. A
validation-selected rerun is still required for a publication-grade claim.

## Key observations before the frozen-text run finishes

Three-seed MELD weighted F1:

| Variant | WF1 (%) |
|---|---:|
| Full | 67.85 +/- 0.06 |
| Text only | 67.50 +/- 0.21 |
| Text + audio | 67.88 +/- 0.17 |
| Text + visual | 67.35 +/- 0.06 |
| No MCA | 67.45 +/- 0.37 |
| No AMM | 67.80 +/- 0.18 |
| No EPIRC | 67.85 +/- 0.06 |
| No Feature Gating + no AG | 67.95 +/- 0.05 |

The single-seed raw concat + MLP control reaches 67.88797 WF1 versus 67.88360
for the same-seed Full model. This is an essential warning that the current
MELD input features make the downstream task almost shallow-model separable.

The visual source artifact contains 3,407 all-zero train vectors. At least 609
nonzero dev vectors and 1,548 nonzero test vectors exactly match train vectors
attached to different utterance text. This repetition is already present in
the CSS/SDT source pickle rather than being introduced by the JSON exporter.
