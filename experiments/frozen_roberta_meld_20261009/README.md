# Frozen RoBERTa-large MELD diagnostic

This experiment tests whether task-supervised text features leave too little
room for downstream multimodal fusion. It keeps the current causal history
text, right truncation and last-non-padding pooling, but extracts features from
the original frozen RoBERTa-large checkpoint without MELD fine-tuning.

The original tokenizer is used unchanged. Speaker strings such as `<s1>` are
therefore decomposed into pretrained tokens; no randomly initialized embedding
rows are added. Audio, visual features, labels, splits, downstream training
configuration and seed 2025 remain unchanged.

Initial comparison: MELD `modal_t` and `full`, one seed. This is a diagnostic,
not a three-seed paper result. Full training starts only after feature shape,
coverage, finiteness and hashes pass verification.

## Completed result

The corrected seed-2025 run gives 48.41 weighted F1 for Full and 31.27 for
text only, a +17.15-point difference. Both selected checkpoints pass exact
fresh replay. Compact artifacts and the validity analysis are in
[`../../results/frozen_roberta_meld_20261009/`](../../results/frozen_roberta_meld_20261009/).

The first attempted runs were invalid because the runner recorded replacement
paths in audit metadata without overwriting the vendor trainer's actual
`feature_paths.meld` entries. They therefore reused the fine-tuned text input
and appeared identical to the original experiment. Commit `ce6265b` fixes the
configuration contract and adds a regression test.
