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
