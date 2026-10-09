# Code and feature-path audit

Audit performed after epoch 1 entered the Test-ranked checkpoint list.

- Local historical and biggpu CSV files have identical SHA256 values for all
  splits: train `4e3907...`, dev `7e3063...`, test `2d3fdb...`.
- The copied RoBERTa-large model, vocabulary, merges and tokenizer files have
  the same hashes as the historical local Hugging Face snapshot.
- The feature exporter is byte-identical to the exporter linked to the active
  paper features (`7f2271...`).
- History strings produced by training and extraction were compared for all
  9,989 train, 1,109 dev and 2,610 test utterances and were equal. All
  `diaX_uttY` keys are unique within each split.
- The epoch-1 candidate reconstructs a 50,274-token RoBERTa model with all 391
  state tensors under strict loading and no missing or unexpected keys.
- Feature post-processing requires exact split keys, expected row counts,
  1,024 dimensions, finite values, and checkpoint/feature hashes.
- A materialized downstream probe confirmed that both
  `runtime_audit.feature_paths` and the vendor trainer's
  `feature_paths.meld.*.text` resolve to the process-scoped ranked feature
  root.

The historical pipeline deliberately has different upstream classification
and downstream export pooling: training uses a final MASK state after left
padding and right-side history retention, while feature export uses the last
non-padding state (normally EOS) with the existing tokenizer path. The ranked
experiment preserves this behavior to remain comparable with the active
paper features. Consequently, upstream Test ranking measures the MASK-based
classifier, while the downstream screen is the decisive measure of the
exported EOS representations.

Checkpoint ranking uses Test WF1 at the user's request. These runs are
diagnostic and cannot be presented as validation-selected main results.
