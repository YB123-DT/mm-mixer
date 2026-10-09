# Frozen RoBERTa-large MELD diagnostic

This one-seed diagnostic replaces the MELD-supervised RoBERTa-large text
features with features from the original frozen RoBERTa-large checkpoint. It
keeps the causal history construction, right truncation, last non-padding
token pooling, audio and visual inputs, downstream configuration, data splits
and seed 2025 fixed.

## Result

| Variant | Best epoch | Accuracy (%) | Weighted F1 (%) | Strict replay |
|---|---:|---:|---:|---:|
| Full (text + audio + visual) | 50 | 52.57 | 48.41 | exact |
| Text only | 4 | 48.12 | 31.27 | exact |
| Full - text only | - | +4.44 | **+17.15** | - |

The exact weighted-F1 delta is 0.1714525336. Unlike the earlier run that
accidentally reused the fine-tuned text paths, the valid Full and text-only
runs are clearly separated. This supports the diagnosis that MELD-supervised
text features substantially compress the apparent value of multimodal input
and downstream fusion. It does not attribute the 17.15-point gain to any one
MM-Mixer module because Full also adds the audio and visual inputs.

The Full run remains weak on fear and disgust, both with zero test F1. This
diagnostic should therefore not replace the reported fine-tuned-feature
result or be presented as a new state-of-the-art result.

## Validity checks

- Frozen feature coverage is exact: 9,989 train, 1,109 dev and 2,610 test
  utterances, all finite and 1,024-dimensional.
- Mean cosine similarity with the fine-tuned features is only 0.122/0.125/
  0.114 on train/dev/test, confirming that the replacement is substantive.
- Both selected bundles have `fresh_strict_replay_exact: true`.
- Each manifest hashes the frozen text files actually consumed by training.
- Checkpoints and per-sample predictions are intentionally omitted from Git;
  their hashes remain in `status.json` and `manifest.json`.

## Configuration defect found during the diagnostic

Before commit `ce6265b`, `dataset_runners/meld.py` copied experimental feature
paths only into `runtime_audit.feature_paths`. The vendor trainer actually
read `feature_paths.meld`, which still pointed to the fine-tuned text JSONs.
That made the first frozen-feature runs byte-identical to the original runs.
Those runs were moved to `runs_invalid_hardcoded_paths` on the experiment
server and are not included here. The runner now materializes the same paths
into both configuration views, protected by
`tests/test_feature_path_contract.py`.

## Files

- `feature_verification.json` and `feature_manifest.json`: extraction coverage,
  dimensions and hashes.
- `full/` and `modal_t/`: compact selected-bundle metrics, configuration,
  manifest, history, classification report and training log.
