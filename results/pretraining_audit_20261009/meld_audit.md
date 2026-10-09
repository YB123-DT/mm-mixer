# MELD upstream feature audit — 2026-10-09

Read-only audit; no features, checkpoints, training or active experiments changed. CPU only. Actual formal paths come from `experiments/concat_mlp_20261009/source_config.json`. Local and biggpu SHA256 match for all nine actual text/audio/visual JSONs (hashes in `meld_features_probe.json`). This is not an audit of `features_roberta_large_ft_current` or full-dialogue variants.

## Corrected critical finding: duplication is already present in the source pickle

The active `features_denseface` dev/test files contain many exact nonzero duplicates across different utterances. A later exhaustive, split-aware reconstruction uniquely matched each target dialogue to its correct CSS source dialogue using ordered utterance IDs and sentences. All 13,707 verifiable exported vectors were identical to the old JSON vectors. Thus the earlier explanation that the JSON exporter confused split-local `diaX_uttY` IDs is refuted: the corresponding correct source dialogues already contain the duplicated vectors.

| Check | train | dev | test |
|---|---:|---:|---:|
| JSON rows | 9989 | 1109 | 2610 |
| All-zero visual vectors | 3407 | 95 | 258 |
| Exactly matches original pickle visual via global dialogue ID = split-local CSV ID | 9989 | 675 | 1740 |
| Direct-ID source sentence matches CSV (case/whitespace normalization) | 9988 | 0 | 0 |
| Direct-ID source sentence mismatches CSV | 1 | 675 | 1740 |
| Direct-ID source does not contain requested utterance | 0 | 434 | 870 |

Of dev's 675 and test's 1740 comparisons against the train dialogue having the same split-local number, **609 dev and 1548 test vectors are nonzero and exactly equal** despite different utterance texts. These are rigorous lower bounds on source-level cross-dialogue duplication (54.91% dev; 59.31% test). They are not evidence that the downstream JSON selected the wrong dialogue. Zero vectors alone remain uninformative.

Example `dia0_utt0`:
- train: “also I was the point person on my company’s transition from the KL-5 to GR-6 system.”, neutral.
- dev: “Oh my God, he’s lost it. He’s totally lost it.”, sadness.
- test: “Why do all you’re coffee mugs have numbers on the bottom?”, surprise.
- All three have the identical nonzero 342-vector, float32 SHA256 `2091c28aec6511b57f1b5b985c2c37a8624c994b8b375312b9b5775f1abda1a3`.
- Original `/home/yangbin/HRM_2/meld_multimodal_features.pkl`, global dialogue 0 utterance 0, has the training sentence and exactly this vector.

Train's one sentence discrepancy: `dia556_utt6` CSV “My son? Pretty serious. Oh hey Katie! What uh, what are you doing here?” versus pickle “What do I do?”. All 9989 train vectors exactly match pickle, including 3407 zeros; zeros were already present upstream.

Early evidence: `meld_visual_collision.json`, `meld_visual_source_sentences.json`; reproducible read-only scripts `meld_visual_probe.py`, `meld_probe.py`. Decisive follow-up: `results/meld_visual_remap_20261009/independent_verification.json` and `source_duplication.json`.

The current local `extract_denseface_features.py` describes sentence-based dialogue remapping. A stricter one-to-one reconstruction confirms that the active JSON values are exactly what correct mapping from this pickle produces. Re-running this exporter cannot remove source-level duplication.

## Text encoder: actual checkpoint and remaining provenance limit

Candidate historical checkpoint exists at `/data2/yb/multimodalERC/MELD/Model/results/roberta_large_ft/best_roberta_text_only.pt`, keys `model`, `tokenizer_len`, `pooling`, `args`. Saved args: RoBERTa-large, seed42, encoder LR1e-6, headLR5e-5, batch4, max10 epochs, patience3, max_length511, pooling=mask, no initial task checkpoint. Tokenizer vocabulary50274.

Adjacent historical metrics: best dev WF1=66.215185, test WF1=66.533773, ACC=67.318008. Historical `roberta_large_ft_train.log` shows dev best at epoch5, stops epoch8, then tests. It supports **dev-selected upstream text checkpoint**, not test-selected pretraining. The present script additionally supports optional diagnostic test-peak export, but that later capability alone does not establish usage in this run. Current training sources read train/dev/test separately (train script lines294–302), backpropagate train loader, choose best dev (399–406).

Important mismatch in current scripts:
- Training `train_roberta_text_only_meld.py:100–114`: keep final511 tokens, append MASK, left pad; representation at MASK (181–184).
- Extraction `extract_roberta_features.py:200–214`: tokenizer default right truncation, automatic special tokens, no appended MASK; pool last non-padding token (normally EOS). This can drop current utterance on long histories.
- History text itself includes only rows through current utterance (`extract_roberta_features.py:20–36`), but speaker numbers are constructed from sorted speakers across the entire dialogue (26–27). This is future speaker-set metadata dependence, not future utterance text/label inclusion.
- Completed CPU replay: the candidate checkpoint encoder loaded strictly, and 9 samples (rows 0,1,5 in each split) reproduced stored text features with cosine approximately 1 and maximum absolute difference < 9.8e-5. This strongly supports the checkpoint and EOS extraction linkage for the sampled vectors; it is not a bitwise full-dataset provenance proof. See `meld_text_probe.json` and source/checkpoint/tokenizer SHA256 in `meld_source_manifest.json`.
- Full CSV/tokenizer scan: extraction inputs exceeding511 tokens: train1/9989, dev0/1109, test7/2610 (maximum516/363/653). Thus truncation-direction mismatch affects only a small subset here; MASK-versus-EOS pooling mismatch applies generally. Performance effect remains unmeasured.
- No evidence was found that test labels received upstream text-training gradients. Historical dev-best epoch5 evidence and final test evaluation are affirmative; optional current test-peak code should not be conflated with actual historical execution.

## Audio and visual provenance

`Dataset/feature_extract.py:390–418` identifies acoustic backbone `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`, a Wav2Vec2 emotion-adapted checkpoint, followed by mean of temporal hidden states. No MELD audio fine-tuning appears in this extraction function. Actual generation manifest/checkpoint hash was not recovered, so code provenance is weaker than the empirical JSON tests.

DenseFace script imports already aggregated 342-vectors from the original pickle (`extract_denseface_features.py:77–83`), not per-frame DenseFace execution; this audit does not establish original DenseFace training data or frame pooling.

## Basic feature integrity

Text train/dev/test all present: 9989/1109/2610,1024-dimensional, no zeros/nonfinite values. Audio has 9988/1112/2615 records: missing train `dia125_utt3`, dev `dia110_utt7`; dev4/test5 extra keys. No nonfinite/zero audio stored vectors. Root audit covers loader handling of missing/extra entries.

Cross-split exact text-vector matches are few and associated with identical current utterance text in sampled matches; they are not analogous to the visual collision finding. Audio no exact cross-split vector matches. Duplicate vectors alone are not a leakage verdict.

## Interpretation

The source-level visual duplication and large number of source zero vectors are concrete reasons to question the effective visual information available to the model. They do **not** prove that all weak ablations stem from this issue. Existing MELD results remain valid for this exact CSS/SDT feature artifact, but cannot establish how the model behaves with freshly extracted visual features. Correcting this requires extraction from original videos, followed by matched retraining; no such extraction or retraining was performed here.
