# IEMOCAP upstream feature comparison audit — 2026-10-09

Read-only audit on `biggpu`; no training, feature replacement, or remote mutation. Scope is comparison evidence for the MELD investigation, not certification that every upstream step is correct.

## Confirmed current artifact lineage

Remote root: `/data2/yb/multimodalERC/IEMOCAP/`.

The actual downstream artifact is `Model_rawaux_textpeak_cssv_v1/artifacts/iemocap_textpeak_audio_cssv.npz`. Its adjacent `manifest.json` records:

- Text and audio inherited from `Model_rawaux_textpeak_v27_fill53/artifacts/fill53_features.npz`.
- Visual vectors replaced with CSS pickle field 8 from `external/CSS/data/iemocap_multimodal_features.pkl`.
- Dimensions T/A/V = 1024/1024/342.
- Source SHA256 `bee09ed676bb2bbd6859a93019e70e428467af95d1d82251d516de83c3005c24`.
- Current artifact SHA256 `88c271b82789438fc8fa197acc453468e4be4c2bd430679c8080ff398811c4ea`.
- CSS SHA256 `368c20fa87ea49fe9f068a5755085538e4d5d44ef4f1431794975c7ddbcbcbe2`.

**Independent CPU checks:** all three actual hashes match the manifest; all dialogue text/audio arrays exactly equal the source arrays, and all visual arrays exactly equal float32 CSS field 8. This establishes inherited artifact identity, not the provenance of the original text/audio encoder checkpoints. Builder evidence: `Model_rawaux_textpeak_cssv_v1/build_cssv_artifact.py:15-21,34-60,73-94`.

## Split and fill53 checks

Actual CSS split: training 120 dialogues / 5,810 utterances, sessions 01–04; test 31 dialogues / 1,623 utterances, session 05. Dialogue ID intersection is empty. No cross-session contamination was found in this downstream split check.

`Dataset/Data` CSV counts are train 4,778 / dev 980 / test 1,622 (7,380 total). `Dataset/Data_7433` counts are train 4,830 / dev 980 / test 1,623 (7,433 total). `Model_rawaux_fill53_css_aligned/artifacts/missing53.csv` has 53 rows: 52 training, one test. Thus fill53 refers to completing missing utterances to the CSS canonical 7,433-turn inventory; it does not itself mean 53 leaked examples. The exact historical generation method for all original V27 fallback vectors remains unproven.

The later `Text_only_officialsplit/build_textpeak_fill53_features.py` explicitly handles 7,380 standard and 53 fallback text features, checks split/dialogue/utterance/class identity, and remaps CSS labels to the encoder label order. It leaves audio/visual inherited. This later script is verified as a candidate reconstruction path, not established as the generator of current V27 text vectors.

## Candidate upstream text pipeline (code confirmed; current V27 linkage unresolved)

`Text_only_officialsplit/train_roberta_text_only_iemocap.py`:

- Lines 72–91: history through target utterance only, speaker-marked; append `</s> Now <sN> feels`. No future utterance text is included in the loop. Speaker numbering uses first-occurrence order in the full dialogue, but future-only speakers cannot change numbering for existing speakers.
- Lines 94–98: retain last 511 tokens, append MASK.
- Lines 101–108: LEFT padding, so final tensor position is the real MASK token, not PAD.
- Lines 150–162: RoBERTa backbone final hidden state `[:, -1, :]`, followed by 1024→768→6 classifier, returns both logits and 1024-dimensional hidden.
- Lines 239–241: separate train/dev/test CSV datasets.
- Lines 353–354,377–411: evaluates dev and test each epoch; writes both dev-selected and test-peak checkpoints. Therefore file names or this script's existence alone do not establish which checkpoint produced current features.

`Text_only_officialsplit/extract_text_features_from_checkpoint.py:345-358` recreates tokenizer/speaker tokens, loads the specified checkpoint, and extracts hidden states with the shared context/tokenization implementation. The candidate mechanism is contextual final-MASK hidden extraction, not mean text pooling.

## Important provenance boundary

Current `Model_rawaux_textpeak_v27_fill53/artifacts/` contains its NPZ but no adjacent original manifest or generation script discovered in this audit. Do not assert its original checkpoint/selection criterion merely from the word `textpeak`.

A nearby, documented **different** branch exists at `Model_rawaux_utterance_history_residual/formal/textpeak6891_fusion_20260724/manifest.json`:

- Explicit checkpoint: `Text_only_officialsplit/reproductions/exact_legacy_6836_20260722/run_02_seed2025_gpu3/best_iemocap_text_only_test_peak.pt`.
- Explicit identifier `roberta-large-testpeak-wf1-68.9088-mask-hidden-1024`; `diagnostic_only: true`.
- It uses current V27 source as OLD input but writes NEW text NPZ with hash `ad5250ca6a4c109a19716d5ecf1fa58d3fbd82de97dda803f2bb3253c65c0d03`.

That hash differs from current V27 (`bee09e...`), and the CSSv manifest names V27 as its source. Thus this adjacent test-selected checkpoint cannot currently be attributed to the actual downstream text feature artifact. Its projection variant also differs (`875d0463...`, 768 projected components zero-padded to 1024), so must not conflate that experiment with the actual input.

## What this permits us to say

- The actual IEMOCAP artifact has verified dimensions, source hashes, CSS-session separation, and exact modality inheritance.
- The candidate text pipeline includes history and supervised emotion fine-tuning; no last-position PAD mistake is present in that code because padding is left-sided.
- The actual original V27 checkpoint, encoder fine-tuning split and selection rule, audio checkpoint/pooling/fallback source, and CSS visual upstream training remain unresolved by available manifest evidence.
- Therefore IEMOCAP is not a verified clean upstream reference against which a MELD issue can simply be inferred. MELD must be audited directly; small ablation gains alone diagnose neither feature corruption nor leakage.
