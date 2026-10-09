# MELD visual provenance reconstruction — 2026-10-09

**Critical finding: this is not a repaired feature set. All 13,707 verified exports equal the corresponding old JSON vectors exactly (root independent verification). Therefore the earlier diagnosis that the JSON exporter introduced split-local-ID mapping errors is not supported. The same cross-dialogue visual duplication already exists in the CSS source pickle. Re-exporting this source cannot resolve that problem.**

For example, source dialogues 0 (train), 1039 (dev) and 1153 (test) contain different first sentences but exactly identical nonzero visual vectors. They are separate arrays without shared memory. Correct-source vectors also equal split-local-numbered source vectors for 609 nonzero dev rows and 1548 nonzero test rows. See `source_duplication.json`. This local source cannot currently be treated as a clean replacement.

The checks below establish identity to the source, not correctness of the vector-to-original-video association.

Local CPU export completed; source hash `7ad660ebd929861b0cbaada5b406223f58787fb1b0dc152aab3746164a355af2`. No remote execution, training, existing-feature overwrite or paper change.

| Split | Official rows | Verified exported rows | Quarantined | Source zeros retained |
|---|---:|---:|---:|---:|
| train | 9989 | 9988 | 1 | 3407 |
| dev | 1109 | 1109 | 0 | 95 |
| test | 2610 | 2610 | 0 | 258 |

All 1432 source dialogues have distinct target dialogue assignments. All dev/test dialogues and 1037 train dialogues match exact ordered `(Utterance_ID, case/whitespace-normalized sentence)` sequences, restricted to the correct source split. Each exported row additionally passes speaker partition and independently checked emotion-label correspondence, finite 342-dimensional shape and exact source vector checks. All 13,707 exported vectors were reloaded from JSON and their float32 byte hashes verified against the source. Existing all-zero vectors remain unchanged; no missing rows were zero-filled.

## Quarantined source discrepancy

`train/dia556_utt6`: official CSV says “My son? Pretty serious. Oh hey Katie! What uh, what are you doing here?” (Ross), while the source says “What do I do?” and assigns the other speaker in the dialogue. The source vector is nonzero. The preceding six ordered utterances uniquely identify source dialogue556; those six verified rows are retained. The mismatching seventh row is omitted, documented in `quarantine.json`. Neither labels nor fuzzy text similarity are used to fill it.

**Status: source mapping reconstructed; source-level duplicated visuals unresolved; train additionally has one quarantined row; not training-ready.** Existing loaders silently filling missing vectors must not consume this output without explicit handling of the quarantined sample. No training settings or dataset labels/text have been changed.

## Artifacts

- `manifest.json`: source/script/output hashes, counts, readiness flag and validation scope.
- `independent_verification.json`: root check confirming zero changed vectors versus old files.
- `source_duplication.json`: correctly mapped source dialogues already contain the observed duplicated visual values.
- `mapping.csv`: every verified target-to-source row, label/sentence validation and vector SHA256; no raw vectors.
- `quarantine.json`: unresolved row and reason.
- Local-only feature JSONs: `outputs/meld_visual_remap_20261009/{train,dev,test}_features/visual_features.json`.
- Exporter and four passing regression tests: `experiments/meld_visual_remap_20261009/`.

Source-copy consistency validates mapping against the CSS pickle, not correctness of the original frame-level extractor or every upstream raw video. Corrected-score effects require separate retraining and have not been measured here.
