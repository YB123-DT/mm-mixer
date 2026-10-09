# Local, split-aware MELD visual export

Purpose: reconstruct and verify a one-to-one dialogue mapping into the CSS global-dialogue pickle. **Independent comparison found all 13,707 exported vectors equal the old JSONs: the prior export-mapping-bug diagnosis was incorrect. Duplicated visual values already occur in the source pickle; this export does not repair them.** No existing features, datasets, training configurations or running jobs are changed.

Run from repository root:

```bash
python -m unittest discover -s experiments/meld_visual_remap_20261009 -p 'test_*.py' -v
python experiments/meld_visual_remap_20261009/remap.py
```

Requires existing NumPy, otherwise only Python standard library. The exporter refuses to overwrite existing output JSONs; use a new `--output` and `--report` directory to replay.

Matching uses exact ordered `(Utterance_ID, normalized sentence)` pairs, with normalization restricted to case and whitespace. Official train/dev match only source train IDs; official test matches only source test IDs. A source dialogue cannot be reused across any splits. Ambiguous or unmatched dialogues fail closed, except the separately documented audited dialogue556 exception. Speaker identity partitions are validated independently. Emotion labels never select mappings: the known source class map is checked only after matching.

Outputs retain the existing split JSON format, with raw 342-vectors copied exactly from the source. Four tests cover split-local collisions, ambiguity rejection, utterance IDs and order, and malformed/missing matches. Real-data checks cover all exported vectors after JSON reload, shapes, finite values, one-to-one dialogues, labels, and speaker partitions.

**Not training-ready:** one official training row is quarantined and omitted. Existing loaders may silently fill missing visual features with zeros; do not point a training job at this export until that row is explicitly resolved or an exclusion/missing-modality policy is approved and recorded. No new zero vectors were introduced. Read `results/meld_visual_remap_20261009/README.md`.

Do not commit outputs/ feature JSONs or upstream datasets. Commit the exporter, tests, mapping metadata, quarantine record and manifest only.
