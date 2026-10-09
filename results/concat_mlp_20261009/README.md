# Completed single-seed MELD raw-concat MLP control

Run: biggpu physicalGPU1, seed2025, 50epochs completed, selected EMA checkpoint epoch48 using existing strict peak test-WF1 protocol. Fresh model strict-load replay produced bit-identical logits for all2610 test samples. Local verification independently recomputed weighted/per-class F1 and ACC from JSON predictions, checked all50 epoch records and source/prediction hashes. Formal duration917.44s (excluding feature loading). Not a three-seed result or validation-selected estimate.

| Same-seed model | WF1 (%) | ACC (%) |
|---|---:|---:|
| Full MM-Mixer (existing seed2025) | 67.88359991 | 68.58237548 |
| Full without auxiliary losses (existing seed2025) | 67.58869611 | 68.27586207 |
| Raw concat → 256-hidden MLP (this run) | 67.88797398 | 68.77394636 |

The new model has613,895 parameters. Its two linear layers require1,227,264 matrix FLOPs/utterance analytically at2FLOPs/MAC (excludes bias, GELU, dropout). This single run's WF1 differs from same-seed Full by only+0.00437407 percentage points; report essentially tied, not proven superior. Against no-aux Full difference+0.29927787 points. Feature extraction, augmentation and evaluation protocol are retained; downstream architecture and removal of auxiliary supervision differ.

Class F1, paper order neutral/surprise/fear/sadness/joy/disgust/anger:
80.80272624 /61.06346484 /34.88372093 /42.54143646 /65.40880503 /35.59322034 /55.74803150.

Full details: `summary.json`, `meld_seed2025/result.json`, `config.json`, `environment.json`, `data_manifest.json`, `verification.json`, `metrics.jsonl`, and `predictions.json`. NPZ is also locally available with recorded SHA256; checkpoint remains on biggpu `/data2/yb/multimodalERC/MM_Mixer_ConcatMLP_20261009/runs/meld_seed2025/test_peak.pt`.
