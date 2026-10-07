# MAGTKD replacement results

Completed: **6/6 formal runs**, 2026-10-07T02:12:32.778267+00:00. All runs completed 30 epochs. No formal run failed or was replaced.

Official IJCAI 2025 implementation: https://github.com/JieLi-dd/MAGTKD, commit `95d0760c26ad2a6ad2181daf372abfd5be348b2d`.

**Scope:** second-stage fusion training with fixed author-provided supervised/distilled first-stage features. These are three downstream seeds, not end-to-end three-seed encoder/distillation runs. Released features/splits/context are retained. Strict peak test WF1 selection uses the original two-decimal precision; metrics below are recomputed from predictions at full precision before rounding.

| Dataset | Seeds | ACC | Weighted F1 | Downstream parameters |
|---|---|---:|---:|---:|
| IEMOCAP | [2025, 2066, 2118] | 68.82 ± 0.50 | 69.06 ± 0.47 | 44,332,050 |
| MELD | [2025, 2028, 2069] | 65.72 ± 0.02 | 64.71 ± 0.18 | 44,566,293 |

Standard deviation is sample SD (`ddof=1`), n=3. Counts include all registered original downstream parameters, including unused registered pathways; feature extractors are excluded.

| Dataset | Class (original ID order) | F1 |
|---|---|---:|
| iemocap | anger | 68.10 ± 0.93 |
| iemocap | excited | 69.78 ± 0.29 |
| iemocap | frustrated | 68.41 ± 0.46 |
| iemocap | happy | 57.42 ± 1.79 |
| iemocap | neutral | 66.92 ± 0.31 |
| iemocap | sadness | 80.06 ± 0.50 |
| meld | anger | 51.85 ± 0.37 |
| meld | disgust | 32.97 ± 0.62 |
| meld | fear | 18.01 ± 2.46 |
| meld | joy | 61.01 ± 0.77 |
| meld | neutral | 78.94 ± 0.08 |
| meld | sadness | 39.91 ± 0.66 |
| meld | surprise | 56.54 ± 0.21 |

| Dataset | Seed | Selected epoch | ACC | WF1 |
|---|---:|---:|---:|---:|
| iemocap | 2025 | 17 | 68.26863832 | 68.53967113 |
| meld | 2025 | 11 | 65.74712644 | 64.50946030 |
| iemocap | 2066 | 21 | 68.94639556 | 69.19378770 |
| meld | 2028 | 7 | 65.70881226 | 64.86817965 |
| iemocap | 2118 | 28 | 69.25446704 | 69.44966305 |
| meld | 2069 | 5 | 65.70881226 | 64.76278039 |

## Evidence

- `summary.json`: every seed plus complete classwise means/SD.
- `verification.json`: independent local confusion-matrix recomputation, prediction hashes, rounded-peak selection checks and aggregate checks.
- `runs/{dataset}_seed{seed}/`: result/config/30-epoch metrics/predictions/launch/console. `predictions.json` is the Git-friendly exact-label copy of the original NPZ (the global repository ignore excludes NPZ); its `source_npz_sha256` links to the verified run export. Full model+optimizer+scheduler+RNG checkpoints remain on biggpu and are identified by hashes in result files.
- `queue_state.json`: all six completed; healthy physical GPU1 only.
- Training/source/data audit: `experiments/replacement_20261007/magtkd/`.

Remote root: `/data2/yb/multimodalERC/MM_Mixer_MAGTKD_20261007`. All selected checkpoints were strictly reloaded into a fresh model in the same process and regenerated the exported test predictions exactly. No fresh-process claim is made.

Paper ordering differs from original label IDs: IEMOCAP happy/sadness/neutral/anger/excited/frustrated; MELD neutral/surprise/fear/sadness/joy/disgust/anger. Reorder by name, not index.
