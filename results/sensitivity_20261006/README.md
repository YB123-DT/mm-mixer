# Projection-view sensitivity results

**Complete: 18/18 new formal runs.** Queue exit code 0; finished 2026-10-06T12:30:18Z. Twelve existing S=1/S=6 reference runs were independently rechecked. All checkpoint/other artifact hashes match; prediction-derived metrics match; every bundle records exact fresh checkpoint replay. No failed, omitted or selected seeds.

Values below are percentages, mean ± sample standard deviation over three seeds (ddof=1). IEMOCAP: 2025, 2066, 2118; MELD: 2025, 2028, 2069. Existing user-authorized `strict_peak_test_wf1` selection is retained.

| S | IEMOCAP Acc | IEMOCAP WF1 | MELD Acc | MELD WF1 |
| --- | --- | --- | --- | --- |
| 1 | 71.58 ± 0.40 | 71.71 ± 0.41 | 68.47 ± 0.10 | 67.78 ± 0.02 |
| 2 | 71.39 ± 0.23 | 71.53 ± 0.28 | 68.51 ± 0.08 | 67.84 ± 0.10 |
| 4 | 71.68 ± 0.23 | 71.77 ± 0.19 | 68.52 ± 0.32 | 67.76 ± 0.24 |
| 6 | 72.03 ± 0.25 | 72.15 ± 0.21 | 68.57 ± 0.06 | 67.85 ± 0.06 |
| 8 | 71.80 ± 0.18 | 71.90 ± 0.22 | 68.48 ± 0.10 | 67.69 ± 0.13 |

## Interpretation

S=6 has the highest mean WF1 among these five settings on both datasets. IEMOCAP S=2/4/8 trails S=6 by 0.62/0.38/0.25 percentage points; S=1 trails by 0.44. The trend is not monotonic: S=2 is below S=1, and S=8 is below S=6.

On MELD, S=2 is effectively tied in mean WF1 with S=6 (67.84 vs 67.85); S=4 and S=8 trail by 0.09 and 0.16 points. The full S range spans only about 0.16 WF1 points. These results support retaining S=6 as a reasonable setting, not a claim that more views consistently help or that S=6 is statistically superior.

S=1 and S=6 are verified existing reference experiments, not newly rerun. S=1 uses the prior post-construction control implementation; S=2/4/8 use constructor-level S. S=6 is bitwise unchanged from the original Full implementation. S-axis hidden width always follows 2S (2/4/8/12/16), so parameter counts vary.

## Evidence and artifacts

- `analysis.json` / `analysis.md`: original frozen analyzer output for all 18 new runs, with every seed, class F1 and confusion counts.
- `comparison.json` / `comparison.csv`: five-S comparison including all reference seeds and paired differences to S=6.
- `reference_analysis.json`: original verified S=1/S=6 reference subset.
- `completion_verification.json`: independent verification of all 30 bundles, including checkpoint hashes and recomputed metrics.
- `verified_metadata/`: per-run configurations, manifests, histories, metrics and statuses; no weights or raw dataset files downloaded.
- `state.json`, `summary.json`, `queue_exit_code.txt`, `finished_at.txt`: final scheduler evidence.

Server: biggpu; physical GPU 7 UUID `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e`. Frozen snapshot SHA256 `689bd97ff0a57fbd5fde0770f0b12c33e235934c7587d1dceee5347219020c67`. Remote new checkpoints remain under `/data2/yb/multimodalERC/MM_Mixer_Sensitivity_20261006/runs/`.
