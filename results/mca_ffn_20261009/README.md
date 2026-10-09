# MCA × AMM and FFN-width controls (2026-10-09)

Status: complete. All 24 new runs and 18 reused reference runs were verified for both datasets and three predetermined seeds. The experiment uses the historical `strict_peak_test_wf1` diagnostic selection rule; these numbers are controlled diagnostics, not validation-selected generalization estimates.

## MCA × AMM four-way control

| Dataset | MCA | AMM | WF1 (mean ± sample SD) |
| --- | --- | --- | ---: |
| IEMOCAP | on | on | 72.15 ± 0.21 |
| IEMOCAP | on | off | 72.00 ± 0.38 |
| IEMOCAP | off | on | 71.74 ± 0.34 |
| IEMOCAP | off | off | 71.54 ± 0.33 |
| MELD | on | on | 67.85 ± 0.06 |
| MELD | on | off | 67.80 ± 0.18 |
| MELD | off | on | 67.45 ± 0.37 |
| MELD | off | off | 67.86 ± 0.26 |

AMM contributes `+0.14 ± 0.19` WF1 points with MCA and `+0.20 ± 0.66` without MCA on IEMOCAP. On MELD, the corresponding effects are `+0.05 ± 0.23` and `-0.41 ± 0.55`. Thus, the four-way control does not support a stable standalone AMM gain on MELD; its behavior depends on the MCA path.

## FFN-width control

The default AMM uses width 1536. All other settings remain fixed.

| FFN width | IEMOCAP WF1 | MELD WF1 |
| ---: | ---: | ---: |
| 256 | 72.02 ± 0.43 | 67.49 ± 0.11 |
| 512 | 72.00 ± 0.40 | 67.46 ± 0.22 |
| 768 | 72.16 ± 0.54 | 67.47 ± 0.26 |
| 1536 (default) | 72.15 ± 0.21 | 67.85 ± 0.06 |

Increasing width above 1536 is reported separately in `results/amm_capacity_20261009`. The compact-width results show no monotonic capacity trend: width 768 is effectively tied with the default on IEMOCAP, whereas all three compact widths trail the default by about 0.36--0.39 points on MELD.

Machine-readable evidence is in `analysis.json`, `comparison.json`, `references.json`, and `state.json`.
