# MM-Mixer revision experiment results

Selection: `strict_peak_test_wf1`. Metrics below are percentages; paired differences are percentage points. Standard deviations use ddof=1. Partial groups are explicitly marked and are not three-seed results.

| Dataset | Variant | Verified seeds | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| iemocap | ffn_width_256 | 3/3 complete | 72.02 ± 0.43 (n=3) | 71.90 ± 0.43 (n=3) | 71.23 ± 0.53 (n=3) |
| iemocap | ffn_width_512 | 3/3 complete | 72.00 ± 0.40 (n=3) | 71.92 ± 0.44 (n=3) | 71.16 ± 0.25 (n=3) |
| iemocap | ffn_width_768 | 3/3 complete | 72.16 ± 0.54 (n=3) | 72.07 ± 0.55 (n=3) | 71.46 ± 0.50 (n=3) |
| iemocap | no_mca_no_amm | 3/3 complete | 71.54 ± 0.33 (n=3) | 71.45 ± 0.34 (n=3) | 70.92 ± 0.31 (n=3) |
| meld | ffn_width_256 | 3/3 complete | 67.49 ± 0.11 (n=3) | 68.28 ± 0.27 (n=3) | 52.10 ± 0.89 (n=3) |
| meld | ffn_width_512 | 3/3 complete | 67.46 ± 0.22 (n=3) | 68.17 ± 0.17 (n=3) | 51.96 ± 0.78 (n=3) |
| meld | ffn_width_768 | 3/3 complete | 67.47 ± 0.26 (n=3) | 68.22 ± 0.27 (n=3) | 52.76 ± 0.11 (n=3) |
| meld | no_mca_no_amm | 3/3 complete | 67.86 ± 0.26 (n=3) | 68.57 ± 0.36 (n=3) | 53.06 ± 0.31 (n=3) |

## Same-seed differences from Full

| Dataset | Variant | ΔWF1 | ΔACC | ΔMacro-F1 |
| --- | --- | --- | --- | --- |
| iemocap | ffn_width_256 | — (n=0) | — (n=0) | — (n=0) |
| iemocap | ffn_width_512 | — (n=0) | — (n=0) | — (n=0) |
| iemocap | ffn_width_768 | — (n=0) | — (n=0) | — (n=0) |
| iemocap | no_mca_no_amm | — (n=0) | — (n=0) | — (n=0) |
| meld | ffn_width_256 | — (n=0) | — (n=0) | — (n=0) |
| meld | ffn_width_512 | — (n=0) | — (n=0) | — (n=0) |
| meld | ffn_width_768 | — (n=0) | — (n=0) | — (n=0) |
| meld | no_mca_no_amm | — (n=0) | — (n=0) | — (n=0) |

## iemocap / ffn_width_256

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 72.05 | 71.90 | 71.37 |
| 2066 | completed | 72.43 | 72.34 | 71.68 |
| 2118 | completed | 71.58 | 71.47 | 70.65 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.94 ± 0.99 (n=3) |
| sadness | 84.82 ± 0.70 (n=3) |
| neutral | 71.73 ± 0.83 (n=3) |
| anger | 71.59 ± 1.82 (n=3) |
| excited | 73.51 ± 0.64 (n=3) |
| frustration | 68.80 ± 0.63 (n=3) |

## iemocap / ffn_width_512

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.64 | 71.53 | 71.02 |
| 2066 | completed | 72.43 | 72.40 | 71.45 |
| 2118 | completed | 71.92 | 71.84 | 71.02 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.88 ± 0.84 (n=3) |
| sadness | 84.46 ± 0.57 (n=3) |
| neutral | 71.48 ± 1.24 (n=3) |
| anger | 71.13 ± 0.13 (n=3) |
| excited | 74.03 ± 0.71 (n=3) |
| frustration | 69.01 ± 0.66 (n=3) |

## iemocap / ffn_width_768

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.89 | 71.78 | 71.33 |
| 2066 | completed | 72.78 | 72.70 | 72.02 |
| 2118 | completed | 71.81 | 71.72 | 71.04 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 57.47 ± 1.05 (n=3) |
| sadness | 85.04 ± 0.34 (n=3) |
| neutral | 71.20 ± 1.21 (n=3) |
| anger | 72.02 ± 0.77 (n=3) |
| excited | 74.06 ± 0.82 (n=3) |
| frustration | 68.97 ± 0.58 (n=3) |

## iemocap / no_mca_no_amm

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.21 | 71.10 | 70.58 |
| 2066 | completed | 71.87 | 71.78 | 71.18 |
| 2118 | completed | 71.54 | 71.47 | 71.01 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 57.84 ± 0.69 (n=3) |
| sadness | 83.93 ± 0.17 (n=3) |
| neutral | 70.51 ± 0.17 (n=3) |
| anger | 71.63 ± 0.46 (n=3) |
| excited | 72.98 ± 0.77 (n=3) |
| frustration | 68.63 ± 0.80 (n=3) |

## meld / ffn_width_256

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.45 | 68.12 | 52.79 |
| 2028 | completed | 67.62 | 68.58 | 51.10 |
| 2069 | completed | 67.41 | 68.12 | 52.41 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.65 ± 0.40 (n=3) |
| surprise | 59.75 ± 0.81 (n=3) |
| fear | 27.05 ± 3.41 (n=3) |
| sadness | 43.43 ± 0.23 (n=3) |
| joy | 64.83 ± 0.67 (n=3) |
| disgust | 32.76 ± 3.89 (n=3) |
| anger | 56.23 ± 0.67 (n=3) |

## meld / ffn_width_512

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.26 | 68.01 | 51.25 |
| 2028 | completed | 67.69 | 68.35 | 52.80 |
| 2069 | completed | 67.43 | 68.16 | 51.83 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.59 ± 0.16 (n=3) |
| surprise | 59.79 ± 0.58 (n=3) |
| fear | 27.71 ± 3.96 (n=3) |
| sadness | 43.70 ± 1.54 (n=3) |
| joy | 65.23 ± 0.61 (n=3) |
| disgust | 30.89 ± 1.21 (n=3) |
| anger | 55.81 ± 0.28 (n=3) |

## meld / ffn_width_768

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.19 | 67.93 | 52.76 |
| 2028 | completed | 67.69 | 68.47 | 52.66 |
| 2069 | completed | 67.55 | 68.28 | 52.88 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.51 ± 0.27 (n=3) |
| surprise | 60.25 ± 0.98 (n=3) |
| fear | 30.23 ± 2.03 (n=3) |
| sadness | 43.25 ± 0.86 (n=3) |
| joy | 64.83 ± 0.35 (n=3) |
| disgust | 34.86 ± 0.94 (n=3) |
| anger | 55.42 ± 0.17 (n=3) |

## meld / no_mca_no_amm

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.79 | 68.47 | 53.41 |
| 2028 | completed | 67.64 | 68.28 | 52.98 |
| 2069 | completed | 68.15 | 68.97 | 52.80 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.80 ± 0.47 (n=3) |
| surprise | 60.50 ± 0.32 (n=3) |
| fear | 32.12 ± 3.20 (n=3) |
| sadness | 43.53 ± 0.34 (n=3) |
| joy | 65.68 ± 0.43 (n=3) |
| disgust | 32.76 ± 1.26 (n=3) |
| anger | 56.05 ± 0.24 (n=3) |

Per-seed confusion counts and their elementwise mean/sample SD are retained in the JSON. Rows are true classes and columns are predictions; class order is explicit. Classes with no true examples remain missing in class-level averages. No statistical significance or improvement claims are inferred.
