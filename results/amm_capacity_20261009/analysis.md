# MM-Mixer revision experiment results

Selection: `strict_peak_test_wf1`. Metrics below are percentages; paired differences are percentage points. Standard deviations use ddof=1. Partial groups are explicitly marked and are not three-seed results.

| Dataset | Variant | Verified seeds | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| iemocap | ffn_width_3072 | 3/3 complete | 71.82 ± 0.52 (n=3) | 71.72 ± 0.53 (n=3) | 71.10 ± 0.60 (n=3) |
| iemocap | ffn_width_6144 | 3/3 complete | 72.10 ± 0.35 (n=3) | 72.01 ± 0.40 (n=3) | 71.35 ± 0.28 (n=3) |
| meld | ffn_width_3072 | 3/3 complete | 67.82 ± 0.10 (n=3) | 68.66 ± 0.15 (n=3) | 52.69 ± 0.67 (n=3) |
| meld | ffn_width_6144 | 3/3 complete | 67.82 ± 0.16 (n=3) | 68.57 ± 0.17 (n=3) | 53.16 ± 0.41 (n=3) |

## Same-seed differences from Full

| Dataset | Variant | ΔWF1 | ΔACC | ΔMacro-F1 |
| --- | --- | --- | --- | --- |
| iemocap | ffn_width_3072 | — (n=0) | — (n=0) | — (n=0) |
| iemocap | ffn_width_6144 | — (n=0) | — (n=0) | — (n=0) |
| meld | ffn_width_3072 | — (n=0) | — (n=0) | — (n=0) |
| meld | ffn_width_6144 | — (n=0) | — (n=0) | — (n=0) |

## iemocap / ffn_width_3072

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.73 | 71.66 | 71.04 |
| 2066 | completed | 72.38 | 72.27 | 71.73 |
| 2118 | completed | 71.35 | 71.23 | 70.52 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.66 ± 1.65 (n=3) |
| sadness | 84.73 ± 0.41 (n=3) |
| neutral | 70.98 ± 0.58 (n=3) |
| anger | 72.05 ± 0.51 (n=3) |
| excited | 73.41 ± 0.93 (n=3) |
| frustration | 68.75 ± 0.30 (n=3) |

## iemocap / ffn_width_6144

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.86 | 71.72 | 71.10 |
| 2066 | completed | 72.50 | 72.46 | 71.65 |
| 2118 | completed | 71.95 | 71.84 | 71.31 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 57.24 ± 0.93 (n=3) |
| sadness | 84.31 ± 0.64 (n=3) |
| neutral | 71.53 ± 1.08 (n=3) |
| anger | 72.09 ± 0.55 (n=3) |
| excited | 73.95 ± 1.26 (n=3) |
| frustration | 69.00 ± 0.58 (n=3) |

## meld / ffn_width_3072

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.73 | 68.51 | 53.41 |
| 2028 | completed | 67.93 | 68.81 | 52.59 |
| 2069 | completed | 67.81 | 68.66 | 52.07 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.92 ± 0.13 (n=3) |
| surprise | 59.85 ± 1.09 (n=3) |
| fear | 28.00 ± 2.94 (n=3) |
| sadness | 44.65 ± 0.41 (n=3) |
| joy | 65.40 ± 0.75 (n=3) |
| disgust | 34.17 ± 4.94 (n=3) |
| anger | 55.83 ± 0.79 (n=3) |

## meld / ffn_width_6144

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.87 | 68.58 | 53.59 |
| 2028 | completed | 67.65 | 68.39 | 53.12 |
| 2069 | completed | 67.95 | 68.74 | 52.77 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.78 ± 0.25 (n=3) |
| surprise | 60.77 ± 0.69 (n=3) |
| fear | 31.27 ± 2.44 (n=3) |
| sadness | 43.04 ± 1.07 (n=3) |
| joy | 65.31 ± 0.13 (n=3) |
| disgust | 34.87 ± 1.70 (n=3) |
| anger | 56.05 ± 0.40 (n=3) |

Per-seed confusion counts and their elementwise mean/sample SD are retained in the JSON. Rows are true classes and columns are predictions; class order is explicit. Classes with no true examples remain missing in class-level averages. No statistical significance or improvement claims are inferred.
