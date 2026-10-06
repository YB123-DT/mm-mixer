# MM-Mixer revision experiment results

Selection: `strict_peak_test_wf1`. Metrics below are percentages; paired differences are percentage points. Standard deviations use ddof=1. Partial groups are explicitly marked and are not three-seed results.

| Dataset | Variant | Verified seeds | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| iemocap | projection_views_2 | 3/3 complete | 71.53 ± 0.28 (n=3) | 71.39 ± 0.23 (n=3) | 70.74 ± 0.29 (n=3) |
| iemocap | projection_views_4 | 3/3 complete | 71.77 ± 0.19 (n=3) | 71.68 ± 0.23 (n=3) | 70.98 ± 0.07 (n=3) |
| iemocap | projection_views_8 | 3/3 complete | 71.90 ± 0.22 (n=3) | 71.80 ± 0.18 (n=3) | 71.16 ± 0.23 (n=3) |
| meld | projection_views_2 | 3/3 complete | 67.84 ± 0.10 (n=3) | 68.51 ± 0.08 (n=3) | 53.48 ± 0.66 (n=3) |
| meld | projection_views_4 | 3/3 complete | 67.76 ± 0.24 (n=3) | 68.52 ± 0.32 (n=3) | 53.19 ± 0.58 (n=3) |
| meld | projection_views_8 | 3/3 complete | 67.69 ± 0.13 (n=3) | 68.48 ± 0.10 (n=3) | 52.73 ± 0.23 (n=3) |

## Same-seed differences from Full

| Dataset | Variant | ΔWF1 | ΔACC | ΔMacro-F1 |
| --- | --- | --- | --- | --- |
| iemocap | projection_views_2 | — (n=0) | — (n=0) | — (n=0) |
| iemocap | projection_views_4 | — (n=0) | — (n=0) | — (n=0) |
| iemocap | projection_views_8 | — (n=0) | — (n=0) | — (n=0) |
| meld | projection_views_2 | — (n=0) | — (n=0) | — (n=0) |
| meld | projection_views_4 | — (n=0) | — (n=0) | — (n=0) |
| meld | projection_views_8 | — (n=0) | — (n=0) | — (n=0) |

## iemocap / projection_views_2

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.35 | 71.23 | 70.45 |
| 2066 | completed | 71.38 | 71.29 | 70.74 |
| 2118 | completed | 71.85 | 71.66 | 71.03 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.45 ± 0.69 (n=3) |
| sadness | 84.05 ± 0.10 (n=3) |
| neutral | 70.66 ± 0.76 (n=3) |
| anger | 71.28 ± 0.86 (n=3) |
| excited | 73.00 ± 0.60 (n=3) |
| frustration | 69.02 ± 0.20 (n=3) |

## iemocap / projection_views_4

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.55 | 71.41 | 70.94 |
| 2066 | completed | 71.91 | 71.84 | 71.06 |
| 2118 | completed | 71.85 | 71.78 | 70.95 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.93 ± 1.44 (n=3) |
| sadness | 84.05 ± 0.61 (n=3) |
| neutral | 71.09 ± 0.53 (n=3) |
| anger | 71.28 ± 0.57 (n=3) |
| excited | 73.60 ± 0.43 (n=3) |
| frustration | 68.95 ± 0.54 (n=3) |

## iemocap / projection_views_8

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.64 | 71.60 | 70.90 |
| 2066 | completed | 72.02 | 71.90 | 71.28 |
| 2118 | completed | 72.03 | 71.90 | 71.31 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.95 ± 0.25 (n=3) |
| sadness | 84.65 ± 0.23 (n=3) |
| neutral | 70.82 ± 0.40 (n=3) |
| anger | 71.82 ± 0.31 (n=3) |
| excited | 73.65 ± 0.77 (n=3) |
| frustration | 69.09 ± 0.34 (n=3) |

## meld / projection_views_2

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.90 | 68.43 | 54.18 |
| 2028 | completed | 67.89 | 68.58 | 52.86 |
| 2069 | completed | 67.72 | 68.51 | 53.41 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.68 ± 0.25 (n=3) |
| surprise | 60.01 ± 0.05 (n=3) |
| fear | 32.16 ± 2.41 (n=3) |
| sadness | 44.52 ± 0.97 (n=3) |
| joy | 65.23 ± 0.30 (n=3) |
| disgust | 35.68 ± 3.30 (n=3) |
| anger | 56.10 ± 0.42 (n=3) |

## meld / projection_views_4

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.54 | 68.16 | 52.69 |
| 2028 | completed | 68.02 | 68.77 | 53.82 |
| 2069 | completed | 67.71 | 68.62 | 53.05 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.71 ± 0.13 (n=3) |
| surprise | 59.69 ± 0.32 (n=3) |
| fear | 31.32 ± 1.55 (n=3) |
| sadness | 44.40 ± 1.18 (n=3) |
| joy | 65.06 ± 0.66 (n=3) |
| disgust | 35.01 ± 1.10 (n=3) |
| anger | 56.12 ± 0.28 (n=3) |

## meld / projection_views_8

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.54 | 68.39 | 52.54 |
| 2028 | completed | 67.80 | 68.58 | 52.99 |
| 2069 | completed | 67.73 | 68.47 | 52.66 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2028 pairing: same-seed verified Full unavailable in this plan.

Seed 2069 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| neutral | 80.78 ± 0.23 (n=3) |
| surprise | 60.56 ± 0.66 (n=3) |
| fear | 29.83 ± 1.16 (n=3) |
| sadness | 43.90 ± 0.93 (n=3) |
| joy | 65.05 ± 0.02 (n=3) |
| disgust | 33.48 ± 2.60 (n=3) |
| anger | 55.51 ± 0.64 (n=3) |

Per-seed confusion counts and their elementwise mean/sample SD are retained in the JSON. Rows are true classes and columns are predictions; class order is explicit. Classes with no true examples remain missing in class-level averages. No statistical significance or improvement claims are inferred.
