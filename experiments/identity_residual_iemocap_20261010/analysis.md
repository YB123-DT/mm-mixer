# MM-Mixer revision experiment results

Selection: `strict_peak_test_wf1`. Metrics below are percentages; paired differences are percentage points. Standard deviations use ddof=1. Partial groups are explicitly marked and are not three-seed results.

| Dataset | Variant | Verified seeds | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| iemocap | residual_no_adaptive_gating | 3/3 complete | 72.00 ± 0.11 (n=3) | 71.90 ± 0.11 (n=3) | 71.14 ± 0.08 (n=3) |
| iemocap | residual_no_cross_attention | 3/3 complete | 71.80 ± 0.36 (n=3) | 71.72 ± 0.37 (n=3) | 70.93 ± 0.28 (n=3) |
| iemocap | residual_no_feature_and_adaptive_gating | 3/3 complete | 71.91 ± 0.20 (n=3) | 71.82 ± 0.18 (n=3) | 71.04 ± 0.19 (n=3) |
| iemocap | residual_no_feature_gating | 3/3 complete | 72.04 ± 0.22 (n=3) | 71.94 ± 0.23 (n=3) | 71.18 ± 0.19 (n=3) |
| iemocap | residual_no_mixer | 3/3 complete | 71.34 ± 0.38 (n=3) | 71.23 ± 0.37 (n=3) | 70.52 ± 0.24 (n=3) |
| iemocap | residual_no_pairwise | 3/3 complete | 71.86 ± 0.17 (n=3) | 71.76 ± 0.19 (n=3) | 71.01 ± 0.18 (n=3) |

## Same-seed differences from Full

| Dataset | Variant | ΔWF1 | ΔACC | ΔMacro-F1 |
| --- | --- | --- | --- | --- |
| iemocap | residual_no_adaptive_gating | — (n=0) | — (n=0) | — (n=0) |
| iemocap | residual_no_cross_attention | — (n=0) | — (n=0) | — (n=0) |
| iemocap | residual_no_feature_and_adaptive_gating | — (n=0) | — (n=0) | — (n=0) |
| iemocap | residual_no_feature_gating | — (n=0) | — (n=0) | — (n=0) |
| iemocap | residual_no_mixer | — (n=0) | — (n=0) | — (n=0) |
| iemocap | residual_no_pairwise | — (n=0) | — (n=0) | — (n=0) |

## iemocap / residual_no_adaptive_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 72.05 | 71.97 | 71.19 |
| 2066 | completed | 71.88 | 71.78 | 71.05 |
| 2118 | completed | 72.08 | 71.97 | 71.18 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.27 ± 0.48 (n=3) |
| sadness | 84.56 ± 0.49 (n=3) |
| neutral | 71.40 ± 0.78 (n=3) |
| anger | 71.56 ± 0.37 (n=3) |
| excited | 73.73 ± 0.22 (n=3) |
| frustration | 69.33 ± 0.23 (n=3) |

## iemocap / residual_no_cross_attention

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 72.21 | 72.15 | 71.24 |
| 2066 | completed | 71.62 | 71.53 | 70.86 |
| 2118 | completed | 71.56 | 71.47 | 70.70 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.03 ± 0.90 (n=3) |
| sadness | 84.34 ± 1.02 (n=3) |
| neutral | 71.00 ± 1.05 (n=3) |
| anger | 71.06 ± 1.01 (n=3) |
| excited | 74.27 ± 0.68 (n=3) |
| frustration | 68.88 ± 0.48 (n=3) |

## iemocap / residual_no_feature_and_adaptive_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.78 | 71.72 | 70.88 |
| 2066 | completed | 71.81 | 71.72 | 70.98 |
| 2118 | completed | 72.14 | 72.03 | 71.25 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 56.22 ± 1.12 (n=3) |
| sadness | 84.28 ± 0.50 (n=3) |
| neutral | 71.53 ± 0.94 (n=3) |
| anger | 71.48 ± 0.58 (n=3) |
| excited | 73.43 ± 0.34 (n=3) |
| frustration | 69.27 ± 0.37 (n=3) |

## iemocap / residual_no_feature_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 72.28 | 72.21 | 71.39 |
| 2066 | completed | 71.87 | 71.78 | 71.04 |
| 2118 | completed | 71.96 | 71.84 | 71.10 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 55.92 ± 0.60 (n=3) |
| sadness | 84.60 ± 0.44 (n=3) |
| neutral | 71.37 ± 0.81 (n=3) |
| anger | 72.13 ± 0.69 (n=3) |
| excited | 73.51 ± 0.62 (n=3) |
| frustration | 69.53 ± 0.62 (n=3) |

## iemocap / residual_no_mixer

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 70.91 | 70.79 | 70.24 |
| 2066 | completed | 71.52 | 71.41 | 70.64 |
| 2118 | completed | 71.60 | 71.47 | 70.67 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 57.05 ± 0.95 (n=3) |
| sadness | 82.96 ± 0.48 (n=3) |
| neutral | 71.81 ± 1.00 (n=3) |
| anger | 70.42 ± 0.53 (n=3) |
| excited | 72.79 ± 0.83 (n=3) |
| frustration | 68.07 ± 0.57 (n=3) |

## iemocap / residual_no_pairwise

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 72.04 | 71.97 | 71.20 |
| 2066 | completed | 71.82 | 71.72 | 70.95 |
| 2118 | completed | 71.72 | 71.60 | 70.86 |

Seed 2025 pairing: same-seed verified Full unavailable in this plan.

Seed 2066 pairing: same-seed verified Full unavailable in this plan.

Seed 2118 pairing: same-seed verified Full unavailable in this plan.

| Class | F1 |
| --- | --- |
| happiness | 55.81 ± 0.64 (n=3) |
| sadness | 84.44 ± 0.59 (n=3) |
| neutral | 71.24 ± 0.81 (n=3) |
| anger | 71.80 ± 0.61 (n=3) |
| excited | 73.72 ± 0.89 (n=3) |
| frustration | 69.04 ± 0.99 (n=3) |

Per-seed confusion counts and their elementwise mean/sample SD are retained in the JSON. Rows are true classes and columns are predictions; class order is explicit. Classes with no true examples remain missing in class-level averages. No statistical significance or improvement claims are inferred.
