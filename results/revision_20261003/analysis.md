# MM-Mixer revision experiment results

Selection: `strict_peak_test_wf1`. Metrics below are percentages; paired differences are percentage points. Standard deviations use ddof=1. Partial groups are explicitly marked and are not three-seed results.

| Dataset | Variant | Verified seeds | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| iemocap | amm_attention | 3/3 complete | 72.04 ± 0.19 (n=3) | 71.92 ± 0.20 (n=3) | 71.36 ± 0.18 (n=3) |
| iemocap | amm_cubemlp | 3/3 complete | 71.95 ± 0.34 (n=3) | 71.82 ± 0.35 (n=3) | 71.16 ± 0.21 (n=3) |
| iemocap | amm_mlp | 3/3 complete | 71.83 ± 0.49 (n=3) | 71.72 ± 0.56 (n=3) | 71.05 ± 0.45 (n=3) |
| iemocap | amm_mlp_no_aux | 3/3 complete | 71.52 ± 0.23 (n=3) | 71.45 ± 0.20 (n=3) | 70.62 ± 0.31 (n=3) |
| iemocap | full | 3/3 complete | 72.15 ± 0.21 (n=3) | 72.03 ± 0.25 (n=3) | 71.37 ± 0.11 (n=3) |
| iemocap | modal_a | 3/3 complete | 54.74 ± 0.65 (n=3) | 54.82 ± 0.49 (n=3) | 53.36 ± 0.84 (n=3) |
| iemocap | modal_av | 3/3 complete | 56.88 ± 0.56 (n=3) | 56.83 ± 0.52 (n=3) | 55.39 ± 0.47 (n=3) |
| iemocap | modal_t | 3/3 complete | 68.06 ± 0.25 (n=3) | 67.98 ± 0.32 (n=3) | 66.82 ± 0.10 (n=3) |
| iemocap | modal_ta | 3/3 complete | 70.60 ± 0.26 (n=3) | 70.53 ± 0.26 (n=3) | 69.91 ± 0.26 (n=3) |
| iemocap | modal_tv | 3/3 complete | 68.70 ± 0.06 (n=3) | 68.60 ± 0.04 (n=3) | 67.65 ± 0.20 (n=3) |
| iemocap | modal_v | 3/3 complete | 33.89 ± 0.66 (n=3) | 34.67 ± 0.68 (n=3) | 32.46 ± 0.59 (n=3) |
| iemocap | no_adaptive_gating | 3/3 complete | 71.71 ± 0.48 (n=3) | 71.58 ± 0.51 (n=3) | 70.95 ± 0.43 (n=3) |
| iemocap | no_auxiliary_loss | 3/3 complete | 71.29 ± 0.41 (n=3) | 71.21 ± 0.48 (n=3) | 70.46 ± 0.28 (n=3) |
| iemocap | no_cross_attention | 3/3 complete | 71.74 ± 0.34 (n=3) | 71.60 ± 0.28 (n=3) | 70.96 ± 0.24 (n=3) |
| iemocap | no_feature_and_adaptive_gating | 3/3 complete | 71.54 ± 0.37 (n=3) | 71.41 ± 0.40 (n=3) | 70.79 ± 0.30 (n=3) |
| iemocap | no_feature_gating | 3/3 complete | 71.85 ± 0.06 (n=3) | 71.72 ± 0.06 (n=3) | 71.08 ± 0.09 (n=3) |
| iemocap | no_feature_mixing | 3/3 complete | 71.95 ± 0.13 (n=3) | 71.82 ± 0.16 (n=3) | 71.19 ± 0.14 (n=3) |
| iemocap | no_mixer | 3/3 complete | 72.00 ± 0.38 (n=3) | 71.88 ± 0.41 (n=3) | 71.21 ± 0.33 (n=3) |
| iemocap | no_modality_mixing | 3/3 complete | 71.99 ± 0.23 (n=3) | 71.86 ± 0.26 (n=3) | 71.27 ± 0.24 (n=3) |
| iemocap | no_pairwise | 3/3 complete | 72.18 ± 0.16 (n=3) | 72.05 ± 0.19 (n=3) | 71.39 ± 0.07 (n=3) |
| iemocap | no_sequence_mixing | 3/3 complete | 71.57 ± 0.12 (n=3) | 71.43 ± 0.13 (n=3) | 70.74 ± 0.03 (n=3) |
| iemocap | one_mixer_block | 3/3 complete | 71.70 ± 0.44 (n=3) | 71.55 ± 0.45 (n=3) | 70.98 ± 0.54 (n=3) |
| iemocap | pairwise_mlp_residual | 3/3 complete | 71.75 ± 0.53 (n=3) | 71.64 ± 0.55 (n=3) | 71.02 ± 0.51 (n=3) |
| iemocap | single_projection_view | 3/3 complete | 71.71 ± 0.41 (n=3) | 71.58 ± 0.40 (n=3) | 70.98 ± 0.52 (n=3) |
| meld | amm_attention | 3/3 complete | 67.49 ± 0.07 (n=3) | 68.42 ± 0.15 (n=3) | 51.73 ± 0.92 (n=3) |
| meld | amm_cubemlp | 3/3 complete | 67.85 ± 0.10 (n=3) | 68.48 ± 0.15 (n=3) | 53.09 ± 0.39 (n=3) |
| meld | amm_mlp | 3/3 complete | 67.56 ± 0.14 (n=3) | 68.33 ± 0.15 (n=3) | 52.63 ± 0.95 (n=3) |
| meld | amm_mlp_no_aux | 3/3 complete | 67.49 ± 0.06 (n=3) | 68.33 ± 0.26 (n=3) | 50.99 ± 1.34 (n=3) |
| meld | full | 3/3 complete | 67.85 ± 0.06 (n=3) | 68.57 ± 0.06 (n=3) | 53.45 ± 0.26 (n=3) |
| meld | modal_a | 3/3 complete | 48.45 ± 0.18 (n=3) | 52.53 ± 0.11 (n=3) | 27.94 ± 0.24 (n=3) |
| meld | modal_av | 3/3 complete | 48.86 ± 0.24 (n=3) | 52.69 ± 0.06 (n=3) | 28.32 ± 0.44 (n=3) |
| meld | modal_t | 3/3 complete | 67.50 ± 0.21 (n=3) | 68.30 ± 0.13 (n=3) | 52.61 ± 0.41 (n=3) |
| meld | modal_ta | 3/3 complete | 67.88 ± 0.17 (n=3) | 68.66 ± 0.17 (n=3) | 53.52 ± 0.52 (n=3) |
| meld | modal_tv | 3/3 complete | 67.35 ± 0.06 (n=3) | 68.11 ± 0.09 (n=3) | 53.15 ± 0.24 (n=3) |
| meld | modal_v | 3/3 complete | 31.71 ± 0.77 (n=3) | 46.05 ± 3.58 (n=3) | 10.41 ± 1.95 (n=3) |
| meld | no_adaptive_gating | 3/3 complete | 67.80 ± 0.18 (n=3) | 68.54 ± 0.14 (n=3) | 53.02 ± 0.32 (n=3) |
| meld | no_auxiliary_loss | 3/3 complete | 67.51 ± 0.13 (n=3) | 68.26 ± 0.02 (n=3) | 52.18 ± 0.61 (n=3) |
| meld | no_cross_attention | 3/3 complete | 67.45 ± 0.37 (n=3) | 68.16 ± 0.23 (n=3) | 52.79 ± 1.34 (n=3) |
| meld | no_feature_and_adaptive_gating | 3/3 complete | 67.95 ± 0.05 (n=3) | 68.65 ± 0.08 (n=3) | 53.66 ± 0.23 (n=3) |
| meld | no_feature_gating | 3/3 complete | 67.83 ± 0.09 (n=3) | 68.56 ± 0.06 (n=3) | 53.13 ± 0.80 (n=3) |
| meld | no_feature_mixing | 3/3 complete | 67.87 ± 0.20 (n=3) | 68.63 ± 0.15 (n=3) | 53.25 ± 0.90 (n=3) |
| meld | no_mixer | 3/3 complete | 67.80 ± 0.18 (n=3) | 68.56 ± 0.12 (n=3) | 52.83 ± 0.38 (n=3) |
| meld | no_modality_mixing | 3/3 complete | 67.72 ± 0.08 (n=3) | 68.44 ± 0.12 (n=3) | 53.01 ± 0.22 (n=3) |
| meld | no_pairwise | 3/3 complete | 67.85 ± 0.06 (n=3) | 68.57 ± 0.06 (n=3) | 53.45 ± 0.26 (n=3) |
| meld | no_sequence_mixing | 3/3 complete | 67.70 ± 0.05 (n=3) | 68.45 ± 0.06 (n=3) | 53.05 ± 0.34 (n=3) |
| meld | one_mixer_block | 3/3 complete | 67.70 ± 0.14 (n=3) | 68.42 ± 0.15 (n=3) | 52.86 ± 1.01 (n=3) |
| meld | pairwise_mlp_residual | 3/3 complete | 67.58 ± 0.30 (n=3) | 68.35 ± 0.33 (n=3) | 52.68 ± 0.50 (n=3) |
| meld | single_projection_view | 3/3 complete | 67.78 ± 0.02 (n=3) | 68.47 ± 0.10 (n=3) | 53.06 ± 0.14 (n=3) |

## Same-seed differences from Full

| Dataset | Variant | ΔWF1 | ΔACC | ΔMacro-F1 |
| --- | --- | --- | --- | --- |
| iemocap | amm_attention | -0.10 ± 0.25 (n=3) | -0.10 ± 0.29 (n=3) | -0.02 ± 0.23 (n=3) |
| iemocap | amm_cubemlp | -0.20 ± 0.33 (n=3) | -0.21 ± 0.37 (n=3) | -0.21 ± 0.21 (n=3) |
| iemocap | amm_mlp | -0.31 ± 0.29 (n=3) | -0.31 ± 0.33 (n=3) | -0.32 ± 0.34 (n=3) |
| iemocap | amm_mlp_no_aux | -0.62 ± 0.07 (n=3) | -0.58 ± 0.09 (n=3) | -0.75 ± 0.26 (n=3) |
| iemocap | modal_a | -17.41 ± 0.53 (n=3) | -17.21 ± 0.41 (n=3) | -18.01 ± 0.74 (n=3) |
| iemocap | modal_av | -15.27 ± 0.40 (n=3) | -15.20 ± 0.30 (n=3) | -15.99 ± 0.36 (n=3) |
| iemocap | modal_t | -4.09 ± 0.46 (n=3) | -4.05 ± 0.56 (n=3) | -4.55 ± 0.21 (n=3) |
| iemocap | modal_ta | -1.54 ± 0.37 (n=3) | -1.50 ± 0.40 (n=3) | -1.46 ± 0.26 (n=3) |
| iemocap | modal_tv | -3.45 ± 0.20 (n=3) | -3.43 ± 0.25 (n=3) | -3.73 ± 0.20 (n=3) |
| iemocap | modal_v | -38.25 ± 0.86 (n=3) | -37.36 ± 0.91 (n=3) | -38.91 ± 0.68 (n=3) |
| iemocap | no_adaptive_gating | -0.43 ± 0.38 (n=3) | -0.45 ± 0.38 (n=3) | -0.42 ± 0.36 (n=3) |
| iemocap | no_auxiliary_loss | -0.85 ± 0.53 (n=3) | -0.82 ± 0.59 (n=3) | -0.92 ± 0.34 (n=3) |
| iemocap | no_cross_attention | -0.40 ± 0.54 (n=3) | -0.43 ± 0.53 (n=3) | -0.41 ± 0.34 (n=3) |
| iemocap | no_feature_and_adaptive_gating | -0.60 ± 0.17 (n=3) | -0.62 ± 0.16 (n=3) | -0.58 ± 0.19 (n=3) |
| iemocap | no_feature_gating | -0.30 ± 0.26 (n=3) | -0.31 ± 0.28 (n=3) | -0.29 ± 0.19 (n=3) |
| iemocap | no_feature_mixing | -0.20 ± 0.20 (n=3) | -0.21 ± 0.23 (n=3) | -0.18 ± 0.17 (n=3) |
| iemocap | no_mixer | -0.14 ± 0.19 (n=3) | -0.14 ± 0.18 (n=3) | -0.16 ± 0.22 (n=3) |
| iemocap | no_modality_mixing | -0.15 ± 0.19 (n=3) | -0.16 ± 0.19 (n=3) | -0.10 ± 0.22 (n=3) |
| iemocap | no_pairwise | +0.03 ± 0.06 (n=3) | +0.02 ± 0.07 (n=3) | +0.02 ± 0.10 (n=3) |
| iemocap | no_sequence_mixing | -0.57 ± 0.11 (n=3) | -0.60 ± 0.13 (n=3) | -0.64 ± 0.10 (n=3) |
| iemocap | one_mixer_block | -0.45 ± 0.49 (n=3) | -0.47 ± 0.49 (n=3) | -0.40 ± 0.54 (n=3) |
| iemocap | pairwise_mlp_residual | -0.39 ± 0.39 (n=3) | -0.39 ± 0.37 (n=3) | -0.35 ± 0.43 (n=3) |
| iemocap | single_projection_view | -0.44 ± 0.20 (n=3) | -0.45 ± 0.16 (n=3) | -0.39 ± 0.42 (n=3) |
| meld | amm_attention | -0.36 ± 0.10 (n=3) | -0.15 ± 0.20 (n=3) | -1.72 ± 0.96 (n=3) |
| meld | amm_cubemlp | -0.00 ± 0.16 (n=3) | -0.09 ± 0.21 (n=3) | -0.36 ± 0.17 (n=3) |
| meld | amm_mlp | -0.29 ± 0.20 (n=3) | -0.24 ± 0.21 (n=3) | -0.83 ± 0.70 (n=3) |
| meld | amm_mlp_no_aux | -0.36 ± 0.06 (n=3) | -0.24 ± 0.31 (n=3) | -2.47 ± 1.26 (n=3) |
| meld | modal_a | -19.40 ± 0.24 (n=3) | -16.04 ± 0.17 (n=3) | -25.51 ± 0.45 (n=3) |
| meld | modal_av | -18.99 ± 0.30 (n=3) | -15.87 ± 0.12 (n=3) | -25.13 ± 0.65 (n=3) |
| meld | modal_t | -0.35 ± 0.22 (n=3) | -0.27 ± 0.17 (n=3) | -0.85 ± 0.22 (n=3) |
| meld | modal_ta | +0.03 ± 0.14 (n=3) | +0.09 ± 0.12 (n=3) | +0.06 ± 0.77 (n=3) |
| meld | modal_tv | -0.50 ± 0.10 (n=3) | -0.46 ± 0.11 (n=3) | -0.31 ± 0.05 (n=3) |
| meld | modal_v | -36.14 ± 0.74 (n=3) | -22.52 ± 3.60 (n=3) | -43.05 ± 1.69 (n=3) |
| meld | no_adaptive_gating | -0.05 ± 0.22 (n=3) | -0.03 ± 0.17 (n=3) | -0.44 ± 0.58 (n=3) |
| meld | no_auxiliary_loss | -0.34 ± 0.07 (n=3) | -0.31 ± 0.08 (n=3) | -1.28 ± 0.47 (n=3) |
| meld | no_cross_attention | -0.40 ± 0.36 (n=3) | -0.41 ± 0.23 (n=3) | -0.67 ± 1.10 (n=3) |
| meld | no_feature_and_adaptive_gating | +0.10 ± 0.09 (n=3) | +0.08 ± 0.07 (n=3) | +0.21 ± 0.35 (n=3) |
| meld | no_feature_gating | -0.02 ± 0.04 (n=3) | -0.01 ± 0.02 (n=3) | -0.32 ± 0.64 (n=3) |
| meld | no_feature_mixing | +0.02 ± 0.25 (n=3) | +0.06 ± 0.21 (n=3) | -0.20 ± 0.79 (n=3) |
| meld | no_mixer | -0.05 ± 0.23 (n=3) | -0.01 ± 0.18 (n=3) | -0.62 ± 0.17 (n=3) |
| meld | no_modality_mixing | -0.13 ± 0.14 (n=3) | -0.13 ± 0.18 (n=3) | -0.44 ± 0.19 (n=3) |
| meld | no_pairwise | +0.00 ± 0.00 (n=3) | +0.00 ± 0.00 (n=3) | -0.01 ± 0.01 (n=3) |
| meld | no_sequence_mixing | -0.15 ± 0.08 (n=3) | -0.11 ± 0.08 (n=3) | -0.40 ± 0.37 (n=3) |
| meld | one_mixer_block | -0.15 ± 0.10 (n=3) | -0.15 ± 0.11 (n=3) | -0.60 ± 0.85 (n=3) |
| meld | pairwise_mlp_residual | -0.27 ± 0.35 (n=3) | -0.22 ± 0.37 (n=3) | -0.77 ± 0.72 (n=3) |
| meld | single_projection_view | -0.07 ± 0.04 (n=3) | -0.10 ± 0.13 (n=3) | -0.39 ± 0.21 (n=3) |

## iemocap / amm_attention

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.91 | 71.78 | 71.26 |
| 2066 | completed | 71.96 | 71.84 | 71.25 |
| 2118 | completed | 72.26 | 72.15 | 71.57 |

| Class | F1 |
| --- | --- |
| happiness | 57.67 ± 0.51 (n=3) |
| sadness | 84.17 ± 0.22 (n=3) |
| neutral | 71.20 ± 0.52 (n=3) |
| anger | 72.24 ± 0.46 (n=3) |
| excited | 73.84 ± 0.46 (n=3) |
| frustration | 69.02 ± 0.17 (n=3) |

## iemocap / amm_cubemlp

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.66 | 71.53 | 70.96 |
| 2066 | completed | 71.86 | 71.72 | 71.13 |
| 2118 | completed | 72.32 | 72.21 | 71.38 |

| Class | F1 |
| --- | --- |
| happiness | 57.41 ± 1.34 (n=3) |
| sadness | 84.34 ± 0.45 (n=3) |
| neutral | 71.20 ± 0.49 (n=3) |
| anger | 71.13 ± 0.69 (n=3) |
| excited | 73.48 ± 0.91 (n=3) |
| frustration | 69.39 ± 0.51 (n=3) |

## iemocap / amm_mlp

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.40 | 71.23 | 70.74 |
| 2066 | completed | 72.36 | 72.34 | 71.57 |
| 2118 | completed | 71.75 | 71.60 | 70.85 |

| Class | F1 |
| --- | --- |
| happiness | 56.26 ± 0.53 (n=3) |
| sadness | 84.69 ± 0.50 (n=3) |
| neutral | 70.98 ± 0.88 (n=3) |
| anger | 72.08 ± 0.54 (n=3) |
| excited | 73.00 ± 0.60 (n=3) |
| frustration | 69.29 ± 0.72 (n=3) |

## iemocap / amm_mlp_no_aux

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.26 | 71.23 | 70.28 |
| 2066 | completed | 71.69 | 71.60 | 70.71 |
| 2118 | completed | 71.62 | 71.53 | 70.88 |

| Class | F1 |
| --- | --- |
| happiness | 55.40 ± 0.38 (n=3) |
| sadness | 84.34 ± 0.58 (n=3) |
| neutral | 70.55 ± 0.72 (n=3) |
| anger | 70.87 ± 1.05 (n=3) |
| excited | 73.33 ± 0.46 (n=3) |
| frustration | 69.22 ± 0.58 (n=3) |

## iemocap / full

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.93 | 71.78 | 71.28 |
| 2066 | completed | 72.35 | 72.27 | 71.49 |
| 2118 | completed | 72.17 | 72.03 | 71.35 |

| Class | F1 |
| --- | --- |
| happiness | 57.33 ± 0.68 (n=3) |
| sadness | 84.59 ± 0.87 (n=3) |
| neutral | 71.69 ± 0.65 (n=3) |
| anger | 71.56 ± 1.25 (n=3) |
| excited | 74.27 ± 0.97 (n=3) |
| frustration | 68.81 ± 0.32 (n=3) |

## iemocap / modal_a

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 54.51 | 54.78 | 52.98 |
| 2066 | completed | 55.47 | 55.33 | 54.32 |
| 2118 | completed | 54.23 | 54.34 | 52.78 |

| Class | F1 |
| --- | --- |
| happiness | 29.82 ± 2.97 (n=3) |
| sadness | 67.34 ± 1.16 (n=3) |
| neutral | 57.53 ± 0.43 (n=3) |
| anger | 58.79 ± 0.78 (n=3) |
| excited | 57.08 ± 0.20 (n=3) |
| frustration | 49.59 ± 0.79 (n=3) |

## iemocap / modal_av

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 56.54 | 56.44 | 55.04 |
| 2066 | completed | 57.52 | 57.42 | 55.92 |
| 2118 | completed | 56.56 | 56.62 | 55.19 |

| Class | F1 |
| --- | --- |
| happiness | 36.62 ± 0.92 (n=3) |
| sadness | 69.43 ± 1.55 (n=3) |
| neutral | 59.33 ± 1.77 (n=3) |
| anger | 54.76 ± 0.94 (n=3) |
| excited | 58.61 ± 1.49 (n=3) |
| frustration | 53.57 ± 1.63 (n=3) |

## iemocap / modal_t

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 68.33 | 68.33 | 66.89 |
| 2066 | completed | 67.84 | 67.71 | 66.71 |
| 2118 | completed | 67.99 | 67.90 | 66.86 |

| Class | F1 |
| --- | --- |
| happiness | 48.91 ± 2.03 (n=3) |
| sadness | 84.33 ± 0.20 (n=3) |
| neutral | 68.87 ± 0.84 (n=3) |
| anger | 65.61 ± 0.94 (n=3) |
| excited | 68.24 ± 0.85 (n=3) |
| frustration | 64.95 ± 0.73 (n=3) |

## iemocap / modal_ta

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 70.78 | 70.73 | 70.06 |
| 2066 | completed | 70.72 | 70.61 | 70.05 |
| 2118 | completed | 70.30 | 70.24 | 69.61 |

| Class | F1 |
| --- | --- |
| happiness | 57.08 ± 0.14 (n=3) |
| sadness | 84.49 ± 0.28 (n=3) |
| neutral | 71.16 ± 0.53 (n=3) |
| anger | 68.69 ± 0.71 (n=3) |
| excited | 72.18 ± 0.60 (n=3) |
| frustration | 65.84 ± 0.94 (n=3) |

## iemocap / modal_tv

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 68.65 | 68.58 | 67.45 |
| 2066 | completed | 68.68 | 68.58 | 67.63 |
| 2118 | completed | 68.76 | 68.64 | 67.86 |

| Class | F1 |
| --- | --- |
| happiness | 51.14 ± 1.41 (n=3) |
| sadness | 84.00 ± 0.82 (n=3) |
| neutral | 67.75 ± 0.75 (n=3) |
| anger | 66.47 ± 0.58 (n=3) |
| excited | 70.39 ± 0.99 (n=3) |
| frustration | 66.13 ± 0.09 (n=3) |

## iemocap / modal_v

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 34.63 | 35.43 | 33.14 |
| 2066 | completed | 33.38 | 34.13 | 32.06 |
| 2118 | completed | 33.67 | 34.44 | 32.20 |

| Class | F1 |
| --- | --- |
| happiness | 33.24 ± 2.21 (n=3) |
| sadness | 38.47 ± 0.51 (n=3) |
| neutral | 30.81 ± 1.78 (n=3) |
| anger | 12.05 ± 1.29 (n=3) |
| excited | 44.42 ± 1.54 (n=3) |
| frustration | 35.80 ± 0.58 (n=3) |

## iemocap / no_adaptive_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.59 | 71.41 | 70.92 |
| 2066 | completed | 72.24 | 72.15 | 71.40 |
| 2118 | completed | 71.31 | 71.16 | 70.54 |

| Class | F1 |
| --- | --- |
| happiness | 57.18 ± 0.37 (n=3) |
| sadness | 84.35 ± 0.25 (n=3) |
| neutral | 71.61 ± 1.08 (n=3) |
| anger | 70.90 ± 0.95 (n=3) |
| excited | 73.67 ± 1.20 (n=3) |
| frustration | 68.01 ± 0.19 (n=3) |

## iemocap / no_auxiliary_loss

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.65 | 71.60 | 70.75 |
| 2066 | completed | 71.38 | 71.35 | 70.43 |
| 2118 | completed | 70.85 | 70.67 | 70.19 |

| Class | F1 |
| --- | --- |
| happiness | 55.06 ± 1.17 (n=3) |
| sadness | 84.66 ± 0.43 (n=3) |
| neutral | 70.24 ± 1.18 (n=3) |
| anger | 71.26 ± 0.30 (n=3) |
| excited | 72.77 ± 0.36 (n=3) |
| frustration | 68.76 ± 0.78 (n=3) |

## iemocap / no_cross_attention

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 72.01 | 71.84 | 71.23 |
| 2066 | completed | 71.36 | 71.29 | 70.77 |
| 2118 | completed | 71.85 | 71.66 | 70.90 |

| Class | F1 |
| --- | --- |
| happiness | 55.85 ± 0.76 (n=3) |
| sadness | 85.18 ± 0.61 (n=3) |
| neutral | 70.66 ± 1.37 (n=3) |
| anger | 72.04 ± 0.27 (n=3) |
| excited | 72.90 ± 1.10 (n=3) |
| frustration | 69.16 ± 0.92 (n=3) |

## iemocap / no_feature_and_adaptive_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.21 | 71.04 | 70.53 |
| 2066 | completed | 71.94 | 71.84 | 71.11 |
| 2118 | completed | 71.48 | 71.35 | 70.72 |

| Class | F1 |
| --- | --- |
| happiness | 57.63 ± 1.23 (n=3) |
| sadness | 83.71 ± 0.18 (n=3) |
| neutral | 71.51 ± 1.04 (n=3) |
| anger | 70.10 ± 1.41 (n=3) |
| excited | 74.25 ± 0.80 (n=3) |
| frustration | 67.54 ± 0.09 (n=3) |

## iemocap / no_feature_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.87 | 71.72 | 71.18 |
| 2066 | completed | 71.78 | 71.66 | 71.03 |
| 2118 | completed | 71.90 | 71.78 | 71.04 |

| Class | F1 |
| --- | --- |
| happiness | 57.11 ± 0.38 (n=3) |
| sadness | 84.65 ± 0.29 (n=3) |
| neutral | 71.21 ± 0.43 (n=3) |
| anger | 71.02 ± 1.52 (n=3) |
| excited | 73.98 ± 0.62 (n=3) |
| frustration | 68.53 ± 0.39 (n=3) |

## iemocap / no_feature_mixing

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.96 | 71.84 | 71.29 |
| 2066 | completed | 72.08 | 71.97 | 71.27 |
| 2118 | completed | 71.81 | 71.66 | 71.03 |

| Class | F1 |
| --- | --- |
| happiness | 57.26 ± 0.42 (n=3) |
| sadness | 84.40 ± 0.32 (n=3) |
| neutral | 71.24 ± 0.57 (n=3) |
| anger | 71.48 ± 1.28 (n=3) |
| excited | 73.90 ± 0.46 (n=3) |
| frustration | 68.88 ± 0.36 (n=3) |

## iemocap / no_mixer

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.67 | 71.53 | 70.91 |
| 2066 | completed | 72.42 | 72.34 | 71.57 |
| 2118 | completed | 71.92 | 71.78 | 71.15 |

| Class | F1 |
| --- | --- |
| happiness | 56.93 ± 0.62 (n=3) |
| sadness | 84.06 ± 0.82 (n=3) |
| neutral | 71.45 ± 0.49 (n=3) |
| anger | 72.01 ± 0.80 (n=3) |
| excited | 73.42 ± 1.26 (n=3) |
| frustration | 69.39 ± 0.38 (n=3) |

## iemocap / no_modality_mixing

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.92 | 71.78 | 71.35 |
| 2066 | completed | 72.25 | 72.15 | 71.46 |
| 2118 | completed | 71.80 | 71.66 | 71.00 |

| Class | F1 |
| --- | --- |
| happiness | 57.53 ± 0.46 (n=3) |
| sadness | 84.47 ± 0.36 (n=3) |
| neutral | 71.24 ± 0.53 (n=3) |
| anger | 71.72 ± 1.27 (n=3) |
| excited | 73.62 ± 0.66 (n=3) |
| frustration | 69.03 ± 0.03 (n=3) |

## iemocap / no_pairwise

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.99 | 71.84 | 71.32 |
| 2066 | completed | 72.30 | 72.21 | 71.39 |
| 2118 | completed | 72.23 | 72.09 | 71.45 |

| Class | F1 |
| --- | --- |
| happiness | 57.20 ± 0.82 (n=3) |
| sadness | 84.82 ± 0.85 (n=3) |
| neutral | 71.77 ± 0.55 (n=3) |
| anger | 71.57 ± 0.96 (n=3) |
| excited | 74.09 ± 0.80 (n=3) |
| frustration | 68.89 ± 0.69 (n=3) |

## iemocap / no_sequence_mixing

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.43 | 71.29 | 70.70 |
| 2066 | completed | 71.65 | 71.53 | 70.74 |
| 2118 | completed | 71.64 | 71.47 | 70.77 |

| Class | F1 |
| --- | --- |
| happiness | 56.59 ± 0.34 (n=3) |
| sadness | 84.10 ± 0.49 (n=3) |
| neutral | 71.22 ± 0.60 (n=3) |
| anger | 70.61 ± 0.55 (n=3) |
| excited | 73.29 ± 0.68 (n=3) |
| frustration | 68.62 ± 0.48 (n=3) |

## iemocap / one_mixer_block

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.91 | 71.78 | 71.33 |
| 2066 | completed | 71.98 | 71.84 | 71.25 |
| 2118 | completed | 71.19 | 71.04 | 70.35 |

| Class | F1 |
| --- | --- |
| happiness | 56.45 ± 1.21 (n=3) |
| sadness | 84.80 ± 0.14 (n=3) |
| neutral | 70.86 ± 0.60 (n=3) |
| anger | 71.92 ± 1.41 (n=3) |
| excited | 73.33 ± 0.21 (n=3) |
| frustration | 68.49 ± 0.37 (n=3) |

## iemocap / pairwise_mlp_residual

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.48 | 71.35 | 70.88 |
| 2066 | completed | 72.36 | 72.27 | 71.59 |
| 2118 | completed | 71.41 | 71.29 | 70.59 |

| Class | F1 |
| --- | --- |
| happiness | 57.18 ± 1.03 (n=3) |
| sadness | 84.29 ± 0.51 (n=3) |
| neutral | 71.16 ± 0.96 (n=3) |
| anger | 71.33 ± 1.07 (n=3) |
| excited | 73.64 ± 1.11 (n=3) |
| frustration | 68.51 ± 0.68 (n=3) |

## iemocap / single_projection_view

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 71.29 | 71.16 | 70.64 |
| 2066 | completed | 72.11 | 71.97 | 71.58 |
| 2118 | completed | 71.72 | 71.60 | 70.73 |

| Class | F1 |
| --- | --- |
| happiness | 57.19 ± 2.19 (n=3) |
| sadness | 84.61 ± 0.41 (n=3) |
| neutral | 71.05 ± 0.46 (n=3) |
| anger | 71.24 ± 0.89 (n=3) |
| excited | 73.23 ± 1.01 (n=3) |
| frustration | 68.58 ± 0.37 (n=3) |

## meld / amm_attention

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.41 | 68.28 | 51.71 |
| 2028 | completed | 67.51 | 68.58 | 50.83 |
| 2069 | completed | 67.55 | 68.39 | 52.66 |

| Class | F1 |
| --- | --- |
| neutral | 80.82 ± 0.28 (n=3) |
| surprise | 59.56 ± 0.19 (n=3) |
| fear | 26.49 ± 4.78 (n=3) |
| sadness | 43.16 ± 1.95 (n=3) |
| joy | 65.10 ± 0.32 (n=3) |
| disgust | 30.98 ± 2.11 (n=3) |
| anger | 56.00 ± 0.76 (n=3) |

## meld / amm_cubemlp

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.81 | 68.39 | 53.51 |
| 2028 | completed | 67.96 | 68.66 | 52.75 |
| 2069 | completed | 67.78 | 68.39 | 53.02 |

| Class | F1 |
| --- | --- |
| neutral | 80.91 ± 0.18 (n=3) |
| surprise | 60.77 ± 1.01 (n=3) |
| fear | 31.38 ± 3.57 (n=3) |
| sadness | 44.63 ± 0.06 (n=3) |
| joy | 65.43 ± 0.36 (n=3) |
| disgust | 33.61 ± 0.28 (n=3) |
| anger | 54.93 ± 0.63 (n=3) |

## meld / amm_mlp

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.49 | 68.24 | 53.70 |
| 2028 | completed | 67.73 | 68.51 | 51.92 |
| 2069 | completed | 67.47 | 68.24 | 52.27 |

| Class | F1 |
| --- | --- |
| neutral | 80.63 ± 0.44 (n=3) |
| surprise | 60.62 ± 0.21 (n=3) |
| fear | 31.09 ± 4.18 (n=3) |
| sadness | 43.37 ± 0.47 (n=3) |
| joy | 65.38 ± 0.13 (n=3) |
| disgust | 32.32 ± 3.53 (n=3) |
| anger | 55.01 ± 0.81 (n=3) |

## meld / amm_mlp_no_aux

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.46 | 68.16 | 51.61 |
| 2028 | completed | 67.46 | 68.62 | 49.45 |
| 2069 | completed | 67.56 | 68.20 | 51.91 |

| Class | F1 |
| --- | --- |
| neutral | 80.67 ± 0.32 (n=3) |
| surprise | 61.47 ± 0.41 (n=3) |
| fear | 22.09 ± 6.41 (n=3) |
| sadness | 42.89 ± 2.16 (n=3) |
| joy | 65.65 ± 0.65 (n=3) |
| disgust | 28.47 ± 3.28 (n=3) |
| anger | 55.69 ± 1.03 (n=3) |

## meld / full

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.88 | 68.58 | 53.75 |
| 2028 | completed | 67.78 | 68.51 | 53.31 |
| 2069 | completed | 67.89 | 68.62 | 53.30 |

| Class | F1 |
| --- | --- |
| neutral | 80.73 ± 0.11 (n=3) |
| surprise | 60.44 ± 0.65 (n=3) |
| fear | 31.63 ± 1.55 (n=3) |
| sadness | 43.11 ± 0.68 (n=3) |
| joy | 65.22 ± 0.52 (n=3) |
| disgust | 36.67 ± 2.94 (n=3) |
| anger | 56.39 ± 0.48 (n=3) |

## meld / modal_a

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 48.38 | 52.53 | 27.77 |
| 2028 | completed | 48.65 | 52.64 | 28.22 |
| 2069 | completed | 48.31 | 52.41 | 27.84 |

| Class | F1 |
| --- | --- |
| neutral | 68.50 ± 0.20 (n=3) |
| surprise | 28.88 ± 0.53 (n=3) |
| fear | 0.00 ± 0.00 (n=3) |
| sadness | 24.62 ± 0.87 (n=3) |
| joy | 31.17 ± 0.49 (n=3) |
| disgust | 0.00 ± 0.00 (n=3) |
| anger | 42.43 ± 1.43 (n=3) |

## meld / modal_av

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 48.64 | 52.68 | 27.95 |
| 2028 | completed | 49.12 | 52.76 | 28.80 |
| 2069 | completed | 48.83 | 52.64 | 28.22 |

| Class | F1 |
| --- | --- |
| neutral | 68.59 ± 0.11 (n=3) |
| surprise | 27.57 ± 1.09 (n=3) |
| fear | 0.00 ± 0.00 (n=3) |
| sadness | 25.30 ± 2.20 (n=3) |
| joy | 32.72 ± 0.61 (n=3) |
| disgust | 0.00 ± 0.00 (n=3) |
| anger | 44.09 ± 1.04 (n=3) |

## meld / modal_t

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.71 | 68.43 | 53.02 |
| 2028 | completed | 67.50 | 68.31 | 52.60 |
| 2069 | completed | 67.29 | 68.16 | 52.20 |

| Class | F1 |
| --- | --- |
| neutral | 80.66 ± 0.12 (n=3) |
| surprise | 60.17 ± 0.43 (n=3) |
| fear | 29.91 ± 0.88 (n=3) |
| sadness | 44.14 ± 1.53 (n=3) |
| joy | 64.96 ± 0.47 (n=3) |
| disgust | 33.71 ± 1.11 (n=3) |
| anger | 54.72 ± 0.59 (n=3) |

## meld / modal_ta

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.81 | 68.58 | 52.92 |
| 2028 | completed | 67.76 | 68.54 | 53.82 |
| 2069 | completed | 68.08 | 68.85 | 53.80 |

| Class | F1 |
| --- | --- |
| neutral | 80.87 ± 0.15 (n=3) |
| surprise | 60.60 ± 0.61 (n=3) |
| fear | 32.29 ± 2.75 (n=3) |
| sadness | 43.84 ± 0.15 (n=3) |
| joy | 65.13 ± 0.41 (n=3) |
| disgust | 36.27 ± 1.61 (n=3) |
| anger | 55.62 ± 0.56 (n=3) |

## meld / modal_tv

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.37 | 68.01 | 53.42 |
| 2028 | completed | 67.39 | 68.16 | 52.97 |
| 2069 | completed | 67.29 | 68.16 | 53.06 |

| Class | F1 |
| --- | --- |
| neutral | 80.43 ± 0.18 (n=3) |
| surprise | 59.97 ± 0.97 (n=3) |
| fear | 33.15 ± 1.20 (n=3) |
| sadness | 41.66 ± 1.09 (n=3) |
| joy | 64.65 ± 0.56 (n=3) |
| disgust | 36.82 ± 2.66 (n=3) |
| anger | 55.36 ± 0.70 (n=3) |

## meld / modal_v

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 32.60 | 41.92 | 12.66 |
| 2028 | completed | 31.27 | 48.12 | 9.28 |
| 2069 | completed | 31.27 | 48.12 | 9.28 |

| Class | F1 |
| --- | --- |
| neutral | 62.63 ± 4.07 (n=3) |
| surprise | 0.00 ± 0.00 (n=3) |
| fear | 0.00 ± 0.00 (n=3) |
| sadness | 0.00 ± 0.00 (n=3) |
| joy | 10.23 ± 17.71 (n=3) |
| disgust | 0.00 ± 0.00 (n=3) |
| anger | 0.00 ± 0.00 (n=3) |

## meld / no_adaptive_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.59 | 68.39 | 52.65 |
| 2028 | completed | 67.92 | 68.66 | 53.20 |
| 2069 | completed | 67.87 | 68.58 | 53.21 |

| Class | F1 |
| --- | --- |
| neutral | 80.83 ± 0.17 (n=3) |
| surprise | 60.53 ± 0.53 (n=3) |
| fear | 30.09 ± 2.04 (n=3) |
| sadness | 44.36 ± 0.82 (n=3) |
| joy | 65.06 ± 0.77 (n=3) |
| disgust | 34.70 ± 1.25 (n=3) |
| anger | 55.56 ± 0.75 (n=3) |

## meld / no_auxiliary_loss

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.59 | 68.28 | 52.67 |
| 2028 | completed | 67.36 | 68.28 | 51.49 |
| 2069 | completed | 67.58 | 68.24 | 52.37 |

| Class | F1 |
| --- | --- |
| neutral | 80.64 ± 0.07 (n=3) |
| surprise | 60.56 ± 0.55 (n=3) |
| fear | 26.75 ± 1.79 (n=3) |
| sadness | 44.13 ± 0.68 (n=3) |
| joy | 64.61 ± 0.53 (n=3) |
| disgust | 32.98 ± 2.05 (n=3) |
| anger | 55.57 ± 0.54 (n=3) |

## meld / no_cross_attention

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.85 | 68.43 | 54.26 |
| 2028 | completed | 67.36 | 68.05 | 52.47 |
| 2069 | completed | 67.14 | 68.01 | 51.64 |

| Class | F1 |
| --- | --- |
| neutral | 80.56 ± 0.29 (n=3) |
| surprise | 60.40 ± 0.91 (n=3) |
| fear | 29.45 ± 3.96 (n=3) |
| sadness | 43.25 ± 1.88 (n=3) |
| joy | 64.76 ± 0.57 (n=3) |
| disgust | 36.23 ± 2.45 (n=3) |
| anger | 54.87 ± 0.85 (n=3) |

## meld / no_feature_and_adaptive_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.89 | 68.58 | 53.66 |
| 2028 | completed | 67.96 | 68.62 | 53.90 |
| 2069 | completed | 68.00 | 68.74 | 53.44 |

| Class | F1 |
| --- | --- |
| neutral | 80.80 ± 0.09 (n=3) |
| surprise | 60.67 ± 0.82 (n=3) |
| fear | 32.70 ± 2.96 (n=3) |
| sadness | 44.00 ± 0.84 (n=3) |
| joy | 65.71 ± 0.18 (n=3) |
| disgust | 36.26 ± 2.32 (n=3) |
| anger | 55.50 ± 0.27 (n=3) |

## meld / no_feature_gating

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.84 | 68.54 | 53.80 |
| 2028 | completed | 67.73 | 68.51 | 52.25 |
| 2069 | completed | 67.92 | 68.62 | 53.35 |

| Class | F1 |
| --- | --- |
| neutral | 80.85 ± 0.09 (n=3) |
| surprise | 60.16 ± 0.40 (n=3) |
| fear | 30.75 ± 3.83 (n=3) |
| sadness | 44.40 ± 0.32 (n=3) |
| joy | 65.38 ± 0.60 (n=3) |
| disgust | 34.85 ± 2.10 (n=3) |
| anger | 55.54 ± 0.51 (n=3) |

## meld / no_feature_mixing

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.92 | 68.66 | 53.79 |
| 2028 | completed | 68.04 | 68.77 | 53.75 |
| 2069 | completed | 67.65 | 68.47 | 52.21 |

| Class | F1 |
| --- | --- |
| neutral | 80.79 ± 0.13 (n=3) |
| surprise | 60.47 ± 0.59 (n=3) |
| fear | 31.40 ± 3.96 (n=3) |
| sadness | 44.10 ± 0.80 (n=3) |
| joy | 65.34 ± 0.35 (n=3) |
| disgust | 34.68 ± 2.42 (n=3) |
| anger | 55.98 ± 0.39 (n=3) |

## meld / no_mixer

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.75 | 68.47 | 53.24 |
| 2028 | completed | 67.99 | 68.70 | 52.49 |
| 2069 | completed | 67.65 | 68.51 | 52.77 |

| Class | F1 |
| --- | --- |
| neutral | 80.84 ± 0.18 (n=3) |
| surprise | 60.90 ± 0.47 (n=3) |
| fear | 30.42 ± 1.70 (n=3) |
| sadness | 43.57 ± 0.61 (n=3) |
| joy | 64.97 ± 0.01 (n=3) |
| disgust | 33.02 ± 2.49 (n=3) |
| anger | 56.12 ± 0.66 (n=3) |

## meld / no_modality_mixing

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.64 | 68.35 | 53.18 |
| 2028 | completed | 67.80 | 68.58 | 52.77 |
| 2069 | completed | 67.71 | 68.39 | 53.08 |

| Class | F1 |
| --- | --- |
| neutral | 80.78 ± 0.22 (n=3) |
| surprise | 59.67 ± 1.03 (n=3) |
| fear | 31.54 ± 3.18 (n=3) |
| sadness | 44.93 ± 0.96 (n=3) |
| joy | 64.76 ± 0.45 (n=3) |
| disgust | 33.51 ± 3.78 (n=3) |
| anger | 55.90 ± 1.06 (n=3) |

## meld / no_pairwise

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.89 | 68.58 | 53.74 |
| 2028 | completed | 67.78 | 68.51 | 53.31 |
| 2069 | completed | 67.89 | 68.62 | 53.29 |

| Class | F1 |
| --- | --- |
| neutral | 80.74 ± 0.10 (n=3) |
| surprise | 60.36 ± 0.77 (n=3) |
| fear | 31.63 ± 1.55 (n=3) |
| sadness | 43.11 ± 0.68 (n=3) |
| joy | 65.28 ± 0.53 (n=3) |
| disgust | 36.67 ± 2.94 (n=3) |
| anger | 56.36 ± 0.44 (n=3) |

## meld / no_sequence_mixing

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.65 | 68.39 | 53.16 |
| 2028 | completed | 67.70 | 68.47 | 52.67 |
| 2069 | completed | 67.75 | 68.51 | 53.33 |

| Class | F1 |
| --- | --- |
| neutral | 80.69 ± 0.18 (n=3) |
| surprise | 60.34 ± 0.48 (n=3) |
| fear | 31.20 ± 2.65 (n=3) |
| sadness | 43.85 ± 0.59 (n=3) |
| joy | 64.92 ± 0.30 (n=3) |
| disgust | 34.50 ± 2.35 (n=3) |
| anger | 55.87 ± 1.35 (n=3) |

## meld / one_mixer_block

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.67 | 68.31 | 53.69 |
| 2028 | completed | 67.57 | 68.35 | 51.73 |
| 2069 | completed | 67.85 | 68.58 | 53.15 |

| Class | F1 |
| --- | --- |
| neutral | 80.70 ± 0.13 (n=3) |
| surprise | 60.20 ± 0.39 (n=3) |
| fear | 29.43 ± 6.27 (n=3) |
| sadness | 43.69 ± 0.99 (n=3) |
| joy | 65.15 ± 0.37 (n=3) |
| disgust | 34.88 ± 2.14 (n=3) |
| anger | 55.94 ± 0.25 (n=3) |

## meld / pairwise_mlp_residual

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.26 | 68.05 | 52.23 |
| 2028 | completed | 67.87 | 68.70 | 52.60 |
| 2069 | completed | 67.60 | 68.31 | 53.22 |

| Class | F1 |
| --- | --- |
| neutral | 80.71 ± 0.29 (n=3) |
| surprise | 60.04 ± 0.80 (n=3) |
| fear | 30.22 ± 1.04 (n=3) |
| sadness | 43.45 ± 0.63 (n=3) |
| joy | 64.83 ± 0.49 (n=3) |
| disgust | 33.81 ± 2.89 (n=3) |
| anger | 55.73 ± 0.24 (n=3) |

## meld / single_projection_view

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | completed | 67.79 | 68.35 | 53.16 |
| 2028 | completed | 67.76 | 68.54 | 52.90 |
| 2069 | completed | 67.79 | 68.51 | 53.13 |

| Class | F1 |
| --- | --- |
| neutral | 80.62 ± 0.10 (n=3) |
| surprise | 60.11 ± 0.31 (n=3) |
| fear | 29.71 ± 1.64 (n=3) |
| sadness | 44.53 ± 0.65 (n=3) |
| joy | 65.21 ± 0.60 (n=3) |
| disgust | 34.99 ± 0.61 (n=3) |
| anger | 56.28 ± 0.59 (n=3) |

Per-seed confusion counts and their elementwise mean/sample SD are retained in the JSON. Rows are true classes and columns are predictions; class order is explicit. Classes with no true examples remain missing in class-level averages. No statistical significance or improvement claims are inferred.
