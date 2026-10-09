# Larger AMM FFN-width controls (2026-10-09)

Status: complete. Both larger widths were evaluated on IEMOCAP and MELD with three predetermined seeds. Results follow the historical `strict_peak_test_wf1` diagnostic selection rule.

| FFN width | IEMOCAP WF1 | Δ from width 1536 | MELD WF1 | Δ from width 1536 |
| ---: | ---: | ---: | ---: | ---: |
| 1536 (default) | 72.15 ± 0.21 | — | 67.85 ± 0.06 | — |
| 3072 | 71.82 ± 0.52 | -0.32 | 67.82 ± 0.10 | -0.03 |
| 6144 | 72.10 ± 0.35 | -0.04 | 67.82 ± 0.16 | -0.03 |

Larger FFNs do not improve the three-seed mean. The weak AMM ablation effect therefore cannot be explained by the default FFN being too narrow. Per-seed results and paired differences are retained in `analysis.json` and `comparison.json`.
