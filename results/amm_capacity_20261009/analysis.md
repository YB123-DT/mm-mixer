# MM-Mixer revision experiment results

Selection: `strict_peak_test_wf1`. Metrics below are percentages; paired differences are percentage points. Standard deviations use ddof=1. Partial groups are explicitly marked and are not three-seed results.

| Dataset | Variant | Verified seeds | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| iemocap | ffn_width_3072 | 0/3 INCOMPLETE | — (n=0) | — (n=0) | — (n=0) |
| iemocap | ffn_width_6144 | 0/3 INCOMPLETE | — (n=0) | — (n=0) | — (n=0) |
| meld | ffn_width_3072 | 0/3 INCOMPLETE | — (n=0) | — (n=0) | — (n=0) |
| meld | ffn_width_6144 | 0/3 INCOMPLETE | — (n=0) | — (n=0) | — (n=0) |

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
| 2025 | running | — | — | — |
| 2066 | queued | — | — | — |
| 2118 | queued | — | — | — |

| Class | F1 |
| --- | --- |
| happiness | — (n=0) |
| sadness | — (n=0) |
| neutral | — (n=0) |
| anger | — (n=0) |
| excited | — (n=0) |
| frustration | — (n=0) |

## iemocap / ffn_width_6144

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | queued | — | — | — |
| 2066 | queued | — | — | — |
| 2118 | queued | — | — | — |

| Class | F1 |
| --- | --- |
| happiness | — (n=0) |
| sadness | — (n=0) |
| neutral | — (n=0) |
| anger | — (n=0) |
| excited | — (n=0) |
| frustration | — (n=0) |

## meld / ffn_width_3072

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | running | — | — | — |
| 2028 | queued | — | — | — |
| 2069 | queued | — | — | — |

| Class | F1 |
| --- | --- |
| neutral | — (n=0) |
| surprise | — (n=0) |
| fear | — (n=0) |
| sadness | — (n=0) |
| joy | — (n=0) |
| disgust | — (n=0) |
| anger | — (n=0) |

## meld / ffn_width_6144

| Seed | State | WF1 | ACC | Macro-F1 |
| --- | --- | --- | --- | --- |
| 2025 | queued | — | — | — |
| 2028 | queued | — | — | — |
| 2069 | queued | — | — | — |

| Class | F1 |
| --- | --- |
| neutral | — (n=0) |
| surprise | — (n=0) |
| fear | — (n=0) |
| sadness | — (n=0) |
| joy | — (n=0) |
| disgust | — (n=0) |
| anger | — (n=0) |

Per-seed confusion counts and their elementwise mean/sample SD are retained in the JSON. Rows are true classes and columns are predictions; class order is explicit. Classes with no true examples remain missing in class-level averages. No statistical significance or improvement claims are inferred.
