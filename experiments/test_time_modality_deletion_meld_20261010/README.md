# MELD test-time modality deletion

This diagnostic evaluates one fixed, trained checkpoint while zeroing missing
raw modality vectors at test time. It does not retrain or alter model weights.
The evaluated subsets are TAV, TV, TA, AV, T, A, and V. TAV must exactly
replay the checkpoint's saved peak-test weighted F1 before the deletion scores
are accepted.

The diagnostic uses the residual, EPIRC-free seed-2025 checkpoint trained with
the re-extracted DenseFace visual features. Results are written to
`results.json` in the remote experiment directory.

## Seed-2025 results

| Kept modalities | Deleted modalities | WF1 | Accuracy | Macro-F1 | ΔWF1 from TAV |
| --- | --- | ---: | ---: | ---: | ---: |
| TAV | None | 68.20 | 68.77 | 53.95 | — |
| TV | A | 66.64 | 67.66 | 52.33 | -1.56 |
| TA | V | 67.52 | 68.51 | 53.15 | -0.68 |
| AV | T | 27.81 | 30.19 | 15.66 | -40.39 |
| T | A, V | 67.15 | 68.58 | 52.81 | -1.05 |
| A | T, V | 25.17 | 28.70 | 11.80 | -43.03 |
| V | T, A | 25.53 | 30.23 | 10.41 | -42.67 |

The TAV replay matches the saved peak-test weighted F1 exactly. These are
interventions on a model trained with all three modalities, so AV/T/A/V are
not comparable to separately trained modality-specific models. Zero inputs
also create a distribution shift. The diagnostic supports actual input
sensitivity, not standalone modality quality or causal attribution.
