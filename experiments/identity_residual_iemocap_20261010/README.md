# IEMOCAP identity-residual matrix (2026-10-10)

Status: completed at 2026-10-10T17:40:37Z; 18/18 formal runs passed artifact and protocol verification with zero failures.

- Server: `biggpu`; physical GPU 6 (`GPU-e4cafb17-818e-216a-b94a-7440063a9153`). Physical GPU 4 is prohibited.
- Frozen source: commit `72f4df0`.
- Architecture: ten learnable residual scales, matching the existing MELD residual model: three FG, three AG, three MCA, and one final query-integration scale. AMM retains its own internal residuals. EPIRC is absent.
- Matrix: Full plus removal of FG, AG, MCA, AMM, and FG+AG; seeds 2025, 2066, and 2118 (18 runs). The queue permits five concurrent jobs on GPU 6 after checking available memory before every launch.
- Selection: strict peak test WF1, following the existing paper protocol.
- Smoke: seed 2025 completed one epoch with exit code 0 and `fresh_strict_replay_exact: true`; the smoke score is excluded from formal results.
- Remote root: `/data2/yb/multimodalERC/MM_Mixer_ResidualIEMOCAP_20261010`.

The persistent queue initially started with two-way concurrency in tmux session `mm_residual_iemocap_20261010`; it was then resumed with five-way concurrency at the user's request. `launch_state.json` is the original launch-time scheduler snapshot.

## Three-seed results

| Variant | WF1 (%) | Change from Full | Accuracy (%) |
|---|---:|---:|---:|
| Full residual model | 71.861 $\pm$ 0.165 | -- | 71.760 $\pm$ 0.188 |
| w/o FG | 72.038 $\pm$ 0.216 | +0.177 | 71.945 $\pm$ 0.233 |
| w/o AG | 72.003 $\pm$ 0.109 | +0.142 | 71.904 $\pm$ 0.107 |
| w/o MCA | 71.797 $\pm$ 0.363 | -0.064 | 71.719 $\pm$ 0.375 |
| w/o AMM | 71.342 $\pm$ 0.378 | -0.520 | 71.226 $\pm$ 0.375 |
| w/o FG and AG | 71.914 $\pm$ 0.200 | +0.052 | 71.822 $\pm$ 0.178 |

AMM is the only component whose removal lowers WF1 for all three seeds, by 1.135, 0.302, and 0.123 points. Removing MCA lowers the mean by 0.064 points but improves seed 2025. Removing FG or AG improves the three-seed mean, so this matrix does not support claiming that either gate independently improves IEMOCAP performance. Scores use the project's strict peak-test protocol.
