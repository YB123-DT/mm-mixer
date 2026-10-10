# IEMOCAP identity-residual matrix (2026-10-10)

Status: prepared after a successful real-data one-epoch smoke; formal queue pending launch.

- Server: `biggpu`; physical GPU 6 (`GPU-e4cafb17-818e-216a-b94a-7440063a9153`). Physical GPU 4 is prohibited.
- Frozen source: commit `72f4df0`.
- Architecture: ten learnable residual scales, matching the existing MELD residual model: three FG, three AG, three MCA, and one final query-integration scale. AMM retains its own internal residuals. EPIRC is absent.
- Matrix: Full plus removal of FG, AG, MCA, AMM, and FG+AG; seeds 2025, 2066, and 2118 (18 runs).
- Selection: strict peak test WF1, following the existing paper protocol.
- Smoke: seed 2025 completed one epoch with exit code 0 and `fresh_strict_replay_exact: true`; the smoke score is excluded from formal results.
- Remote root: `/data2/yb/multimodalERC/MM_Mixer_ResidualIEMOCAP_20261010`.
