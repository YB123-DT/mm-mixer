# AMM increased capacity — 2026-10-09

Hypothesis: increasing AMM FFN capacity may improve emotion recognition. This is unproven; all planned positive and negative outcomes are retained.

12 formal runs: FFN widths 3072 and 6144, original IEMOCAP seeds 2025/2066/2118 and MELD seeds 2025/2028/2069. Original Full width1536 is reused from results/revision_20261003/analysis.json; references.json fixes six original runs before new training. First pair is width3072 seed2025 on each dataset. S6, D256, depth2, all other blocks/losses/features/optimizer/EMA and original100/50epoch budgets remain fixed. Only both AMM blocks' FFN hidden widths change.

| FFN width | AMM parameters | IEMOCAP actual total | MELD actual total |
|---|---:|---:|---:|
|1536 original|1974614|6358082|5712070|
|3072|3550550|7934018|7288006|
|6144|6702422|11085890|10439878|

Frozen base1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab plus controls.patch. Local isolated source outputs/amm_capacity_20261009/code; source manifest snapshots426files. Patch reuses prior tested FFN replacement after originalFull initialization, preserving all nonFFN initial tensors. Unused no_mca_no_amm support is inherited but no joint-deletion run is in this plan. Unrelated dirty working-tree code is excluded. No dependency added, no paper edits.

MELD CPU builder has394755 feature_aligners parameters that original actual no_alignment trainer removes; CPU probe totals are not actual training totals. IEMOCAP checkpoint state includes1536 residual-scale buffers excluded from parameter counts. verify_smoke.py checks actual saved shapes/counts, artifact hashes and strict reload.

CPU validation:6probes passed, including originalFull bitexact comparisons on both datasets, newFFN shape assertions, unchanged nonFFN initialization, optimizer inclusion and finite gradients/updates for all8FFN tensors. Four one-epoch real-data smokes must pass before formal launch. Smoke scores are not formal results.

Server biggpu physicalGPU0 UUID GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45. GPU4 prohibited. Other MCA×AMM/FFN queue remains onGPU1. New queue sharesGPU0 between two independent runs with fixed batch32, existing original precision. InitialGPU0 free32495MiB; minfree6000MiB, no other process at preparation. Existing /data2/yb/reproduction_envs/s0 environment. Remote root /data2/yb/multimodalERC/MM_Mixer_AMM_CAPACITY_20261009. Persistent tmux amm_capacity_12runs; run_pipeline.sh fills slots and produces verified analysis at completion.

Checkpoint selection retains explicitly user-requested strict peak testWF1 for comparison to existing baseline. These are test-selected diagnostics, not independent validation-selected estimates. collect.sh retrieves current verified analysis/state; summarize.py pairs same seeds and verifies label hashes. Incomplete groups show individual results only, no three-seed aggregate. Do not silently resume failed partial training as complete.

Validation completed: all4real-data one-epoch smokes passed strict fresh checkpoint replay and artifact hashes. Saved checkpoint shapes verify both Linear layers in both blocks; actual totals match table above. All426source hashes verified. Reference18artifact hashes reverified remotely. Maximum observed sharedGPU0 smoke memory1448MiB, utilization34%; two formal jobs fit with ample margin.

Formal queue launched 2026-10-09T12:43:07Z in tmux amm_capacity_12runs, schedulerPID1620671. Status running, not a completed comparison. Follow launch_verification.json and collect.sh for live evidence. Existing other queue is untouched.
