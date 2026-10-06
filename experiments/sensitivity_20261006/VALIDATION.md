# S sensitivity implementation validation

Base: `1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab`. Only the frozen snapshot is modified; live dataset runners/config remain untouched.

- Variants: `projection_views_2`, `projection_views_4`, `projection_views_8`.
- IEMOCAP seeds: 2025, 2066, 2118; MELD seeds: 2025, 2028, 2069.
- Existing `launch_revision.py plan --matrix sensitivity` generates 18 formal runs. Execution and analysis accept every job; configuration comparison confirms no changes outside variant/S/derived S-MLP width.
- Both original dataset implementations use S → 2S → S, not hidden width 128. New hidden widths are 4, 8, 16; original S=6 remains hidden 12. D=256, feature FFN=1536, two blocks, original order, optimizer, loss, data and budgets remain unchanged.
- S is passed into encoder construction, including MELD's candidate class used inside formal training. No post-construction replacement is used for sensitivity.
- `test_sensitivity.py` verifies original-commit versus patched Full with identical seed: all state tensors and forward logits are bitwise equal on both datasets. All six new dataset/S combinations pass shape, finite forward, finite/nonzero gradients, real optimizer membership, and actual parameter-update checks for split/S/modality/feature weights.
- `verification.json` records this evidence. Its parameter counts are constructor-level; MELD's downstream trainer subsequently disables its legacy alignment module, just as in Full. These constructor counts are not final training parameter counts.
- Original revision-control test module: 4 tests pass locally. Broad local unittest run: 31 pass, 1 skip, queue-test import unavailable because local environment lacks pytest. No dependency was installed; the existing remote s0 environment passed the full revision pytest suite: **52 passed, 1 skipped in 88.34 s** (analysis, controls, efficiency, integration, queue).
- `prepare.py` reconstructs the snapshot using git archive + sensitivity.patch. An independently rebuilt snapshot produced exactly the same 427-file hash. `snapshot.json` excludes itself, pycache and pytest cache.

Commands (from repository root):

```bash
python experiments/sensitivity_20261006/prepare.py --destination /absolute/new/empty/snapshot
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python experiments/sensitivity_20261006/test_sensitivity.py
/home/yangbin/miniconda3/envs/multimodalerc310/bin/python experiments/sensitivity_20261006/check_plan.py
```

After any intentional frozen-source modification, regenerate the patch from the base commit and run `python experiments/sensitivity_20261006/prepare.py --manifest-only` before syncing. Do not refresh provenance in a running experiment tree.

A one-epoch data/trainer smoke command (not a formal result):

```bash
cd /data2/yb/multimodalERC/MM_Mixer_Sensitivity_20261006/code
TMPDIR=/data2/yb/tmp/mmsens CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e /data2/yb/reproduction_envs/s0/bin/python run.py run --dataset iemocap --variant projection_views_2 --seed 2025 --output-root /data2/yb/multimodalERC/MM_Mixer_Sensitivity_20261006/smoke --epochs 1
```

Recheck resource ownership before execution. Host GPU 4 is forbidden. No formal training is launched by these preparation/validation scripts.

## Real-data smoke verification

Both IEMOCAP/S=2 and MELD/S=8 completed one real-data epoch on biggpu GPU 7 (UUID above), returned exit code 0, and exported complete peak bundles. All artifact hashes were independently recalculated and matched; both status files record `fresh_strict_replay_exact: true`. Runtime architecture records show S=2/8 and subspace hidden=4/16 with D=256, FFN=1536, two blocks. Evidence: `smoke_verification.json`. These one-epoch outputs are excluded from formal results. The 18 formal runs were subsequently authorized and launched at 2026-10-06T05:44:51Z; see README.md for persistent queue details.
