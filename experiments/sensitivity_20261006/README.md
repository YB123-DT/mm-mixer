# Projection-view sensitivity, 2026-10-06

Status: **completed, 18/18 formal runs, zero failures**. Finished `2026-10-06T12:30:18Z`, queue exit code `0`. Eighteen explicitly authorized runs completed on `biggpu` using the frozen Full implementation from `1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab` plus `sensitivity.patch`.

| Setting | Value |
| --- | --- |
| Projection views S | 2, 4, 8 |
| Datasets | IEMOCAP, MELD |
| IEMOCAP seeds | 2025, 2066, 2118 |
| MELD seeds | 2025, 2028, 2069 |
| Training budget | IEMOCAP 100 epochs; MELD 50 epochs |
| Selection | Existing user-authorized `strict_peak_test_wf1` |
| Batch size | 32, unchanged |
| Architecture | D=256; FFN=1536; two blocks; S-axis hidden=2S |
| GPU whitelist | Host GPU 7, `GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e` only; GPU 4 forbidden |
| Concurrency | Two jobs per GPU; 6000 MiB minimum free memory checked before each launch |
| Threads / temp | One OMP/MKL/OpenBLAS/NumExpr thread; `TMPDIR=/data2/yb/tmp/mmsens` |

Original S=6 and S=1 are separate existing experiments; no historical S=4/8 run is reused in this new matrix. Sensitivity changes the projection output size and its derived S-axis MLP dimensions, including parameter count. It is not a fixed-parameter-capacity comparison. All other training settings inherit Full.

## Frozen code and verification

- Local snapshot: `outputs/sensitivity_20261006/code`
- Remote root: `/data2/yb/multimodalERC/MM_Mixer_Sensitivity_20261006`
- Remote snapshot: `code/`
- Python: `/data2/yb/reproduction_envs/s0/bin/python`
- Snapshot content SHA256: `689bd97ff0a57fbd5fde0770f0b12c33e235934c7587d1dceee5347219020c67`
- `prepare.py` reconstructs the same snapshot; `snapshot.json` records individual file hashes.
- Both datasets' S=6 state and logits match the base commit bitwise. All six new configurations pass forward/backward and actual optimizer update checks. Existing revision tests: 52 passed, 1 skipped.
- IEMOCAP/S=2 and MELD/S=8 one-epoch real-data smoke runs passed strict fresh-checkpoint replay and all artifact hash checks. These outputs are under `smoke/` and are never reused for formal runs.

## Persistent execution

Launch time: `2026-10-06T05:44:51Z`.

- Pipeline PID: `197886`.
- Scheduler PID: `197891`.
- Launch: `nohup setsid bash ./run_pipeline.sh > pipeline.log 2>&1 < /dev/null &`, in remote `pipeline/`.
- Checked before launch: no existing sensitivity process; GPU 7 had 17,494 / 32,768 MiB allocated to an unrelated process and 0% compute utilization. This leaves approximately 15 GiB; each verified smoke used less than 1 GiB additional GPU memory.
- `plan.json` contains all 18 concrete commands. `run_pipeline.sh` validates it before starting the existing locked scheduler, with duplicate-output protection, per-run provenance and artifact validation.
- Runtime status: remote `pipeline/state.json`; ongoing metrics: `pipeline/summary.json`.
- Job logs: remote `pipeline/logs/{dataset}_{variant}_seed{seed}.log`.
- Checkpoints/results: remote `runs/{dataset}/{variant}/seed{seed}/best_peak/`.
- When the queue terminates, the pipeline runs `analyze_revision.py`, saving `pipeline/analysis.json` and `pipeline/analysis.md`, plus `queue_exit_code.txt` and `finished_at.txt`. Failures remain visible; no automatic retries or result replacement.

All formal runs and twelve existing S=1/S=6 reference bundles were independently verified, including checkpoint hashes and prediction-derived metrics. The frozen analysis was rerun and matched exactly. Results and per-seed metadata are saved in `results/sensitivity_20261006/`; see its README and comparison.csv.

## Historical startup evidence

At 2026-10-06T05:46:37.733768+00:00, the scheduler reports **2 running, 16 queued, 0 failed**. IEMOCAP/S=2/seed2025 completed epoch 1/100 with finite train/test loss. The two jobs together increased GPU memory from 17,494 to 18,590 MiB (about 1.1 GiB); observed GPU utilization was 11%. `launch_verification.json` records run PIDs, exact commands, GPU identity and the observed epoch lines. These are startup checks, not completed-experiment scores.

## Completion verification

Final state: 18 completed, 0 failed, 0 queued, 0 running. `analysis.complete=true`, all three seeds included for every dataset/S setting, every task returned 0. `verify_completed.py` independently recomputed the frozen analysis and checked all checkpoint/other artifact hashes plus prediction-derived metrics for 18 new and 12 reference runs. `summarize.py` produces the complete S=1/2/4/6/8 table without seed filtering. No checkpoint weights or raw dataset files were downloaded.

Mean WF1 (IEMOCAP / MELD): S=2 **71.53 / 67.84**, S=4 **71.77 / 67.76**, S=8 **71.90 / 67.69**. Existing S=6 Full: **72.15 / 67.85**. S=6 has the highest observed mean, but MELD S=2 is nearly tied; improvements are not monotonic in S and no significance claim is made.
