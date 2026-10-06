#!/usr/bin/env bash
set -euo pipefail
root=/data2/yb/multimodalERC/MM_Mixer_Baselines_20261006
py=/data2/yb/reproduction_envs/s0/bin/python
"$py" -u "$root/code/queue.py" --plan "$root/plans/12_runs.json" --state "$root/queue/state.json"
CUDA_VISIBLE_DEVICES='' "$py" "$root/code/summarize.py" --plan "$root/plans/12_runs.json" --output "$root/summary.json"
