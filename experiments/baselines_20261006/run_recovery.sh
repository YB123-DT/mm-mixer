#!/bin/bash
set -euo pipefail
export TMPDIR=/data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/tmp
py=/data2/yb/reproduction_envs/s0/bin/python
"$py" -u /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/code/queue_recovery.py --plan /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/plans/recovery_12_runs.json --state /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/queue/recovery_state.json
CUDA_VISIBLE_DEVICES='' "$py" /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/code/summarize.py --plan /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/plans/recovery_12_runs.json --output /data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/summary.json
