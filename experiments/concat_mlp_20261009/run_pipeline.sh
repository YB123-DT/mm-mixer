#!/usr/bin/env bash
set -euo pipefail
ROOT=/data2/yb/multimodalERC/MM_Mixer_ConcatMLP_20261009
export CUDA_VISIBLE_DEVICES=GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/data2/yb/tmp/concat_mlp
cd "$ROOT/code"
PY=/data2/yb/reproduction_envs/s0/bin/python
"$PY" -c 'import json; x=json.load(open("../smoke/result.json")); assert x["status"]=="completed" and x["fresh_strict_replay_exact"] and x["smoke_only"]'
set +e
"$PY" -u train.py --out "$ROOT/runs/meld_seed2025" > "$ROOT/train.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$ROOT/exit_code.txt"
exit "$rc"
