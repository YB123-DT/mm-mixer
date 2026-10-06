#!/usr/bin/env bash
set -euo pipefail
ROOT=/data2/yb/multimodalERC/MM_Mixer_AuxWeights_20261006
export TMPDIR=/data2/yb/tmp/mmauxw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e
cd "$ROOT/code"
for item in 'iemocap aux_equal' 'meld aux_double'; do
 read -r dataset variant <<< "$item"
 /data2/yb/reproduction_envs/s0/bin/python run.py run --dataset "$dataset" --variant "$variant" --seed 2025 --epochs 1 --output-root "$ROOT/smoke" > "$ROOT/checks/smoke_${dataset}.log" 2>&1
done
