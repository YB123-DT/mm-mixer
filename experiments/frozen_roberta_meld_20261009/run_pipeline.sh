#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/yb/multimodalERC/MM_Mixer_FrozenRoBERTa_MELD_20261009
PY=/data2/yb/reproduction_envs/s0/bin/python
GPU_UUID=GPU-e4cafb17-818e-216a-b94a-7440063a9153
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

test -f "$ROOT/features/manifest.json"
test -f "$ROOT/feature_verification.json"
mkdir -p "$ROOT/runs" "$ROOT/logs"
cd "$ROOT/downstream_code"

run_one() {
    local variant="$1"
    set +e
    "$PY" -u run.py run --dataset meld --variant "$variant" --seed 2025 \
        --output-root "$ROOT/runs" > "$ROOT/logs/${variant}.log" 2>&1
    local status=$?
    set -e
    printf '%s\n' "$status" > "$ROOT/logs/${variant}.exit"
    return "$status"
}

run_one modal_t &
text_pid=$!
run_one full &
full_pid=$!
printf '%s\n' "$text_pid" > "$ROOT/logs/modal_t.pid"
printf '%s\n' "$full_pid" > "$ROOT/logs/full.pid"

status=0
wait "$text_pid" || status=1
wait "$full_pid" || status=1
printf '%s\n' "$status" > "$ROOT/pipeline.exit"
exit "$status"
