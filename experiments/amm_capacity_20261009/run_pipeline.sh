#!/usr/bin/env bash
set -euo pipefail
ROOT=/data2/yb/multimodalERC/MM_Mixer_AMM_CAPACITY_20261009
PYTHON=/data2/yb/reproduction_envs/s0/bin/python
export TMPDIR=/data2/yb/tmp/amm_capacity
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$TMPDIR" "$ROOT/pipeline"
cd "$ROOT/code"
"$PYTHON" "$ROOT/pipeline/verify_smoke.py" > "$ROOT/pipeline/prelaunch_verification.txt"
"$PYTHON" launch_revision.py validate --plan "$ROOT/pipeline/plan.json" > "$ROOT/pipeline/validated_commands.json"
set +e
"$PYTHON" launch_revision.py run --plan "$ROOT/pipeline/plan.json" --state "$ROOT/pipeline/state.json" --server biggpu --gpu 0:GPU-43d98f5a-edab-1498-e9db-eeeb2d909d45 --per-gpu 2 --min-free-mib 6000 --threads 1 --poll-seconds 20
QUEUE_EXIT=$?
set -e
printf '%s\n' "$QUEUE_EXIT" > "$ROOT/pipeline/queue_exit_code.txt"
"$PYTHON" analyze_revision.py --plan "$ROOT/pipeline/plan.json" --state "$ROOT/pipeline/state.json" --output-json "$ROOT/pipeline/analysis.json" --output-md "$ROOT/pipeline/analysis.md"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/pipeline/finished_at.txt"
exit "$QUEUE_EXIT"
