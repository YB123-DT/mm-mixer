#!/usr/bin/env bash
set -euo pipefail
ROOT=/data2/yb/multimodalERC/MM_Mixer_ResidualIEMOCAP_20261010
PYTHON=/data2/yb/reproduction_envs/s0/bin/python
export TMPDIR=/data2/yb/tmp/residual_iemocap
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p "$TMPDIR" "$ROOT/pipeline"
cd "$ROOT/code"
"$PYTHON" launch_revision.py validate --plan "$ROOT/pipeline/plan.json" > "$ROOT/pipeline/validated_commands.json"
set +e
"$PYTHON" launch_revision.py run --plan "$ROOT/pipeline/plan.json" --state "$ROOT/pipeline/state.json" --server biggpu --gpu 6:GPU-e4cafb17-818e-216a-b94a-7440063a9153 --per-gpu 2 --min-free-mib 6000 --threads 1 --poll-seconds 30
QUEUE_EXIT=$?
set -e
printf '%s\n' "$QUEUE_EXIT" > "$ROOT/pipeline/queue_exit_code.txt"
"$PYTHON" analyze_revision.py --plan "$ROOT/pipeline/plan.json" --state "$ROOT/pipeline/state.json" --output-json "$ROOT/pipeline/analysis.json" --output-md "$ROOT/pipeline/analysis.md"
date -u +%Y-%m-%dT%H:%M:%SZ > "$ROOT/pipeline/finished_at.txt"
exit "$QUEUE_EXIT"
