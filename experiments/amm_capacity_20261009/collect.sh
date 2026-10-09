#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
LOCAL=$(cd "$HERE/../.." && pwd)/results/amm_capacity_20261009
REMOTE=/data2/yb/multimodalERC/MM_Mixer_AMM_CAPACITY_20261009
mkdir -p "$LOCAL"
STAMP=$(date -u +%Y%m%dT%H%M%S)
ssh biggpu "cd '$REMOTE/code' && /data2/yb/reproduction_envs/s0/bin/python analyze_revision.py --plan '$REMOTE/pipeline/plan.json' --state '$REMOTE/pipeline/state.json' --output-json '$REMOTE/pipeline/analysis_$STAMP.json' --output-md '$REMOTE/pipeline/analysis_$STAMP.md'"
scp "biggpu:$REMOTE/pipeline/analysis_$STAMP.json" "$LOCAL/analysis.json"
scp "biggpu:$REMOTE/pipeline/analysis_$STAMP.md" "$LOCAL/analysis.md"
scp "biggpu:$REMOTE/pipeline/state.json" "$LOCAL/state.json"
python "$HERE/summarize.py" "$LOCAL"
