#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=/data2/yb/multimodalERC/MM_Mixer_MAGTKD_20261007
PYTHON=/data2/yb/reproduction_envs/s0/bin/python
export TMPDIR=/data2/yb/tmp/magtkd OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PYTHON" - "$TASK_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for suffix in ['i','m']:
 r=json.loads((root/f'smoke_{suffix}'/'result.json').read_text())
 assert r['smoke_only'] and r['checkpoint_predictions_verified'] and r['smoke_updated_parameters']
assert len(json.loads((root/'data'/'audit.json').read_text()))==6
PY
"$PYTHON" "$TASK_ROOT/code/run_queue.py" --plan "$TASK_ROOT/plan.json" --state "$TASK_ROOT/queue_state.json"
"$PYTHON" "$TASK_ROOT/code/summarize.py" --plan "$TASK_ROOT/plan.json" --output "$TASK_ROOT/summary.json"
