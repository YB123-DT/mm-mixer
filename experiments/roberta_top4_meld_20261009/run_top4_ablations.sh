#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/yb/multimodalERC/MM_Mixer_RoBERTa_Top4_MELD_20261009
PYTHON=${DOWNSTREAM_PYTHON_BIN:-/data2/yb/reproduction_envs/s0/bin/python}

"$PYTHON" -u "$ROOT/code/run_top4_ablations.py" \
  --experiment-root "$ROOT" \
  --python "$PYTHON" \
  --max-parallel 4
