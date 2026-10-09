#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/yb/multimodalERC/MM_Mixer_RoBERTa_Top4_MELD_20261009
PYTHON=${PYTHON_BIN:-python}
MODEL=${ROBERTA_MODEL:-/home/yangbin/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59}
CSV=${MELD_CSV_DIR:-/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw}

"$PYTHON" -u "$ROOT/code/extract_top4.py" \
  --experiment-root "$ROOT" \
  --python "$PYTHON" \
  --model "$MODEL" \
  --csv-dir "$CSV" \
  --batch-size 8
