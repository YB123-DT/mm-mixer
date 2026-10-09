#!/usr/bin/env bash
set -euo pipefail

ROOT=/data2/yb/multimodalERC/MM_Mixer_RoBERTa_Top4_MELD_20261009
MODEL=${ROBERTA_MODEL:-/home/yangbin/.cache/huggingface/hub/models--roberta-large/snapshots/722cf37b1afa9454edce342e7895e588b6ff1d59}
CSV=${MELD_CSV_DIR:-/data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw}
PYTHON=${PYTHON_BIN:-python}

"$PYTHON" -u "$ROOT/code/train_top4.py" \
  --meld_csv_dir "$CSV" \
  --model_name "$MODEL" \
  --model_type roberta \
  --output_dir "$ROOT/checkpoints" \
  --batch_size 4 \
  --grad_accum_steps 1 \
  --epochs 10 \
  --lr 1e-6 \
  --head_lr 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --max_grad_norm 1.0 \
  --max_length 511 \
  --pooling mask \
  --context_mode history \
  --class_weight none \
  --label_smoothing 0.0 \
  --early_stop_patience 3 \
  --seed 42 \
  --track_test_peak \
  --test_top_k 4
