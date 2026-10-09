#!/usr/bin/env bash
set -euo pipefail
ROOT=/data2/yb/multimodalERC/MM_Mixer_ConcatMLP_IEMOCAP_20261009
DEST=results/concat_mlp_iemocap_20261009
mkdir -p "$DEST"
for name in result.json metrics.jsonl predictions.npz verification.json cache_verification.json; do
  scp "biggpu:$ROOT/runs/iemocap_seed2025/$name" "$DEST/$name"
done
scp "biggpu:$ROOT/exit_code.txt" "$DEST/exit_code.txt"
