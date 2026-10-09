#!/usr/bin/env bash
set -u

ROOT=/data2/yb/multimodalERC/MM_Mixer_RoBERTa_Top4_MELD_20261009
LOG="$ROOT/logs/monitor_30m.log"

while true; do
  {
    date -u +'%Y-%m-%dT%H:%M:%SZ'
    for name in train extract_top4 downstream; do
      if test -f "$ROOT/logs/$name.exit"; then
        printf '%s.exit=' "$name"
        cat "$ROOT/logs/$name.exit"
      else
        printf '%s.exit=pending\n' "$name"
      fi
    done
    if test -f "$ROOT/checkpoints/test_top_k.json"; then
      cat "$ROOT/checkpoints/test_top_k.json"
    fi
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory \
      --format=csv,noheader | grep GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a || true
    df -h "$ROOT" | tail -n 1
    printf '%s\n' '---'
  } >> "$LOG" 2>&1

  if test -f "$ROOT/logs/downstream.exit"; then
    break
  fi
  sleep 1800
done
