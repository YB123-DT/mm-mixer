#!/usr/bin/env bash
set -u
root=/data2/yb/multimodalERC/MM_Mixer_Runtime_20261006
export CUDA_VISIBLE_DEVICES='-1' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data2/yb/tmp/mmrt
for model in dialoguernn ada2i mmgcn mmdfn m3net sdt css ecerc confilmer; do
 for dataset in iemocap meld; do
  out="$root/smoke/${model}_${dataset}.json"
  if [ -f "$out" ]; then continue; fi
  /data2/yb/reproduction_envs/s0/bin/python "$root/code/measure.py" --model "$model" --dataset "$dataset" --cpu-smoke --output "$out" > "$root/smoke/${model}_${dataset}.log" 2>&1
  echo "$model $dataset $?" >> "$root/smoke/exit_codes.txt"
 done
done
