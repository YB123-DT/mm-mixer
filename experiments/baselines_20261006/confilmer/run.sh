#!/usr/bin/env bash
set -euo pipefail
# Physical GPU must be supplied by the common scheduler, GPU 4 is forbidden.
: "${CUDA_VISIBLE_DEVICES:?Set physical healthy GPU}"
if [[ ",${CUDA_VISIBLE_DEVICES}," == *,4,* ]]; then exit 64; fi
dataset=${1:?IEMOCAP or MELD}; seed=${2:?seed}; output=${3:?absolute output dir}
root=/data2/yb/multimodalERC/MM_Mixer_Baselines_20261006/source/confilmer
py=/data2/yb/reproduction_envs/s0/bin/python
export PYTHONHASHSEED=$seed OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
cd "$root"
case "$dataset" in
 IEMOCAP) cfg=(--dropout 0.5 --epochs 80 --num_L 5 --num_K 4);;
 MELD) cfg=(--dropout 0.4 --epochs 15 --num_L 3 --num_K 3 --use_modal);;
 *) exit 64;;
esac
exec "$py" -u train_our.py --base-model GRU --lr 0.0001 --batch-size 16 \
 --graph_type hyper --graph_construct direct --multi_modal --mm_fusion_mthd concat_DHT \
 --modals avl --Dataset "$dataset" --norm BN --seed "$seed" --output-dir "$output" "${cfg[@]}" "${@:4}"
