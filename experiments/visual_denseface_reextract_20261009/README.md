# MELD DenseFace re-extraction

This directory provides a reproducible replacement for the unavailable
DenseFace checkpoint used by earlier preprocessing. It retains the same
architecture and output contract: DenseNet-BC-100, growth rate 12, dense
blocks of 16/16/16 layers, compression 0.5, and a 342-dimensional vector
before the classifier.

This is a **locally retrained model with the same architecture**, not the
authors' unpublished checkpoint. Results produced by it must be described as
re-extracted/retrained DenseFace features rather than the original released
features.

## 1. Rebuild the FER2013 image CSV

The FER2013 images used here come from the public `AutumnQiu/fer2013`
Hugging Face mirror. This is not an official image download. The FER+ vote
annotations in `fer2013new.csv` come from Microsoft's official FER+ release.
Rebuild the standard CSV in the required train/valid/test order with:

```bash
python experiments/visual_denseface_reextract_20261009/prepare_fer2013_csv.py \
  --train-parquet /path/to/train-00000-of-00001.parquet \
  --valid-parquet /path/to/valid-00000-of-00001.parquet \
  --test-parquet /path/to/test-00000-of-00001.parquet \
  --output /path/to/fer2013.csv \
  --manifest /path/to/fer2013_csv_manifest.json
```

The conversion requires exactly 28,709/3,589/3,589 rows, writes Usage as
Training/PublicTest/PrivateTest, and verifies every split's first sample and
the serialized Usage counts. The manifest hashes all three Parquet inputs and
the resulting CSV.

## 2. Train on FER+

The loader aligns the official `fer2013.csv` image rows with
`fer2013new.csv`. It takes the argmax over all ten vote columns, removes a
sample when `unknown` or `NF` wins (and removes all-zero vote rows), and keeps
the remaining eight emotion classes. Official Usage values map as follows:
`Training -> train`, `PublicTest -> dev`, and `PrivateTest -> test`.
The grayscale mean and population standard deviation are computed from the
kept FER+ training images for every run and stored in both the checkpoint and
manifest; no borrowed normalization constants are used for training.

```bash
python experiments/visual_denseface_reextract_20261009/train.py \
  --fer2013-csv /data2/yb/multimodalERC/DenseFace_retrain_20261009/data/raw/fer2013.csv \
  --fer2013new-csv /data2/yb/multimodalERC/DenseFace_retrain_20261009/data/raw/fer2013new.csv \
  --output-dir /data2/yb/multimodalERC/DenseFace_retrain_20261009/training/seed42 \
  --seed 42
```

The default optimization matches the recorded MEmoBert setup:
Adam with learning rate `1e-3` and weight decay `0`. `best_dev.pt` is selected
by development macro-F1, with early stopping after eight non-improving epochs.
`metrics.json` stores
the full training history and the single final test evaluation.
`manifest.json` records the input hashes, split/label policy, seed, model
contract, environment, and checkpoint hash. Add `--smoke` for a two-sample,
one-epoch plumbing check; it is not a valid trained model.

## 3. Select the target-speaker face track

Each `local_meld/{split}_video/<utterance>/` directory can contain multiple
`face_*.mp4` tracks. Extraction never silently averages different people.
Track selection has this precedence:

1. `--selection-manifest tracks.json`, mapping utterance IDs to a face file;
2. the only face track when exactly one exists;
3. `--selector-plugin package.module:function`; or
4. the existing MELD LipSyncNet adapter when both `--lipsync-utils` and
   `--lipsync-checkpoint` are supplied.

A manifest can be a flat mapping or contain `train`, `dev`, and `test`
sub-mappings. A selector plugin receives
`(utterance_id, utterance_dir, candidate_paths, audio_path)` and returns a
candidate path/name or `None`.

The available local LipSyncNet can be selected with:

```bash
--lipsync-utils /data2/yb/multimodalERC/MELD/Model/utils.py \
--lipsync-checkpoint /data2/yb/multimodalERC/MELD/Dataset/Data/lipsync_model_meld.pth
```

## 4. Compute MELD train face statistics

The original MEmoBert/OpenFace preprocessing reports task-specific grayscale
statistics, but those values describe its OpenFace crops. The `local_meld`
videos here are MTCNN tracks with a different pixel distribution. Compute the
mean and population standard deviation from the valid target-speaker frames of
the MELD **training split** using the same track selector and 64x64 grayscale
pipeline used for extraction:

```bash
python experiments/visual_denseface_reextract_20261009/compute_meld_stats.py \
  --local-meld /data2/yb/multimodalERC/MELD/Dataset/Data/local_meld \
  --output /data2/yb/multimodalERC/DenseFace_retrain_20261009/meld_train_face_stats.json \
  --lipsync-utils /data2/yb/multimodalERC/MELD/Model/utils.py \
  --lipsync-checkpoint /data2/yb/multimodalERC/MELD/Dataset/Data/lipsync_model_meld.pth
```

Do not substitute the published OpenFace values `67.61417/37.89171`. The
statistics JSON records frame/pixel counts, selection failures, selector
inputs, and whether it was computed on a bounded subset. `--max-samples` is
only for a smoke run and its result must not be used for formal extraction.

## 5. Four-GPU sharded statistics and extraction

Sharding is deterministic: the sorted utterance list is assigned by
`position % num_shards`. The shards are mutually exclusive and their union is
the unsharded list. The following example runs one process per GPU:

```bash
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python experiments/visual_denseface_reextract_20261009/compute_meld_stats.py \
    --local-meld /data2/yb/multimodalERC/MELD/Dataset/Data/local_meld \
    --output /tmp/meld_stats_shard_${i}.json \
    --lipsync-utils /data2/yb/multimodalERC/MELD/Model/utils.py \
    --lipsync-checkpoint /data2/yb/multimodalERC/MELD/Dataset/Data/lipsync_model_meld.pth \
    --num-shards 4 --shard-index $i --device cuda &
done
wait

python experiments/visual_denseface_reextract_20261009/merge_meld_stats.py \
  --inputs /tmp/meld_stats_shard_{0,1,2,3}.json \
  --output /data2/yb/multimodalERC/DenseFace_retrain_20261009/meld_train_face_stats.json

for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python experiments/visual_denseface_reextract_20261009/extract_meld.py \
    --local-meld /data2/yb/multimodalERC/MELD/Dataset/Data/local_meld \
    --checkpoint /data2/yb/multimodalERC/DenseFace_retrain_20261009/training/seed42/best_dev.pt \
    --normalization-stats /data2/yb/multimodalERC/DenseFace_retrain_20261009/meld_train_face_stats.json \
    --output-root /data2/yb/multimodalERC/DenseFace_retrain_20261009/features_sharded \
    --lipsync-utils /data2/yb/multimodalERC/MELD/Model/utils.py \
    --lipsync-checkpoint /data2/yb/multimodalERC/MELD/Dataset/Data/lipsync_model_meld.pth \
    --num-shards 4 --shard-index $i --device cuda --resume &
done
wait

python experiments/visual_denseface_reextract_20261009/merge_meld_features.py \
  --shard-roots \
    /data2/yb/multimodalERC/DenseFace_retrain_20261009/features_sharded/shard_000_of_004 \
    /data2/yb/multimodalERC/DenseFace_retrain_20261009/features_sharded/shard_001_of_004 \
    /data2/yb/multimodalERC/DenseFace_retrain_20261009/features_sharded/shard_002_of_004 \
    /data2/yb/multimodalERC/DenseFace_retrain_20261009/features_sharded/shard_003_of_004 \
  --metadata-root /data2/yb/OpenDataLab___MELD/raw/MELD/MELD.Raw \
  --output-root /data2/yb/multimodalERC/DenseFace_retrain_20261009/features
```

The statistics merger requires every shard index and combines the stored
pixel sums and squared sums, rather than averaging shard means. The feature
merger rejects duplicate or unexpected keys and invalid 342-D/non-finite
vectors. It checks keys against each MELD CSV, adds explicit zero vectors for
CSV utterances whose video directory is missing, and records all missing and
zero-vector keys.

## 6. Extract MELD features without sharding

```bash
python experiments/visual_denseface_reextract_20261009/extract_meld.py \
  --local-meld /data2/yb/multimodalERC/MELD/Dataset/Data/local_meld \
  --checkpoint /data2/yb/multimodalERC/DenseFace_retrain_20261009/training/seed42/best_dev.pt \
  --output-root /data2/yb/multimodalERC/DenseFace_retrain_20261009/features \
  --normalization-stats /data2/yb/multimodalERC/DenseFace_retrain_20261009/meld_train_face_stats.json \
  --lipsync-utils /data2/yb/multimodalERC/MELD/Model/utils.py \
  --lipsync-checkpoint /data2/yb/multimodalERC/MELD/Dataset/Data/lipsync_model_meld.pth \
  --resume
```

Every valid frame of the selected face track is converted to grayscale,
resized to 64x64, and mapped to 342 dimensions. Only decoded valid frames are
averaged. Missing faces, unresolved target tracks, empty videos, and decoding
or model errors receive an explicit 342-dimensional zero vector. Their reason
is retained in `denseface_progress.jsonl` and summarized in
`extraction_stats.json`. Final files follow the existing contract:
`{split}_features/visual_features.json`.

Use `--max-samples N` for a bounded smoke/subset run. Use a separate output
directory for smoke runs so their subset JSON cannot be mistaken for a full
feature set. Resume is driven by the append-only per-utterance progress file.
The extraction manifest embeds the MELD train mean/std, its population record,
and the statistics-file hash.
