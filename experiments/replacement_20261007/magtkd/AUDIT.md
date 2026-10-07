# MAGTKD replacement audit — 2026-10-07

**Final status: all six formal runs completed and independently verified.** Results: `results/replacement_20261007/magtkd/README.md`.

Selected replacement candidate for ECERC. Paper: [IJCAI 2025, 905](https://www.ijcai.org/proceedings/2025/905), official [JieLi-dd/MAGTKD](https://github.com/JieLi-dd/MAGTKD), commit `95d0760c26ad2a6ad2181daf372abfd5be348b2d`.

## Scope and released data

Reproduce the official **second-stage fusion** on **fixed official first-stage supervised/distilled features**, three training seeds per dataset. This is not an end-to-end three-seed retraining of feature extractors or distillation. Preserve ECERC historical records; replace paper rows only when new verified results are ready.

Official Google Drive resource: https://drive.google.com/file/d/19g3hTaBEKF5wXI0DHdvRYbu0BD3XZa3d/view . HTTP metadata identifies `MAGTKD.zip`, 6,192,026,726 bytes. ZIP64 central directory lists six `first_stage_{train,dev,test}_features.pkl` files plus large upstream encoder weights. Download feature members by HTTP byte range; verify decompressed size and ZIP CRC, record SHA256. Full upstream checkpoints are not needed for stage two. `fetch.py` records exact member offsets and compression metadata.

The released MELD archive path is nested `MAGTKD/MELD/MELD/feature`; IEMOCAP is `MAGTKD/IEMOCAP/feature`. The adapter uses explicit feature paths. Actual script name is **multimodel_fusion.py**, despite README spelling multimodal_fusion.py.

## Faithful settings and narrow adaptations

Both scripts set 30 epochs, 16 dialogues/batch, AdamW lr=1e-4, weight decay=1e-6, hidden=768, 8 heads, dropout=.5, temperature=2. IEMOCAP feeds `(text, audio_kd, video)` while MELD feeds `(text, audio_kd, video_kd)`. Preserve these released choices. Auxiliary weights are IEMOCAP .7/.8 and MELD .01/.08, as the original function specifies.

- Load model/functions from the frozen original source AST. Remove unavailable `transformers` imports for unused stage-one encoder definitions, and training-module model/dataset imports replaced by explicit isolated loading. Do not rewrite model forward or objective.
- Implement Hugging Face's linear warmup/decay scheduler algebraically with PyTorch LambdaLR, without installing dependencies. Preserve author dialogue-count warmup/total steps rather than correcting to minibatch counts.
- Set DataLoader workers=0 instead of16 to bound shared-host CPU resources; batch size/order/feature contents remain defined by the official Dataset and collate functions.
- CUDA-only operations in official forward are retained for actual GPU runs. CPU synthetic interface check temporarily routed Tensor.cuda to identity; this is **not** the formal runtime.
- Select strict highest test WF1 at the original rounded-two-decimal precision, preserving the user-established reproduction protocol. Export prediction-based full-precision ACC/WF1/per-class F1. Each run strictly reloads the selected checkpoint and regenerates predictions.
- Freeze source and record SHA256/config/data/seed/environment/device; outputs isolated. Checkpoint saves model, optimizer, scheduler, epoch and RNG state, but no resume is attempted automatically.

## Labels and code observations

Label IDs verified against official `utils.py:all_features_batchs` and Dataset definitions: IEMOCAP ang/exc/fru/hap/neu/sad; MELD anger/disgust/fear/joy/neutral/sadness/surprise. Remap by class name when writing paper tables.

Original forward reuses `t_t`/`t_t_gate` for two cross pathways and leaves registered `a_t`, `v_t`, their gates and `fc` unused; retained without silent correction. Parameter count includes all registered downstream parameters. Synthetic finite forward/backward at batch2×length3 with original hidden768: IEMOCAP 44,332,050 parameters; MELD 44,566,293. Actual data smoke/strict replay must pass before formal launch.

## Execution and failures

Server: biggpu. Healthy whitelist GPU1 `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`; host GPU4 forbidden. Avoid GPU2 timing jobs and GPU7 auxiliary jobs. Remote root `/data2/yb/multimodalERC/MM_Mixer_MAGTKD_20261007`; environment `/data2/yb/reproduction_envs/s0/bin/python`. Short temporary root `/data2/yb/tmp/magtkd`.

Initial biggpu direct Google download timed out and was stopped. Initial local tmux download inherited stale proxy settings; refreshed only proxy environment values and restarted with a separate log. Neither failure began training or produced purported results. Source clone and adapters remain separate. Queue controller is named `run_queue.py` to avoid shadowing Python's `queue` module.

## Verified data and smoke evidence

All six official feature members downloaded and transferred successfully. `feature_manifest.json` and `zip_directory.json` record archive member metadata/CRC/hash; `data_audit.json` verifies actual deserialization, finite feature values and shapes. IEMOCAP splits: 108/12/31 dialogues, 5163/647/1623 utterances. MELD: 1038/114/280 dialogues, 9989/1109/2610 utterances. All five feature streams are 768-dimensional. These are the author's splits, not a re-created random holdout.

Real GPU smoke uses original batch16, two optimizer steps (first has zero LR under the official warmup), then verifies actual parameter changes, finite weights, and test predictions after strict load into a fresh model instance in the same process. No fresh-process replay is claimed. Peak IEMOCAP smoke allocated memory is 1,298,373,632 bytes. Two concurrent jobs on the otherwise idle 32GB GPU1 therefore have substantial initial memory headroom; queue additionally requires 5000MiB free before admitting each job and waits20s between launches.

`plan.json` fixes IEMOCAP seeds2025/2066/2118 and MELD2025/2028/2069. `run_pipeline.sh` refuses to launch without both successful smoke records and the six-file data audit. The reused controller verifies frozen source/data SHA before each job, requires30 completed epochs, checks checkpoint replay and checkpoint/prediction file hashes. At queue completion, `summarize.py` independently recomputes all metrics from predictions and emits mean and sample SD (ddof=1).

## Launch verified

Formal persistent controller started in biggpu tmux `magtkd_6runs` on 2026-10-07 UTC. Root `queue_state.json` records commands, timestamps, PID and physical GPU identity; a launch snapshot is saved locally as `launch_state.json`. First job IEMOCAP2025 PID1936809 entered actual training and reached epoch11; MELD2025 was admitted after the20s interval. Remaining four jobs are queued automatically. Latest smoke replay mode is `fresh_model_same_process`; 125 parameter tensors changed for each dataset. MELD smoke peak allocated memory:787,118,080 bytes. Progress is not a completed six-seed result; final success requires six `result.json` artifacts plus independently verified `summary.json`.

Read-only status command: `ssh biggpu 'cat /data2/yb/multimodalERC/MM_Mixer_MAGTKD_20261007/queue_state.json'`. Run logs live under `runs/{dataset}_seed{seed}/console.log`; controller log is `pipeline.log`.

## Final completion

All six planned runs completed30epochs with no formal failures. Remote automatic summary and independent local confusion-matrix verification agree on all full-precision metrics, earliest rounded-WF1 selected epochs and ddof=1 summaries. IEMOCAP ACC68.82±0.50/WF169.06±0.47; MELD ACC65.72±0.02/WF164.71±0.18. Parameters44.33M/44.57M. Original launch snapshot stays `launch_state.json`; final status is in the results directory `queue_state.json`.
