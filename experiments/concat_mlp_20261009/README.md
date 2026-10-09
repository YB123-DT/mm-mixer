# MELD raw-concatenation MLP control — 2026-10-09

User requested a single training run with only concatenated raw features and one MLP. Predeclared architecture: text1024 + audio1024 + visual342 → Linear(2390,256) → GELU → Dropout(0.2) → Linear(256,7), **613,895 parameters**. No modality projections, normalization, feature/adaptive gating, MCA, AMM, EPIRC, query pooling or auxiliary classifiers/losses. Hidden width is fixed before results, with no search.

## Data and protocol

Frozen `vendor/meld/multiattn.py` and feature configuration come verbatim from commit `1b8b1ff`; copies are `source_multiattn.py` and `source_config.json`. The source module supplies the original MELDDataset, augmentation, collate and loss reference; none of its fusion architectures is instantiated. Same original train/dev/test CSVs, author feature paths, labels and missing-feature zero filling. Feature and CSV SHA256/sample identities are recorded in `data_manifest.json`.

MELD only, seed2025, 50 epochs, batch32 utterances, accumulation2, AdamW wd9.97646370349997e-5, clip1. Hidden linear uses original base fusion LR2.0832826726482106e-5; output classifier uses original classifier multiplier2 (4.166565345296421e-5). Scheduler reproduces original 20% warmup + cos(2πprogress), total steps defined as 50×number of minibatches, but stepped per optimizer update, including original partial-accumulation behavior. Original training feature augmentation is retained, workers0. EMA decay0.999 is updated only on full accumulation steps; dev/test and checkpoint selection use EMA, matching original training.

Main objective is unweighted CE plus Poly correction alpha1.9508055462649292, gamma1.402459950741192, times differentiable per-sample focal coefficient exponent2.5. No auxiliary terms. Value/gradient are checked exactly against the frozen vendor loss with **empty auxiliary dict**, not `None` (which bypasses focal reweighting). Verification saves/restores RNG so it does not alter the training stream.

The existing explicitly user-authorized **strict peak test-WF1** selection protocol is preserved; this is not validation-selected performance. Report one seed without implying a three-seed mean.

## Execution

Server `biggpu`, isolated root `/data2/yb/multimodalERC/MM_Mixer_ConcatMLP_20261009`, Python `/data2/yb/reproduction_envs/s0/bin/python`. Physical GPU1 UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049` maps to process cuda:0; GPU4 excluded. Prelaunch GPU1 used996MiB at1–3% utilization with two independent MM-Mixer jobs, permitting this lightweight third process. Other jobs are untouched.

Persistent tmux `concat_mlp_meld_20261009`: final four-minibatch real-data smoke → fresh strict prediction replay → full50 epochs. `run_pipeline.sh` refuses formal launch if smoke failed. Training saves per-epoch metrics, effective configuration, environment/source hashes, data manifest, optimizer/scheduler/raw-model/EMA/RNG peak checkpoint and prediction NPZ; a fresh model strictly reloads selected EMA and must reproduce logits bit-for-bit.

An initial non-EMA implementation was identified before formal training proceeded. Its loading-only launch was stopped and preserved remotely under `pre_ema_aborted`; no formal score is used from it. A subsequent smoke predating RNG isolation is also retained remotely. Only the final source-hash-matched smoke and formal run are evidence.

Status: completed all50 epochs, selected EMA epoch48, exact fresh replay passed, exit0. Test WF1=67.88797397685946%, ACC=68.77394636015326%; see `results/concat_mlp_20261009/summary.json`. Same-seed Full WF1=67.88359990807896%, so this single run is effectively tied; no multi-seed superiority claim.
