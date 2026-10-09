# IEMOCAP raw-concatenation MLP control — 2026-10-09

User requested the same plain-concat MLP diagnostic on IEMOCAP. One seed2025, no hyperparameter search. Architecture: raw text1024 + audio1024 + visual342 → Linear(2390,256) → GELU → Dropout(.2) → Linear(256,6), **613,638 parameters**. No modality projection, normalization, gating, MCA, AMM, EPIRC, pooling or auxiliary loss.

## Frozen data and actual training semantics

`source_dataset.py`, `source_multiattn.py`, and `source_config.json` are byte-identical to commit `1b8b1ff` vendor source. Fill53Dataset loads the same CSS metadata and packed textpeak/audio/CSS-visual features as the Full reference:5810 training and1623 test utterances, class order happiness/sadness/neutral/anger/excited/frustration. Original feature augmentation and missing-feature representation are unchanged. File hashes and ordered sample identities are recorded.

Batch32, accumulation2, up to100epochs, original earlystop30 and LRplateau5×.5. AdamW weightdecay.0002, hiddenLR.00006/classifierLR.00012: actualFull fusionLR.00003×mixer_multiplier2 with outputclassifier multiplier2. Original LambdaLR warmup20% then cos(2πprogress), totalsteps100×182minibatches but advanced per optimizer update. Gradientclip1, EMA.999, evaluation and strictpeaktestWF1 selection use EMA. As with originaltrainer, odd finalmicrobatch has optimizer/scheduler step but noEMA; trainloader182batches so no odd remainder. This is the user's existing test-selected diagnostic, not validation-selected performance.

Original main objective: class-weighted per-sampleCE + Poly correction, batch mean, then multiplied by mean detached focal coefficient and main weight.45. Auxiliary dict empty, avoiding originalcriterion(None) shortcut. Criterion uncertainty logvariance remains zero because originaloptimizer excludes loss parameters. Value and gradient equality to frozen original are explicitly checked, with RNG state restored. No auxiliary objective. Dropout.2 matches the actual Full architecture and previous MELD plainMLP, even though legacy JSON contains fusion_dropout.1.

## Execution and evidence

Serverbiggpu, physicalGPU1 UUID`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`, GPU4excluded. Isolated root `/data2/yb/multimodalERC/MM_Mixer_ConcatMLP_IEMOCAP_20261009`, environment `/data2/yb/reproduction_envs/s0/bin/python`. Existing two lightweightGPU1runs remain untouched; preflight memory1010MiB utilization2–17% permits this smallthirdprocess.

Four-minibatch real smoke checks finite loss/gradients, allfourMLP parameter tensors updated, and fresh strictreload exacttestlogits. Formalrun uses a new seed-reset process and independent directory. Peakcheckpoint includes raw/EMAmodel,optimizer,scheduler,RNG and config. All epochmetrics, source/environment/datahashes, predictions and finalstrictreplay are saved. Initial smoke before earlystopconfiguration correction remains remotely `smoke_pre_earlystop`; only finalsource smoke is launch evidence.

Raw NPZ arrays are cached in host RAM once to avoid repeated decompression. The original dataset augmentation/copying path is unchanged. Original-vs-cache train/test batches are bit-identical, post-augmentation Python/NumPy/Torch RNG states match and cache arrays are not mutated. Cached smoke yields exactly the same loss, logits and prediction hash as uncached smoke. This is an I/O optimization, not changed features/preprocessing.

Status: completed all100epochs, EMA peak epoch82, exit0, fresh strictcheckpoint replay exact. WF1=70.12560602033261%, ACC=70.11706715958103%. Local confusion-matrix metric recomputation, prediction SHA256 and epoch-selection checks pass. No TeX modified.
