# GS-MCC official-code audit (2026-10-06)

Status: deferred; no training started. Server ownership: biggpu; GPU 4 forbidden.

Paper: AAAI 2025, https://ojs.aaai.org/index.php/AAAI/article/view/33242
Official repository: https://github.com/FuchenZhang/GS-MCC
Audited commit: `38c4038a7738f9bf7b3132c3e99a126e1cf1f28d`.
Local unmodified source: `/data2/yb/paper/baseline_sources_20261006/GS-MCC`.

## Why not launch three seeds yet

- README points to absent `train.py`; actual entry is `train_fourier.py`.
- Entry overwrites `CUDA_VISIBLE_DEVICES=0, 1`; remove override before any GPU execution.
- MELD initializes loss with undefined `loss_weights` (train_fourier.py 366–373).
- Dataset code expects IEMOCAP 12-field multimodal pickle and MELD 10-field pickle; model audio/visual constructor dimensions remain hardcoded 342/1582 for both datasets. MELD expects text/audio/visual widths 600/300/342, so defaults fail.
- Forward converts time-first inputs to batch-first but calls all three `nn.LSTM` instances with default `batch_first=False` (FourierGNNmodel.py 771–776, 843–852), causing recurrence across examples.
- `batch_graphify` keeps all padded nodes for each dialogue, but accumulates edge offsets using true dialogue lengths (375–400). With unequal lengths, later-dialogue edges target wrong nodes.
- Epoch accuracy/weighted F1 ignore padding mask (train_fourier.py 180–181), whereas final classification report applies it. Checkpoint selection uses the unmasked test F1.
- Repository selects peak test F1 over 250 default epochs and resets RNG inside each train/eval call. No multi-seed CLI or checkpoint outputs.
- Published paper describes coordinated high/low-frequency contrastive losses. Available entry instead trains main and unimodal NLL plus unimodal-to-fused KL; no explicit LFCL/HFCL computation was found. FGN receives `[nodes, hidden_features]`, applies FFT along hidden features, rather than across conversation nodes (FourierGNNmodel.py 579–607, 627–630). Repairing runtime alone would not establish faithful paper reproduction.
- Original model uses bidirectional recurrent modules and default future graph window 10. It is not a history-only baseline; setting future window zero does not remove recurrence leakage.

## Available assets

Read-only check confirmed biggpu `/data2/yb/reproduction_envs/s0/bin/python` has torch, torch_geometric, pandas, sklearn; no installation needed.
Existing IEMOCAP candidate: `/data2/yb/paper/tsne_baselines_20260728/SDT/data/iemocap_multimodal_features.pkl` (12 fields, 120/31 dialogue train/test split, text1024/audio1582/visual342).
Existing MELD candidate: `/data2/yb/paper/tsne_baselines_20260728/MMGCN/MELD_features/MELD_features_raw1.pkl` (10 fields, 1152/280 dialogue train/test split, text600/audio300/visual342).
No data transformation, new dependency, GPU smoke run, or formal training was performed. Choose another reproducible recent baseline before considering a separately labelled reconstruction.
