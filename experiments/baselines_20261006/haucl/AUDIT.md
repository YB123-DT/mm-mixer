# HAUCL preliminary audit (2026-10-06)

Paper: ACM Multimedia2024, https://arxiv.org/abs/2408.00970.
Paper's original repository https://github.com/yzjred/-HAUCL currently returns404.
A repository under coauthor Ziming Zhao's account is available: https://github.com/zhziming/HAUCL, commit6a295b878f4faacac4b0cb5539debf2fdc056bb8. Its README links the matching paper and M3NET features. No trained results claimed here.

MELD feature files already exist on biggpu under `/data2/yb/paper/tsne_baselines_20260728/M3NET/MELD_features/`: `MELD_features_raw1.pkl` and `meld_features_roberta.pkl`. Loader requires raw file named `MELD_features_raw.pkl`; an isolated symlink can provide the name after checking structures.

Important limitations before launching:
- Only MELD loader, model initialization and loss branch are published. Selecting IEMOCAP yields undefined model; do not invent missing dataset hyperparameters and call it paper reproduction.
- hypergraph.py computes both `out1_aug` and `out2_aug` from `out1`, so nominal two-view contrastive learning compares projections of the same branch. Keep this behavior explicitly labelled released-code reproduction, or obtain an author-verified correction.
- Model has bidirectional text GRU and dialogue-wide hyperedges; full-context/offline comparison, not history-only.
- Default train budget is10epochs, batch12, dim512, graph_dim512, Adam lr5e-5/wd3e-5, dropout0.4, CL0.5 and generative0.1. Entry selects peak test weightedF1 and resets RNG inside every train/eval call; no CLI seed/checkpoint writer.
- Required PyG/torch_scatter dependencies are already expected in existing graph baseline environment, but no actual import/forward smoke test has yet been performed.

Status: deferred after screening; main task prioritizes two runnable2025 methods. MELD-only released-code reproduction would still need smoke verification and explicit caveats. No formal training started. Any future wrapper must preserve safe CUDA_VISIBLE_DEVICES, explicit seeds2025/2028/2069, separate outputs, source hash, metrics and checkpoint selection record. Server biggpu only, physical GPU4 forbidden.
