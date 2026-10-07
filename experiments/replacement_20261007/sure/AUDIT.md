# SURE screening — 2026-10-07

Official repository: https://github.com/swaggy66/SURE
Venue independently confirmed by ICASSP official accepted program: ICASSP 2026, paper 4540.
Audited commit: dc42e5053e940e7f82d84ba36d912b59cebefd4d.
Source clone: /data2/yb/paper/baseline_sources_20261007/SURE.

Not selected; no training or dependency installation.

- train.py imports MaskedNLLLoss, MaskedKLDivLoss and Transformer_Based_Model from model, but model/ has no __init__.py exporting them; actual model class is in main.py and the loss classes are not supplied in the visible source.
- main.py constructs DialogueMoE without importing it.
- model/moe.py Router.forward uses masked_probs without defining it. Reconstructing the intended routing normalization would change a central module without an authoritative implementation.
- train_or_eval_model initializes labels=[] but never appends labels before np.concatenate(labels).
- Hardcoded CUDA_VISIBLE_DEVICES=0 and absolute IEMOCAP path need operational adaptation; these are not the main blockers.
- README advertises both datasets; run.sh only gives IEMOCAP settings. Default train.py exposes MELD, but the preceding failures prevent a faithful runnable baseline without reconstructing implementation.

Because a separate IJCAI 2025 candidate (MAGTKD) provides complete stage-two code and author-released features, prioritize that candidate rather than guessing missing SURE behavior.
