# MELD RoBERTa-large Test Top-4 diagnostic

This diagnostic reproduces the historical MELD RoBERTa-large training setup
and retains the four epochs with the highest test weighted F1. The resulting
checkpoints will be used to extract four text-feature sets and measure whether
MM-Mixer ablation gaps change with upstream text strength.

Configuration: history-only context, right truncation to 511 tokens plus a
final MASK token, MASK-state pooling, seed 42, batch size 4, encoder learning
rate 1e-6, classifier/projection learning rate 5e-5, weight decay 0.01,
10 epochs and dev early-stopping patience 3.

The Top-4 ranking uses test labels at the user's request. These checkpoints
are diagnostic controls and must not be described as validation-selected
publication results.

After training, `extract_top4.py` reads the ranked checkpoint index, extracts
history-aware EOS-pooled features for train/dev/test, verifies exact split
coverage, 1,024-dimensional finite vectors and records checkpoint/feature
hashes. This deliberately matches the existing downstream feature-export
semantics even though upstream training uses the final MASK representation.

The initial downstream screen runs seed 2025 for Full, text only, no AMM,
no EPIRC, no MCA, and no Feature Gating plus no Adaptive Gating on every
ranked feature set. Four runs share one healthy GPU. Every selected bundle
must pass exact replay and contain the expected text-feature hash before it is
included in `screening_results.csv`.
