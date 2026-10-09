# External review prompt

Please audit the linked MM-Mixer MELD implementation and result files. The
specific failure to explain is that removing most modules changes weighted F1
by only a few tenths of a point; some removals slightly improve it, and the
EPIRC removal produces effectively identical predictions/metrics.

Answer these questions using concrete code references:

1. Are Feature Gating, Adaptive Gating, MCA, AMM, EPIRC and auxiliary heads
   actually on the forward/optimizer path in Full training?
2. For each ablation, does the implementation remove only the intended
   operation while keeping inputs, capacity and initialization comparable?
3. Do residual/bypass paths allow the remaining network to reconstruct nearly
   the same mapping after a module is removed?
4. Does zero-initialized EPIRC output, its optimizer partition, or its input
   source explain why `no_pairwise` is indistinguishable from Full?
5. Are the three modality branches numerically balanced after projection and
   normalization, or can the history-aware, MELD-supervised text vector
   dominate gradients and query pooling?
6. Given that text+audio matches Full and the visual artifact has many zero and
   repeated vectors, which conclusions can and cannot be drawn about AMM?
7. How much can per-variant test-peak epoch selection inflate ablations and
   compress paired differences?
8. Propose the smallest decisive experiment sequence. Prioritize diagnostics
   that distinguish an implementation bug, redundant parameterization,
   feature dominance and genuinely ineffective interaction modules. Do not
   suggest broad hyperparameter searches before checking these mechanisms.

Please separate confirmed defects from plausible hypotheses, and state what
evidence would falsify each hypothesis.
