# Corrected Full: code-to-Method alignment

| Manuscript component | Frozen implementation | Status |
|---|---|---|
| Three gated utterance-level input vectors | `vendor/*/multiattn.py` | Active |
| Modality-centric cross-attention | `vendor/*/multiattn.py` | Active |
| Two alternating sequence/modality/feature mixer blocks | `vendor/iemocap/factorized_mixer/model.py` (`HO_WO_TAV` -> `M4_LR2_D2_H1536`); `vendor/meld/model.py` (`MX_LR1`) | Active on both |
| Learned modality-axis routing `W_r` | Bias-free `nn.Linear(3,3)` in both mixer implementations | Active on both |
| Pairwise residual | IEMOCAP `StructuredHighOrderCross`; MELD `PairwiseCross` | AV/TV/TA active; TAV disabled |
| Three unimodal auxiliary losses | Dataset-specific training loss | Active |

The released pairwise modules allocate a fourth rank-`R` input slot and a
`Linear(4R,D)` projection for compatibility with the earlier interaction
parameterization. That fourth slot is **fixed to zero** in corrected Full;
only three pairwise Hadamard products contribute. The manuscript Method
now writes the effective input as `[r_AV; r_TV; r_TA; 0_R]`.

The architecture is shared, but the final training objectives are *not*
identical:

- IEMOCAP: class-weighted PolyLoss averaged over the batch, multiplied by
  the batch-mean detached focal factor. Effective task coefficients are
  `main/T/A/V = 0.45/0.33/0.11/0.11`.
- MELD: per-sample, unweighted-main PolyLoss multiplied by a differentiable
  per-sample focal factor, then averaged. The auxiliary PolyLoss remains
  class weighted. Effective task coefficients are `1/1/1/1`.

The IEMOCAP legacy config still sets `use_uncertainty=true`, but the loss
module's `log_vars` are not included in the fusion optimizer. They remain
zero, yielding the fixed coefficients above; this is **not** a learned
uncertainty-weighting result. The MELD runner encodes the effective unit
weights directly. `skip_pretrain=true` in both published configurations:
the reported training runs consume the pre-extracted features rather than
fine-tuning the feature extractors inside this repo.

The manuscript's main IEMOCAP table uses the corrected Full peak-test
three-seed subset. Its MELD main table has been updated to the same
corrected Full peak-test protocol. Older ablation/efficiency measurements
in the draft are **not** part of this release and should be re-audited
before being attributed to this frozen configuration.

Provenance caveat: the eight result manifests retain the source hashes from
training time. Current released source differs in some runner, config,
and ablation-enabled model files after the runs. The score/config artifacts
match their manifest hashes, but **byte-identical source replay has not been
established** for every seed.
