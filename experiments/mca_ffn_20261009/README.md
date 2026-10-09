# MCA × AMM and FFN width controls — 2026-10-09

Base commit: `1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab`, identical to original revision controls. Code is extracted into an isolated snapshot; unrelated working-tree changes are excluded. `controls.patch` is the complete source difference.

New runs: `no_mca_no_amm`, `ffn_width_256`, `ffn_width_512`, `ffn_width_768`; both datasets, three original seeds each, 24 runs. Reuse original Full/no MCA/no AMM reference runs; do not rerun or select references by outcome. Width1536 is original Full. Fixed S6/D256/depth2, losses, learning rates, batch32, input features, and original100/50epoch budgets.

The joint deletion starts from the original no-AMM path retaining EPIRC and disables MCA through the same `disable_channel_attention` method as the original individual deletion. FFN widths replace the two Linear layers in each of two existing blocks after original Full initialization, preserving every non-FFN initial tensor. New FFN tensors use ordinary PyTorch Linear initialization with the existing RNG stream; this is width replacement, not weight truncation. All widths retain GELU and feature256 output.

Server biggpu, physicalGPU1 UUID`GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`; GPU4 excluded. Start2concurrent, same batch and precision. Remote root `/data2/yb/multimodalERC/MM_Mixer_MCA_FFN_20261009`. Existing environment `/data2/yb/reproduction_envs/s0/bin/python`, no dependencies added. Previous18runs consumed5.5GB;24estimated7.3GB against117GBfree at preparation.

Selection remains explicitly user-requested strict peak test WF1, matching reference runs. These are test-selected diagnostic comparisons, not validation-selected generalization estimates. Preserve all outcomes; no result-dependent extra widths or seeds.

Validation: original Full tensor state/output bitexact regression; actual shape/switch checks; FFN optimizer membership and finite gradients/updates; retained EPIRC active parameters update while historical unused paths may have no gradient. Eight one-epoch real-data smokes validate full test prediction publication and fresh strict checkpoint replay, separate from formal results.

Initial test-harness correction: added variant declarations add false metadata switches to Full config; compare tensor hashes, architecture and parameter count rather than entire metadata dictionary. No model change was needed. Unused EPIRC parameters are not falsely required to receive gradients.

MELD parameter reconciliation: CPU `build_variant_model` helper registers6,106,825 parameters, but the actual unchanged trainer config has `no_alignment:true` and calls `disable_alignment()` before optimizer creation. Exactly394,755 `feature_aligners` parameters (15 tensors across v_a/v_t/a_t) disappear; all other checkpoint/helper shapes match. Formal Full is5,712,070, widths256/512/768 are4,398,790/4,661,446/4,924,102; joint-off2,947,952. CPU probe counts are helper-stage counts, not paper parameter totals. The eight real-data smoke checkpoints verify actual training constructors and fresh reload.

Smoke-verifier correction: MELD can publish multiple immutable bundles during final replay. Audit the canonical `best_peak` pointer rather than falsely asserting only one historical bundle exists. Completed valid smoke runs are reused; no formal run is selected from smoke results.

Validation completed: 10 CPU probes, all8real-data one-epoch smokes, canonical checkpoint hashes, fresh strict replay, actual FFN shapes and deletion paths, and all426snapshot source hashes passed. IEMOCAP checkpoints include1,536 constant residual-scale buffer elements for two active Mixer blocks; these are excluded from registered parameter counts.

Formal launch: persistent tmux `mca_ffn_24runs`, 2026-10-09 on biggpu GPU1, maximum2concurrent. Queue automatically fills freed slots. Current execution is running, not a completed result. `pipeline/state.json` tracks24jobs; completion requires queue exit0 plus verified complete bundles. `run_pipeline.sh` automatically writes `analysis.json` and `analysis.md` after queue completion. Root comparison report combines the verified reused references with these new results.
