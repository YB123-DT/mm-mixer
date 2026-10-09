# IEMOCAP plain concat + MLP result

Completed seed2025,100epochs, selected EMAepoch82, strictpeaktestWF1 diagnostic. No auxiliary losses; architecture2390→256→6,613638parameters.

| Model, same seed2025 | WF1 (%) | ACC (%) | Parameters |
|---|---:|---:|---:|
| MM-Mixer Full |71.92654|71.78065|6,358,082|
| Raw concat + MLP |70.12561|70.11707|613,638|

PlainMLP is1.80093WF1points and1.66359ACCpoints below Full in this one run. This differs from the previous MELD near-tie; it does not identify the contribution of any one module because all fusion modules and auxiliary losses were removed together. One seed does not establish significance or stability. No additional runs or tuning were performed.

Test classF1 in original order: happiness53.23741, sadness82.54620, neutral68.76640, anger71.09827, excited72.72727, frustration67.41573.

Formal elapsed475.27seconds after loading. All100 epochmetrics preserved; bestepoch recomputed. Fresh model strictcheckpoint reload reproduces all1623test logits exactly. Local metric recomputation from predictions independently passes at1e-12 tolerance. Source/data/environment/effectiveconfiguration and cache-equivalence checks are in `experiments/concat_mlp_iemocap_20261009/`. Checkpoint stays on biggpu `/data2/yb/multimodalERC/MM_Mixer_ConcatMLP_IEMOCAP_20261009/runs/iemocap_seed2025/test_peak.pt`, not Git.

Recheck from repositoryroot: `python experiments/concat_mlp_iemocap_20261009/verify_results.py` (NumPy only). Paper unchanged.
