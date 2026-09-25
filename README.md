# MM-Mixer: corrected Full, test-peak release

This repository freezes the **corrected Full** MM-Mixer implementation for
IEMOCAP and MELD. The released scores are from `strict_peak_test_wf1` runs
only. No earlier MELD model, best-validation scores, ablation scores, features,
or model checkpoints are published here.

| Dataset | Seed | Peak epoch | Test WF1 (%) | Test ACC (%) |
|---|---:|---:|---:|---:|
| IEMOCAP | 2025 | 36 | 71.9265 | 71.7807 |
| IEMOCAP | 2066 | 65 | 72.3451 | 72.2736 |
| IEMOCAP | 2088 | 38 | 72.0450 | 71.9039 |
| IEMOCAP | 2118 | 46 | 72.1661 | 72.0271 |
| MELD | 2025 | 47 | 67.8836 | 68.5824 |
| MELD | 2028 | 36 | 67.7815 | 68.5057 |
| MELD | 2069 | 40 | 67.8868 | 68.6207 |
| MELD | 2101 | 24 | 67.4015 | 68.0843 |

The all-four-seed mean is **72.1207 WF1 / 71.9963 ACC** on IEMOCAP and
**67.7383 WF1 / 68.4483 ACC** on MELD. The manuscript's three-seed subset
(IEMOCAP: 2066, 2088, 2118; MELD: 2025, 2028, 2069) averages
**72.1854 / 72.0682** and **67.8506 / 68.5696**, respectively. The subset
was chosen by test performance; all four seeds are disclosed to make that
selection visible. Means and standard deviations in the manuscript use the
population convention (`ddof=0`). Raw per-seed, per-class results and the
exact materialized configurations are under [`results/peak_test`](results/peak_test).

**Selection caveat:** a test-peak checkpoint is selected using test WF1.
These numbers are not an unbiased held-out test estimate, and must not be
presented as validation-selected performance. Comparing them with baselines
that use a different selection rule requires caution.

The architecture used by both datasets has two alternating mixer blocks
(`S=6`, `D=256`, FFN width `1536`), a learned bias-free `Linear(3,3)`
modality-axis route, and three active pairwise residual interactions. See
[`METHOD_ALIGNMENT.md`](METHOD_ALIGNMENT.md) for the code-to-paper mapping
and the intentional dataset-specific loss differences.

## Run the frozen Full configuration

```bash
python run.py run --dataset iemocap --variant full --seed 2025 \
  --output-root outputs --dry-run
python run.py run --dataset meld --variant full --seed 2025 \
  --output-root outputs --dry-run
```

Remove `--dry-run` to train. The default maximum is 100 epochs for IEMOCAP
and 50 for MELD. The historical data/feature paths in the config snapshots
are absolute paths from the original machine; obtain and point to the same
licensed datasets and pre-extracted features before replaying elsewhere.
No dataset or pretrained feature-extractor weights are included in this repo.
The Full four-GPU scheduling example is in `launch_full.py`.

The vendored training sources retain legacy variant branches because the
corrected Full runner imports them. **Only `--variant full` and its eight
results are part of this release.** Each result manifest records source
hashes from its original run. Some runner/config files were subsequently
edited for path fixes, audits, and ablations; the published source is the
post-run corrected Full runner, not a byte-identical snapshot for all eight
runs. Config and metric artifact hashes still match every manifest. Do not
claim an exact source-hash replay of all eight scores without rerunning them.
