# VEGA official implementation audit — 2026-10-07

- Official source: https://github.com/dkollias/VEGA
- Commit: `a2779205cca64ade5f3f0563283c58310a33e20a`
- Publication reported by official README: ACM MM 2025, DOI `10.1145/3746027.3755340`.
- Local clone: `/data2/yb/paper/baseline_sources_20261007/VEGA`.

## Decision

Do not select as a complete two-dataset ECERC replacement at present. IEMOCAP has an official six-class configuration, feature file and visual anchors. MELD has code branches but no released MELD features, seven-class visual anchors or dedicated hyperparameter configuration in this repository. Inventing missing class anchors would change a central part of the method.

## Released resources, verified HTTP 200 attachment metadata

| Resource | Bytes | URL |
|---|---:|---|
| IEMOCAP.pkl | 98,389,497 | https://drive.google.com/file/d/1dx4yikoU8hYZ7FxyrRwzcANzaJBcg90N/view |
| 35_anchor.pt | 451,354 | https://drive.google.com/file/d/1F-ajsUUHihO0RgREros5AJUiptuVOoIl/view |
| 35_anchor.zip | 15,164,404 | https://drive.google.com/file/d/1DOmYn6tISoEPJ4PQDD4F-gB1M58G-NS1/view |

Actual content not downloaded or deserialized; sizes come from direct download Content-Length, not an inferred Drive page size.

## Implementation evidence

- `configs/iemocap_config.py` is the only dataset configuration. `run.py` always loads it, including when `--Dataset MELD` is selected.
- `dataloader.py` defines both IEMOCAPDataset and MELDDataset, expecting seven-part multimodal pickles. MELD path is `data/meld.pkl`, but README only links `IEMOCAP.pkl`.
- README documents six anchor labels: happy, sad, neutral, anger, excited, frustration. MELD needs neutral, surprise, fear, sad, happy, disgust, anger.
- `vega_utils/anchor_utils.py` uses `anchor/{expr_img_folder}_anchor.pt` for both datasets and returns any structurally valid cache without checking requested labels. Reusing the IEMOCAP cache for MELD is invalid.
- `model.py` draws anchor features using dataset-specific emotion labels; absent MELD classes cannot be solved by changing a data path.

## Existing environment

`biggpu:/data2/yb/reproduction_envs/s0/bin/python` has torch, numpy, pandas, sklearn, tqdm, PIL, pytz; transformers is absent (find_spec check). `main.py` imports transformers scheduler unconditionally despite scheduler=False default. A lazy import would remove that unused dependency for training with ready anchors; building new anchors still requires transformers and CLIP weights.

No dependency installed, no source edits, no GPU use, no formal training. IEMOCAP-only training is plausibly possible with official ready features/anchors and a narrowly documented import adaptation; full model forward was not tested because the candidate fails the required two-dataset resource gate.
