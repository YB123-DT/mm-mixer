# GatedxLSTM replacement audit — 2026-10-07

## Decision

**Reject as the immediate ECERC replacement.** The released experiment is IEMOCAP **four-class**, not the required IEMOCAP six-class / MELD seven-class pair. Adapting the protocol, repairing feature extraction and implementing a MELD entry point would be substantial reconstruction rather than a straightforward reproduction. No training or package installation was performed.

## Sources and provenance

- Official repository: https://github.com/glam-imperial/GatedxLSTM
- Inspected commit: `7f82ed155d11179bba69cbe01119fa583a12317a`
- Local checkout: `/data2/yb/paper/baseline_sources_20261007/GatedxLSTM`
- Paper: https://arxiv.org/abs/2503.20919 (submitted 2025-03-26); abstract explicitly describes IEMOCAP four-class evaluation. Repository README states acceptance at ACII 2025.
- Included `ded_code.zip` was listed; it contains an older DED package and `main2.py`, not evidence of a documented complete MELD reproduction pipeline.

## Code findings

Paths below are relative to the official checkout.

1. `Dialogical-Emotion-Decoding/main.py` fixes `selected_classes = [0, 5, 6, 8]`, maps them to four labels and constructs the final model with four classes (line 388). It loads only IEMOCAP CLAP arrays. MELD-specific DED helper files exist, but no complete MELD feature/training entry point is documented or present among the unpacked source files.
2. Effective splitting is dialogue-random 80/10/10 (`main.py:326–359`), overwriting earlier utterance-random splits. This is not MM-Mixer's fixed six-class IEMOCAP split. Seed 42 is hard-coded. The classifier checkpoint is selected by validation weighted F1 (`main.py:447–450`).
3. Main requires `clap_with_oppo_IEMOCAP_audio_features_512.npy` and the corresponding text file. Neither is included or linked for download. README requires licensed IEMOCAP raw data and local extraction. Included DED pickle files are not these CLAP feature arrays.
4. `data/preprocess.py:257,295` uses 768-dimensional zero placeholders for 512-dimensional CLAP vectors. The audio branch saves to a **text** filename (`:285`); the text branch loads the **audio** array (`:289`) and overwrites that same text output (`:322`). Consequently the documented preprocessing does not produce the required audio filename and does not faithfully build separate text features.
5. `main.py:148` assigns a plain Python list to `self.stackmodel`; the caller passes `[stack_model1, stack_model2, stack_model3]` (`:388`). This is not `nn.ModuleList`, so those submodels are not registered through the final module. `optim.Adam(final_model.parameters(), ...)` at `:393` therefore excludes their parameters; final-model train/eval mode also does not recurse into that list. This is a source-level observation, not a claim about the authors' private training implementation.
6. Training saves `best_model_2.pt` (`main.py:449`), while inference loads `try.pt` (`:466`), which is not supplied. The published entry point cannot automatically evaluate its just-trained best checkpoint.

## Context and label audit

- Feature assembly uses current and preceding utterances plus prior partner features. It does not itself establish a future-text dependency.
- DED defaults to two copies of the complete dialogue (`ded/arguments.py`, `ded/beam_search.py:62`), then returns predictions from the last copy. Thus this decoder is not strictly history-only.
- `main.py:500–509` builds `emo_dict` from **test-loader ground-truth labels**. The same dictionary is then passed to transition-bias estimation (`:609,615`) and used for decoding (`:628`). The intended session exclusion is also faulty: `ded/utils.py:45` compares a sliced list of utterance IDs with a session string. Under the released flow, this creates a test-label-dependent transition prior. Repairing it would require a separately documented protocol correction.

## Existing biggpu environment

Read-only check using `/data2/yb/reproduction_envs/s0/bin/python` and `importlib.util.find_spec`:

| Package | Available |
|---|---|
| torch | yes |
| pandas | yes |
| sklearn | yes |
| tensorflow | no |
| xlstm | no |
| laion_clap | no |

The repository requirement file lists only `numpy==1.17.4` and `joblib`, so it does not capture actual imports. No new dependencies were installed, no GPU was used, and no formal run was submitted.

## Handoff

Prefer a different official 2025 baseline with both standard label spaces, released feature files, complete train/evaluate scripts and compatible installed dependencies. Preserve this audit to avoid repeating this candidate search.
