# HRG-SSA preliminary audit (2026-10-06)

Status: deferred, no GPU run or new dependency installation.
Official source: https://github.com/cgao-comp/HRG-SSA, commit b7f4b88559b9e2e35620a068c397679345ace7a3.
Paper: IJCAI 2025, Hybrid Relational Graphs with Sentiment-laden Semantic Alignment for Multimodal Emotion Recognition in Conversation.

The README requires T5-base plus author-provided Google Drive audio/video feature dictionaries, rather than only the existing MM-Mixer pooled arrays. T5-base was not found in the bounded remote model/cache directories checked. Existing s0 environment lacks transformers and sentencepiece. README dependencies request transformers4.41.2 and torch2.1.2.

More importantly:
- main.py81 indexes gat_config in the pretrained config, but standard T5 config does not contain it. model.py also requires edge relation configuration. No pretrained_model/config.json is distributed. Do not guess missing architecture settings.
- main.py overwrites explicit data_dir and model_dir. Multi-seed execution needs isolated working directories because model_dir excludes seed and backbone/config.json is modified in place.
- reader.py273 and433 append a suffix of the gold response (emotion/sentiment labels) into history. runner.py825 calls process_batch(batch,None,None) during prediction; the constructed prediction histories are not passed. Therefore released test inputs include gold historical label information. This must be resolved and labelled before comparing to ordinary history-only systems.
- Reader defaults has_dev=False for both datasets, so dev is loaded from test. Training evaluates that loss each epoch; entry later predicts from the last10 checkpoints. Selection policy needs explicit recording.
- model.load_model uses ignore_mismatched_sizes=True and hardcoded pretrained_model/pytorch_model.bin decoder initialization. Any adaptation must inspect mismatches and preserve intended initializations.

No wrapper is published as runnable because missing assets/configuration and label-history protocol affect validity, not just path portability. Server ownership remains biggpu; GPU4 prohibited. Official Google Drive folder metadata probe returned no output after several minutes and was terminated; no successful download or inaccessible-folder conclusion is claimed.
