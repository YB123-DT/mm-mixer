# MFCRE source audit (2026-10-06)

Status: deferred; no training, dependency installation, or GPU use.
Repository: https://github.com/rhoqwomda/MFCRE
Commit: 650358ec0a0d68ca6e4138c8568d4462166e805a
Source clone: /data2/yb/paper/baseline_sources_20261006/MFCRE
Paper identification supplied for screening: Contextual xLSTM-Based Multimodal Fusion for Conversational Emotion Recognition, Pattern Analysis and Applications2025, DOI10.1007/s10044-025-01508-8. The README title matches; direct publisher-code attribution remains to verify before any reported reproduction.

## Blocking implementation inconsistencies

1. Model/model.py constructs `DialogueCRN(model_dim, n_classes, hidden_dim, num_layers, device)`, while Model/DialogueCRN.py defines `(n_features=200, n_classes=7, dropout=0.2, cuda_flag=False, reason_steps=None)`. Thus `reason_steps` becomes a torch.device and `self.steps[0]` immediately fails. In addition, default hidden_dim1024 is passed as dropout. Fixing the call requires deciding the intended architecture, not just a path change.
2. Model/DialogueCRN.py imports `ViLBlock` from `vision_lstm`, absent from repository and requirements. It invokes the block with LSTM-style `(input, hidden_state)` and expects `(q,h)`; the intended custom implementation/version is not given.
3. MultiEMO.forward accesses `self.n_speakers` but constructor never sets it.
4. DialogueCRN.forward expects `(U_s,U_p,seq_lengths)`, while parent passes modality features, speaker masks and utterance masks. It returns class log-probabilities, whereas parent treats output as hidden features for a convolution. This is another structural interface discrepancy.
5. Parent forward concatenates three transformer outputs then slices the first third, discarding the two cross-modal outputs. Do not silently replace this behavior with a presumed paper implementation.
6. Training hardcodes CUDA_VISIBLE_DEVICES=1 and seed2023; multi-seed and safe GPU binding need a wrapper. MELD official command600epochs, IEMOCAP100; entry selects maximum test weighted F1 (ties latest). Training makes a seed-dependent5% dialogue holdout.

## Inputs

Requires MultiEMO feature files `Data/{IEMOCAP,MELD}/{Speakers,TextFeatures,AudioFeatures,VisualFeatures}.pkl`, with input widths768/512/1000. README links https://github.com/TaoShi1998/MultiEMO. These differ from MM-Mixer cached features; adapting widths would be an adapted-input baseline, not original numbers. Data loading has no need for new feature extraction if original MultiEMO assets exist, but it does not resolve the model interface blockers above.

No runnable wrapper is claimed. Server ownership would be biggpu; physical GPU4 remains prohibited.
