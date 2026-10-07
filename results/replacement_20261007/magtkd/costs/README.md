# MAGTKD downstream stage-two parameters and FLOPs

The counted model is the released `Transformer_Based_Model` after strict loading of the completed seed-2025, 30-epoch training run's selected checkpoint. Stage-one encoders, extraction and distillation training are excluded: official first-stage features are fixed inputs. All registered stage-two parameters and the full released stage-two forward (including auxiliary outputs) are retained.

Counting convention matches the paper's existing matrix/convolution table: 2 FLOPs per multiply-add; no bias, elementwise, normalization, nonlinear, softmax, scatter or backward/loss operations. Batch size is the released 16 dialogues. Sum the padded batch costs across the full test set, then divide by the number of valid utterances. Input feature and padding settings differ from MM-Mixer, so this is not a controlled architecture-only comparison.

Measurement is CPU-only on biggpu with one CPU thread; CUDA calls map to CPU and MKLDNN is disabled to expose recurrent matrix operations. Opaque attention/RNN operators are rejected. The no-grad reference and instrumented forward are checked for all six returned tensors on every batch, with fixed RNG state. Labels and CPU predictions are also compared with the completed GPU run's exported predictions.

IEMOCAP uses the author's actual `(text, audio_kd, video)` inputs; MELD uses `(text, audio_kd, video_kd)`. The script refuses to read a checkpoint until the final non-smoke result confirms 30 epochs and strict prediction replay, then checks checkpoint/source/data hashes before loading.

| Dataset | Parameters | MFLOPs / valid utterance | Valid test utterances |
|---|---:|---:|---:|
| IEMOCAP | 44,332,050 | 145.411854 | 1,623 |
| MELD | 44,566,293 | 183.280562 | 2,610 |

Both datasets: every forward difference is exactly zero, and CPU predictions exactly match saved GPU predictions. `counter_executed_iemocap.py` preserves the exact measured script bytes (its old inherited module docstring is cosmetic; the output counting convention and implementation describe MAGTKD). The maintained script corrects that docstring. MELD used the current maintained script. All 2 IEMOCAP and 18 MELD test batches passed; no opaque recurrent or attention operators occurred. `summary.json` provides the table-ready fields.
