# FLOPs without EPIRC

Same matrix/convolution-only counting convention and default eval forward as results/flops_20261004. Two FLOPs per MAC; excludes pretrained encoders and elementwise operations. Includes auxiliary/diagnostic computations actually executed.

| Dataset | Full MFLOPs | No EPIRC MFLOPs | Reduction |
|---|---:|---:|---:|
| iemocap | 68.997504 | 68.669824 | 0.4749% |
| meld | 68.721024 | 68.606336 | 0.1669% |

Strict checkpoint loading, artifact/feature identity verification, batch1/batch32 agreement, and exact reference logits passed. The adapter adds no_pairwise-specific config identity validation; original counter and frozen training source remain unchanged. No training or paper changes.
