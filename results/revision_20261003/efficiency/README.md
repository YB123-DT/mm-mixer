# 已完成的 1,000 话语推理效率测量

8/8 项于 2026-10-04 20:44:33 UTC 前完成；本地于 2026-10-06 重新核验并归档。两个数据集均覆盖 Full、AMM→MLP、AMM→Attention、AMM→CubeMLP-style，使用 seed 2025 checkpoint。

| Dataset | Variant | 1,000 utterances (s, median) | Utterances/s | Peak allocated MiB |
|---|---|---:|---:|---:|
| iemocap | amm_attention | 0.3942 | 2536.8 | 57.89 |
| iemocap | amm_cubemlp | 0.4203 | 2379.2 | 59.04 |
| iemocap | amm_mlp | 0.3741 | 2672.9 | 52.19 |
| iemocap | full | 0.4391 | 2277.4 | 59.03 |
| meld | amm_attention | 0.3667 | 2727.0 | 55.09 |
| meld | amm_cubemlp | 0.3965 | 2521.9 | 56.23 |
| meld | amm_mlp | 0.3440 | 2906.9 | 49.47 |
| meld | full | 0.3761 | 2658.9 | 56.22 |

## 计时口径

biggpu 宿主 GPU 2，Tesla V100-SXM2-32GB，UUID `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`；未使用 GPU 4。FP32、eval/no_grad、batch 32 个话语，取各测试集按原顺序的前 1,000 个目标，不重采样或复制；31 个完整 batch 加末尾 8 个。20 batch 预热，10 次完整 1,000 话语 sweep，报告中位数。不是三种子计时均值，也不是 batch=1 请求延迟。

预提取输入已驻留 GPU，每个 sweep 前后 CUDA 同步；包括默认 forward 实际计算的辅助输出，排除预训练特征提取、数据加载、H2D、参数审计、预测导出和指标计算。峰值为 PyTorch allocated memory，包括模型、全部选定输入和保留状态；不是 nvidia-smi 全卡显存，也不是独立激活显存。

8 个命令的起止时间不重叠，全部 returncode=0。每次运行的启动前、计时前、结束后三次检查均无其他 GPU compute 进程；这些是离散检查，不能声称全程连续监控。

## 核验与限制

逐个结果与服务器汇总完全一致，服务器与本地 SHA-256 相同；重算10次计时的中位数、吞吐量与摊销毫秒值一致；同数据集4种配置使用相同目标ID，显存字段关系正常，全部 strict=True 加载checkpoint。源码比较中已加载文件匹配，部分历史训练入口或未加载文件为 `not_loaded_or_ambiguous`，因此原始 `all_recorded_sources_match` 为 false；没有明确的已加载文件哈希冲突。保留该状态，不能将本次效率检查写成历史预测的重新复现。

仍未测：统一空闲GPU条件下的训练每epoch耗时/训练吞吐/训练峰值显存，以及其他完整baseline在此1,000话语协议下的统一计时。历史baseline速度与本表边界不同，不能直接拼接排序。参数量和matrix/convolution FLOPs是另外的计数结果。

原始8份JSON保留全部配置、checkpoint/数据/源码哈希和计时序列；`efficiency_summary.json`、`pipeline_status.json`为服务器原件，`provenance.json`记录本次独立核验和文件来源。
