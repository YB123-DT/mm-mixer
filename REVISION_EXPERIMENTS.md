# 补充实验代码与运行说明

代码与三种子矩阵已完成：144 个位置全部验证通过，8 项推理效率测量也已完成，见 [运行记录](EXPERIMENT_RUN_20261003.md) 与 [最新缺项核查](EXPERIMENT_STATUS_20261006.md)。
下面的参数量是结构统计，不代表性能提升。

## 固定协议

- 服务器：`biggpu`；宿主 GPU 4 禁用，按设备编号和 UUID 指定健康卡。
- 选择方式：沿用 `strict_peak_test_wf1`，在结果中明确标注 test-peak。
- IEMOCAP 种子：`2025, 2066, 2118`；MELD：`2025, 2028, 2069`。
- 种子包含已有单种子消融，再按允许的种子编号补齐；不按性能筛选。
- 保留当前输入特征、batch size 32、学习率、损失与早停规则；最大 epoch 分别为 100 和 50。
- 逐次保留结果，汇总报告均值与样本标准差（`ddof=1`）；缺失或失败不能标成完整三种子结果。
- README 中历史四种子结果及其 `ddof=0` 统计保留原定义，本轮另行汇总。

## 实验矩阵

基础矩阵共 17 种配置：`full`；`no_adaptive_gating`、`no_cross_attention`、
`no_mixer`、`no_pairwise`、`no_auxiliary_loss`、`no_feature_gating`、
`no_sequence_mixing`、`no_modality_mixing`、`no_feature_mixing`、`one_mixer_block`；
`modal_t`、`modal_a`、`modal_v`、`modal_ta`、`modal_tv`、`modal_av`。
两数据集各三种子，共 102 个位置。Full 同时代表 T+A+V，不重复计算。

新增七种配置均已接入两个数据集的生产 runner：

| variant | 目的与具体变动 |
|---|---|
| `amm_mlp` | 三个模态向量展平，两层普通残差 MLP，恢复三个输出分支；替换投影扩展和 AMM 块 |
| `amm_attention` | 保留 S=6 学习投影，两个八头自注意力块联合处理模态×视图 token |
| `amm_cubemlp` | 保留学习投影，使用 CubeMLP-style 固定 S→M→D、双层非线性映射、残差后沿对应轴归一化 |
| `single_projection_view` | S=1；保留两个 AMM 块和三个轴，S 轴 MLP 为 1→2→1 |
| `no_feature_and_adaptive_gating` | 同时关闭 Feature Gating 和 Adaptive Gating，补齐两级门控四格对照 |
| `pairwise_mlp_residual` | 在相同输入与残差加法位置，用输出零初始化的普通 MLP 替换 EPIRC |
| `amm_mlp_no_aux` | 与 `amm_mlp` 完全相同的初始化和结构，关闭辅助损失，补齐结构×辅助损失四格对照 |

七种配置新增 42 个位置，全矩阵共 144 个位置。位置数不等于需要新跑的次数；旧结果必须逐项审计后才能复用。

AMM 替换块的参数预算包括原 split 和两个 AMM block，不包括保持不变的 EPIRC：

| 模块 | 可训练参数 | 与目标差值 |
|---|---:|---:|
| 原 AMM | 1,974,614 | — |
| 普通 MLP（hidden=641） | 1,975,042 | +428 |
| Attention（FFN=1024） | 1,974,272 | −342 |
| CubeMLP-style（FFN=1538） | 1,974,726 | +112 |
| 普通残差 MLP（hidden=48） | 49,456 | 相对 EPIRC 活跃参数 49,504：−48 |

EPIRC 注册参数为 82,368，其中部分旧三元交互参数和输出列未参与当前前向。
残差 MLP 匹配实际活跃容量，并同时记录注册参数差异，不用闲置参数凑数。
CubeMLP-style 依据[原文 Eq. 2–4](https://arxiv.org/html/2207.14087)实现轴向残差单元；
它使用本模型的学习视图、GELU 和视图均值池化，是结构适配对照，不是原始时序 CubeMLP 完整模型的复现。

## 生成和运行计划

先将已验证代码放到独立、后续不再修改的代码目录。计划中的 `code_root` 必须指向该目录，
所有任务使用独立输出路径。不要在已有任务运行时覆盖其源代码。
例如在仓库内从已提交版本导出到全新目录（不混入工作区未提交内容）：

```bash
mkdir /absolute/path/frozen_code
git archive HEAD | tar -x -C /absolute/path/frozen_code
python -c 'import json, pathlib, subprocess; pathlib.Path("/absolute/path/frozen_code/snapshot.json").write_text(json.dumps({"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "source": "git archive HEAD", "working_tree_diff": ""}, indent=2))'
```

```bash
python launch_revision.py plan --matrix full --plan /absolute/path/jobs.json \
  --code-root /absolute/path/frozen_code --output-root /absolute/path/runs \
  --python /data2/yb/reproduction_envs/s0/bin/python
python launch_revision.py validate --plan /absolute/path/jobs.json
```

`--matrix base` 生成 102 个位置，`--matrix controls` 生成 42 个位置；生成计划不会启动训练。
需要补跑时可从计划中移除已另行核实的格点；总结果汇总仍须包含固定三种子的记录。
旧 bundle 可通过 job 的 `reuse_bundle` 显式指定，队列会再次验证配置、源码、特征和产物哈希。
历史源码已变或协议无法核实的产物不会因为有一个分数就自动接受。

在 biggpu 上按实际 `nvidia-smi` 输出填写 UUID，再启动持久队列：

```bash
python launch_revision.py run --plan /absolute/path/jobs.json \
  --state /absolute/path/queue/status.json --server biggpu \
  --gpu 1:GPU-ACTUAL-UUID --gpu 2:GPU-ACTUAL-UUID \
  --per-gpu 2 --min-free-mib 6000 --threads 1 --poll-seconds 20
```

队列保存 PID、启动时间、GPU、命令、日志及环境版本，支持重启后接管原进程。
已有不完整输出会标记失败并保留，不能自动覆盖或盲目重跑。
进程退出码 0 不足以判成功：必须具备完整 bundle、严格重载一致标记及通过哈希/指标检查。
单轮试跑的配置预算不满足正式 100/50 epoch 预算，不能纳入正式结果；正常早停不要求跑满预算。

## 汇总指标与效率

```bash
python launch_revision.py summarize --plan /absolute/path/jobs.json \
  --state /absolute/path/queue/status.json --output /absolute/path/summary.json
python analyze_revision.py --plan /absolute/path/jobs.json \
  --state /absolute/path/queue/status.json \
  --output-json /absolute/path/analysis.json --output-md /absolute/path/analysis.md
```

分析脚本重新核对预测与指标，输出 WF1、ACC、macro-F1、逐类 F1、每种子的混淆矩阵，
以及与相同种子 Full 的配对差值。缺失或失败会保留，不据此宣称提升或显著性。

效率对比使用 Full、`amm_mlp`、`amm_attention`、`amm_cubemlp` 已保存的 checkpoint：

```bash
python measure_revision_efficiency.py --artifact-dir /absolute/path/run/best_peak \
  --dataset meld --variant amm_attention --seed 2025 --server biggpu \
  --gpu GPU-ACTUAL-UUID --output /absolute/path/efficiency/meld_attention.json
```

默认使用真实 test 集前 1000 个不同话语，batch 32（末批 8）、20 批预热、10 次整轮同步计时。
只计 GPU 常驻预提取特征后的默认模型前向（包括默认计算的辅助输出），不含特征提取、加载或 H2D。
报告耗时、吞吐、PyTorch 峰值显存以及参数统计；正式比较必须使用同一空闲 GPU。
脚本拒绝宿主 GPU 4、存在其他计算进程的 GPU、不完整产物和覆盖已有输出。
`--smoke-samples 33 --repeats 3` 只用于检查流程，其输出标记为非正式效率测试。

## 验证与当前边界

新增测试覆盖两个数据集的真实 builder、前向/反向、优化器参数覆盖及更新、严格重建评测，
以及 Full 权重/输出/RNG 不变、门控与三轴开关、模态输入掩蔽、队列恢复和结果缺失处理。
两个数据集各完成一次旧基础配置的单轮 GPU 流程检查；它们仅验证训练与产物链路，不是正式结果。
新对照正式训练和 8 项效率测量现已完成，结果见 `results/revision_20261003/`；不再重复启动原矩阵。

2026-10-03 验证记录：工作区原有及新增检查 84 项通过，另 10 项结果分析测试通过；
仅包含本次提交内容的独立副本为 90 项通过、1 项跳过。
跳过项需要未纳入 Git 的历史代码快照，其 Full 权重、输出和 RNG 精确对比已在工作区通过。
Python 编译检查和 Git 差异格式检查通过；144 个计划位置均可构造对应正式配置。

训练数据、checkpoint、日志与本地历史审计文件保留在实验目录，不提交到 Git。
本轮没有修改 TeX、PDF 或论文中的实验表格。
