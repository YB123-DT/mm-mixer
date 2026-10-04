# CSS / SDT 下游前向计算量

日期：2026-10-04。服务器：biggpu；实际计算设备 CPU，单线程，未使用任何 GPU。

| 模型 | 数据集 | MFLOPs / 有效话语 | 实例注册参数 | 模型来源 |
|---|---|---:|---:|---|
| CSS | IEMOCAP | 200.959989 | 50,452,563 | 原训练日志配置、原代码重建；旧权重缺失 |
| CSS | MELD | 286.097237 | 53,398,634 | 原训练日志配置、原代码重建；旧权重缺失 |
| SDT | IEMOCAP | 289.795924 | 79,687,704 | 实际保存权重，严格加载 |
| SDT | MELD | 385.877876 | 78,386,204 | 实际保存权重，严格加载 |

口径为 1 MAC = 2 FLOPs，仅矩阵乘与卷积。包括默认 forward 执行的全部分类头、注意力矩阵运算和 padding 位置计算，不包括特征提取、反向、逐元素运算、偏置加法、归一化、激活和 softmax。与 `measure_revision_flops.py` 使用同一个原始 counter。

按历史速度测试设置，每批 32 个对话，顺序遍历完整测试集：IEMOCAP 1623 个有效话语、MELD 2610 个有效话语。分母为有效话语数，分子含 padding 开销；这里的每话语均值依赖测试对话长度和分批，不等于独立单话语输入的复杂度。JSON 保留每批长度、形状、padding 数和原始运算计数。

SDT MELD 实际模型参数为 78,386,204，与旧手稿 68.95M 不同。本记录只描述严格加载上述历史保存模型所得结果，不以旧手稿数替换实际实例。

## CSS 权重与计算图说明

原 IEMOCAP seed61080 / MELD seed10073 的 checkpoint 目录还在，但指定 `.pt` 已缺失。因此 CSS 结果明确标为 `architecture_reconstructed_missing_checkpoint`；按原 `train.log` 首行 Namespace 的结构配置重建，固定初始化种子 7。两者实际参数量均精确匹配原日志。配置摘录保存在同目录 `css_*_config_excerpt.txt`，完整日志只记录路径与 SHA256，不提交大日志。

审计实际 `model.py`：顶层 forward 的选择取决于固定 dataset / speaker 数。三路单模态 encoder 以同一张量同时传入自身，`equal` 分支固定；该分支中的 attention 与 feed-forward 矩阵形状固定。共享 cross-attention 按固定层数执行三路。融合中的 polynomial 循环由固定 order 控制，学习门控是稠密逐元素加权，不跳过矩阵运算。故这些 forward 的矩阵/卷积数量由输入形状及配置决定，不依赖训练权重数值；这里不能提供训练权重下预测正确性或性能验证。

## CPU 适配与验证

原代码只在补齐 speaker index 的整数常量处把设备写死为 `.cuda()`。适配器临时将此调用留在 CPU，并断言张量为一维 int32 且值仅为对应数据集 padding speaker 2/9；其他 `.cuda()` 类型会触发断言。未改原仓库模型、矩阵运算或 mask 逻辑。

每个完整测试批均比较普通 `no_grad` 与 counter 的 grad-enabled 前向分类输出，要求 `rtol=1e-4, atol=1e-5`。不反向、不训练，也不声称做了 CPU/GPU 数值一致性测试。CSS 是初始化模型的计数/普通前向一致性验证，SDT 是训练 checkpoint 的一致性验证。

## 复现

将根目录的 `measure_baseline_transformer_flops.py` 和 `measure_revision_flops.py` 放在远程 `/data2/yb/paper/tsne_baselines_20260728/`。示例：

```bash
cd /data2/yb/paper/tsne_baselines_20260728
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /data2/yb/reproduction_envs/s0/bin/python measure_baseline_transformer_flops.py --model sdt --dataset iemocap --output /tmp/sdt_iemocap_flops_new.json
```

分别选 `--model css/sdt` 与 `--dataset iemocap/meld`，隔离进程执行。输出路径已存在会拒绝覆盖。最终远程结果及日志目录：`/data2/yb/paper/tsne_baselines_20260728/flops_transformers_verified/`，持久任务 tmux `flops_transformers_verified`。各 JSON 保存源代码、数据、日志/配置来源和可用 checkpoint 的 SHA256。
