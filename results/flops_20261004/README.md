# MM-Mixer 下游前向 FLOPs（2026-10-04）

| 数据集 | FLOPs / 话语 | MFLOPs / 话语 | 注册参数 |
|---|---:|---:|---:|
| IEMOCAP | 68,997,504 | 69.00 | 6,358,082 |
| MELD | 68,721,024 | 68.72 | 5,712,070 |

统计冻结版本 `1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab` 的 Full、seed2025，
在 biggpu 的 CPU 上加载正式保存的 checkpoint 和真实预提取输入，严格匹配权重。
这里记录 MM-Mixer 的计算量；其他模型的完整测试集统计与比较边界见
[基线汇总](BASELINES.md)，原始数据见 [baseline_comparison.csv](baseline_comparison.csv)。

## 口径

一次乘加计 2 FLOPs，计入矩阵乘法和卷积（本模型实际计数为 addmm/mm/bmm），
包括注意力投影以及 QK、AV 运算。排除预训练特征提取器、反向传播、偏置相加、
逐元素运算、激活、归一化、softmax 和数据搬运。因此是明确限定的主要运算 FLOPs，
不是所有浮点指令的总数，也不是毫秒或 FLOPS/s。

使用未经裁剪的默认 eval 前向，包括该前向实际执行的辅助分类头和诊断计算。
不是移除训练辅助输出后部署模型的最小 FLOPs。输入维度与正式特征一致，详见 JSON。
MELD 按正式 runner 调用 disable_alignment() 删除 feature_aligners 后再加载权重；
这里删除的分支不等于论文的 MCA 模块。构造后、删除前参数为 6,106,825，
正式实例为 5,712,070。以前的 6.11M 是构造阶段口径，不能冒充正式实例参数量。

## 验证及复现

脚本：[measure_revision_flops.py](../../measure_revision_flops.py)。
使用已有 torch 2.2.2 的 FlopCounterMode，无新增依赖。
在 batch=1 和 batch=32 下逐话语计数完全一致，计数前向与 no_grad 参考 logits
最大绝对差均为 0。Linear 与交叉注意力的手算自检通过。
强制 math SDPA，并检查未展开的融合注意力/RNN 运算，避免静默漏计主要矩阵运算。
JSON 保留已计数运算、未计数运算调用、checkpoint/特征/脚本哈希及输入形状。

远程复现（以下 ROOT 为项目目录）：

```bash
ROOT=/data2/yb/multimodalERC/MM_Mixer_Revision_20261003
PY=/data2/yb/reproduction_envs/s0/bin/python
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 "$PY" "$ROOT/automation/measure_revision_flops.py" --self-test
# 输出必须使用新路径，脚本拒绝覆盖已有结果。
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 "$PY" "$ROOT/automation/measure_revision_flops.py" \
  --code-root "$ROOT/code/revision_1b8b1ff" \
  --artifact-dir "$ROOT/runs/revision_1b8b1ff/iemocap/full/seed2025/best_peak" \
  --dataset iemocap --output /tmp/mm_mixer_iemocap_flops_new.json
```

MELD 将两个 iemocap 参数替换为 meld。原始结果见同目录两个 JSON。
本次不修改 Main.tex 或 PDF。
