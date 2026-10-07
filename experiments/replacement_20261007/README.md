# ECERC 替换计划与筛查记录（2026-10-07）

作者要求将 ECERC 换成另一种近期基线。优先选择 **MAGTKD（IJCAI 2025）**，官方数据及两个数据集真实 GPU smoke 已通过；6 次正式训练全部完成，预测与成本核验通过，已纳入正文 v020。

- 论文：Multi-modal Anchor Gated Transformer with Knowledge Distillation for Emotion Recognition in Conversation。
- 出处：IJCAI 2025，8141–8149，DOI 10.24963/ijcai.2025/905。
- 官方代码：https://github.com/JieLi-dd/MAGTKD，commit `95d0760c26ad2a6ad2181daf372abfd5be348b2d`。
- 官方特征：README 的 Google Drive `19g3hTaBEKF5wXI0DHdvRYbu0BD3XZa3d`；原作者发布的第一阶段蒸馏特征，用于第二阶段融合训练。
- 目标：IEMOCAP 六分类 seeds 2025/2066/2118，MELD 七分类 seeds 2025/2028/2069，共 6 次正式运行。
- 沿用既定 test-peak 选择口径并明确记录，保留原训练超参数与全部种子；输出逐类 F1、ACC、WF1、样本标准差及模型成本。
- 第一阶段特征固定，不将第二阶段三种子称为端到端全流程三种子重训。
- 服务器归属 biggpu；GPU4 禁用；不改主工作区 MM-Mixer 未提交代码。
- 运行入口与状态见 [MAGTKD 记录](magtkd/AUDIT.md)。六份官方特征通过 ZIP CRC 与 SHA256 校验，原测试集分别 1623/2610 话语。两个数据集实际 GPU smoke、参数更新和 fresh-model 同进程严格重载预测检查通过。正式6次运行使用 tmux `magtkd_6runs`、宿主 GPU1 UUID `GPU-56b14af1-00dc-4542-e2d8-5bba1dd39049`、最多2并发；六次均完成30轮，三种子汇总见 results/replacement_20261007/magtkd/summary.json，启动证据见子目录 launch_state.json。

## 其他筛查候选

- VEGA（ACM MM2025）：当前公开资源缺少完整 MELD 特征、七类 anchors 和设置；见 [审计](vega/AUDIT.md)。
- GatedxLSTM（ACII2025）：四分类 IEMOCAP，缺 MELD 入口和所需特征，且有实现/评价问题；见 [审计](gatedxlstm/AUDIT.md)。
- SURE（ICASSP2026）：公开源码缺失入口导出/损失定义，路由变量未定义等；见 [审计](sure/AUDIT.md)。
- 以前筛过的 GS-MCC、HRG-SSA、MFCRE、HAUCL 问题见 `experiments/baselines_20261006/`，不重复启动失败路径。

## 论文替换边界

三种子预测重算与成本核实完成，Main.tex 的两数据集结果行、参数/FLOPs行、近期基线说明及相应引用已替换为 MAGTKD，完整14页 PDF 编译及表格视觉检查通过，前一版归档至 paper/archive/v019。ECERC 的历史实验产物保留为审计记录，不删除。其他原始基线分数不改。

## 最终复现结果

| 数据集 | ACC | WF1 | 参数量/M | MFLOPs/话语 |
|---|---:|---:|---:|---:|
| IEMOCAP | 68.82±0.50 | 69.06±0.47 | 44.33 | 145.41 |
| MELD | 65.72±0.02 | 64.71±0.18 | 44.57 | 183.28 |

均值与样本标准差来自全部三个预定种子。成本使用全测试集、原始 batch16 对话设置计数，2 FLOPs/MAC，仅计下游矩阵/卷积。
