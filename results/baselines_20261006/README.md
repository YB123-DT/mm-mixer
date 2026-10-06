# 2025 基线三种子复现结果

12/12 正式运行完成。逐次检查模型/数据集/种子、测试样本数，并由保存的预测重算 Accuracy、Weighted F1、Macro F1，与结果一致。

| Model | Dataset | Accuracy | Weighted F1 |
|---|---|---:|---:|
| ConFilMER | iemocap | 69.52 ± 0.82 | 69.49 ± 0.50 |
| ECERC | iemocap | 70.24 ± 0.67 | 70.41 ± 0.67 |
| ConFilMER | meld | 67.01 ± 0.18 | 65.86 ± 0.19 |
| ECERC | meld | 67.14 ± 0.21 | 65.86 ± 0.44 |

均值 ± 样本标准差（ddof=1），每组 3 seeds。沿用 test-peak WF1 选轮协议，不是验证集选轮的独立测试分数。
原版特征和上下文：ECERC 及 ConFilMER 均不能标为统一 history-only；ConFilMER MELD 使用 train+dev。详见 ../../experiments/baselines_20261006/README.md 和各方法 AUDIT.md。
系统盘满与临时 socket 路径问题的失败尝试单独保留；最终使用 recovery_plan.json 指定的完整运行。
自动汇总曾被同目录 queue.py 遮蔽 Python 标准库阻塞；修复导入路径后汇总核验全部通过。训练结果未改变。
完整逐种子数据、预测哈希和来源路径见 summary.json。
