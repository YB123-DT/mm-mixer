# 可用于论文的图注草稿

## 图 1

**Same words, different annotated emotions.** Three illustrative MELD clips show the same speaker, Ross, uttering the exact same current text, “Hey!”, with different dataset emotion labels: joy (train/dia645_utt13), sadness (test/dia163_utt2), and neutral (train/dia700_utt0). Each panel contains an unaltered video frame and the min/max envelope of the existing 16 kHz clip audio, using common time and PCM amplitude scales. Examples were selected by matching the current words and speaker, seeking distinct labels, and checking target-speaker visibility; model predictions were not used. Identical current words do not imply identical model text features, which also encode history. These examples illustrate the task setting rather than establish a benefit of AMM or the necessity of non-text modalities. Cite the [MELD dataset paper](https://aclanthology.org/P19-1050/).

中文说明：图 1 展示同一个说话人使用相同当前话语时存在不同情绪标注。标签来自原始数据，未修改画面，也未使用模型预测筛选。当前文本相同不代表历史增强的文本输入相同，不能据此断言文本模态无法判断、非文本模态必不可少或 AMM 已解决这些样例。

## 图 2

**Learned projection views and axis-wise interaction in AMM.** (a) Each 256-dimensional modality-branch representation, following the preceding MCA stage, is mapped into six learned projection views. The six projection maps are shared across modality branches. (b) Two residual mixer blocks use projection–modality–channel and modality–projection–channel update orders, respectively, before averaging over the projection axis. Projection mixing operates within each branch; modality mixing exchanges branch representations at each projection/channel position using a learned 3-by-3 matrix shared across those positions; channel mixing applies a feed-forward transformation within each projected vector. Layer normalization and residual additions are omitted for clarity. Individual projection views have no predefined semantic roles; their usefulness is a design hypothesis to be assessed by controlled experiments.

中文说明：图 2 按现有实现画出展开、分轴处理、交替顺序与均值聚合。六个投影未被指定为语义、语气、表情等固定角色，模态路由也不是按视角动态选择权重。
