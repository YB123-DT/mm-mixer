# MM-Mixer 上游特征审计（2026-10-09）

本次只读核查上游特征、训练/导出代码、保存检查点及当前训练输入；没有替换特征、修改论文、重训或中断既有实验。

## 结论与证据边界

**MELD 使用的 CSS/SDT 源 pickle 内部存在已证实的跨对话视觉向量重复；文本微调与导出的表示读取方式也不一致。重新按台词建立正确对话映射后，13,707 条可核对向量与旧 JSON 全部相同，因此此前归因于 JSON 导出时的 split-local ID 错配是不正确的。不能仅凭这些发现断言训练使用了测试标签。**

### 视觉：重复发生在 CSS/SDT 源 pickle 内部

实际使用的 `features_denseface/{train,dev,test}_features/visual_features.json` 与本地审计文件逐个 SHA256 相同。验证集中609/1109条（54.91%）、测试集中1548/2610条（59.31%）与训练集中同局部编号、不同台词的样本具有完全相同的非零视觉向量。进一步按完整台词序列将每个 split 对话唯一映射到正确的 CSS 源对话后，13,707 条可核对向量仍与旧 JSON 逐条相同。源 pickle 中不同的正确对话本身就保存了这些重复向量；不是 JSON 导出器选错了对话。

例如 `dia0_utt0`：训练台词为 “also I was the point person …”，验证台词为 “Oh my God, he's lost it …”，测试台词为 “Why do all … coffee mugs …”，三个不同视频样本却使用同一个非零视觉向量。

- 早期碰撞检查与例子：`meld_visual_collision.json`；其“导出映射错误”解释已被后续完整重映射否定。
- 正确映射与旧 JSON 完全相同、源内重复的最终证据：`../meld_visual_remap_20261009/independent_verification.json` 和 `../meld_visual_remap_20261009/source_duplication.json`。
- 源 pickle 台词逐行核对：`meld_visual_source_sentences.json`。
- 全部输入覆盖、零向量、维度、哈希：`meld_features_probe.json`。
- 609/1548是非零且不同台词的确定异常数量，不把正常零向量重复当成错配证据。
- 全局ID直接查找还在验证675行、测试1740行返回了与目标不同的源台词；其余行不能据此证实来源。训练3407条零视觉向量原样继承自源pickle，零本身不证明映射错误。

### 文本：MASK微调与非MASK导出不一致

保存的RoBERTa检查点配置为 `pooling=mask`；候选微调实现截取末尾上下文并追加MASK。当前导出实现不追加MASK，读取最后一个非padding token（通常EOS）。9条实际样本的CPU重算与已导出特征余弦相似度约1，最大绝对误差小于0.0001，为当前导出路径提供抽样支持，并非全量逐元素相等证明。

截断方向也不同：微调保留末尾，导出保留开头。长度扫描发现训练1条、验证0条、测试7条超过511 tokens，因此截断差异影响少量长输入，不能用它解释所有样本；MASK/EOS读取差异则不限于长输入。

证据：`meld_text_probe.json`、`meld_text_probe.py`。检查点的具体选择与其训练/导出证据边界见 `meld_audit.md`。

### 下游并没有再次微调这些编码器

当前Full配置 `skip_pretrain=true`，`use_context_encoders`缺省为false，融合训练直接读取JSON向量。`prepare_fusion_features` 在无额外编码器时原样返回输入。不能把下游跳过图编码器预训练说成RoBERTa从未微调。证据：`downstream_input_evidence.json`。

### IEMOCAP 对照

已验证当前NPZ的文本/音频与上游打包数组完全相同，视觉与CSS源字段完全相同；下游训练Session01–04、测试Session05，无对话ID交集。本次没有发现MELD那种split局部编号冲突。但当前V27文本/音频原始检查点链仍不完整，不能宣称上游全部无问题；相邻test-peak分支不能冒充当前输入的来源。详见 `iemocap_audit.md`。

## 对已有实验的影响

- 已报告的MELD结果确实是在这套实际输入上得到；各变体共用输入不等于输入本身正确。
- 这些结果描述的是使用当前 CSS/SDT 源视觉向量的设置。源内大量完全重复和零向量可能削弱视觉信息，但仅重新映射无法修复；需要从原始视频重新提取后才能测量影响。
- 当前仍运行的MELD容量/消融队列同样沿用旧特征，应标识为旧输入诊断，不与后续修正输入结果混合。
- 后续需要从MELD原始视频重新提取视觉特征，并让文本训练/导出共用上下文和pooling实现；验证新输入后再建立新的Full及关键消融对照。本次未实施重新提取或训练。
