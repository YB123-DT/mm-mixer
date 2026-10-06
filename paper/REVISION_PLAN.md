# MM-Mixer：两位审稿人的逐点、逐句修改清单

> 2026-10-06 实验状态更新：本文件保留原审稿修改方案。实际完成/补跑状态见 [实验核查报告](../EXPERIMENT_STATUS_20261006.md)；144 个消融位置、S=2/4/8 的 18 次运行和两种近期基线已完成，不能将下文旧待办框当作当前运行状态。

更新时间：2026-10-01。用途：按编号逐项修改论文；本文件是修改方案；已落实的文字修改见下方进度，**未标记完成的建议不表示已写入 TeX，文字完成也不表示建议实验已经完成**。

**本轮范围（按作者最新安排）**：先处理故事、动机、方法解释和写作定位。主表/消融表口径及实验数字由作者后续解决；R2-5 保留为待办记录，不作为当前改写的前置条件，不修改现有数字。

## 0. 使用方式与来源

- 当前稿：[Main.tex](Main.tex)、[Supplement.tex](Supplement.tex)。定位基准为 Git 提交 `bd4d5e7fcd963fac095711892ebfec1fa4531b00`，论文版本 `paper-v008`。
- 下文 `Main.tex:行号` / `Supplement.tex:行号` 链接指向该提交，后续修改不会改变链接中的原句。行号对应源文件；同一行有多个句子时，另列句子开头，便于在编辑器中搜索。
- R1 = **Program Committee 8gGo**；R2 = **Program Committee XkY6**。依据是作者于 2026-09-25 在会话中提供的中文评审全文，已重新读取核对。R1 的缺点原本未编号，此处按原文顺序编号；R2 保留原编号，并合并同编号的 Rebuttal 追问。
- [GitHub Issue #1](https://github.com/YB123-DT/mm-mixer/issues/1) 是此前的整理与核查稿，同时含 AI Review，**不是两份评审原文**。本清单的 R1/R2 只对应两位真人；额外发现的问题单列为 C 项。
- 初建本清单时只记录方案，未改 `Main.tex`、`Supplement.tex` 或 PDF；后续修改进度另列如下。你在聊天里给出的三段 Implementation Details 是**拟替换稿**；当前磁盘文件仍有未合并的 A2、注释掉的输入/损失说明和旧种子表述。

编号后可以记录两种进度：`[ ] 文字完成`、`[ ] 证据完成`。需要实验的问题，即使文字改好，也不能直接算已解决。优先级：**P1 本轮主线与方法说明；P2 后续证据与扩展；暂缓 作者后续处理**。实验待办只列设计，不在本轮启动。

### 已落实进度：v009（2026-10-01）

按作者本轮要求，只修改 Introduction 的前四项写作问题；第 5 项贡献列表暂不修改。

- [x] 场景衔接：在 `Main.tex:187` 末尾，从示例过渡到一般的话语级多模态表示组织问题。
- [x] 问题表述：`Main.tex:196` 缩窄整体向量融合的适用范围，去掉缺少直接证据的“线索被淹没/稀释”推断，保留明确的轴向组织问题。
- [x] AMM 主线：`Main.tex:200` 加入多头注意力的多投影启发与真实文献引用；解释投影视角和三个轴的分工，保留非时间序列说明。
- [x] 模块关联：`Main.tex:198–202` 说明 AG/MCA 为核心交互准备表示，查询聚合形成主融合表示，EPIRC 通过低秩乘积作残差补充。
- [ ] 贡献列表：本轮按作者要求保留，后续单独修改。

这些修改部分落实 R1-1、R2-1、R2-2、R2-3 的引言解释，不代表消融、容量控制或其他实验问题已解决。摘要、实验总结、数值、Supplement 和图注保持本地原稿；聊天中讨论的摘要替换稿未在本次引言任务中一并写入。详细操作及编译结果见 [TEX_CHANGELOG.md](TEX_CHANGELOG.md)。下文仍使用 v008 的原句与固定提交链接，便于逐句对照。

### 全部意见索引

| 编号 | 审稿问题 | 主要修改位置 | 优先级 / 当前状态 |
|---|---|---|---|
| [R1-1](#r1-1) | 组件多、整体动机和关联不清 | 标题、摘要、Introduction、Overview、Conclusion | P1；已有场景和示意图，主线未统一 |
| [R1-2](#r1-2) | 两级门控重叠 | Feature Gating、AG、联合消融 | P1；有单项消融，联合证据未在稿件中给出 |
| [R1-3](#r1-3) | 收益较小是否值得复杂流程 | 贡献、主结果、消融、效率 | P1；需公平替换与成本证据 |
| [R1-4](#r1-4) | 数据集少、缺真实环境验证 | 数据集、结果范围、限制 | P2；场景插图不等于新增评测 |
| [R1-5](#r1-5) | 域外 / 跨数据集泛化 | 新实验与限制 | P2；当前稿无对应结果 |
| [R1-6](#r1-6) | 1000 样本推理耗时、参数比较 | Model efficiency、参数表 | P1；只有参数表 |
| [R1-M1](#r1-m1) | 编码器特征提取细节 | Implementation Details 第 1 段 | P1；只列模型名与维度不够 |
| [R1-M2](#r1-m2) | 时序输出如何池化为单向量 | 输入表示、提取配置说明 | P1；池化与截断规则待查证 |
| [R1-M3](#r1-m3) | 压缩时序是否丢失信息 | 输入限制、AMM 解释 | P1；不能说投影展开恢复时间信息 |
| [R2-1](#r2-1) | 对话任务定位、迁入基线、接入对话模块 | 场景→问题→方法、对照/讨论 | P1；三个子问题均需回应 |
| [R2-2](#r2-2) | S 轴含义及其必要性 | AMM、公式、替换消融 | P1；核心主线 |
| [R2-3](#r2-3) | EPIRC 交互形式还是容量带来收益 | EPIRC、公式、控制实验 | P1；公式还有实现口径问题 |
| [R2-4](#r2-4) | 同标签辅助监督、损失权重依据 | Learning Objective、Implementation、敏感性 | P1；需说明假设与选择依据 |
| [R2-5](#r2-5) | 主表和消融 Full 不一致 | 两个主表、两个消融表、种子/选模协议 | **暂缓；作者后续处理，本轮不改数字** |
| [R2-6](#r2-6) | 架构贡献与训练目标贡献混淆 | 替换式消融、消融结论 | P1；需控制训练目标及尺度 |
| [R2-7](#r2-7) | 参数量不足以证明计算效率 | 与 R1-6 共用一份效率实验 | P1；避免重复做 |
| [R2-8](#r2-8) | 近期基线覆盖不足 | Related Work、Baselines、主表 | P1；需解释纳入标准并补比较 |

### 建议先后顺序

1. **从摘要开始改故事**：R1-1 → R2-1 → R2-2；沿用你确定的 AMM 多投影视角主线。
2. **补齐模块分工与训练依据**：R1-2 → R2-3 → R2-4，以及 R1-M1/M2/M3 的输入解释。可查明的方法事实先写清，未知细节列待核实。
3. **合并 Implementation Details**：按 C1 整理三个自然段，修正术语/补缺失内容；涉及最终运行批次、数字和表格的部分留待作者统一。
4. **形成实验待办**：R2-6、R1-3、R1-6/R2-7、R2-8，再处理 R1-4/R1-5；本轮只记录，不运行。
5. **作者后续处理 R2-5**，届时再同步涉及数字的摘要末句、结果分析、表注和结论。这不阻塞现在改摘要动机和方法主线；当前避免新增更强的性能主张。

## 1. 全文统一主线

建议固定为：**对话情绪判断需要结合已编码的历史文本与当前音视频；在给定语句级向量的前提下，AMM 用多个可学习投影构造表示视角，再分别组织视角内/间的变换与跨模态交换。** 这里的目标是设计一种融合结构，而非恢复原始时序。

全文的叙述层次应当是：

1. **任务场景**：短回应、招呼等当前词语相近，但上下文及声学/视觉线索不同。现有 “Hey!” 场景图可保留。
2. **研究问题**：给定已有语句级表示，如何组织其内部变换与跨模态交换？避免空泛地说“更好挖掘细微情绪”。
3. **主要设计**：同一分支向量 → 多个学习投影视角 → S/M/D 分轴交互 → 聚合。S 是学习投影坐标，M 是分支/模态，D 是通道。
4. **配套组件**：FG/AG 校准输入，MCA 在展开前交换整向量信息，EPIRC 在聚合后补充低秩乘性残差，辅助头提供训练正则。各自说明作用，但不把每个组件都包装成并列主创新。
5. **证据问题**：这些视角及其混合是否比相同输入、相近容量的普通 MLP/attention 融合更有效？这是实验要回答的，不能从场景图直接推出。

多头注意力的多投影思想可以作启发，依据是其将输入投影到多个表示子空间后并行计算的设计；AMM 使用的是自己的共享投影与轴向算子。对应文献：[Attention Is All You Need，§3.2.2](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS2)。分轴算子的相关基础应讨论 [MLP-Mixer](https://arxiv.org/abs/2105.01601)。

“多个 LoRA 并行”目前只适合作为讨论设计时的类比：现有 AMM 的展开是稠密线性映射，没有 LoRA 的冻结基座权重与低秩增量参数化，也没有已核实的多 LoRA 路由实现。[LoRA 原论文](https://arxiv.org/abs/2106.09685)不直接证明本模型的多视角收益。建议正文以多头投影为启发；若以后要写某一种并行 LoRA 方法，先明确对应论文与实际相似点。

## 2. 第一位审稿人：8gGo

<a id="r1-1"></a>

### R1-1｜组件过多，缺少统一动机与核心贡献

**原意见要点**：组件串联很多，只用“各自有一点帮助”的消融，未说明为什么需要这些组件、彼此为何有关联。对应推荐理由中的“提炼更简单明确的核心贡献”。

**本点要改的内容**：将 AMM 的多投影视角及轴向组织作为中心；其他组件按它们在流程中的位置解释。仅改名字或增加“novel”不能解决这一点。

#### 摘要逐句

| 句子位置与原句锚点 | 具体操作 / 建议内容 |
|---|---|
| [Main.tex:165](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L165)，`Multimodal Emotion Recognition in Conversation (MERC) aims to identify...` | **保留任务定义**；统一使用 emotion，后面不在 emotion/sentiment 间无理由切换。 |
| [Main.tex:167](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L167)，第 1 句 `Existing methods fuse unimodal representations...` | **替换缺口句**。当前把所有方法概括为统一视角且忽略 S/D，范围过大；还把模态内操作也称为 cross-modal。建议围绕本研究问题写：`Given utterance-level modality representations, we study how multiple learned projections can organize within-branch processing and cross-modal exchange for emotion classification.` 若要批评已有方法，先在 Related Work 给出具体对象与依据。 |
| 同行，第 2 句 `Feature utilization from multiple perspectives can help...` | **删除或与上一句合并**。这是未经比较支持的收益判断，且重复“multiple perspectives”。用具体设计代替“更准确捕获情绪”的泛化承诺。 |
| [Main.tex:169](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L169)，第 1 句 `To address this issue, we propose a simple yet strong...` | **替换方法总述**，直接点明 learned projection views 与 axis-wise fusion；暂去掉 `simple yet strong`，例如：`We propose MM-Mixer, a fusion framework that expands each modality-centric representation into multiple learned projection views and organizes their interaction along separate axes.` |
| 同行，第 2 句 `Specifically, the Adaptive Gating module is first deployed...` | **移动到 AMM 核心句之后并压缩**，作为准备步骤。不要让 AG 占据摘要第一个核心机制位置，也不要在没有噪声实验时断言 `reduces irrelevant information`。 |
| 同行，第 3 句 `To extract multimodal sentiment cues from multiple perspectives, an Axis-wise Multimodal Mixer...` | **改为精确轴分工**：`The mixer combines learned views within each branch, exchanges information across branches at corresponding view coordinates, and transforms feature channels.` 不能写三个轴都直接跨模态；也不能用一个固定的 `in sequence` 掩盖两块不同顺序。 |
| 同行，第 4 句 `Finally, a simple fusion module enhanced with Explicit Pairwise...` | **压成支持性机制一句**：查询聚合得到主表示，低秩成对乘积提供加性残差。去掉 `simple`、`all different`、`compact yet discriminative` 等堆叠修饰。 |
| 同行，第 5 句 `Extensive experiments on the IEMOCAP and MELD datasets show...consistently outperforms...` | **性能结果句留待作者后续统一**，本轮先处理摘要前面的动机与方法句。最终应限定为实际评测对象、协议和指标；不在本轮新增更强的领先主张。 |

#### Introduction、贡献和结论逐句

| 句子位置与原句锚点 | 具体操作 / 建议内容 |
|---|---|
| [Main.tex:95](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L95)，`A Simple Yet Strong Baseline...` | **最后同步标题**。建议候选 `MM-Mixer: Learned Projection Views for Multimodal Emotion Recognition in Conversations`。标题是否改由主线最终版本决定；避免摘要改了而标题仍强调“简单强基线”。 |
| [Main.tex:187](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L187)，`Even a one-word greeting...` 至 `These examples motivate considering...` | 前三句陈述样本事实可保留；最后一句只引出历史与多模态证据。**在该段之后新增桥接句**：本文研究的是这些证据已经编码为向量之后的融合组织。不要直接接“因此需要六个视角”。详见 R2-1。 |
| [Main.tex:196](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L196)，第 1 句 `Despite this progress, many existing MERC frameworks still treat...` | **限定问题对象**为采用语句级向量的融合设置；不要把“输出一个向量”当成既有方法内部没有多头或子空间操作的证据。 |
| 同行，第 2 句 `Although such representations may contain rich affective information...` | **改成研究问题**：是否可以通过多学习投影及共享的分轴操作，更有效地利用这些已编码表示。避免把较粗粒度必然低效写成事实。 |
| 同行，第 3 句 `Subtle but valuable cues may be obscured...` | **删除或改成待验证假设**。现有结果没有直接测量线索“被掩盖/稀释”；不能当成已证实失效机制。 |
| 同行，第 4 句 `Consequently, the resulting multimodal representation may be insufficiently sensitive...` | **合并为设计目标**，不再从上一句推出确定的敏感性缺陷。可写“我们关注同一分支的多个投影视角如何与其他分支交换信息”。 |
| [Main.tex:198](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L198)，`To address this limitation...` / `MM-Mixer first applies...` / `After the modality-centric...` | **调整顺序**：先一句核心思想与启发，再一句多投影构造，再交代前置校准与 MCA。此处叙述顺序突出 AMM，但不能把实际计算顺序写错。 |
| [Main.tex:199](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L199)，`Here, the latent sequence...` 及三句轴说明 | **统一术语**为 learned projection views / projection axis，并保留不是时间序列的说明；分别写 within-branch view mixing、cross-branch exchange、channel transformation。具体改法见 R2-2。 |
| [Main.tex:200](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L200)，`To obtain a compact yet discriminative...` 和后面的 `One main path...` | **两句压成一至两句**，解释主聚合与乘性残差的分工；不再反复说 simple/unified/all different features。 |
| [Main.tex:211](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L211)，贡献第 1 项 `We propose MM-Mixer...` | **改成主要贡献**：给定语句级表示，构造多投影视角并组织轴向交互；与下一项不要重复列框架名与 AMM 名。 |
| [Main.tex:212](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L212)，贡献第 2 项 `We introduce an Axis-wise...` | **合并到第 1 项或具体化**为共享投影坐标、S/M/D 分工。不要仅将三轴名称重述一遍。 |
| [Main.tex:213](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L213)，贡献第 3 项 `We develop an Explicit Pairwise...` | **降低并列主创新的分量**：作为 AMM 后的支持机制；若保留独立贡献，必须补 R2-3 的形式/容量控制证据。 |
| [Main.tex:215](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L215)，贡献第 4 项 `Extensive experiments...impressive performance...` | **改成可核实的评估贡献**，只列实际完成的比较/消融；不能提前写尚未做的替换和效率实验。 |
| [Main.tex:278](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L278)，Overview 中 `As illustrated in Figure...` 附近 | **用准备表示→核心交互→输出聚合三个阶段解释流程**；图中的 FG/AG/MCA 均保留实际依赖。所有组件重要程度不必平均分配。 |
| [Main.tex:1025](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1025)，`We presented MM-Mixer...` 及其后逐模块总结 | **结论先收束 AMM 思想**，再用一句说明配套组件；最终性能范围由 R2-5、R1-6 决定，不继续列一串模块来代替结论。 |

**完成标准**：摘要、Introduction、贡献、方法开头和结论都能回答同一个问题——“为什么把单个向量组织为多个学习视角，以及怎样对这些视角进行多模态交互”。

- [ ] 文字完成；场景→研究问题→AMM 的过渡连贯。
- [ ] 核心收益已有 R2-2/R2-6 的对照证据；否则保留设计假设措辞。

<a id="r1-2"></a>

### R1-2｜两级门控是否重复

**原意见要点**：多个门控阶段解决的问题可能重叠，妨碍识别核心贡献。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:300](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L300)，`Since the input modalities have different feature dimensions...` | 将维度投影与 FG 分开说明：投影统一维度，FG 的系数仅依赖本分支表示。不能单凭投影宣称语义分布已经对齐。 |
| [Main.tex:317](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L317)，`Here, it is used to suppress irrelevant channels...` | 改为可观察的操作：对单模态通道作输入相关的缩放；“无关通道被抑制”是设计意图，未做归因分析前不当作验证结果。 |
| [Main.tex:335](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L335)，`Given the filtered unimodal features, the AG module constructs...` | **在公式前加对比句**：`Feature gating conditions on one branch alone, whereas AG uses a shared reference computed from all three branches to recalibrate each branch.` |
| [Main.tex:350](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L350)，`The shared reference then conditions a separate channel gate...` | 说明这里的 context 是当前三分支融合参考，不是新建的对话历史编码器；两级输入不同不等于已证明两级均必要。 |
| [Main.tex:997](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L997)，`...demonstrating that they provide complementary rather than redundant contributions.` | **删除这个强结论**。可先写“所列单项移除结果用于评估各组件在 Full 配置中的边际影响”；是否互补留给联合对照。 |
| [Main.tex:1008](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1008)，`Removing Feature Gating...` 和 `Removing adaptive gating...` | 本轮先处理解释和结论强度，数字留待作者后续统一。四格结果就绪后再报告差值与波动，不只把两个下降并排写来证明不重复。 |

**实验待办**：同一特征、种子集合、训练目标、选模规则下做 FG×AG 四格。AG 的关闭范围要写清：共享参考构造与后续条件门控是否一起关闭；与已有 `w/o AG` 实现一致。

| 配置 | FG | AG |
|---|---|---|
| Full | 开 | 开 |
| 仅 FG | 开 | 关 |
| 仅 AG | 关 | 开 |
| 无两级门控 | 关 | 关 |

优先复核前三行已有实验，缺失的再补；不能假定旧结果可直接复用。报告参数变化及逐种子差值。四格可衡量“另一门控存在时是否仍有增益”，但不能证明一般意义的非冗余。若双门控优势不稳定，正文应如实缩小主张，并据结果讨论简化。

- [ ] 两级条件输入、关闭范围写清。
- [ ] 四格结果可追溯；删除未经证实的“互补而非冗余”。

<a id="r1-3"></a>

### R1-3｜小幅提升是否值得复杂流程

**原意见要点**：约 71.5→72.2 的提升，相对于复杂度可能不足。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:205](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L205)，`...achieves impressive performance...` | 去掉主观程度词，交代比较范围。提升幅度不能靠措辞放大。 |
| [Main.tex:899](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L899)，`It outperforms the CSS model by 0.56\% and 0.68\%...` | 对齐后以 **percentage points** 报告绝对差值；报告每种子/不确定性，不凭 mean±std 宣称统计显著。 |
| 同行，`These results suggest that the axis-wise interaction and explicit pairwise...` | 主表只能比较完整系统，**不能分解归因到 AMM/EPIRC**。此句改为整体结果描述，组件贡献移到控制实验。 |
| [Main.tex:906](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L906)，`The largest gains over CSS occur for fear and disgust...` 及下一句 `The results suggest that preserving axis-specific information...` | 类别差值可以在核实后报告；“保留弱线索”的机制解释需要类别级消融或额外分析，不能只由完整模型的类 F1 推出。 |
| [Main.tex:912](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L912)，`These results indicate that structured axis-wise interaction...` | 在尚无时间统计时，只说相对指定基线的下游参数与分数；补 R1-6 后再讨论实际成本收益。 |

**需补证据**：复用 R2-2 的等输入/近容量替换、R2-6 的目标控制，以及 R1-6 的耗时；不必为此再建一套重复实验。若要报告统计检验，使用相同样本的预测和合适的配对设计，说明方法；三次重复的标准差本身不是显著性检验。

- [ ] 所有提升采用清楚的差值口径，并限制在表内比较范围。
- [ ] 用准确率/成本对照回答取舍，而非把小幅提升写成巨大优势。

<a id="r1-4"></a>

### R1-4｜两个数据集不足以支撑广泛适用的“强基线”

**原意见要点**：希望更多数据集、不同条件，特别是真实环境场景。

| 位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:852](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L852)，`We evaluate MM-Mixer on two widely used...` | 当前如实保留“两套数据”；只有新增评测完成后才扩展。新数据需与任务标签、模态可用性、对话结构相容。 |
| [Main.tex:187](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L187)，三个 `Hey!` 样例 | **保留为说明图**。它们来自已有 MELD，不能算新增真实环境泛化评测，也没有模型预测结果。 |
| [Main.tex:1035](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1035)，`Future work will address minority emotions and longer contexts.` | **扩写限制**：当前验证范围仅两套基准，尚未建立自然采集新域、噪声/缺失模态等条件下的可靠性。要做哪一种条件就具体写哪一种。 |

**证据待办**：选定可复现的第三个对话多模态情绪数据集或明确的真实环境评测设置，固定处理规则与基线。缺失模态/噪声实验可作为补充，但不能冒充新数据集的真实环境结果。未完成前，此条只能标为“限制已交代，扩展验证未完成”。

- [ ] 区分说明样例、条件扰动和独立新数据集评测。
- [ ] 扩展实验完成，或明确记录未解决的范围限制。

<a id="r1-5"></a>

### R1-5｜域外鲁棒性和跨数据集泛化

**原意见要点**：跨数据集时是否仍然有效？

**当前没有可修改的对应实验句子**，需要新增；不能只改“robust”一类词来回答。

- 在 [Main.tex:1016](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1016) 的模态消融段后，若实验完成，新增 `Cross-dataset evaluation` 段/小节。
- 在 [Main.tex:1035](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1035) 附近先补一句范围限制：分别在两套数据上训练测试，不等于从一套训练迁移到另一套。

**设计必须先写清**：训练源域、评测目标域、情绪标签映射、目标域是否用于调参/适配、特征提取器是否已接触目标域标签、各模型相同的选模规则。IEMOCAP/MELD 类别并不一一相同，不能直接把六类分类头拿去评七类；标签合并或共同类别筛选须在看测试结果前确定。若使用已在目标域监督微调的特征，不应声称严格零样本跨域泛化。

建议结果表字段：源域→目标域 / 标签设置 / 是否目标域适配 / 基线 / ACC、WF1（必要时 Macro-F1）/ 运行来源。

- [ ] 文字明确当前证据不包含跨域评估。
- [ ] 如要关闭此意见，提供可追溯跨域结果；只写 Future work 不等于实证回应。

<a id="r1-6"></a>

### R1-6｜补 1000 样本推理时间与运行成本

**原意见要点**：复杂模型应给运行时，明确举例为 1000 样本推理耗时，并比较参数量。与 R2-7 共用此项。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:843](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L843)，`Trainable downstream parameters in millions. Pretrained feature extractors are excluded.` | **保留统计边界**，补参数是否包含辅助头/未参与前向的参数。训练可训练参数与部署活跃参数可分别报；核对所用实例。 |
| [Main.tex:909](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L909)，`Model efficiency.` | 测时前标题改为 `Downstream parameter count.`；测时完成后才用完整效率标题。 |
| [Main.tex:910](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L910)，`MM-Mixer only uses 6.36M and 6.11M...` | 用实际统计值，去掉 `only`；增加同表的时间/计算量列及单位，或另列效率表。 |
| [Main.tex:911](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L911)，`Consequently, MM-Mixer reduces the trainable parameter count...` | 参数百分比只解释参数节省；不要拿它推导速度提升。 |
| [Main.tex:1033](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1033)，`Experiments ... with fewer parameters than recent multimodal baselines.` | 限定为实际比较的指定基线及下游统计口径；没有新测量时不能改为 lower computational cost。 |

**建议一张表完成两位审稿人的要求**：模型 / 下游训练参数 / FLOPs 或 MACs（说明计数约定）/ 固定 1000 个目标话语的总推理时间 / 吞吐 / 单样本延迟 / 每 epoch 训练时间 / 峰值显存。可优先完成参数+千样本时间，再补其他列；未测列不得填估计值。

**测量说明必须写在表注或设置段**：同一硬件与精度、固定 batch size、预热与重复次数、GPU 同步、数据/特征是否预加载、是否包含特征提取。对话基线预测这 1000 个目标话语时须保留所需历史，不能截掉上下文让它变快。报告仅下游耗时就明确叫 downstream inference；端到端成本另算。

- [ ] 测量配置与样本子集可复现。
- [ ] 1000 样本时间与参数比较已补；计算量/训练时间按实际完成情况报告。

<a id="r1-m1"></a>

### R1-M1｜补预训练特征提取过程

**原意见**：“论文缺少一些关于预训练编码器特征提取过程的细节。”

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:873](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L873)，`For each utterance, we use RoBERTa-large...` | 保留模型名，但补实际 checkpoint/来源、是否曾针对数据集微调、在何数据划分微调。只说 pretrained 不能说明特征的训练经历。 |
| [Main.tex:877](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L877)，`1024, and 342.` 后的注释句 | **将预提取说明写成正文**：`All unimodal representations are pre-extracted before downstream multimodal training.` 这不等于编码器从未微调；两件事分别说明。 |
| 同一段，在预提取句之后 | **加入**：`The textual representation includes the preceding dialogue context, whereas the acoustic and visual representations describe the target utterance.` 与 Overview 保持一致。 |
| [Supplement.tex:49](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Supplement.tex#L49)，`We use pre-extracted RoBERTa-large...` | 迁入正文后删重复基础说明；如果保留补充节，只放正文放不下的 checkpoint、预处理与完整配置，并加回指。 |

**需要查证的来源**：发布的 `config.json` / `manifest.json` 只证明融合阶段消费哪些特征、`skip_pretrain` 与输入维度；详细提取协议还要追到实际生成这些特征的代码和运行记录。两数据集的特征来源应分别核实，不能只凭最后维度相同就当成提取流程完全相同。

- [ ] 正文明确预提取、历史/当前话语范围及提取器来源。
- [ ] checkpoint、微调数据与真实提取记录对应。

<a id="r1-m2"></a>

### R1-M2｜每个编码器的 T×D 输出如何变成一个向量

**原意见**：“当视频、音频或文本编码器的输出仍然具有时间维度时，作者究竟如何得到一个代表整条语句的单一向量？”

**添加位置**：[Main.tex:877](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L877) 的维度与预提取说明后，或用一句话指向保留的详细提取附表。R1-M1 的一句“pre-extracted”不能回答池化问题。

| 必须查清的对象 | 要写进文中的具体信息（当前不能猜填） |
|---|---|
| 文本 → 1024 | 历史拼接方式、说话人/话语标记、最大 token 数、截断方向、长历史处理、所取层与 `[CLS]`/mean/其他池化规则。 |
| 音频 → 1024 | 输入预处理及采样率、编码器输出层、有效帧 mask、时间池化、长短音频/空音频处理。 |
| 视频 → 342 | 人脸/画面选择、帧采样、编码器输出含义、跨帧聚合、多人/无脸/缺帧处理；不能未查实现就说342是普通隐藏层维度。 |
| 特征与标签对齐 | dialogue/utterance ID 的对应方法、缺失模态处理，以及特征缓存所属数据划分。 |

另定位 [Main.tex:868](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L868) 的 `We use the full dialogue history, up to 109...`：核查“历史话语数量”与“编码器实际能读入的 token 范围”是否一致。若存在截断，要把使用的历史范围和编码截断分别交代。

- [ ] 三个模态各有明确的聚合规则与证据。
- [ ] 109/32 的含义、token 截断和实际输入范围一致。

<a id="r1-m3"></a>

### R1-M3｜承认时序压缩的边界，不把 AMM 当作时序恢复

**原意见要点**：将 T×D 压成向量可能丢失有用信息。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:199](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L199)，`Here, the latent sequence comprises learned elements...` | 保留“非时间序列”，并明确这是给定向量的学习重表示；不能用 sequence 命名暗示还原声学/视觉帧。 |
| [Main.tex:406](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L406)，`a shared linear map expands one D-dimensional vector...` | 公式后补：`The expansion reorganizes information in the input representation; it does not recover temporal observations discarded during feature extraction.` |
| [Main.tex:1035](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1035)，未来工作句 | 加入对预聚合表示的依赖与时序粒度限制。不要声称本次已测量压缩损失程度。 |

若补实验，需要从同一提取器保留时序输出，对比实际池化设置或具有时序输入的对照，并控制额外计算预算。该实验改变输入粒度，属于独立扩展；不能拿 S=1/6 的投影消融代替。

- [ ] 清楚区分观测时序长度与学习视角数。
- [ ] 信息损失尚未实测时以限制表述；不凭无损/有损猜测下结论。

## 3. 第二位审稿人：XkY6

<a id="r2-1"></a>

### R2-1｜对话任务定位，以及两个容易漏掉的追问

**原意见要点**：方法更像通用增强组件组合。Rebuttal 还问：这些增强迁入基线是否也有效？MM-Mixer 加入对话专属机制是否还能获益？这两问不能只用一段动机替代。

#### A. 任务定位：哪些句子改

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:187](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L187)，`These examples motivate considering conversational context...` | 后接“本文聚焦这些信息被编码为语句级表示之后的融合”；保留图注 [Main.tex:192](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L192) 关于相同当前词语不等于相同文本特征的说明。 |
| [Main.tex:196](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L196)，问题段四句 | 改成“语句级表示条件下的融合组织”问题，接 R1-1；不要虚构新对话拓扑、说话人记忆或时序恢复模块。 |
| [Main.tex:222](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L222)，`MM-Mixer does not introduce a new dialogue topology...` | **保留并解释分工**：历史由文本表示承载，本文贡献在后续融合。可以说这种分工与对话编码器在接口上可组合，不能说组合后的收益已证实。 |
| [Main.tex:260](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L260)，`The textual feature represents the target utterance...` 附近 | 在输入定义中保持 text=target+history，audio/visual=target，且排除未来。细节与 R1-M2 一起补齐。 |
| [Main.tex:367](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L367)，`The MCA module comprises three directional cross-attention branches.` | 段尾补 MCA/AMM 分工：MCA 先对三分支整向量作输入相关注意力；AMM 再在学习视角张量上作分轴处理。MCA 后各分支已含其他模态信息，不能称其为纯单模态表示。 |

#### B. 将相同增强加到基线上是否也有收益

**添加位置**：[Main.tex:994](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L994) Ablation Study 内，新建替换/迁移对照段。可选一个适配接口的基线或普通融合骨架，比较原版、加相同辅助目标、加相同校准前端，保持输入特征和选模规则一致。

重点是拆开“通用训练/前处理收益”和“AMM 特定组织方式收益”。若迁入基线也有效，应如实写通用收益；不能据此仍将全部提升归于 AMM。该对照与 R2-6 共用，不重复运行。

#### C. 接入更专门的对话机制是否有帮助

**添加位置**：[Main.tex:1035](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1035) 附近的讨论/限制；若完成实验，再在 Ablation Study 给结果。

可讨论同一因果对话编码器接普通融合与 AMM 的组合，固定输入/未来屏蔽等规则。当前只能说接口允许研究这种组合，实际提升未知。不能为了回复而把未实现的对话模块写入现有模型。

- [ ] A：任务范围与上下文来源写清。
- [ ] B：迁入基线的通用增强问题有结果或明确未验证说明。
- [ ] C：对话专属模块组合问题有结果或明确未验证讨论。

<a id="r2-2"></a>

### R2-2｜S 个学习视角是什么，为什么要交互

**原意见要点**：D→S×D 中 S 的含义不清；与普通特征变换有何区别；已有 M/D 混合为何还需要 S 混合？这是当前改写的核心。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:391](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L391)，`AMM aims to mine the sentiment cues along different dimensions...` | **替换动机**为多学习投影与分轴组织；不要先假设三条轴本来就存在。S 是模型主动构造的轴。 |
| [Main.tex:393](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L393)，`The projection axis indexes different learned transformations...` | **保留**，并作为整个小节的术语基准；后文不再混用 sequence/token/subspace 来暗示不同实体。 |
| [Main.tex:406](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L406)，`Sequence construction.` 和 `a shared linear map expands...` | 标题建议改为 `Learned projection views.`；补逐视角等价式 `z^0_{m,s}=W^{(s)}h^m+b^{(s)}`，其中每个 `W^{(s)}` 是 D×D，S 个矩阵组成当前 D→SD 线性映射。不是复制同一向量 S 次。 |
| [Main.tex:416](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L416)，`the parameters ... are shared by T, A, and V...` | 解释“不同 s 用不同投影；同一 s 的参数在三个分支共享”。共享坐标提供一致参数化，**不保证语义对齐、正交或独立**。 |
| [Main.tex:421](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L421)，`learned-sequence element s` | 换成 learned projection view；维度、索引与固定 V/A/T 顺序保持一致。 |
| [Main.tex:426](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L426)，`Sequence mixing applies an MLP along the S axis...` | 改为 `Projection-view mixing...`，解释它组合**同一分支的不同学习视角**，M 混合交换对应视角的不同分支信息，D 混合变换通道。不要只列算子名字而不解释索引固定关系。 |
| [Main.tex:447](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L447)，`The bias-free routing matrix...` | 明确每块一个可学习静态 3×3 矩阵，在该块所有 s/d 坐标共享；不写成每个样本、每个视角拥有动态路由。 |
| [Main.tex:466](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L466)，`We stack two mixer blocks...` 附近 | 说明 S→M→D 和 M→S→D 是两个不同顺序；是否优于同序堆叠需要实验，不能只从操作顺序推导最优性。 |
| [Main.tex:398](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L398)，AMM 图注 `Modality mixing uses one learned...` | 改为 `Each block uses its own learned ... shared across projection/channel positions.` 避免读成两个块共享同一矩阵。图中视角不能标成“语气视角/悲伤视角”等已发现语义角色。 |
| [Main.tex:944](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L944)，`w/o Sequence Mixing`；[Supplement.tex:117](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Supplement.tex#L117) 的图注说明 | 若统一改名，同步表格与文字；旧图标签暂不重画时保留明确映射。“去掉 S 混合”不能写成“取消多投影展开”。 |

**Related Work 也要随这条主线修改**：

- [Main.tex:227](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L227)，`according to the internal structure of the utterance-level representation`：改为主动构造 learned view axis，避免暗示原向量中已有可观察的 S 结构；后一句的 learned-sequence 同步统一术语。
- [Main.tex:230](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L230)，`MLP-Mixer...separates token mixing from channel mixing`：在段后补区别，MLP-Mixer 的输入 token 与本模型从整向量生成的学习视角不同；本模型另保留模态轴。低秩乘法部分说明 EPIRC 是主聚合之外的残差，与将低秩融合用作主要融合算子的定位不同；不要把已知的乘法算子本身声称为首次提出。

**可写进方法的解释骨架**：

> The S views are learned transformations of the same input representation. View mixing combines these transformations within a branch, whereas modality mixing exchanges information across branches at corresponding view coordinates. This factorization defines a structured parameterization of fusion; its benefit over an unstructured feature transformation must be evaluated under controlled input and capacity settings.

这段回答“机制差别”，**不能单独回答“为何更好”**。单个线性展开本身可等价写成一个更宽线性层；价值假设在随后非线性混合、共享方式和交互结构，不能声称增加了新的观测信息。

**实验最小集合（计划，不是已完成结果）**：

| 对照 | 检验的问题 | 控制要求 |
|---|---|---|
| Full S=6 vs S=1 | 多视角展开是否值得 | 记录容量差；单独这组不能排除容量作用 |
| S=6，关闭 S 混合 | 跨视角混合是否有边际作用 | 保留展开、M/D 及其他组件，明确关闭位置 |
| AMM vs 普通残差 MLP 融合 | 分轴结构是否优于普通变换 | 相同输入、聚合输出接口、目标；匹配或报告参数/FLOPs |
| AMM vs 合理 attention 融合 | 与已有交互结构比较 | 相同输入/训练，说明容量与调参预算 |
| 可选 S=2/4/8 与同序双块 | 视角数和交替顺序是否敏感 | 验证集选择；不按测试集挑最好设置 |

已有 `w/o MCA` 或“以 Mixer 替换 MCA”的实验，如接口不同，只能回答前置注意力的作用，不能自动代替 AMM 多视角对照。

- [ ] S 的构造、共享、无预设语义和非时间含义写清。
- [ ] 至少有直接针对“展开”和“结构”的对照；缺失时不写必要性已证明。

<a id="r2-3"></a>

### R2-3｜EPIRC 的价值是否来自交互形式

**原意见要点**：只有整体移除不足，需解释 Hadamard 与残差的优势，并排除仅因增加容量而获益。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:526](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L526)，`Although the main fusion path adaptively aggregates...` 附近 | 精确写成：主路径通过学习查询聚合；残差在低秩投影后显式构造 AV/TV/TA 的双线性乘积，提供可学习加性校正。“explicit”不等于普通 MLP/attention 不可能表达交互。 |
| [Main.tex:534](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L534)，`interaction between two modalities is modeled as the product between two mixer outputs...` 附近 | 应写**低秩投影后的向量**的逐元素乘积，而不是直接把原 D 维 mixer outputs 相乘。 |
| [Main.tex:547](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L547)，`r_cat=[r_AV;r_TV;r_TA]` | 当前实现保留第 4 个恒零 R 维槽，输出层为 4R→D。可显式写 `[r_AV;r_TV;r_TA;0_R]`；或清楚声明公式使用等价有效 3R 子矩阵，实际参数统计仍按真实实现。**不可把零槽称为启用的三模态乘积。** |
| [Main.tex:553](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L553)，`W_o and b_o are initialized to zero...` 附近 | 保留初始化时残差为零的事实；不能由此直接声称收敛更快、更稳定，除非有训练曲线/初始化对照。 |
| [Main.tex:999](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L999)，`...further showing that axis-specific interaction and explicit...contribute complementary information.` | 改为相应配置下的分数变化；“互补信息”需要联合/替换证据。 |
| [Main.tex:906](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L906)，fear/disgust 的增益解释 | 没有逐类别 EPIRC 消融时，删去对少数类改善的模块级归因。 |

**优先对照**：Full pairwise residual / 无残差 / 相近参数的普通 MLP residual（同样输入与加法位置，必要时同样零初始化）。有余力再做逐对移除 AV、TV、TA，以及秩 R/初始化敏感性；不必一开始穷举所有组合。分析使用统一来源、逐种子差值；若要谈少数类则必须给相应类别结果。

- [ ] 公式、有效交互项与实现维度一致。
- [ ] 容量控制与形式对照完成，或将结论限制为整体模块的表内影响。

<a id="r2-4"></a>

### R2-4｜同标签辅助监督是否合理，权重为什么这样设

**原意见要点**：不同模态可能表达不同线索，同一整体标签可能抑制模态特性；需解释主/辅权重策略。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:617](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L617)，`Beyond the ... we also deploy an auxiliary loss...` 附近 | 说明辅助头的训练目的为保留对**话语级任务标签**有用的信息，是一种正则化假设；不能说三模态各自真实情绪一定一致。 |
| [Main.tex:628](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L628)，`These classifiers are used only during training...` 所在段 | 在训练专用说明后补局限：单模态证据可能含歧义/冲突，整体标签并非独立的模态标签；本设计不保证保留所有模态专属情绪信息。不能写梯度冲突已经发生，除非测量。 |
| [Main.tex:609](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L609)，`Here, omega_i denotes the focal modulation coefficient...` | 补两数据集的实际定义，见 C2。没有明确定义，读者无法判断主损失和辅助损失的权重差别。 |
| [Main.tex:637](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L637)，`Therefore, the total learning objective is defined...` 附近 | 明确固定任务系数与类别权重、focal 系数是三个不同概念；不要把固定损失系数写成自动学习的不确定性权重。 |
| [Main.tex:891](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L891)，`On IEMOCAP, the coefficients...` 与 `On MELD...equally weighted.` | 数字可按实际配置保留，但**补选择依据**：候选范围、依据哪个验证指标、是否数据集分别选择。现有发布结果有 test-peak 背景，不能凭写作改称在验证集选出。 |
| [Main.tex:895](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L895)，整行损失实现说明目前被 `%` 注释 | 将核实后的两数据集差异放入三段 Implementation 的第 3 段；`class-balanced Poly loss` 建议改为更准确的 `class-weighted PolyLoss`，同时给出类别权重计算方式。 |

**建议新增文字的含义**：我们把辅助任务用作任务相关正则，不把整体情绪标签当成每个模态的独立真值；用损失权重控制其影响，并通过实验检查净收益。不能以“弱模态梯度被保护”等未测机制替代证据。

**实验待办**：主任务不变，比较无辅助/等权辅助/当前权重，并作小范围权重敏感性；条件允许再分 T/A/V 辅助移除。控制整体 loss 尺度，避免关辅助时同时增大主任务梯度。若调查模态冲突，可加梯度或冲突样本分析，但不将其列为已经观察到的结论。

- [ ] 同标签假设及其限制明确。
- [ ] 权重真实选择流程可追溯；各类“权重”定义不混淆。

<a id="r2-5"></a>

### R2-5｜主表与消融 Full 不一致：已记录，作者后续处理

**原意见要点**：MELD Table 2 与 Table 4 的 Full 不一致，消融值甚至超过主表均值加标准差；明确哪一个对应主设置。

**本轮暂缓**：按作者最新安排，此项只保留审稿问题、定位与后续检查内容；不核改表格、不重算实验数字，也不把它作为文字改写的前置条件。下列是建立清单时记录的当前值，按 **ACC / WF1** 排列：

| 数据集 | 当前主表 | 当前组件消融 Full | 当前模态消融 T+A+V |
|---|---|---|---|
| IEMOCAP | 72.07±0.15 / 72.19±0.12 | 72.03 / 72.16 | 72.03 / 72.16 |
| MELD | 68.36±0.25 / 67.62±0.23 | 68.77 / 68.01 | 68.77 / 68.01 |

| 位置与原句/表格锚点 | 要改的内容 |
|---|---|
| [Main.tex:751](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L751)，IEMOCAP `MM-Mixer` 行；[Main.tex:802](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L802)，MELD `MM-Mixer` 行 | 建立每个数值→run/config/seed/checkpoint/prediction 的对应关系，明确这些行来自哪个实验版本。**不要现在把其中一行直接改成另一个表的数字。** |
| [Main.tex:926](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L926)，`Full MM-Mixer`；[Main.tex:985](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L985)，`T+A+V` | 若是单种子、另一特征批次或另一模型版本，写明真实来源；若来源不可核实，不当作已验证的正式消融。Full 和各变体必须在可比协议下。 |
| [Main.tex:757](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L757)、[809](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L809)，`All entries are reported as mean ± standard deviation...` | 补重复次数、seed、std 约定及表格来源；若不同模型来源不同，表注逐类说明。 |
| [Main.tex:961](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L961)、[989](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L989)，两个消融表注 | 加结果是单次还是多次均值，及相同 Full 设置的索引。若采用不同重复范围，解释而不掩盖。 |
| [Main.tex:894](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L894)，`All experiments are run on an NVIDIA TESLA V100 with three seeds [0, 1, 42]...` | 发布 run 的种子与此句不一致，先核对应哪批结果；V100 也要有环境记录。不能保留一个通用实验模板句覆盖所有运行。 |
| [Main.tex:857](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L857)，`We follow the standard train, validation, and test splits...` | 数据划分说明之后增加**实际选模规则**。训练/验证/测试划分存在，不代表 checkpoint 是按验证集选择。 |
| [Main.tex:997](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L997)，`...retraining it in the same training and evaluation settings.` | 必须由记录证实 same settings；包括特征、loss、各参数组学习率、种子、checkpoint 选择，不只 epochs/batch 相同。 |
| [Main.tex:899](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L899)、[906](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L906)、[999](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L999)、[1002](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1002)、[1016](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1016)，所有数值差值段 | 表格确定后统一重算差值、类别最优/次优、粗体和结论；不可只改表而保留旧分析。 |

**已核实的发布记录，供追溯，不作为直接替换指令**：

- [发布说明](../README.md)与 [results/peak_test](../results/peak_test)含 IEMOCAP seeds `2025,2066,2088,2118`，MELD seeds `2025,2028,2069,2101`。
- 发布 README 明确三种子子集为 IEMOCAP `2066,2088,2118`、MELD `2025,2028,2069`；对应均值为 **WF1/ACC：72.1854/72.0682、67.8506/68.5696**。四种子均值为 **72.1207/71.9963、67.7383/68.4483**。这与当前 MELD 主表和消融表都不能直接画等号。
- 发布 manifest 明确 `selection=strict_peak_test_wf1`；README 也披露三种子子集按测试表现选出。**这不能写成无偏、验证集选模的 held-out 测试结果。** 如正式主结果改用验证集选模，应找到相应产物或按固定协议重做，不能改一句描述冒充另一种协议。
- 当前公开结果包不含旧消融来源；发现另有输出目录也不等于证明表内每一行已重现，需逐一对应。

“消融单次结果高于主表均值+std”本身不构成统计学矛盾；应解释来源、重复范围与设置。若确属不同实验，应披露，不可为了表面一致强行填同一个 Full。

- [ ] 每行结果有来源，正式主设置唯一明确。
- [ ] 选模、seed、std、数据与模型版本在正文/表注中一致。
- [ ] 所有依赖表格的结论同步更新。

<a id="r2-6"></a>

### R2-6｜架构和损失分别贡献多少

**原意见要点**：IEMOCAP 去辅助损失下降较大，性能可能来自训练目标；需要替换式消融。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:997](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L997)，`We evaluate the contribution...by removing it...` | 扩展为移除与替换两类实验；写明替换了什么、保留了哪些输入/前端/训练目标。 |
| [Main.tex:999](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L999)，`removing auxiliary loss causes the largest degradation of 1.10...` | 来源确认后可以报告现象；不能从不同移除幅度直接算“架构贡献百分之几/损失贡献百分之几”，模块之间可能有交互。 |
| [Main.tex:1013](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L1013)，`Finally, removing MCA or Unimodal losses...confirming the complementary roles...` | 改为有限的观测描述；联合互补需对应四格，loss 优势需对照同架构。 |
| [Main.tex:956](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L956)，`w/o Auxiliary Loss` 行 | 在表注注明主损失系数/整体尺度是否保持。当前 no-aux 路径应核查，避免关闭辅助同时改主损失权重。 |

**建议分两步，避免一次换太多因素**：

1. **固定目标，换融合结构**：AMM 与近容量 MLP/attention 对照都使用相同主/辅目标；由此检查结构作用。复用 R2-2。
2. **固定结构，换目标**：在 AMM 与一个替换骨架上，分别对照只含主目标与主+辅助目标。若进一步用普通 CE 对比现有主损失，另列一组，明确同时改变了 focal/Poly/class weighting 中哪些项。

| 融合结构 | 固定主目标，不加辅助 | 同一主目标，加辅助 |
|---|---|---|
| 对照融合骨架 | 待核实/运行 | 待核实/运行 |
| AMM | 待核实/运行 | 待核实/运行 |

训练目标的相对系数和整体尺度分开控制。删除辅助时保留原主系数作严格删除，若另作归一化控制就单独标注。当前工作区的 no-aux runner 保留 IEMOCAP 主系数0.45、MELD主系数1；这只能说明当前实现，不能反推旧表的运行情况。实施时还应保留正常 loss 路径，不能将 `aux_logits` 设为 `None` 触发提前返回而同时跳过 focal。记录优化器组、学习率和调参预算，避免把训练变化归给结构。

- [ ] 替换式对照完成并可追溯。
- [ ] 归因语言限定到对照真正能回答的问题。

<a id="r2-7"></a>

### R2-7｜效率不能只看参数

**原意见要点**：需要 FLOPs、训练或推理成本，参数少不足以证明总体计算成本低。

**修改位置**：[Main.tex:909](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L909) 的效率标题、[910](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L910)–912 的解释、[843](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L843) 的表注，与 R1-6 共用修改。

**专门检查**：R1-6 的千样本推理时间回答运行时；R2-7 还指出 FLOPs 与训练成本缺失。至少在最终报告中逐项说明哪些实测、哪些未测，并限制效率结论。参数更少不保证更快，小矩阵算子、访存和多阶段调用均可能影响实际耗时。

- [ ] 用同一效率表回应两位审稿人。
- [ ] 参数、计算量、训练与推理成本没有混为同一种指标。

<a id="r2-8"></a>

### R2-8｜增加近期基线覆盖，解释比较边界

**原意见要点**：近期量化比较较少，IEMOCAP 领先有限，因此需要更有竞争力的参照。

| 句子位置与原句锚点 | 要改的内容 |
|---|---|
| [Main.tex:225](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L225)，Related Work 中 GS-MCC / EMART / DnR 等 | 已经讨论的方法不等于已经评测。逐项核查论文年份、任务/模态、代码与特征可用性，决定哪些适合加进主实验；不能为“近期”只加引用。 |
| [Main.tex:860](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L860)，`We compare MM-Mixer with representative methods from three mainstream...` | 补具体纳入标准：同任务/模态、可复现性、是否能适配因果设置、来源。保留代表性旧方法，同时补相关新方法。 |
| [Main.tex:863](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L863)，`For fair comparison, we retain their models with their released data, features, and hyperparameters...` | **必须改**：各自特征/超参数保留并不等于控制所有变量。区分按发布设置复现的系统级比较与同特征同协议的结构比较；记录因果改造，不能同时说完全无修改。 |
| [Main.tex:866](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L866)，`Evaluation is conducted under a history-only evaluation protocol...` | 若主张所有基线因果，需给每种序列/图/attention 方法如何屏蔽未来的证据。仅 MM-Mixer 输入无未来不够。 |
| [Main.tex:757](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L757)、[809](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L809)，主表 caption | 每组标注复现/原文报告、输入特征、上下文方向与选模规则。不能把原论文异协议数字混入一个表后称统一公平排名。 |
| [Supplement.tex:3](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Supplement.tex#L3)，A1 baseline 简介 | 补新增模型的机制简介；如排除特定强基线，正文/补充材料说明可核实的原因。 |

**本阶段先列候选，不把未复现的论文数字写成实测**：可从当前 Related Work 已列方法开始核查，选择最相关且协议可对齐者。新增基线后更新最优/次优及所有 `highest` / `outperforms` 句子。R2-8 的完整回应需要实证比较或明确的覆盖限制，引用列表变长本身不足。

- [ ] 纳入/排除理由与结果来源透明。
- [ ] 至少完成所选近期方法的有效比较；未完成项单列。

## 4. 随审稿修改一起处理的一致性问题（不是新增的真人审稿条目）

以下 C 项来自本次稿件/实现核查和已有记录，不冒充 R1/R2 原文要求。方法解释可以先改；涉及实验批次/数值的部分随 R2-5 由作者后续统一。

### C1｜把 A2 合入正文：保留一个标题、三个自然段

**位置**：[Main.tex:872](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L872)，`Implementation Details.`。

| 正文段落 | 从哪里补、改什么 | 对应审稿意见 |
|---|---|---|
| 第 1 段：输入表示 | 现有提取器/维度 + Supplement 49–56 的预提取与历史范围；补查证后的提取/池化规则。109/32 已在 Evaluation 写过，可回指而不重复。 | R1-M1/M2 |
| 第 2 段：架构配置 | 现有 D=256、8 heads、S=6、hidden=1536、R=32；补 `uses two mixer blocks`。残差系数和分类头实现按 C3 交代。 | R1-1、R2-2/3 |
| 第 3 段：优化与复现 | AdamW、**参数组学习率**、weight decay、batch、epochs、**fusion dropout**、两数据集损失差异、真实 seed/硬件/选模规则。 | R2-4/5/6 |

**需要纠正的现有句子**：

- [Main.tex:884](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L884)，`using base learning rates of 3×10^-5 ... and 4.17×10^-5...`：不能省略不同参数组。发布 manifest 的有效分组如下；如最终使用另一批实验，以那批记录为准。
- [Main.tex:888](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L888)，`The dropout rate is set to 0.2...`：改为 fusion dropout，避免暗示每个 dropout 层都为0.2。
- [Main.tex:894](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L894)：核实硬件与种子后重写，见 R2-5。
- [Supplement.tex:46](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Supplement.tex#L46)：合并确认无遗漏后删除重复 A2；若保留细节附表则重新命名并避免复述同样三段。**不要删除 A1 baseline 和 B1 可视化。**

| 参数组（以 runtime audit 的分组名为准） | IEMOCAP | MELD |
|---|---:|---:|
| projection | 3.00×10^-5 | 1.0416413×10^-5 |
| mixer / remaining | 6.00×10^-5 | 2.0832827×10^-5 |
| cross_attention | 6.00×10^-5 | 未单列；按其实际所属组 |
| pairwise_cross | 1.20×10^-4 | 未单列；按其实际所属组 |
| classifier | 1.20×10^-4 | 4.1665653×10^-5 |

来源：[IEMOCAP manifest](../results/peak_test/iemocap/seed2025/manifest.json)、[MELD manifest](../results/peak_test/meld/seed2025/manifest.json) 的 `runtime_audit.learning_rates`。这张表是分组学习率事实，不把 `classifier` 自动等同于所有辅助头；各组参数归属需与实现对应。

- [ ] 三段正文完成，A2 重复内容清理，运行事实已核对。

### C2｜补全 focal 系数和类别权重

**位置**：[Main.tex:600](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L600)–617 的主损失公式与定义。建议保留统一形式，但增加数据集对应的具体定义：

- IEMOCAP：`omega_i = stopgrad(mean_j (1-p_{j,y_j})^gamma_f)`，批内样本共享；主 PolyLoss 使用类别权重。
- MELD：`omega_i = (1-p_{i,y_i})^gamma_f`，概率参与反向传播；主 PolyLoss 的 `w_y=1`。
- 两者的辅助目标都使用 class-weighted PolyLoss；给出实际类别权重构造与归一化、是否仅作用于 CE 项，不以含糊的 “balanced” 替代公式。
- 主/辅固定任务系数见 Implementation，不能宣称存在已学习的不确定性权重。

当前统一公式可容纳两种 omega，但不写定义会掩盖关键差别。数学定义需与当前 loss 代码及对应运行 manifest 一起核对。

- [ ] 两数据集目标及梯度路径明确，公式和文字一致。

### C3｜方法细节与有效参数化

| 位置 / 原句锚点 | 具体动作 |
|---|---|
| [Main.tex:426](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L426)，`three pre-normalized residual operators`，以及 `LN_S` / `LN_M` | 说明下标标识算子，实际 LayerNorm 均沿 D 特征维归一化；不要让读者误认为 LN_S 沿 S、LN_M 沿 M。也可统一改为 LN 并注明各算子独立参数。 |
| [Main.tex:362](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L362)，`epsilon_g preserves a direct contribution...`；[386](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L386)，`epsilon_a controls the query residual` | 报告实际运行的门控/查询残差系数；当前发布实现基值为0.1/0.2，仍需对应最终配置，不能把敏感性实验的覆盖值混入 Full。 |
| [Main.tex:594](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L594)，`a linear classifier is used...` | 当前实现平均三个线性头 logits。可写实际三头平均式；或声明正文 W_c/b_c 为平均后的等价仿射头，在 Implementation 交代真实三头参数化。不要误称平均概率或多模型 ensemble。 |
| [Main.tex:524](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L524)，`back to the common feature space.where ...` | 删重复 vec 定义，修正缺空格；向量化定义保留一次。 |

实现依据：[MELD model.py](../vendor/meld/model.py)、[MELD multiattn.py](../vendor/meld/multiattn.py)、[IEMOCAP mixer](../vendor/iemocap/factorized_mixer/model.py) 与对应输入/目标运行记录。

- [ ] 归一化轴、矩阵共享范围、残差系数和分类头表述一致。

### C4｜结果叙述、可视化与文档不得相互代替证据

- [Main.tex:899](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Main.tex#L899)、906、999–1016：差值统一使用百分点；`confirming`、`demonstrating`、`complementary` 逐句检查能否由对应对照支撑。
- [Supplement.tex:89](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Supplement.tex#L89)、100：t-SNE 属定性展示，不能代替视角语义解释、联合消融或因果归因。更新模型/结果版本时要检查旧可视化是否还对应该模型。
- [Supplement.tex:139](https://github.com/YB123-DT/mm-mixer/blob/bd4d5e7fcd963fac095711892ebfec1fa4531b00/paper/Supplement.tex#L139)：保留“相同 per-class cap 和设置”的准确措辞；跨模型测试顺序可能不同，不能改成所有图严格使用同一批样本，除非重新对齐样本 ID。
- [METHOD_ALIGNMENT.md](../METHOD_ALIGNMENT.md) 曾写“正文已写0_R”“MELD表已更新”，但当前 Main.tex 仍有上述失配。修改真正落地后再同步该文档，不能把文档中的“已完成”当成完成证据。
- Issue #1 另有 AI Review 指出的 MELD 若干基线类 F1/WF1 聚合疑点。此项**不是 R1/R2 原始意见**，可在 R2-5 的逐预测重算中一并核查；不直接照抄独立估算替换表内数值，也不据此断言结果不真实。

- [ ] 图、表、文字与证据版本对应；不扩大定性图的解释范围。

## 5. 按点修改时的记录模板

每次只处理一个编号或一组重叠编号，直接在下面追加；这一份 MD 保持为当前清单，不另存一堆“最终版”副本。

```text
处理编号：
对应审稿问题：
修改的原句 / 公式 / 表格行：
修改后内容：
证据来源或实验状态：
尚未解决的部分：
同步检查：摘要 / 引言 / 方法 / 实验 / 结论 / Supplement / 图注
TeX_CHANGELOG 记录、编译结果与 Git 提交（实际改稿后填写）：
```

后续实际改 TeX 时，按已有约定先归档旧版、更新 `TEX_CHANGELOG.md`、编译核对并推送。当前文档提交只记录修改方案，不产生新的论文 PDF 版本。

## 6. 覆盖与自检

- **贡献**：R1-1、R2-1/2 回答“主贡献是什么”；目前文字方案齐备，结构优势仍须控制实验。
- **清晰度**：每项含原句/表格/新增插入位置；新增段落明确标识，不伪装成已有句子。
- **实验强度**：R1-2、R2-2/3/4/6 列出了直接对应问题的对照，均为待核实或待完成。
- **评估完整性**：R1-4/5/6、R2-7/8 覆盖范围、域外、效率和近期基线；没有用场景图替代真实评测。
- **设计与事实一致性**：方法说明以实现为依据；R2-5 及实验数字已按作者要求暂缓，不阻塞主线改写，也不据未核实结果新增强主张。

原意见覆盖：**R1 六条缺点 + 三条次要问题；R2 八条缺点 + 八条同编号 Rebuttal 反馈**。R1 推荐理由没有新增问题，已归入 R1-1/3/4/6；R2 推荐理由对应其八项，已逐项包含。两位审稿人的优点不需要人为制造修改任务。
