# TeX 操作日志

本文件追加记录 TeX 修改与归档操作；`Main.tex` 是主稿。`paper/` 顶层仅保留当前版本，旧版按 `archive/vNNN/` 保存，每个完成版本验证后提交并推送到 GitHub。

## 2026-09-30 · v003：保存已完成的图稿预览版本

1. `MM-mixer-with-figures.tex` 从已有 `MM-mixer-revised.tex` 派生。在 Introduction 第一段后加入场景说明、场景图及 `fig:motivation_scenes`；在 Axis-wise Multimodal Mixer 第一段后加入结构说明、示意图及 `fig:amm_projection_axes`。
2. 该派生版本保留原有摘要、实验数字与其余正文，使用原稿已有的 `meld` 引用键；完整 PDF 经编译为 9 页，场景图与 AMM 图位于第 2、5 页。
3. 按用户最新目录要求，将两个 `includegraphics` 路径改为 `paper/` 根目录中的 PDF 文件名；图的内容与这一版的宽幅布局保持一致。重新编译验证路径与交叉引用。
4. `MM-mixer-revised.tex` 是接管前已存在的稿件，本次登记并上传，不将其既有正文修改归为本轮新增修改。
5. 用户提供的 `Main.tex` 当前与原始 `MM-mixer.tex` 字节一致；`Supplement.tex` 当前仍是独立文档。这两个输入先保存到版本历史，下一版在 Main 中整合。

这一版的场景图为宽幅三列，保留作历史版本；用户要求的单栏版将在 v004 中加入 Main。

## 2026-09-30 · v004：Main 主稿、单栏场景与补充材料

1. `Main.tex`：以用户提供的原稿为底稿，在 Introduction 第一段后加入一段 MELD 场景说明和 `fig:motivation_scenes`。使用普通 `figure` 与 `width=\columnwidth`，引用根目录 `01_same_words_scenes_single_column.pdf`。三组样例改为上下排列，原生尺寸为 84 × 72 mm；标题直接写明当前台词为 “Hey!”。
2. `Main.tex`：在 Axis-wise Multimodal Mixer 第一段后加入投影视角说明和 `fig:amm_projection_axes`，引用根目录 `02_amm_projection_axes.pdf`。说明学习投影不代表观察到的时间帧或预先指定的语义角色。
3. `Main.tex`：在原有唯一的 `\bibliography{aaai2027}` 前加入 `\FloatBarrier`、`\appendix`、Supplementary Material 标题和 `\input{Supplement}`。补充材料共用 Main 的前导、编号和参考文献。
4. `Supplement.tex`：移除独立文档的 documentclass、包导入、标题/作者、maketitle、begin/end document 和单独 bibliography，保留 A1、A2、B1 的正文与三张 t-SNE 图。原始独立文件保存在 `paper-v003`。
5. `Supplement.tex`：将跨模型图注的 “the same class-stratified test samples” 改为 “class-stratified test samples with the same per-class cap”。依据既有 `model-comparison-tsne-diagnostics.json`，不同仓库的测试顺序有差异，不能声称不同模型逐样本一致。
6. `Supplement.tex`：在组件消融图注说明面板中的 “Sequence Mixing” 指学习投影轴上的混合，并核对引号闭合；将 `Figure~~\ref` 中重复的不可断空格修正为一个。
7. `Supplement.tex`：三张双栏图的放置参数从 `[t]` 改为 `[tp]`，允许独立图页。仅允许页顶时，浮动屏障之后的参考文献第一栏出现异常拉伸；独立临时编译验证 `[tp]` 后间距正常，全文从 13 页变为 12 页。移除重复屏障的对照未解决问题，未采用；AAAI 模板未修改。
8. 保留 `Main.tex` 的原有摘要、正文、公式、实验表和数值；此次修改是图的接入与补充材料整合。原始 `MM-mixer.tex` 不变，TeX 保持 CRLF 换行。

编译和验证结果：`latexmk` 成功生成 12 页完整稿，逐页视觉核查通过。场景图在第 1 页，AMM 图在第 5 页，补充材料正文从第 9 页开始，三张 t-SNE 图在第 10–11 页，唯一参考文献在第 12 页。33 个标签无重复，引用目标完整，25 个引用键均存在，六份图文件全部接入；没有未定义引用或超宽盒。编译仍有常规 underfull 排版提示，逐页检查没有裁切或异常间距。

移除三处新增块后，Main 与 `paper-v003` 保存的用户原稿逐字节一致。Main、Supplement、样式、参考文献及六张图的 SHA256 见 [`Main-build-manifest.json`](archive/v004/Main-build-manifest.json)。输出为 [`Main.pdf`](archive/v004/Main.pdf) 和 [`MM-Mixer-v004-main-with-supplement.pdf`](archive/v004/Main.pdf)，两份 PDF 字节一致。版本标签：`paper-v004`；后续完成的 Main 修改沿用“记录—编译—核查—提交—推送”流程。

## 2026-09-30 · v005：场景图简化标注并重新导出全文

1. 按用户要求，从当前单栏场景图删除三处 “Clip audio envelope” 副标题和三处 train/test 样本编号；波形区域略微上移并增高，利用删除文字后留下的空间。原生尺寸仍为 84 × 72 mm，保留情绪标签、共同坐标轴和历史文本说明。
2. 修改图的生成脚本 `figures/motivation/build_figures.py`，重新导出根目录 `01_same_words_scenes_single_column.pdf` 及图稿目录的 PNG/SVG。样本编号与来源仍保存在 `source_data.csv` 和完整图注记录中；CSV、三个原始帧及历史宽幅图均不变。
3. 本版没有修改 `Main.tex` 或 `Supplement.tex`。替换 Main 引用的图资产后，沿用既有 `latexmk` 命令重新编译，更新 `Main.pdf` 并保存 `MM-Mixer-v005-scene-clean.pdf`。两份输出字节一致，均为 12 页。
4. 验证删除的文字未出现在当前单栏图或全文第一页；核对原生尺寸、数据来源和图像导出。人工核查单栏图及全文第一页通过，第 2–12 页渲染像素与 v004 一致。最终编译无未定义引用、超宽盒或 LaTeX 错误，仍有 7 条常规 underfull 提示。

本版图稿校验见 [`single_column_validation_report.json`](archive/v005/figures/motivation/single_column_validation_report.json)，编译依赖和输出哈希见 [`Main-build-manifest.json`](archive/v005/Main-build-manifest.json)。版本标签：`paper-v005`。

## 2026-09-30 · v006：同步 Main 的场景正文和图注

1. `Main.tex:187`：重写 Introduction 中的场景段落，明确这是同一说话人 Ross 的三个独立话语样本，每个样本的当前台词都仅标注为 “Hey!”，对应 joy、sadness、neutral。以简短问候的情绪解读引出对历史上下文、声音表达和面部表情的联合考虑，不声称 AMM 已正确识别这些样例。
2. `Main.tex:192`：同步 `fig:motivation_scenes` 的图注，将 `clip-audio envelopes` 改为 `audio waveform`，删除 PCM 术语及重复的示例说明；明确每行是原始视频帧与对应音频波形，共同时间与幅度标尺仅指音频面板。历史对话说明只在图注中保留一次，正文不再重复。
3. 本版只替换以上两处文字。反向还原这两段后，Main 与 v005 逐字节一致；CRLF 换行不变。`Supplement.tex`、原始稿、六张图以及实验表和数值均保持原样。
4. 重新编译并核查 12 页全文。33 个标签无重复，12 个引用目标与 25 个引用键均完整，六张图文件均存在；最终编译没有未定义引用、超宽盒或 LaTeX 错误，保留 5 条常规 underfull 提示。文字变化导致第 1–9 页重新排版，逐页视觉检查通过，第 10–12 页与 v005 渲染像素一致。

输出为 [`Main.pdf`](archive/v006/Main.pdf) 与 [`MM-Mixer-v006-scene-text.pdf`](archive/v006/Main.pdf)，两者字节一致。源文件和输出哈希见 [`Main-build-manifest.json`](archive/v006/Main-build-manifest.json)，版本标签：`paper-v006`。

## 2026-10-01 · v007：删除场景图注中的音频说明

1. `Main.tex:192`：按用户要求，整句删除 “Each row shows an original video frame and the corresponding audio waveform, with shared time and amplitude scales across the audio panels.”，不再将 `clip-audio envelopes` 换成另一个音频术语写在图注中。
2. 图注保留三个 Ross 样本、当前台词 “Hey!”、数据集情绪标签和历史上下文说明。本次只删除这一句；反向还原后 Main 与 v006 逐字节一致，CRLF 换行及其余编译输入均不变。
3. `latexmk` 成功生成 12 页全文；33 个标签无重复，12 个引用目标、25 个引用键和六张图均完整。最终编译没有未定义引用、超宽盒或 LaTeX 错误，保留 7 条常规 underfull 提示。逐页检查重新排版的第 1–9 页，第 10–12 页与 v006 渲染像素一致。

输出为 [`Main.pdf`](archive/v007/Main.pdf) 与 [`MM-Mixer-v007-caption-short.pdf`](archive/v007/Main.pdf)，两者字节一致。哈希与核查记录见 [`Main-build-manifest.json`](archive/v007/Main-build-manifest.json)，版本标签：`paper-v007`。

## 2026-10-01 · v008：顶层只保留当前版本，历史稿分别归档

1. 按用户最新要求整理目录。`paper/` 顶层仅保留 Main、Supplement、当前完整 PDF、当前图文件、编译依赖与索引；不再保留多个带版号的完整 PDF 或旧稿 TeX。
2. 将 v001–v007 分别存放于 `archive/v001/` 到 `archive/v007/`。归档完整 PDF 统一命名为 `Main.pdf`；每版对应的源文件、图文件、样式和参考文献取自相应 Git 标签，并核对原有 SHA256。作者原稿移动到 `archive/v001/MM-mixer.tex`，字节不变。
3. v003 的宽幅场景图、旧 PNG/SVG、原始图稿核查记录及 `exports/` 元数据移入相应归档目录。更新当前 README、版本索引、历史链接和工作约定，明确以后先归档上一版，再更新顶层当前稿。
4. 本次没有修改 Main 或 Supplement 的 TeX 内容、当前图文件、正文数字及当前完整 PDF。只整理文件位置和文档；当前 PDF 与 v007 字节一致。
5. 核对七份归档的源码、PDF 和依赖哈希，所有引用的图文件与补充材料均存在。将当前 Main 编译到临时目录验证，仍为 12 页，所有页面与保留的当前 PDF 渲染像素一致，无未定义引用、超宽盒或 LaTeX 错误；有 7 条既有 underfull 提示。

当前稿见 [Main.pdf](Main.pdf)，旧稿见 [archive/](archive/README.md)。本版清单见 [Main-build-manifest.json](Main-build-manifest.json)，版本标签为 `paper-v008`。当前完整 PDF 只保留 `Main.pdf`，不另存一个带版号的顶层副本。

## 2026-10-01 · v009：引言的场景衔接与多投影视角主线

1. 修改前将 v008 的 `Main.tex`、`Supplement.tex`、`Main.pdf`、参考文献、样式、图文件及图稿来源复制到 `archive/v008/`，共 26 个原文件；逐文件与 `paper-v008` 核对字节并生成归档哈希清单。
2. `Main.tex:187`：保留三个 Hey! 样本及其说明，段末增加一句桥接，将场景引向编码后的模态表示如何交互这一一般问题。模型任务仍是话语级 MERC，不限定为短话语。
3. `Main.tex:196`：将“许多现有方法”泛化批评缩窄为以整体向量为交互单位的融合设计；删除 obscured、diluted、insufficiently sensitive 等缺少直接证据的解释，明确问题是如何组织多个学习视角、区分模态内处理与跨模态交换。
4. `Main.tex:198`：AG 使用共享多模态参考校准已有特征，MCA 准备模态分支表示；为后续多视角交互建立联系。
5. `Main.tex:200`：补入多头注意力的多投影设计启发，将 latent sequence 解释为 learned projection views；分别说明投影轴内分支视角混合、模态轴跨分支交换、隐藏特征轴通道变换。保留“不是有时间顺序的话语序列”的说明，不声称恢复时间信息或赋予视角固定语义。
6. `Main.tex:202`：将重复的 simple fusion / dual path 描述压缩为两句，明确查询聚合形成主融合表示，EPIRC 以低秩成对乘性交互提供残差补充。
7. `aaai2027.bib`：新增 `vaswani2017attention`。作者、题名、会议、卷号和年份来自 NeurIPS 官方 BibTeX，并对照原论文第 3.2.2 节确认多投影设计。与当前文献表的字段风格一致，省略可选的编者和出版社信息；原有条目不变。来源见 `Main-build-manifest.json` 的 citation_provenance。
8. 范围核验：引言第一段、Figure 1 图注与图文件、摘要、实验总结、贡献列表、Related Work 及其后所有 TeX、Supplement 全部保持原字节。主稿继续使用 CRLF。此次不修改或启动实验；表格口径仍由作者后续处理。
9. 更新版本索引、README 和逐句清单中的进度；清单的原句定位继续固定到 v008，未把文字调整标记为实验问题已解决。

编译及验证：`latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build/intro-v009 -jobname=Main Main.tex` 成功。完整稿为 **13 页**：新增文献后最后一个参考条目自然续排到第 13 页，没有修改模板字号、页边距或强制压缩间距。33 个标签无重复，12 个交叉引用目标和 26 个文献键均解析，六份图文件全部存在；无未定义引用或超宽盒，有 5 条 underfull 提示。检查全稿缩略图并详细检查引言、正文衔接、补充材料衔接和参考文献页，没有裁切或重叠；第 10–11 页与 v008 像素一致。独立文字复核通过，八份归档清单哈希通过。

当前文件为 [Main.tex](Main.tex) 和 [Main.pdf](Main.pdf)；旧稿为 [archive/v008/](archive/v008/Main.tex)。版本标签为 `paper-v009`，编译与范围检查记录见 [Main-build-manifest.json](Main-build-manifest.json)。

## 2026-10-02 · v010：Figure 1 替换为 AMM 三轴交互原理图

1. 修改前将 v009 的 `Main.tex`、`Supplement.tex`、`Main.pdf`、六张图、参考文献、样式、编译清单及图稿来源复制到 `archive/v009/`，共 26 个原文件；每个文件与 `paper-v009` 逐字节核对，并记录 SHA256。历史 PDF 保留原始字节。
2. `Main.tex:187`：将原有三个 Ross “Hey!” 样本的场景说明替换为一句对 AMM 轴向交互图的引导，不再用真实场景作为 Figure 1。
3. `Main.tex:189`：保留普通 `figure[t]` 和 `width=\columnwidth`，图片路径改为 `01_amm_axis_mixing.pdf`，标签改为 `fig:amm_axis_mixing`。图注说明三轴张量、各轴向量映射在其余位置共享、映射在 mixer block 内依次应用，并明确投影视角不是时间步；说明为清晰起见省略归一化和残差相加。
4. 修改范围只包含原场景段及该图的路径、图注、标签。反向还原这一连续块后，Main 与 v009 逐字节一致；摘要、Despite 段、后续方法介绍、实验总结、贡献、其余正文及 Supplement 均保持原字节。主稿继续使用 CRLF。
5. 更新 README、版本索引和归档索引，v009 路径指向 `archive/v009/`，新版本登记为 v010。图稿生成、完整稿编译和发布由本轮主流程完成。

编译及验证：`latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build/amm-v010 -jobname=Main Main.tex` 成功，完整稿为 **13 页**。33 个标签无重复，12 个交叉引用目标、26 个引用键及 6 张图均解析；无未定义引用、超宽盒或 LaTeX 错误，保留 7 条 underfull 提示。检查独立图稿、全文缩略图和第 1–2 页，单栏图无裁切或重叠；独立图稿/第一页复核得分 95/100。与 v009 像素一致的页面为 [10, 11, 12, 13]。图以 SVG 为可编辑源，PyMuPDF 导出矢量 PDF，检查确认 0 个位图和可提取文字。完整旧图已在 v009 归档，旧场景 PDF 从顶层移除。哈希和检查结果见 `Main-build-manifest.json`；没有运行训练或修改实验代码。

当前入口为 [Main.tex](Main.tex)，旧稿为 [archive/v009/](archive/v009/Main.tex)。版本标签为 `paper-v010`。
