# TeX 操作日志

本文件追加记录所有 TeX 修改；`Main.tex` 是后续主稿。最终 TeX/PDF 保存在本目录，每个完成版本编译后提交并推送到 GitHub。

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

移除三处新增块后，Main 与 `paper-v003` 保存的用户原稿逐字节一致。Main、Supplement、样式、参考文献及六张图的 SHA256 见 [`Main-build-manifest.json`](Main-build-manifest.json)。输出为 [`Main.pdf`](Main.pdf) 和 [`MM-Mixer-v004-main-with-supplement.pdf`](MM-Mixer-v004-main-with-supplement.pdf)，两份 PDF 字节一致。版本标签：`paper-v004`；后续完成的 Main 修改沿用“记录—编译—核查—提交—推送”流程。
