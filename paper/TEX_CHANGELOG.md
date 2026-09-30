# TeX 操作日志

本文件追加记录所有 TeX 修改；`Main.tex` 是后续主稿。最终 TeX/PDF 保存在本目录，每个完成版本编译后提交并推送到 GitHub。

## 2026-09-30 · v003：保存已完成的图稿预览版本

1. `MM-mixer-with-figures.tex` 从已有 `MM-mixer-revised.tex` 派生。在 Introduction 第一段后加入场景说明、场景图及 `fig:motivation_scenes`；在 Axis-wise Multimodal Mixer 第一段后加入结构说明、示意图及 `fig:amm_projection_axes`。
2. 该派生版本保留原有摘要、实验数字与其余正文，使用原稿已有的 `meld` 引用键；完整 PDF 经编译为 9 页，场景图与 AMM 图位于第 2、5 页。
3. 按用户最新目录要求，将两个 `includegraphics` 路径改为 `paper/` 根目录中的 PDF 文件名；图的内容与这一版的宽幅布局保持一致。重新编译验证路径与交叉引用。
4. `MM-mixer-revised.tex` 是接管前已存在的稿件，本次登记并上传，不将其既有正文修改归为本轮新增修改。
5. 用户提供的 `Main.tex` 当前与原始 `MM-mixer.tex` 字节一致；`Supplement.tex` 当前仍是独立文档。这两个输入先保存到版本历史，下一版在 Main 中整合。

这一版的场景图为宽幅三列，保留作历史版本；用户要求的单栏版将在 v004 中加入 Main。
