# 当前论文文件

当前修改与编译入口是 [Main.tex](Main.tex)，它调用 [Supplement.tex](Supplement.tex)，完整 PDF 是 [Main.pdf](Main.pdf)。

两位审稿人的逐句修改待办见 [REVISION_PLAN.md](REVISION_PLAN.md)：按 R1/R2 编号列出原句、位置、改法和证据需求；表格口径问题按作者安排留待后续处理。引言前四项文字修改已在 v009 写入 TeX；当前 v012 优化 Figure 1 的长方体布局，将 S/M/D 示例分别放在顶部、右侧和前面，简化标题、标签与轴线。v011 已移除下半映射流程；本次不修改 TeX 正文或图注。贡献列表及其他待办仍按清单逐项处理，进度见清单开头。

`paper/` 顶层只保留当前版本及其依赖：主稿、补充材料、完整 PDF、当前图文件、AAAI 样式与参考文献、版本索引和操作日志。`MM-Mixer.pdf` 是当前稿引用的架构图，完整论文请打开 `Main.pdf`。

旧版本按版号分开保存在 [archive/](archive/README.md)：每个目录都有对应的源码、PDF、图文件和编译依赖。当前 Figure 1 为单栏 AMM 三轴张量图 [01_amm_axis_mixing.pdf](01_amm_axis_mixing.pdf)，图稿在 `figures/amm_axis_mixing/`；正文原有 AMM 图稿在 `figures/motivation/`。旧场景图及其来源随 v009 存入归档；中间构建文件在不上传的 `build/`。

```text
paper/
  Main.tex / Supplement.tex / Main.pdf    当前稿
  当前图文件、样式、参考文献和索引
  figures/amm_axis_mixing/               新 Figure 1 图稿
  figures/motivation/                     正文原有 AMM 图稿
  archive/v001/ ... archive/v011/         历史版本
  build/                                 临时编译与验证
```

从本目录编译当前稿：

```bash
mkdir -p build/main
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
  -outdir=build/main -jobname=Main Main.tex
cp build/main/Main.pdf Main.pdf
```

以后修改前先将上一版的源码、PDF 和对应依赖存入 `archive/vNNN/`，再更新顶层当前文件。不要在顶层额外生成带版号的完整 PDF。完成后更新 [TeX 操作日志](TEX_CHANGELOG.md)、[版本索引](VERSIONS.md) 和 `versions.json`，编译验证后提交并推送。

作者原稿在 [archive/v001/MM-mixer.tex](archive/v001/MM-mixer.tex) 中逐字节保留，SHA256 为 `7e9d5470c30331704a1baadfa9031906cbda5c0f322bb4fbefe3f6646f447064`。原独立 Supplement 在 [archive/v003/Supplement.tex](archive/v003/Supplement.tex)。归档目录保留历史内容；历史复跑的输出写入 `paper/build/`，避免覆盖归档 PDF。
