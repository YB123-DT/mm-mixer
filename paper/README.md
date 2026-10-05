# 当前论文文件

当前唯一 TeX 文件是 [Main.tex](Main.tex)，包含正文与全部补充材料，不再调用外部 TeX。完整 PDF 是 [Main.pdf](Main.pdf)。

两位审稿人的逐句修改待办见 [REVISION_PLAN.md](REVISION_PLAN.md)：按 R1/R2 编号列出原句、位置、改法和证据需求；表格口径问题按作者安排留待后续处理。引言前四项文字修改已在 v009 写入 TeX；当前 v017 修改 Conclusion：突出 AMM 的学习投影视角与分轴交互，压缩支持模块说明，并交代两数据集及泛化验证限制。正文与补充材料继续共用唯一 Main.tex。贡献列表及其他待办仍按清单逐项处理，进度见清单开头。

`paper/` 顶层只保留当前版本及其依赖：单一主稿（含补充材料）、完整 PDF、当前图文件、AAAI 样式与参考文献、版本索引和操作日志。`MM-Mixer.pdf` 是当前稿引用的架构图，完整论文请打开 `Main.pdf`。

旧版本按版号分开保存在 [archive/](archive/README.md)：每个目录都有对应的源码、PDF、图文件和编译依赖。当前 Figure 1 为单栏 AMM 三轴张量图 [01_amm_axis_mixing.pdf](01_amm_axis_mixing.pdf)，图稿在 `figures/amm_axis_mixing/`；正文原有 AMM 图稿在 `figures/motivation/`。旧场景图及其来源随 v009 存入归档；中间构建文件在不上传的 `build/`。

```text
paper/
  Main.tex / Main.pdf    当前稿
  当前图文件、样式、参考文献和索引
  figures/amm_axis_mixing/               新 Figure 1 图稿
  figures/motivation/                     正文原有 AMM 图稿
  archive/v001/ ... archive/v016/         历史版本
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
