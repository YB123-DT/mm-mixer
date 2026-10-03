# AMM 多投影视角概念图预览

本目录保存一份当前预览：二维模态表示经过共享学习投影与 reshape，形成三维多视角张量。本图尚未替换 `paper/Main.tex` 中的 Figure 1；当前完整论文仍为 v012。

## 文件

- `01_amm_projection_views_preview.svg`：可编辑源图。
- `01_amm_projection_views_preview.pdf`：单页纯矢量 PDF，原生尺寸 86 × 48.375 mm。
- `01_amm_projection_views_preview.png`：450 dpi 预览。
- `build_figure.py`：使用现有 PyMuPDF 从 SVG 导出 PDF/PNG，无新增依赖。
- `figure_contract.json`：张量、投影共享范围、体素索引及省略项。
- `validation_report.json`：导出与结构检查结果、视觉审查结果。

## 图意

左侧是 MCA 后的三个模态分支，省略 batch 后形状为 `M × D = 3 × 256`。同一个学习线性投影 `D → S·D` 作用于三个分支，再 reshape 为 `M × S × D = 3 × 6 × 256`。S 是学习投影视角，不是时间步或 attention heads。

蓝、紫、橙三根不相交的体素条分别标示 S、M、D 轴上的混合方向。每条固定另外两个索引；用不透明的顶面和侧面表达体积，并省略相邻体素之间的内部面。左、右两侧均用 7 格示意实际的 256 维 D 轴；S 为 6 格，M 为 3 格。

引线仅说明轴向操作，不表示三个并行分支。真实运算为 S 轴 MLP、M 轴无偏置 Linear、D 轴 MLP。两块的交替顺序、归一化、残差及池化由方法部分说明，本概念图不展开。

## 备选图注

```latex
Concept of AMM. Aligned modality vectors are expanded by a shared
learned projection and reshaped into an $M\times S\times D$ tensor.
Colored fibers indicate mixing along the learned-view, modality,
and hidden-feature axes. $S$ indexes learned views rather than time
steps; feature cells are drawn schematically.
```

## 重新生成

在仓库根目录执行：

```bash
python paper/figure_previews/amm_projection_views/build_figure.py
```

该命令覆盖本目录的 SVG、PDF、PNG 和图意契约。修改生成器或 SVG 后需要重新核对图意及验证报告。
