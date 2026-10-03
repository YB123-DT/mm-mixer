# AMM 三轴交互图

当前 Figure 1 的源文件是 `01_amm_axis_mixing.svg`，矢量 PDF 为 `../../01_amm_axis_mixing.pdf`，PNG 为本目录预览。

- 长方体省略 batch，表示 `M × S × D = 3 × 6 × 256`。
- 模态轴从原点向上依次为 V、A、T；S 是学习投影视角，不是时间步。
- 长方体补齐六个面和十二条边；背面透出，隐藏网格使用浅色线。
- 蓝、紫、橙三条填色格子分别沿 S、M、D 轴，固定另外两个索引，并在空间上错开。每条仅一格厚，端面同色标明格子的厚度。
- 三组索引为 S：`m=2,d=6`；M：`s=1,d=0`；D：`m=0,s=5`，从 0 开始计数。三组格子互不相交。
- 图只展示张量及轴向向量；下半部分的映射、块顺序和池化流程已经移除。真实算子的宽度仅保留在 `figure_contract.json` 中供核对。
- D 轴实际为 256 维，图中用 8 格示意；S 为 6 格，M 为 3 格。图中没有实验数据或性能含义。

源代码依据：`vendor/meld/model.py` 与 `vendor/iemocap/factorized_mixer/model.py` 的 MixerBlock / FactorizedMixerEncoder。

从仓库根目录重新生成 SVG、PDF、PNG（会覆盖手工改过的 SVG）：

```bash
python paper/figures/amm_axis_mixing/build_figure.py
```

手工修改 SVG 后，仅重新导出 PDF 和 PNG：

```bash
python paper/figures/amm_axis_mixing/build_figure.py --export-only
```

导出使用现有 PyMuPDF，无新增依赖。PDF 保留文字及矢量路径。图的原生尺寸为 86 × 49.88 mm；论文以单栏宽度引用。修改模型含义或尺寸后，需同步核对 `figure_contract.json` 和图注；自动导出不会验证人工修改后的语义。
