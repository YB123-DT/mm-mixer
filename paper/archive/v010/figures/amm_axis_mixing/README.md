# AMM 三轴交互图

当前 Figure 1 的源文件是 `01_amm_axis_mixing.svg`，矢量 PDF 为 `../../01_amm_axis_mixing.pdf`，PNG 为本目录预览。

- 长方体省略 batch，表示 `M × S × D = 3 × 6 × 256`。
- 模态轴从原点向上依次为 V、A、T；S 是学习投影视角，不是时间步。
- 三条彩色向量分别固定另外两个索引，展示 S-MLP、M-Linear、D-MLP。
- 映射在同一块的其他位置共享，两个块使用各自的参数，并按图示顺序执行。
- LayerNorm 均沿 D；归一化和残差连接未在图内展开。
- D 的网格数仅作示意，图中没有实验数据或性能含义。

源代码依据：`vendor/meld/model.py` 与 `vendor/iemocap/factorized_mixer/model.py` 的 MixerBlock / FactorizedMixerEncoder。

从仓库根目录重新生成 SVG、PDF、PNG（会覆盖手工改过的 SVG）：

```bash
python paper/figures/amm_axis_mixing/build_figure.py
```

手工修改 SVG 后，仅重新导出 PDF 和 PNG：

```bash
python paper/figures/amm_axis_mixing/build_figure.py --export-only
```

导出使用现有 PyMuPDF，无新增依赖。PDF 保留文字及矢量路径。图的原生尺寸为 86 × 80.84 mm；论文以单栏宽度引用。修改模型含义或尺寸后，需同步核对 `figure_contract.json` 和图注；自动导出不会验证人工修改后的语义。
