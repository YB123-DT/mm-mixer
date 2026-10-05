# AMM 三轴交互图

当前 Figure 1 的源文件是 `01_amm_axis_mixing.svg`，矢量 PDF 为 `../../01_amm_axis_mixing.pdf`，PNG 为本目录预览。

- 长方体省略 batch，表示 `M × S × D = 3 × 6 × 256`。
- 模态轴从原点向上依次为 V、A、T；S 是学习投影视角，不是时间步。
- 长方体显示顶面、前面、右侧面的网格；十二条边中九条可见边为实线、三条隐藏边为细灰虚线。取消内部/背面穿透网格。
- 蓝、紫、橙三条填色格子分别沿 S、M、D 轴，位于顶面、右侧面、前面，固定另外两个索引，并在空间上错开。每条仅一格厚，不再跨面添加端帽。
- 三组索引为 S：`m=2,d=4`；M：`s=1,d=7`；D：`m=0,s=5`，从 0 开始计数。三组格子互不相交。
- 三轴共用从后下左角 O 出发的三条隐藏边，在轮廓外衔接短实线箭头。调整深度投影，使 O 与前面竖网格错开。只保留轴端尺寸和 T/A/V，删除顶部标题及重复 S/M/D 字母，统一使用无衬线字体。
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

导出使用现有 PyMuPDF，无新增依赖。PDF 保留文字及矢量路径。由于现有 SVG 导出器不支持 `stroke-dasharray`，隐藏边由显式断段路径绘制，保证 PDF 中也呈现虚线。图的原生尺寸为 86 × 46.44 mm；论文以单栏宽度引用。修改模型含义或尺寸后，需同步核对 `figure_contract.json` 和图注；自动导出不会验证人工修改后的语义。
