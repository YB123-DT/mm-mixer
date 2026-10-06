# 2025 baseline reproduction — 2026-10-06

用户要求：优先增加近期 baseline，复现三个种子查看分数。本批选定两篇 2025 年论文，
两个数据集各三个种子，共 **12 次正式训练**。本目录不修改论文 TeX 或现有消融代码。

## 选定方法

| 方法 | 出处 | 官方代码 | Epoch 上限 I/M | Batch（对话）I/M |
|---|---|---|---:|---:|
| ECERC | ACL 2025 | https://github.com/TAN-OpenLab/ECERC | 200 / 40 | 64 / 32 |
| ConFilMER | ICASSP 2025 | https://github.com/G22-web/ConFilMER | 80 / 15 | 16 / 16 |

- IEMOCAP seeds：2025、2066、2118；MELD seeds：2025、2028、2069。
- 保留原作者模型、特征与默认训练配置；环境兼容修改和指标导出详见各 `AUDIT.md`。
- 沿用用户既定 `strict_peak_test_wf1`，是测试集峰值，不是验证集选轮后的独立测试结果。
  原代码按两位小数的 WF1 比较 checkpoint；记录该规则，最终指标从预测以全精度重算。
  ECERC 另保存原验证集 WF1 选择对应的结果；不开展超参数搜索、不择优报告种子。
- 三个种子全部完成且预测指标核验通过后，才生成该组均值与样本标准差（ddof=1）。

## 比较边界

这是 **原系统／发布代码复现**，不是与 MM-Mixer 相同特征、相同上下文的消融对照。

1. ECERC 使用独立的 emotion/semantic 两路文本特征。四个官方预处理文件已经下载，
   哈希见 `ecerc/data_manifest.json`。IEMOCAP 保留官方训练内部 10% 对话 dev；MELD
   为 9989/1109/2610 条 train/dev/test。原模型全空 mask 行会泄漏未来信息；MELD
   上游音视频命名与切分顺序不一致。保留并明确记录，不将其标作严格 history-only。
2. ConFilMER 使用四层 RoBERTa 特征，IEMOCAP train/test=5810/1623；MELD
   train/test=11098/2610（训练合并通常的 train/dev）。双向 GRU 和整段对话图使用未来
   上下文，跨 batch 节点的相似度还使输出依赖批次组成。数据哈希见
   `confilmer/data_manifest.json`。
3. 不把本批结果直接混入标题声称“统一 history-only”的主表，也不把原版代码错误修复
   后的新变体冒充原方法。后续文章应标明原设置和训练数据差异。

## 可追踪运行

- 服务器：`ssh biggpu`。
- 根目录：`/data2/yb/multimodalERC/MM_Mixer_Baselines_20261006`。
- Python：`/data2/yb/reproduction_envs/s0/bin/python`；复用现有环境，没有安装新依赖。
- 持久会话：`mm_mixer_baselines_20261006`。
- GPU 白名单：物理 3 / `GPU-cab071a3-de66-5a82-35d8-9f8b5b731e7a`；
  物理 5 / `GPU-fa1e8bfd-85d8-9599-f804-7c88b9c71b62`。物理 GPU 4 禁用。
- 每 GPU 最多两个本批任务。ConFilMER 启动要求至少 24000 MiB 空闲，ECERC 6000 MiB；
  每次启动间隔 20 秒重新查看显存，不改变 batch size。
- `plan.json` 保存全部命令、输出目录、每个源文件的哈希；远端为 `plans/12_runs.json`。
  `environment.json` 保存环境和设备信息。源码在专用远端目录冻结，不引用主工作区修改。
- 状态：远端 `queue/state.json`；控制器日志：`queue/controller.log`。
- 单次运行：`runs/{model}_{dataset}_seed{seed}/`，含 `launch.json`、`console.log`、
  配置、checkpoint、逐轮指标、`predictions.npz` 和 `result.json`。
- 自动汇总：队列结束后 `summarize.py` 检查样本量、身份及预测重算指标，输出根目录
  `summary.json`。失败与缺失保留，不自动重试或覆盖产物。
- `run_pipeline.sh` 是持久会话入口。不要在该会话运行期间重复启动；中断控制器后先
  检查 PID，队列会拒绝自动重跑标记为 running 的任务。

## 启动前验证

两方法、两数据集均通过真实数据单批训练/反向/测试检查。ECERC 两数据集及最终版
ConFilMER MELD 通过 checkpoint 严格重载预测一致性。ConFilMER 的 PyG 超图兼容
改动通过稠密参照测试（最大误差 2.98e-8）。正式每次训练结束仍会重载核验。
这些单批分数不作为正式实验结果。

调度器测试覆盖坏卡拒绝、GPU UUID 核对、冻结源码修改拒绝、种子身份核对和 smoke
结果不能充当正式训练。运行：

```bash
python -m unittest discover -s experiments/baselines_20261006 -p test_queue.py -v
```

## 其他候选

GS-MCC（AAAI 2025）、HRG-SSA（IJCAI 2025）、MFCRE（2025）和 HAUCL（2024）
已做源码筛查；分别存在实现/接口/配置/数据分支问题，未正式训练。证据见各目录
`AUDIT.md`。DRKF（ACM MM 2025）发布默认是四分类 IEMOCAP 且需要增强原始音频，
当前没有即用 MELD 路径，因此未纳入这批六分类/七分类复现。

本批训练正在运行时，文档只报告状态；正式均值和标准差以完整汇总为准。
