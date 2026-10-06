# 三种子补充实验运行记录

状态（2026-10-06 复核）：**训练及预测结果核验全部完成，8 次效率测试也已完成并归档**。
144 个位置全部通过验证：119 次新训练、25 次历史结果复用；48 个数据集/配置组合均齐 3 种子。
完整数值和结论见 [三种子结果](results/revision_20261003/README.md)。

## 固定版本与协议

- 服务器：`ssh biggpu`。
- 训练代码提交：`1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab`。
- 冻结代码目录：`/data2/yb/multimodalERC/MM_Mixer_Revision_20261003/code/revision_1b8b1ff`。
- 启动前核验 `snapshot.json` 中全部 111 个文件哈希；后续工作区修改不进入正在运行的任务。
- Python：`/data2/yb/reproduction_envs/s0/bin/python`；Python 3.10.20、torch 2.2.2+cu121。
- IEMOCAP seeds：`2025, 2066, 2118`；MELD seeds：`2025, 2028, 2069`。
- 使用 `strict_peak_test_wf1`；batch size 32，最大 epoch 分别 100/50，保留原早停和数据集各自损失配置。
- 每次运行记录生效配置、源码和数据哈希、环境、GPU UUID、命令、PID、日志；完成须通过产物与严格重载验证。

## 复用与补跑

完整矩阵为 24 种配置 × 2 个数据集 × 3 种子，共 144 个位置。

| 阶段 | 新训练 | 旧结果复用 | 说明 |
|---|---:|---:|---|
| Phase 1 | 95 | 0 | 53 个基础消融缺失/失效/无法核实位置，加 42 个新增对照 |
| Phase 2 | 24 | 25 | 对旧候选再次执行正式协议和完整产物验证；不通过者重跑 |
| 总计 | 119 | 25 | 144 个位置；Full 与 T+A+V 只计一次 |

历史语义审计曾接受 49 个候选；启动时严格检查仅 25 个通过。
另外 21 个缺匹配的历史源码，3 个 MELD Full 虽找到旧字节存档，但原路径的源码哈希不匹配，
因此全部保留旧结果并新建运行，不修改旧 manifest 或放宽验证。
旧 MELD Feature Gating 消融中开关未生效的记录也已排除。

复用计划逐条保存原 bundle 路径和历史 `config_contract_sha256`，仍须满足 100/50 的正式配置预算，
以及 test-peak、特征/源码/产物哈希和预测指标检查。单轮试跑不纳入正式结果。

## GPU 与队列

使用宿主 GPU **0、1、2、3**，通过 UUID 显式绑定；**GPU 4 禁用**。
先以每卡 2 个任务启动，再升到 3 个；三任务时每卡占用不足 2 GiB、采样利用率约 14–50%，
主机有 80 个 CPU 核心和充足内存，随后设为每卡最多 4 个任务。
调整仅重启调度器，原训练进程被接管，没有重启训练或修改 batch size。

实验根目录：`/data2/yb/multimodalERC/MM_Mixer_Revision_20261003`。

| 内容 | 根目录下相对路径 |
|---|---|
| Phase 1 计划 | `plans/phase1_95_jobs.json` |
| Phase 1 状态 | `queue/phase1/status.json` |
| Phase 1 日志 | `queue/phase1/logs/` |
| 启动命令与并发调整记录 | `queue/phase1/launch_record.json` |
| Phase 2 计划 | `plans/phase2_49_jobs.json` |
| Phase 2 状态 | `queue/phase2/status.json` |
| 单次训练产物 | `runs/revision_1b8b1ff/{dataset}/{variant}/seed{seed}/` |
| 后续流程状态 | `pipeline/pipeline_status.json` |
| 全部结果汇总 | `pipeline/analysis.json`、`pipeline/analysis.md`（已生成并核验） |
| 效率结果 | `pipeline/efficiency/`、`pipeline/efficiency_summary.json`（8/8 已完成，本地已归档） |

Phase 1 的持久会话名为 `mm_mixer_revision_phase1`。
后续控制器在 Phase 1 结束后运行 Phase 2，合并全部 144 个位置并重查预测/指标，
再对两个数据集的 Full、MLP、Attention、CubeMLP-style（seed 2025）测量效率。
八次效率测量固定在同一张空闲健康 GPU；GPU 忙时等待，不与训练共卡计时。
失败和缺失项保留；调度器异常退出时标记需要处理，不盲目重新训练。

Phase 1 于 19:39 UTC 完成 95/95；Phase 2 于 22:03 UTC 完成 49/49。
合并分析于 22:03:36 UTC 正常结束，`analysis.json` 的 `complete` 为 `true`。
本地独立重算全部组的 WF1、ACC、Macro-F1 均值、样本标准差及同种子差值，与报告一致。
效率控制器已于 2026-10-04 20:44:33 UTC 进入 `completed`，8 个效率命令均 returncode=0。
使用宿主 GPU 2（UUID `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`，V100-SXM2-32GB），
八个命令顺序执行，起止时间不重叠；每项三次进程快照检查均无其他 GPU compute 进程。
FP32、eval/no_grad，前 1000 个测试目标、batch 32、20 batch 预热、10 次 sweep；
报告中位耗时而非三种子计时均值。输入预先驻留 GPU，排除特征提取、加载与 H2D。
详细数值、显存定义、源码匹配边界及哈希核验见
[效率结果](results/revision_20261003/efficiency/README.md)。
统一训练成本和其他完整 baseline 的相同 1000 话语计时仍未补齐。

## 本地审计位置

本地 `outputs/revision_20261003/` 保存 `reuse_audit.{json,md}`、
`reuse_strict_gate.{json,md}`、阶段计划、启动记录和单轮流程检查证据。
训练中间结果留在服务器并按需同步；checkpoint、数据和日志不进入 Git。
最终分类指标的 JSON 和 Markdown 已按服务器原始字节保存到 `results/revision_20261003/`，
对应 SHA-256 与冻结训练版本见该目录的 `provenance.json`。
8 项原始效率 JSON、服务器汇总和控制器完成快照同步至 `results/revision_20261003/efficiency/`；
其 `provenance.json` 保存服务器来源、SHA-256 和本次独立核验。论文是否收录各项以当前 Main.tex 为准。
