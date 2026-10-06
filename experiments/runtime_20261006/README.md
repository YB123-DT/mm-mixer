# 九个外部基线的固定 1000 话语直接运行时间

所属服务器 biggpu；固定宿主机 GPU 2，UUID
`GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`。GPU 4 禁用，GPU 7 留给已授权训练。
同一物理卡是此前 MM-Mixer Full/AMM 替换计时使用的卡；最终比较还须核对环境与线程条件。
本目录不使用旧完整测试集时间乘比例伪造 1000 话语计时。

- 范围：DialogueRNN、Ada2I、MMGCN、MM-DFN、M3Net、SDT、CSS、ECERC、ConFilMER，IEMOCAP/MELD 共 18 项。
- 每个模型保留其原特征、测试对话顺序与保存权重。依次取完整对话，最后一个对话只保留凑足 **1000 个有效话语**所需的前缀，然后调用原 collate 重新生成 padding/mask。图由原模型基于裁剪后的输入构造。预提取特征本身的上游上下文不重算。
- 因各方法测试对话顺序/特征可能不同，不声称输入完全相同。部分非因果模型最后一个对话的未来上下文被截短；这是明确的固定规模工作负载，不用于重新报告分类性能。
- 每批 32 **个对话**；ConFilMER 使用原训练的 16 个对话（保存模型的 1000 超边参数限制，不改权重尺寸）。记录实际每批长度与 padding。不是 32 个话语一批。
- 输入先驻留 GPU，eval/inference_mode，循环所选输入做 20 批预热、10 次实际 1000 话语前向，每遍 CUDA 同步后计时，再同步结束。CPU 单线程。保留默认 forward 实际执行的辅助/诊断分支，排除特征提取、磁盘读取和 H2D。
- GPU 必须没有其他 compute 进程、显存低于 500 MiB、利用率不超过 2%，连续三次观察合格才启动。每项启动前后再检查竞争进程；出现竞争会记失败而非有效测量。非独占调度环境不能保证采样间绝无外部进程，因此保留观察证据。
- CSS 的旧 checkpoint 缺失，严格标为 **原配置初始化结构计时**；其他模型加载现有保存权重。MM-DFN/M3Net 直接反序列化完整保存模型，其他可用 state_dict 严格加载。

代码：`adapters.py` 仅复制已验证 FLOPs 适配器的加载部分；不修改任何模型源代码。
`measure.py` 保存源代码、权重、特征文件哈希、环境、1000 话语选择清单、逐批 mask 长度、真实重复耗时。
`run_queue.py` 加进程锁、原子状态文件、逐项日志及退出码，不覆盖已有结果，不自动重试同一失败。
CPU smoke 会核查完整 1000 条输入构造，再对首个完整对话执行原模型；smoke 不当作正式 GPU 耗时。

远程目录：`/data2/yb/multimodalERC/MM_Mixer_Runtime_20261006`。
环境：`/data2/yb/reproduction_envs/s0/bin/python`；临时目录 `/data2/yb/tmp/mmrt`。

```bash
# 持久队列；需要先同步本目录代码至远程 code/
ssh biggpu 'tmux new-session -d -s mm_runtime_1000_20261006 "/data2/yb/reproduction_envs/s0/bin/python /data2/yb/multimodalERC/MM_Mixer_Runtime_20261006/code/run_queue.py > /data2/yb/multimodalERC/MM_Mixer_Runtime_20261006/queue.log 2>&1"'
```

任务状态以远程 `state.json`、每项结果 JSON 和退出码共同判断。排队不等于完成。
