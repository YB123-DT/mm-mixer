# 系统盘满后的恢复

状态检查发现两个 ConFilMER IEMOCAP 任务停在 48/80、45/80 轮，日志为
`OSError: [Errno 28] No space left on device: /tmp/pymp-*`。进程仍存活但没有继续训练。
`df` 确认系统盘 100%，/data2 尚有约 249 GB；不是 inode 耗尽。

- 保留全部旧日志与最佳模型，仅终止本队列控制器和两个卡死任务及其子进程。
- 原 checkpoint 是最佳轮次模型与优化器，不含完整 RNG/数据加载状态，因此重新以
  原种子开始两个独立运行，输出后缀 `_restart_tmpfix`；不称为断点续训。
- ECERC IEMOCAP seeds 2025、2066 已完成，重载预测验证通过，直接复用。
- 每个新任务设置 `/data2/yb/tmp/mmbl/<输出路径哈希前12位>` 下独立
  `TMPDIR`/`TEMP`/`TMP`；写入测试通过。首次使用完整实验目录作为临时路径过长，
  触发 `AF_UNIX path too long`，该失败尝试也保留，改用短路径后再次启动。
  两次恢复后的输出后缀为 `_restart_tmpfix_shorttmp`。
- 队列在大任务显存暂不足时继续检查后续小任务，避免队头阻塞空闲槽位。
- 模型、数据、种子、训练超参数和预算均未改变。GPU 白名单仍为 3、5，4 禁用。

新的执行入口是 `run_recovery.sh`；远端代码为 `code/queue_recovery.py`。
计划为 `plans/recovery_12_runs.json`（本地 `recovery_plan.json`），状态为
`queue/recovery_state.json`，日志为 `queue/recovery_controller.log`。
tmux 会话为 `mm_mixer_baselines_20261006_recovery`。
原 `queue/state.json` 保留为 `interrupted_disk_full`，不是当前进度。

仍为原定 12 个种子位置：2 个已完成结果复用，其余 10 个继续执行。失败尝试单独保留。
最终汇总路径仍为远端根目录 `summary.json`。
