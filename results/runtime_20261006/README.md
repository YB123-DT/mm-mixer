# 外部基线 1000 话语直接计时：已排队，尚无正式结果

2026-10-06，9 个外部模型 × 两数据集，共 18 项正式计时已进入持久队列。
宿主机 `biggpu` GPU 2（UUID `GPU-a8bb25f8-e771-1975-ef10-fdc0679488a4`）当前被其他训练占用，因此状态为 `waiting_for_idle_gpu`，**0/18 正式计时完成**。

- tmux：`mm_runtime_1000_20261006`；启动 PID：`936168`。
- 远程根目录：`/data2/yb/multimodalERC/MM_Mixer_Runtime_20261006`。
- 实时状态：远程 `state.json`；本目录 `queue_snapshot.json` 只是交付时快照。
- 独立CPU预检：`smoke/`，检查严格1000目标/标签/原mask构造；首个完整对话的分类预测数量与标签对齐、所有返回张量finite。**不能当作GPU计时结果。**
- 正式前向每批再检查输出数量/finite；固定20批预热、10遍实际1000话语，记录median/mean/raw时间。

前一轮增加队列脚本时，名称 `queue.py` 遮蔽 Python 标准库，引起后5项导入失败。
已更名 `run_queue.py` 并重跑增强预检；失败证据留在远程 `smoke_v1/`，没有隐去或把失败算通过。

协议、上下文截断/输入差异、CSS缺权重、来源及复现命令详见
[实验说明](../../experiments/runtime_20261006/README.md)。不将原历史全测试集归一化速度冒充本次1000话语实测。

最终预检：**18/18通过**。ConFilMER两项初次因原入口要求非空CUDA_VISIBLE_DEVICES在导入前失败；按旧CPU FLOPs方式改为`-1`（不暴露GPU），归档旧日志后两项重跑均通过。`smoke/exit_codes.txt`同时保留失败与修复后退出码。每项1000目标/标签/mask构造、首个完整对话的分类数量及全部返回张量finite均核对通过。正式GPU计时仍**0/18，排队等待GPU2空闲**。
