# 辅助损失权重敏感性（R2-4）

服务器固定 biggpu；只用宿主机 GPU 7（GPU-c38d9fe1-0b58-158f-a289-32d21e96df2e），GPU 4 禁用。代码从 Full commit `1b8b1fff90e19d793a99c0d0cf01c4bfd3cf51ab` 冻结，快照位于 `outputs/aux_weights_20261006/code`，远程 `/data2/yb/multimodalERC/MM_Mixer_AuxWeights_20261006/code`。主工作区模型代码不修改。

正式 15 次：IEMOCAP aux_half/aux_double/aux_equal 各 seeds 2025,2066,2118；MELD aux_half/aux_double 各 seeds 2025,2028,2069。Full/no_aux 复用既有实验，不重复训练。

| 数据集/配置 | main | T | A | V |
|---|---:|---:|---:|---:|
| I Full（参考） | .45 | .33 | .11 | .11 |
| I aux_half | .45 | .165 | .055 | .055 |
| I aux_double | .45 | .66 | .22 | .22 |
| I aux_equal | .45 | .55/3 | .55/3 | .55/3 |
| M Full（参考） | 1 | 1 | 1 | 1 |
| M aux_half | 1 | .5 | .5 | .5 |
| M aux_double | 1 | 2 | 2 | 2 |

不归一化。half/double 改变总目标尺度，不解释为恒定总尺度下的分配消融。aux_equal 保持 IEMOCAP 辅助总预算 .55。S=6、D=256、FFN=1536、blocks=2、R=32，批量和100/50轮训练、历史 peak-test-WF1 选择协议保持 Full；不作为独立验证集调参证据。

发现原 MELD loss 虽接收 aux 数值，但最终硬编码单位权重；隔离补丁让数值系数进入实际损失，权重为1时保持原加法分支。原 Full 模型初始化/输出及 Full 损失位一致；所有5配置验证生效配置、解析加权损失、有限非零梯度；模型前反向和优化器更新通过，证据见 model_verification.json、loss_iemocap.json、loss_meld.json。本地无所需python环境的尝试保留为 local_environment_unavailable.log，不是通过证据。

真实数据1轮 smoke：I aux_equal、M aux_double；由现有runner保存 checkpoint 并 fresh strict replay 检查。正式队列使用 launch_revision.py，2并发，至少6GB空闲，独立run目录，短TMPDIR=/data2/yb/tmp/mmauxw；每run配置/数据/源码哈希由原产物协议保存，快照 snapshot.json 提供额外全文件哈希。退出后自动 analyze_revision.py 汇总。

启动：`nohup bash /data2/yb/multimodalERC/MM_Mixer_AuxWeights_20261006/pipeline/run_pipeline.sh > /data2/yb/multimodalERC/MM_Mixer_AuxWeights_20261006/pipeline/queue.log 2>&1 < /dev/null &`

状态：准备、数值验证与两数据集真实 smoke 均通过（fresh_strict_replay_exact=true）。正式 15run 已在 2026-10-06 15:22 UTC 启动，持久 shell PID 921128；2并发，13排队，首批 IEMOCAP aux_half 的 seeds 2025/2066 均已输出第1/100轮训练和测试日志，尚无正式完成结果。状态与 PID 以 launch_verification.json 为准。
