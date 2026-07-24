# 2026-07-24 ACT A-Sanity 收尾与后续实验指南

## 实验目标

固定已完成的 A-Sanity 结果，明确下一轮 A0/A1/A2/A3 的实验边界、可复现入口和同源对照关系，避免后续实验把训练差异误归因为推理策略变化。

## 实验思路

- 把 A-Sanity 的离线与闭环结论与下一轮计划分离：先冻结已得事实，再把三组后续实验作为执行计划。
- 统一基线条件：任务、演示条数、帧数、batch size、更新步数、seed、VAE 设置保持一致。
- A1/A2/A3 的推理差异采用 source-matched 对照，避免跨 checkpoint 的混淆。
- A0/A1 采用单步/chunking 对照，A2 对比是否加入 delta-loss，A3 对比是否开启 temporal ensemble。
- 同一实验的离线与闭环都要回到同一份 `init_state_id` 选择策略，以保证公平性。

## 可直接执行命令

- 后续三组实验启动和评测的基础命令见对应指南：
  - [ACT 单步预测与 Action Chunking 指南](act_action_chunking.md)
  - [ACT 动作差分损失指南](act_delta_action_loss.md)
  - [ACT Temporal Ensemble 指南](act_temporal_ensemble.md)
- 对应脚本入口通常是：
  - `bash scripts/experiments/act/train_step0_xxx.sh ...`
  - `bash scripts/experiments/act/train_act.sh ...`
  - `bash scripts/experiments/act/eval_offline.sh ...`
  - `bash scripts/experiments/act/eval_libero.sh ...`

具体参数与路径请以对应指南中的“可直接执行命令”块为准，避免复用旧参数导致结果不可比。

## 结果落点

- 实验输出目录：`outputs/act/...`
- 离线指标：`outputs/.../offline_eval/metrics.json` 与 `outputs/.../offline_eval/run_config.json`
- 闭环指标：`outputs/.../closed_loop_eval/*/metrics.json` 与 `outputs/.../closed_loop_eval/*/eval_info.json`
- 全局索引与结果归并：`reports/experiment_log.md`
- 指标名解释：`reports/metrics.md`

## 判断标准

- 三类实验（A0/A1、A2、A3）均有对应的 Offline 与 Closed-loop 对照，并在 `reports/experiment_log.md` 更新。
- A0/A1、A2、A3 的比较采用 source-matched 对照，不允许把不同 source checkpoint 的结果混写为同一结论。
- 指标可复现：核心离线指标与闭环成功率、稳定性均能在 `metrics.json` 中定位且与指南中定义一致。
- 异常闭环 case 有明确归因，写入失败案例并给出可执行修复方向。

## 模式

- 本文件当前主要记录 `Offline` + `Closed-loop` 的收敛与交付状态，后续实验结果按实际评测类型在 `reports/experiment_log.md` 做分类登记。

## 重要进展

- A-Sanity 已完成五条 demonstration、859 帧的 teacher-forced offline 诊断，并完成
  20 次正式 LIBERO closed-loop rollout；20k checkpoint 的闭环成功率为 18/20（90%）。
- 20k checkpoint 的 first-action MAE 为 0.02076，delta-action MAE 为 0.00972；这些是
  离线动作预测诊断指标，不能代替模拟器任务成功率。
- 零更新 B0、完整训练 loss 和两次失败 rollout 的分类没有保留，因此 A-Sanity 不能给出
  严格的训练前后对照或完整失败分析。
- 后续 development 实验固定扩展到同一任务的 20 条 demonstrations、3233 帧，保持
  batch size 8、20,000 updates、seed 1000 和 `use_vae=false`。
- A0/A1 Action Chunking、A2 动作差分损失和 A3 Temporal Ensemble 的设计已完成独立审查，
  三份实施指南已经写好。
- A0/A1 已完成通用 step-0 initializer、清晰的统一训练入口、两个便捷别名和可复现的
  normalized proprioception noise；A2 policy 插件和 A3 inference override 仍待实现。

## 总结范围

本次总结收尾 A-Sanity，并为后续三个 ACT 小实验提供明确的实施入口。离线评测和 Rerun
可视化只作为支撑能力简要说明，详细内容继续以已有专项日志为准。

完整 A-Sanity 操作和结果见
[ACT A-Sanity 实验指南](act_a_sanity.md)。离线标定与可视化细节见
[2026-07-23 ACT 离线评测与标定可视化](2026-07-23-ACT离线评测与标定可视化.md)。

## A-Sanity 实测结果

A-Sanity 使用公开的 LIBERO Object task 8：
`pick up the chocolate pudding and place it in the basket`。

| 实验字段 | 配置或结果 |
|---|---|
| Task suite | `libero_object` |
| Dataset size | 5 demonstrations，859 frames |
| Observation | 两路 256×256 RGB + 8D state |
| Action space | 7D relative EEF action |
| Action horizon | `chunk_size=20`，`n_action_steps=5` |
| Control frequency | 10 Hz |
| Train loss | 未保留完整日志，无法补录 |
| First-action MAE | 0.02076 |
| Delta-action MAE | 0.00972 |
| Teacher-forced action jerk | 0.01146 |
| Closed-loop success rate | 18/20（90%） |
| Closed-loop action TV | 0.02195 |
| Closed-loop action jerk | 0.02210 |
| Model-query latency p50/p95 | 18.28/24.47 ms |
| Failure cases | 2 次失败，尚未逐帧分类 |

100-step checkpoint 的 first-action MAE 为 0.16989，delta-action MAE 为 0.02365，只能作为
近似的早期训练参考。它不是零更新 B0，不能据此声称获得了严格的 step-0 到 20k 配对结果。

在当前证据范围内，可以确认：

- 20k checkpoint 在五条训练 demonstration 上的 open-loop action error 已明显降低；
- 20 次正式 LIBERO rollout 中成功 18 次；
- teacher-forced offline 指标和 closed-loop success 回答不同问题，两者均需保留。

## 后续实验设计

后续三个实验继续使用同一个 LIBERO task。Development 数据固定为该任务按 episode ID
排序后的前 20 条 demonstrations，共 3233 帧。这样既能延续 A-Sanity，又不会直接把实验规模
扩展到全部 LIBERO tasks。

公共训练条件：

| 项目 | 固定值 |
|---|---|
| Dataset size | 20 demonstrations，3233 frames |
| Observation | 两路 256×256 RGB + 8D state |
| Action space | 7D relative EEF action |
| Control frequency | 10 Hz |
| Batch size | 8 |
| Updates | 20,000 |
| Seed | 1000 |
| VAE | false |

三组实验分别回答：

1. A0/A1：单步预测与 action chunking 相比，是否能在保持成功率的同时减少动作抖动；
2. A2：增加动作差分损失后，预测动作变化是否更接近 demonstration，闭环 jerk 是否下降；
3. A3：同一个 source checkpoint 使用 Temporal Ensemble 后，是否比每帧普通重规划更稳定。

A3 增加 source-matched Replan control。A3 若使用 A2 checkpoint，就必须和同一个 A2
checkpoint 的 `n_action_steps=1, ensemble=None` 对比，不能把 A1 与 A2 的训练差异错误归因
给 Temporal Ensemble。

完整设计见
[ACT 三组实验指南设计](../docs/superpowers/specs/2026-07-23-act-chunking-delta-ensemble-guides-design.md)。

## 已完成的实验指南

### A0/A1：单步预测与 Action Chunking

[ACT 单步预测与 Action Chunking 指南](act_action_chunking.md)说明：

- A0 使用 `chunk_size=1, n_action_steps=1`；
- A1 使用 `chunk_size=20, n_action_steps=5`；
- 两者固定相同数据、seed、训练步数和主要网络规模；
- 统一执行 offline、clean closed-loop 和 normalized proprioception noise 评测；
- 主要判定指标为 closed-loop action jerk 和 clean/noisy success rate。

### A2：动作差分损失

[ACT 动作差分损失指南](act_delta_action_loss.md)说明：

- 在仓库内实现 `act_delta` policy，不修改安装环境中的 LeRobot；
- 使用 `delta_loss_weight=0.1`；
- 只在相邻两个 action 都有效时计算 delta loss；
- A1 与 A2 从完全相同的 step-0 model tensors 开始；
- evaluator 需要同时识别 ACT family 和 `act_delta` concrete type。

### A3：Temporal Ensemble

[ACT Temporal Ensemble 指南](act_temporal_ensemble.md)说明：

- 不重新训练，加载 A1 或 A2 的已有 checkpoint；
- Standard 使用 checkpoint 原始 `n_action_steps=5`；
- Replan control 使用 `n_action_steps=1`、不启用 ensemble；
- A3 使用 `n_action_steps=1, temporal_ensemble_coeff=0.01`；
- A3 与 Replan control 必须使用同一个 source checkpoint 和相同 init states。

这三份文件是实施指南，不是已经完成的 20-demo 实验结果。A0/A1 启动与
`proprio_gaussian` 扰动已经实现；本地 A2 policy 插件和 A3 运行时 inference override
仍待实现。

## 离线评测与可视化基础

现有 offline evaluator 已能在 demonstration 上执行 teacher-forced action chunk prediction，
并在 Rerun 中同步展示两路 recorded images、Panda/EEF 3D 几何、GT/Pred action、逐维曲线、
metrics 和多个可切换 recording。这些能力可以用来观察过拟合程度、动作误差和预测 chunk。

离线 Pred action 没有在模拟器执行，future chunk 也不是未来真实机器人轨迹。任务是否成功、
执行动作是否抖动以及控制器实际响应，仍必须通过 LIBERO closed-loop rollout 判断。

## 当前限制

- A-Sanity 的零更新 B0 checkpoint 没有保留，100-step 只能作为近似参考；
- 训练终端日志没有保留，无法补录完整 train loss 曲线；
- 两次 closed-loop 失败尚未分类；
- 20-demonstration 的 A0、A1 和 A2 尚未训练；
- Temporal Ensemble 尚未完成本地 evaluator inference override；
- 三份指南定义的是接下来的实施步骤，不能提前写成实验结论。

## 下一步

1. 使用已经实现的 Action Chunking 启动脚本与 evaluator；
2. 使用固定 20 条 demonstrations 完成 A0/A1 的 20k 训练、offline 和 clean/noisy
   closed-loop 对比；
3. 按动作差分损失指南实现本地 `act_delta` policy，并从 A1 step-0 权重生成 A2 初始化；
4. 完成 A2 训练和同条件评测；
5. 为 closed-loop evaluator 增加推理参数覆盖，使用同源 checkpoint 完成 Replan 与
   Temporal Ensemble 对比；
6. 将每次正式实验的必填字段、failure cases 和结论继续写入
   [experiment_log.md](experiment_log.md)。
