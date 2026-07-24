# 指标说明（reports/metrics.md）

> 规则来源：`src/robot_practice/evaluation/metrics.py`。

## 说明

- 指标文件请优先看 `metrics.json`（机器可读）；本文件用于人类阅读。
- 离线与闭环的指标定义与字段不同，不要跨场景直接比较。

## Offline 指标（ACT teacher-forced）

- `first_action_mae`: 演示首动作与预测首动作的平均绝对误差。
- `per_dimension_mae`: 首动作每维 MAE。
- `chunk_mae_by_horizon`: 预测 chunk 每个 horizon 的 MAE。
- `delta_action_mae`: 邻时刻动作差分 MAE（按有效帧统计）。
- `teacher_forced_action_tv`: 连续两步执行命令（前 6 维机械臂）差分绝对值平均值。
- `teacher_forced_action_jerk`: 连续两阶差分（二阶差分）绝对值平均值（proxy）。
- `inference_latency_p50_ms` / `p95_ms`: 离线模型推理延迟分位数。

## Closed-loop 指标（LIBERO 闭环）

- `action_tv`: 执行动作（前 6 维）相邻步 L1 平均，越低越平滑。
- `action_jerk`: 执行动作（前 6 维）二阶差分绝对值平均，越低抖动越少。
- `mean_gripper_toggles`: 每个 rollout 中执行动作第 7 维从负数变为非负数或反向变化的
  次数，再对 episode 取平均。当前实现使用零阈值且没有 deadband/持续帧去抖，衡量的是
  夹爪命令符号切换，不等于 measured gripper 的物理完整开合次数。
- `num_successes` / `success_rate`: 成功率（`init_state_id` 对齐的正式闭环 run）。
- `avg_sum_reward`, `avg_max_reward`: 奖励聚合。
- `policy_step_latency_p50_ms` / `p95_ms`: 控制步耗时分位数。
- `model_query_latency_p50_ms` / `p95_ms`: 网络推理耗时分位数。

## 相关字段

- `run_config.json`: 命令与配置快照（参数来源与有效参数）。
- `metrics.json`: 结构化实验指标（离线 / 闭环）。
- `eval_info.json`: 逐 episode 的简表（闭环）。
- `rollouts.parquet` / `predictions.parquet`: 每步原始执行序列（闭环 / 离线）。
