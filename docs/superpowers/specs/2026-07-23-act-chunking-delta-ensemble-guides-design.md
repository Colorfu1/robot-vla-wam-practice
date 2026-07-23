# ACT Chunking、动作差分损失与 Temporal Ensemble 实验指南设计

## 1. 目标

为已经完成 A-Sanity 的 ACT 实验补充三份可以直接照着实施的中文指南：

1. A0 单步预测与 A1 Action Chunking 对比；
2. A2 在 A1 上增加动作差分损失；
3. A3 使用已有 A1 或 A2 checkpoint 做 Temporal Ensemble。

指南只描述实验思路、需要实现的本地文件、测试、训练命令、评测命令和结果判定，不修改已安装
的 LeRobot 源码，也不把生成数据、checkpoint 或评测产物纳入版本控制。

## 2. 文档产物

新增三份报告：

```text
reports/act_action_chunking.md
reports/act_delta_action_loss.md
reports/act_temporal_ensemble.md
```

每份指南统一包含：

1. 实验问题和假设；
2. 固定变量与唯一变量；
3. 需要新增或修改的本地文件；
4. 分步骤实现方法和关键伪代码；
5. 最小测试；
6. 可复制的训练和评测命令；
7. 指标说明、结果表和通过标准；
8. 常见失败检查。

## 3. 固定数据与训练条件

三个实验使用 A-Sanity 的同一个 LIBERO 单任务：

| 字段 | 值 |
|---|---|
| Suite | `libero_object` |
| Simulator task ID | `8` |
| Dataset task index | `29` |
| Language | `pick up the chocolate pudding and place it in the basket` |
| LeRobot dataset | `lerobot/libero` |
| Local root | `datasets/lerobot/lerobot/libero` |
| Observation | 两路 256×256 RGB + 8D state |
| Action | 7D relative EEF action |
| Control frequency | 10 Hz |

Development 阶段固定使用该任务按 episode ID 排序后的前 20 条 demonstrations：

```text
823, 826, 834, 840, 841,
845, 847, 869, 875, 878,
910, 918, 920, 929, 944,
948, 949, 953, 966, 972
```

验证流程稳定后，final 阶段再扩展到该任务的前 50 条 demonstrations。

训练公共配置：

| 字段 | 值 |
|---|---|
| Batch size | `8` |
| Updates | `20000` |
| Seed | `1000` |
| `use_vae` | `false` |
| WandB | 默认关闭，可显式开启 |

视觉骨干、Transformer 规模、数据处理和其余优化参数保持一致。

## 4. A0 与 A1

唯一实验变量：

| 配置 | `chunk_size` | `n_action_steps` |
|---|---:|---:|
| A0 single-step | 1 | 1 |
| A1 chunking | 20 | 5 |

指南建议未来实现：

```text
scripts/experiments/act/a0_single_step.sh
scripts/experiments/act/a1_chunking.sh
```

两个启动脚本沿用 `a_sanity.sh` 的接口和安全检查，支持：

```text
--steps=N
--policy-path=PATH
--print-command
```

`chunk_size` 会改变 query embedding 数量，因此 A0 与 A1 不可能共享所有参数完全相同的
step-0 state dict。公平性要求是相同 seed、相同骨干和 Transformer 规模、分别保存 step-0
checkpoint，并记录两者参数量。

## 5. A2 本地策略插件

A2 不修改 Conda 环境中的 LeRobot 文件。指南描述在仓库内新增：

```text
src/robot_practice/policies/__init__.py
src/robot_practice/policies/act_delta/__init__.py
src/robot_practice/policies/act_delta/configuration_act_delta.py
src/robot_practice/policies/act_delta/modeling_act_delta.py
src/robot_practice/experiments/act_delta_checkpoint.py
scripts/experiments/act/a2_delta_loss.sh
```

`ACTDeltaConfig` 继承 `ACTConfig`，注册策略类型 `act_delta`，增加非负配置字段：

```python
delta_loss_weight: float = 0.0
```

正式 A2 实验传入 `delta_loss_weight=0.1`。默认值为零，使本地策略在未开启附加损失时保持
原始 ACT 语义。

`ACTDeltaPolicy` 继承 `ACTPolicy`，保持原始 ACT L1/KL 计算不变，并在一次 model forward
得到的 `actions_hat` 上增加：

\[
\Delta \hat a_t=\hat a_{t+1}-\hat a_t
\]

\[
\Delta a_t=a_{t+1}-a_t
\]

\[
L_{\Delta a}
=
\frac{
\sum_{t\in valid}
\left\|\Delta\hat a_t-\Delta a_t\right\|_1
}{
\#valid
}
\]

\[
L=L_{ACT}+\beta L_{\Delta a},\qquad \beta=0.1
\]

关键 padding 逻辑：

```python
valid = ~batch["action_is_pad"]
delta_pred = actions_hat[:, 1:] - actions_hat[:, :-1]
delta_target = batch[ACTION][:, 1:] - batch[ACTION][:, :-1]
delta_valid = valid[:, 1:] & valid[:, :-1]
delta_error = (delta_pred - delta_target).abs()
delta_loss = (
    delta_error * delta_valid.unsqueeze(-1)
).sum() / delta_valid.sum().clamp_min(1)
```

分母是有效相邻动作对数量，七个动作维度按向量 L1 norm 求和。只有相邻两步都有效时才计入。
没有有效相邻动作或 `chunk_size=1` 时，附加损失为有限的零。

`loss_dict` 增加：

```text
delta_action_loss
delta_loss_weight
loss
```

训练命令使用 LeRobot 的第三方策略发现参数加载本地包，不复制上游 ACT 实现。

### 5.1 A1/A2 相同初始化

A1 和 A2 网络张量结构相同。为了严格共享初始化：

1. 创建 A1 step-0 checkpoint；
2. 本地 checkpoint 准备工具创建 A2 初始化目录；
3. 只把 policy 类型改为 `act_delta` 并加入 `delta_loss_weight`；
4. 原样复用模型权重和 normalization processor；
5. 逐张量确认 A1/A2 state dict 完全相等；
6. 不覆盖或修改原始 A1 checkpoint。

A2 checkpoint 类型为 `act_delta`。现有 evaluator 未来需要在反序列化配置前导入本地插件，
接受 `ACTConfig` 子类，并继续复用当前 processor、指标和可视化代码。

## 6. A3 Temporal Ensemble

A3 不训练新模型。对同一个 A1 或 A2 checkpoint 做三组闭环评测：

| 配置 | `n_action_steps` | `temporal_ensemble_coeff` | 作用 |
|---|---:|---:|---|
| Standard | 5 | `None` | checkpoint 原始执行方式 |
| Replan control | 1 | `None` | 控制每步重新规划的影响 |
| A3 Ensemble | 1 | `0.01` | 测量 Temporal Ensemble |

Replan control 是必要对照。A3 相对它只改变 ensemble，可以避免把每步重新规划带来的变化错误
归因于 Temporal Ensemble。

指南描述为现有闭环评测增加两个可选参数：

```text
--n-action-steps=N
--temporal-ensemble-coeff=C
```

涉及：

```text
src/robot_practice/evaluation/common.py
src/robot_practice/evaluation/libero_act.py
scripts/experiments/act/eval_libero.sh
```

规则：

- 两个参数默认不提供，保持 checkpoint 行为；
- 只覆盖本次运行的内存 config，不修改 checkpoint；
- 覆盖必须在 policy 构造前完成；
- `n_action_steps` 必须在 `[1, chunk_size]`；
- 使用 Temporal Ensemble 时 `n_action_steps` 必须为 1；
- 控制台和 `run_config.json` 同时记录 checkpoint 值与最终生效值。

Temporal Ensemble 是连续模型调用之间的重叠 chunk 聚合，只在 closed-loop 中有意义。普通
offline chunk prediction 不执行跨帧聚合，因此 A3 不重复计算没有新增含义的 offline MAE。

## 7. Noisy Evaluation

现有 `--perturbation` 默认保持 `none`，并新增一种固定开发扰动：

```text
--perturbation=proprio_gaussian
--proprio-noise-std=0.01
```

噪声只加入已经通过 checkpoint preprocessor 归一化的 8D `observation.state`，不改变模拟器
真实状态、图像、`env.step()`、成功判定或 Rerun 中的实测机器人。

要求：

- 每个 episode 使用由实验 seed 派生的独立确定性噪声源；
- 相同参数重复执行得到相同噪声序列；
- `run_config.json` 和 `metrics.json` 记录模式、标准差和噪声 seed；
- Rerun 区分真实 state 与模型收到的 noisy normalized state。

## 8. 评测矩阵

Offline 使用全部 20 条训练 demonstrations。Closed-loop 使用相同任务、`seed=1000` 和相同
20 个 `init_state_id`，不使用缩短的 `--max-steps`。

最小闭环矩阵：

| 配置 | Clean | Proprio noise 0.01 |
|---|---|---|
| A0 single-step | 20 rollouts | 20 rollouts |
| A1 chunking | 20 rollouts | 20 rollouts |
| A1 replan control | 20 rollouts | 20 rollouts |
| A2 delta loss | 20 rollouts | 20 rollouts |
| A3 ensemble | 20 rollouts | 20 rollouts |

统一记录：

- train loss；
- first-action MAE；
- delta-action MAE；
- clean/noisy success rate；
- action TV；
- action jerk；
- gripper toggles；
- model query 数量；
- policy/model-query p50 和 p95 latency；
- failure cases。

闭环 TV 和 jerk 只计算执行 action 的前六个机械臂维度。TV 是平均绝对一阶差分；jerk 是平均
绝对二阶差分，是固定 10 Hz 下的抖动代理，不是带物理单位的真实 jerk。夹爪用 toggles 单独
衡量。

## 9. 判定标准

### A0 到 A1

- clean success rate 下降不超过 5 个百分点；
- closed-loop action jerk 至少降低 15%；
- noisy success rate 不低于 A0。

### A1 到 A2

- clean success rate 下降不超过 5 个百分点；
- closed-loop action jerk 至少降低 15%；
- offline `delta_action_mae` 低于 A1。

### A1 Replan 到 A3

- clean success rate 下降不超过 5 个百分点；
- action jerk 至少降低 15%，或者 noisy success rate 至少提高 5 个百分点；
- 同时报告额外模型调用和延迟成本。

未达到标准时，报告“当前结果不支持假设”，不选择性省略不利指标。20 个 rollouts 属于
development 规模；结果接近门槛时需要增加 seeds 或 rollout 数量。

## 10. 测试与验收

指南要求未来实现至少覆盖：

- A0/A1 启动脚本参数、固定数据和 dry-run 命令；
- `delta_loss_weight=0` 与原始 ACT loss 一致；
- delta target、padding、空有效对和 backward；
- A1/A2 初始化 state dict 完全相同；
- `act_delta` checkpoint 保存后可被 offline/closed-loop evaluator 加载；
- evaluation override 不修改 checkpoint；
- Temporal Ensemble 的配置约束；
- `none` 扰动保持当前结果路径；
- proprio noise 可复现且只改变 policy input；
- `run_config.json`、`metrics.json` 和命令行日志包含有效配置。

三份指南完成后做文档级检查：

- 所有仓库路径真实或明确标为“待新增”；
- 参数名称与设计一致；
- 训练和评测命令从仓库根目录执行；
- 不包含本机绝对路径、session ID、访问凭证或生成产物；
- 所有实验结果最终按 `AGENTS.md` 的必填字段写入
  `reports/experiment_log.md`。
