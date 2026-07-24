# ACT 实验三：Temporal Ensemble

## 1. 实验要回答什么

Temporal Ensemble 在每个控制步骤重新预测一个 action chunk，并把多个历史 chunk 对当前时刻
的重叠预测做指数加权。

实验问题：

> 对同一个已经训练好的 ACT checkpoint，每帧重新规划并组合重叠预测，是否能减少动作抖动并
> 提高 noisy observation 下的稳定性？

A3 不重新训练。它只改变 checkpoint 的运行时推理配置。

> 当前状态：LeRobot 0.4.4 已经实现 ACT Temporal Ensemble，但当前本地
> `eval_libero.sh` 还不能覆盖 checkpoint 的推理参数。下面的命令需要先完成本指南第 3 节。

## 2. 必须使用 source-matched 对照

先选择一个 source checkpoint：

```text
A1 checkpoint
或者
A2 checkpoint
```

然后对同一个 checkpoint 做三次评测：

| 配置 | `n_action_steps` | `temporal_ensemble_coeff` |
|---|---:|---:|
| Source Standard | 5 | `None` |
| Source Replan control | 1 | `None` |
| A3 Ensemble | 1 | `0.01` |

为什么需要 Replan control：

- Standard 与 A3 同时改变了重新规划频率和 ensemble；
- Replan control 与 A3 都是每帧调用模型；
- 因此 A3 对比 Replan control 才能单独衡量 ensemble。

如果 A3 使用 A2 checkpoint，对照必须是同一个 A2 checkpoint 的 Replan control，不能拿
A1 Replan 与 A2 Ensemble 比较。

## 3. 需要实现的文件

```text
src/robot_practice/evaluation/common.py
src/robot_practice/evaluation/libero_act.py
scripts/experiments/act/eval_libero.sh
tests/test_evaluation_common.py
tests/test_libero_evaluation.py
```

不修改 LeRobot 的 `ACTTemporalEnsembler`。

### 3.1 增加运行时参数

在 `LiberoEvaluationConfig` 和 argparse 中增加：

```text
--n-action-steps=N
--temporal-ensemble-coeff=C
```

Python 字段使用可选值：

```python
n_action_steps: int | None = None
temporal_ensemble_coeff: float | None = None
```

两者都不传时，保持 checkpoint 原始行为。

### 3.2 在 policy 构造前覆盖 config

修改 `load_act_checkpoint()`，让它接收 inference overrides。处理顺序必须是：

1. 从 checkpoint 读取原始 config；
2. 保存原始 `n_action_steps` 和 `temporal_ensemble_coeff`；
3. 应用本次运行时 override；
4. 校验有效配置；
5. 使用有效配置构造 policy；
6. 加载原始模型权重和 processor；
7. 不写回 checkpoint。

必须在构造 policy 前覆盖，因为 `ACTPolicy.__init__()` 会根据
`temporal_ensemble_coeff` 决定是否创建 `ACTTemporalEnsembler`。

校验规则：

```text
1 <= n_action_steps <= chunk_size
temporal_ensemble_coeff 必须是有限数
启用 temporal ensemble 时 n_action_steps 必须等于 1
```

不要在用户只传 coefficient 时静默把 `n_action_steps` 改成 1。配置冲突应直接报错，并告诉
用户同时传：

```text
--n-action-steps=1
```

### 3.3 记录原始值和有效值

控制台至少打印：

```text
checkpoint n_action_steps
checkpoint temporal_ensemble_coeff
effective n_action_steps
effective temporal_ensemble_coeff
```

`run_config.json` 同时保存：

```json
{
  "policy": {
    "checkpoint_inference": {
      "n_action_steps": 5,
      "temporal_ensemble_coeff": null
    },
    "effective_inference": {
      "n_action_steps": 1,
      "temporal_ensemble_coeff": 0.01
    }
  }
}
```

现有 `ActionPlanTracker` 和 Rerun 已经能区分：

- normal action queue；
- discarded chunk tail；
- Temporal Ensemble 模式下每帧新预测的 chunk。

因此不需要伪造“未来执行轨迹”。

## 4. 最小测试

```bash
pytest -q \
  tests/test_evaluation_common.py \
  tests/test_libero_evaluation.py
```

必须覆盖：

- 不传 override 时有效值等于 checkpoint；
- `n_action_steps=1` 且 coefficient 为空时使用普通每帧重规划；
- `n_action_steps=1, coefficient=0.01` 时创建 ensembler；
- coefficient 非空但 `n_action_steps>1` 时拒绝；
- `n_action_steps>chunk_size` 时拒绝；
- checkpoint 中的 `config.json` 在评测前后字节不变；
- 每个 episode 调用 `policy.reset()`，ensemble 历史不会跨 episode；
- `run_config.json` 同时记录原始值和有效值；
- A1 与 A2 source checkpoint 都可以使用相同 override。

先做一帧 smoke test：

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="<source_policy>" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=1 \
  --max-steps=1 \
  --n-action-steps=1 \
  --temporal-ensemble-coeff=0.01 \
  --output-dir=outputs/act/a3_smoke
```

smoke test 只能验证代码可运行，不能计算正式成功率。

## 5. 选择 source checkpoint

默认先使用 A1：

```bash
source_policy="outputs/act/a1_chunking/<timestamp>/checkpoints/020000/pretrained_model"
source_name="a1"
```

如果 A2 已经明显优于 A1，也可以改成：

```bash
source_policy="outputs/act/a2_delta_loss/<timestamp>/checkpoints/020000/pretrained_model"
source_name="a2"
```

下面所有命令必须复用同一个 `source_policy`。

## 6. Clean closed-loop 对比

### 6.1 Source Standard

不传 override：

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${source_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir="outputs/act/a3_temporal_ensemble/${source_name}_standard_clean"
```

### 6.2 Source Replan control

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${source_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --n-action-steps=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir="outputs/act/a3_temporal_ensemble/${source_name}_replan_clean"
```

### 6.3 A3 Ensemble

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${source_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --n-action-steps=1 \
  --temporal-ensemble-coeff=0.01 \
  --perturbation=none \
  --seed=1000 \
  --output-dir="outputs/act/a3_temporal_ensemble/${source_name}_ensemble_clean"
```

正式评测不要传 `--max-steps`。

## 7. Noisy closed-loop 对比

三条命令保持相同 source、seed 和 init states，只把扰动改为：

```text
--perturbation=proprio_gaussian
--proprio-noise-std=0.01
```

Replan control：

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${source_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --n-action-steps=1 \
  --perturbation=proprio_gaussian \
  --proprio-noise-std=0.01 \
  --seed=1000 \
  --output-dir="outputs/act/a3_temporal_ensemble/${source_name}_replan_noisy"
```

A3 Ensemble：

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${source_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --n-action-steps=1 \
  --temporal-ensemble-coeff=0.01 \
  --perturbation=proprio_gaussian \
  --proprio-noise-std=0.01 \
  --seed=1000 \
  --output-dir="outputs/act/a3_temporal_ensemble/${source_name}_ensemble_noisy"
```

如需完整主表，再用相同扰动运行 Source Standard；判断 ensemble 本身时，优先比较 Replan 与
Ensemble。

## 8. 查看结果

```bash
rerun "outputs/act/a3_temporal_ensemble/${source_name}_replan_clean/rollout.rrd"
rerun "outputs/act/a3_temporal_ensemble/${source_name}_ensemble_clean/rollout.rrd"
```

重点比较：

| 配置 | Clean SR | Noisy SR | Action TV | Action Jerk | Gripper toggles | Model queries | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source Standard | | | | | | | |
| Source Replan | | | | | | | |
| A3 Ensemble | | | | | | | |

解释结果：

- Standard → Replan：每帧重新规划造成的变化；
- Replan → Ensemble：Temporal Ensemble 本身造成的变化；
- Standard → Ensemble：最终部署配置的总体变化。

支持 Temporal Ensemble 假设的 development 门槛：

- A3 clean success rate 相比 source-matched Replan 下降不超过 5 个百分点；
- action jerk 至少降低 15%，或者 noisy success rate 至少提高 5 个百分点；
- 同时报告增加的模型调用次数和 latency。

A3 每一步都调用模型，`num_model_queries` 明显增加是预期成本，不是 bug。Temporal Ensemble
只在 closed-loop 连续调用中生效，普通 offline chunk MAE 不会因为启用它而产生新的实验含义。

最后把 source checkpoint、有效推理配置、clean/noisy 指标、latency 和失败案例写入
`reports/experiment_log.md`。
