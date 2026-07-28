# ACT 实验三：Temporal Ensemble

## 实验目标

- 在固定训练好的 source checkpoint 上，验证 Temporal Ensemble（`--temporal-ensemble-coeff`）是否比
  `--n-action-steps=1` 的重规划基线更能降低动作抖动并提升 noisy 条件下稳定性。

## 实验思路

三组对照必须共用同一 source checkpoint：

1. Source Standard（默认 `n_action_steps=5`）；
2. Source Replan（`n_action_steps=1`）；
3. A3 Ensemble（`n_action_steps=1` + `temporal_ensemble_coeff=0.01`）。

若目标是 isolation，对比优先 `Source Replan` 与 `A3 Ensemble`。

## 可直接执行命令

```bash
cd "$(git rev-parse --show-toplevel)"
conda activate robot-practice
export PYTHONNOUSERSITE=1

bash scripts/experiments/act/eval_libero.sh \
  --policy-path=outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model \
  --suite=libero_object --task-id=8 --n-episodes=20 --num-visualized-episodes=1 \
  --perturbation=none --seed=1000 --output-dir=outputs/act/a3_temporal_ensemble/a1_standard_clean

bash scripts/experiments/act/eval_libero.sh \
  --policy-path=outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model \
  --suite=libero_object --task-id=8 --n-episodes=20 --num-visualized-episodes=1 \
  --n-action-steps=1 --perturbation=none --seed=1000 \
  --output-dir=outputs/act/a3_temporal_ensemble/a1_replan_clean

bash scripts/experiments/act/eval_libero.sh \
  --policy-path=outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model \
  --suite=libero_object --task-id=8 --n-episodes=20 --num-visualized-episodes=1 \
  --n-action-steps=1 --temporal-ensemble-coeff=0.01 --perturbation=none --seed=1000 \
  --output-dir=outputs/act/a3_temporal_ensemble/a1_ensemble_clean
```

## 结果落点

- `outputs/act/a3_temporal_ensemble/<a1|a2>_standard_clean/metrics.json`
- `outputs/act/a3_temporal_ensemble/<a1|a2>_replan_clean/metrics.json`
- `outputs/act/a3_temporal_ensemble/<a1|a2>_ensemble_clean/metrics.json`
- 每组额外保留 `rollouts.parquet` 与 `rollout.rrd`（若开启可视化）。

## 判断标准

- 在同一 source checkpoint 下对比 `A3 Ensemble` 与 `Source Replan` 的 `success_rate`；
- 优先关注 `action_jerk`、`action_tv` 与 `gripper_toggles` 的方向性改善；
- 所有正式对比为 full `n-episodes=20`；`max-steps` 只用于 smoke。

## 模式

- Closed-loop（基础）
- Smoke（单步/快速可运行性）

## 1. 实验要回答什么

Temporal Ensemble 在每个控制步骤重新预测一个 action chunk，并把多个历史 chunk 对当前时刻
的重叠预测做指数加权。

实验问题：

> 对同一个已经训练好的 ACT checkpoint，每帧重新规划并组合重叠预测，是否能减少动作抖动并
> 提高 noisy observation 下的稳定性？

A3 不重新训练。它只改变 checkpoint 的运行时推理配置。

> 当前状态：`eval_libero.sh` 已支持评测专用的 `--n-action-steps` 和
> `--temporal-ensemble-coeff` override，Source Replan 与 A3 Ensemble 均可直接运行。

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

## 3. Temporal Ensemble 运行时接口

```text
src/robot_practice/evaluation/common.py
src/robot_practice/evaluation/libero_act.py
scripts/experiments/act/eval_libero.sh
tests/test_evaluation_common.py
tests/test_evaluation_interfaces.py
```

不修改 LeRobot 的 `ACTTemporalEnsembler`。

### 3.1 运行时参数状态

当前支持：

```text
--n-action-steps=N
```

它只覆盖本次评测，不修改 checkpoint，并在 `run_config.json` 的
`input.policy_execution` 中记录 checkpoint 原值和实际生效值。

```text
--temporal-ensemble-coeff=C
```

最终两个 Python 字段都使用可选值：

```python
n_action_steps: int | None = None
temporal_ensemble_coeff: float | None = None
```

不传对应 override 时，保持 checkpoint 原始行为。

### 3.2 Temporal Ensemble 必须在 policy 构造前覆盖 config

实现会在 policy 构造前应用 `temporal_ensemble_coeff` override。处理顺序是：

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
  "input": {
    "policy_execution": {
      "checkpoint_n_action_steps": 5,
      "effective_n_action_steps": 1,
      "n_action_steps_overridden": true,
      "checkpoint_temporal_ensemble_coeff": null,
      "effective_temporal_ensemble_coeff": 0.01,
      "temporal_ensemble_coeff_overridden": true
    }
  }
}
```

现有 `ActionPlanTracker` 和 Rerun 已经能区分：

- normal action queue；
- discarded chunk tail；
- Temporal Ensemble 模式下每帧新预测的 chunk。

因此不需要伪造“未来执行轨迹”。

## 4. 实现验证

```
pytest -q \
  tests/test_evaluation_common.py \
  tests/test_evaluation_interfaces.py
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

```
bash scripts/experiments/act/eval_libero.sh \
  --policy-path="outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model" \
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

```
source_policy="outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model"
source_name="a1"
```

如果后续完成 A2，可以把这两个变量替换为 A2 的真实 checkpoint 相对路径和名称；A2 的
Replan/Ensemble 对照仍必须共用同一个 checkpoint。

下面所有命令必须复用同一个 `source_policy`。

## 6. Clean closed-loop 对比

`--num-visualized-episodes=M` 会把前 M 个 rollout 分别保存为可点击切换的独立 Rerun
Recording；每个 Recording 使用自己的从 0 开始的时间轴。要查看全部 20 个 rollout，
把下面命令中的值从 `1` 改为 `20`。

### 6.1 Source Standard

不传 override：

```
bash scripts/experiments/act/eval_libero.sh \
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

```
bash scripts/experiments/act/eval_libero.sh \
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

```
bash scripts/experiments/act/eval_libero.sh \
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

```
bash scripts/experiments/act/eval_libero.sh \
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

```
bash scripts/experiments/act/eval_libero.sh \
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

同时打开 Replan/Ensemble 两个独立 Viewer 时使用不同端口：

```
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9876 \
  "outputs/act/a3_temporal_ensemble/${source_name}_replan_clean/rollout.rrd"

conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9877 \
  "outputs/act/a3_temporal_ensemble/${source_name}_ensemble_clean/rollout.rrd"
```

`--detach-process` 不会自动更换端口；若两个 Viewer 都使用 `9876`，第二份数据会进入第一个
Viewer。把两个路径放进同一条命令也只会打开一个 Viewer。`--persist-state` 保存布局；
同一 application ID 的并行窗口共享持久化布局，最后关闭者可能覆盖布局状态。

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

## 9. 已完成结果：A1 Temporal Ensemble

结果标签：`Formal`、`Closed-loop`、`Clean`。

评测任务为 `libero_object/task_id=8`：
`pick_up_the_chocolate_pudding_and_place_it_in_the_basket`。三组结果均使用 seed
1000～1019 和 init state 0～19。A1 Replan 与 A1 Ensemble 使用同一个 A1 checkpoint，
因此二者是判断 Temporal Ensemble 效果的主对照；A0 来自另一个训练配置，只作为辅助基线。

### 9.1 正式结果

| 配置 | Clean SR ↑ | Action TV ↓ | Action Jerk ↓ | Mean gripper toggles | Model queries | Policy p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| A0 single-step（辅助基线） | 9/20（45%） | 0.01495 | 0.01786 | 3.45 | 4590 | 11.11 ms |
| A1 Replan（主对照） | 4/20（20%） | 0.01879 | 0.02264 | 1.80 | 5149 | 10.09 ms |
| A1 + Temporal Ensemble 0.01 | 18/20（90%） | 0.01519 | 0.00413 | 2.80 | 3475 | 10.66 ms |

A1 Ensemble 相比 source-matched A1 Replan：

- 成功率从 20% 提高到 90%，增加 70 个百分点；
- Action Jerk 降低约 81.8%；
- Action TV 降低约 19.1%；
- policy p95 latency 增加约 0.57 ms；
- A1 Replan 成功的 4 个初始化在 Ensemble 中仍然全部成功，另外有 14 个初始化由失败变为成功。

三组都在每个实际控制步查询一次模型。总 `Model queries` 会受到成功 episode 提前结束的
影响，不能直接解释为单步推理成本；单步成本应比较 p95 latency。

相对 A0 辅助基线，A1 Ensemble 的成功率提高 45 个百分点，Action Jerk 降低约 76.9%；
但 A0 与 A1 使用不同训练配置，这部分差异不能单独归因于 Temporal Ensemble。

结果文件：

- A0：`outputs/act/a0_single_step/closed_clean/metrics.json`
- A1 Replan：`outputs/act/a1_chunking/closed_clean/metrics.json`
- A1 Ensemble：`outputs/act/a3_temporal_ensemble/a1_ensemble_clean/metrics.json`
- A1 Ensemble 可视化：
  `outputs/act/a3_temporal_ensemble/a1_ensemble_clean/rollout.rrd`

### 9.2 问题与现象

失败 episode 为 2 和 16，二者均运行到 280 步 time limit：

- Episode 2：从可视化看，机械臂已经接近抓起巧克力布丁，但没有完成稳定抓取。这是抓取阶段
  的精度失败现象；仅凭视频不能确定根因是接触位置、闭合时机还是其他控制误差。
- Episode 16：机械臂完成搬运和释放，但巧克力布丁落在篮筐边缘，没有满足模拟器成功条件。
  这是放置阶段的精度失败现象；仅凭视频不能确定是目标位置、释放时机还是接触动力学导致。

这两次失败分别发生在抓取和放置阶段，不是同一种失败，也没有观察到明显的全程轨迹崩溃。
当前剩余问题更接近空间精度与操作裕量不足，而不是已经被 Action Jerk 指标反映出的高频动作
抖动。该判断来自可视化现象，不等同于已证明的根因。

`Mean gripper toggles` 从 Replan 的 1.80 增加到 2.80 不能直接视为退化。成功完成
“抓取—搬运—释放”本身需要夹爪状态切换，而 Replan 的低切换次数同时伴随较低成功率。

### 9.3 当前结论与限制

在这个 A-Sanity 单任务上，`chunk_size=20`、每步重规划和
`temporal_ensemble_coeff=0.01` 的组合通过了预先设定的 development 门槛：clean success
没有下降，Action Jerk 的降幅远大于 15%，并且成功率显著提高。结果支持 Temporal Ensemble
能够缓解当前 A1 policy 的重规划不稳定问题。

该结论只覆盖一个 LIBERO task、20 个固定初始状态和 clean observation。它不能证明跨任务
泛化，也不能证明 noisy observation 下同样稳定；Noisy closed-loop 仍需按第 7 节单独评测。

### 9.4 完整复现命令

下面命令使用本次 A1 checkpoint，并写入新的复现目录，避免覆盖上述正式结果：

```bash
bash scripts/experiments/act/eval_libero.sh \
  --policy-path=outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=20 \
  --n-action-steps=1 \
  --temporal-ensemble-coeff=0.01 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a3_temporal_ensemble/a1_ensemble_clean_repeat
```

## 10. 已完成结果：A2 + Temporal Ensemble

结果标签：`Formal`、`Closed-loop`、`Clean`。

本次复用已经训练完成的 A2 `act_delta` checkpoint，不重新训练。两组都评测
`libero_object/task_id=8`、20 个 rollout、seed 1000、init state 0～19，且未加入观测扰动：

- A2 Standard：checkpoint 默认 `n_action_steps=5`，不启用 Ensemble；
- A2 Ensemble：`n_action_steps=1, temporal_ensemble_coeff=0.01`。

### 10.1 正式结果

| 配置 | Clean SR ↑ | Action TV ↓ | Action Jerk ↓ | Mean gripper toggles | Model queries | Policy p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| A2 Standard（`n=5`） | 14/20（70%） | 0.02344 | 0.02403 | 14.65 | 804 | 8.67 ms |
| A2 + Temporal Ensemble 0.01（`n=1`） | 18/20（90%） | 0.01517 | 0.00418 | 2.30 | 3162 | 11.63 ms |

A2 Ensemble 相比 A2 Standard：

- 成功率从 70% 提高到 90%，增加 20 个百分点；
- Action TV 降低 35.3%；
- Action Jerk 降低 82.6%；
- 原始 gripper command 过零次数从 14.65 降到 2.30，降低 84.3%；
- policy p95 latency 从 8.67 ms 增至 11.63 ms；
- Ensemble 每个控制步都调用模型；Standard 每 5 步调用一次，因此总 model queries
  从 804 增至 3162。两组 episode 长度不同，不能用总调用数计算单次推理速度。

Standard 失败的 init states `0, 1, 4, 5, 8, 14` 在 Ensemble 中全部成功；但 Standard
成功的 init states `16, 17` 在 Ensemble 中退化为失败。Ensemble 的两次失败都运行到
280 步 time limit，失败阶段尚未逐帧分类。因此它提高了总体成功率，但并非对每个初始化
都单调改善。

结果文件：

- A2 Standard：`outputs/act/a2_delta_loss/run_20000_01/closed_clean_n5/metrics.json`
- A2 Ensemble：
  `outputs/act/a2_delta_loss/run_20000_01/closed_clean_ensemble_001/metrics.json`
- A2 Ensemble 可视化：
  `outputs/act/a2_delta_loss/run_20000_01/closed_clean_ensemble_001/rollout.rrd`

### 10.2 当前结论与证据边界

A2 + Temporal Ensemble 是当前已测配置中更好的 clean closed-loop 组合：成功率更高，
机械臂动作 TV/Jerk 更低，夹爪命令符号也明显更稳定。就最终部署配置而言，这个提升不是
“略微平滑”，而是 smoothness 指标上的显著变化；成功率则提高了 4/20。

但是，当前缺少同一 A2 checkpoint 的严格 Replan control：

```text
n_action_steps=1
temporal_ensemble_coeff=None
```

Standard 与 Ensemble 同时改变了 `n_action_steps` 和是否启用 Ensemble，所以现有结果只能
证明“A2 Standard → A2 Ensemble 组合配置”改善，不能把全部差异严格归因于 Temporal
Ensemble 本身。要回答纯粹的 Ensemble 因果效果，仍需按第 6.2 节补跑 A2 Replan control，
再以 Replan → Ensemble 为主对照。

此外，本结果只覆盖 clean observation、一个任务和固定 20 个 init states；Noisy
Closed-loop 尚未执行。

### 10.3 完整复现命令

```bash
bash scripts/experiments/act/eval_libero.sh \
  --policy-path=outputs/act/a2_delta_loss/run_20000_01/checkpoints/020000/pretrained_model \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=20 \
  --n-action-steps=1 \
  --temporal-ensemble-coeff=0.01 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a2_delta_loss/run_20000_01/closed_clean_ensemble_001_repeat
```
