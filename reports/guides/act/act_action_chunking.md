# ACT 实验一：单步预测与 Action Chunking

## 实验目标

- 比较 A0 (`chunk_size=1,n_action_steps=1`) 与 A1 (`chunk_size=20,n_action_steps=5`) 在同数据、
  同步数、同 optimizer 配置下的 Offline 和 Closed-loop 性能差异，并验证实现不改变核心网络配置。

## 实验思路

1. 用固定 seed 和同一 20 条 demonstration 分别初始化 A0/A1 step-0 checkpoint；
2. 分别训练到同样步数；
3. 同一 evaluator 对照 offline 与 closed-loop 指标；
4. 通过 `action_chunking` 对照组分离“chunk 训练”与“closed-loop 执行频率”两个影响。

## 可直接执行命令

```bash
cd "$(git rev-parse --show-toplevel)"
conda activate robot-practice
export PYTHONNOUSERSITE=1

bash scripts/experiments/act/action_chunking.sh --variant=a0 --steps=0
bash scripts/experiments/act/action_chunking.sh --variant=a1 --steps=0
bash scripts/experiments/act/action_chunking.sh --variant=a0 --steps=20000
bash scripts/experiments/act/action_chunking.sh --variant=a1 --steps=20000

bash scripts/experiments/act/eval_offline.sh --policy-path=outputs/act/a0_single_step/.../checkpoints/020000/pretrained_model \
  --suite=libero_object --task-id=8 --dataset-root=datasets/lerobot/lerobot/libero \
  --episodes 823 826 834 840 841 845 847 869 875 878 910 918 920 929 944 948 949 953 966 972 \
  --output-dir=outputs/act/a_action_chunking/a0_offline
```

## 结果落点

- `outputs/act/a0_single_step/<timestamp>/checkpoints/020000/pretrained_model`
- `outputs/act/a1_chunking/<timestamp>/checkpoints/020000/pretrained_model`
- `outputs/act/a0_single_step/<timestamp>/eval.../` 与 `outputs/act/a1_chunking/.../eval.../` 下的：
  `run_config.json`, `eval_info.json`, `metrics.json`, `rollout.rrd`, `rollouts.parquet`, `videos/`

## 判断标准

- Offline 指标（`first_action_mae`, `delta_action_mae`）与 A0/A1 对比满足预设优先序；
- Closed-loop `success_rate` 仅按 formal run（20 episodes）评估，smoke 仅验证可运行；
- 动作平滑性通过 `action_tv`、`action_jerk` 对比，减少越好；指标读取自 `metrics.json`。

## 模式

- Offline
- Closed-loop
- Smoke（用于可运行性 sanity check）

## 1. 实验要回答什么

比较以下两种 ACT：

| 配置 | 训练监督 `chunk_size` | checkpoint 默认执行 `n_action_steps` | 含义 |
|---|---:|---:|---|
| A0 | 1 | 1 | 每次只预测并执行一个动作 |
| A1 | 20 | 5 | 每次预测 20 个动作，执行前 5 个后重新规划 |

参数语义必须分开理解：

- `chunk_size` 决定训练样本包含多少步 GT action、模型预测多少步，以及 loss 覆盖多少步；
- `n_action_steps` 不参与 `ACTPolicy.forward()` 的训练 loss，只会写入 checkpoint，作为不传
  eval override 时的默认执行长度；
- 因此 A1 训练时仍然对 20 步预测和 20 步 GT 计算 loss，不是只计算前 5 步。

> 闭环评测提示：`eval_libero.sh` 默认使用 checkpoint 中的 `n_action_steps`。需要隔离
> “chunk 训练”和“连续执行多步”的影响时，给同一个 A1 checkpoint 传
> `--n-action-steps=1`；这只覆盖本次评测，不修改 checkpoint。完整命令见 7.2 节。

实验问题：

> 在相同数据、训练步数和主要网络规模下，A1 是否能在不明显降低任务成功率的前提下，让动作比
> A0 更连续？

当前 A-Sanity 已经证明 ACT 能在五条轨迹上学习这个任务。本实验不再验证“能不能训练”，而是
用 20 条轨迹比较单步预测和 action chunking。

> 实施状态：A0/A1 启动脚本、step-0 initializer、`proprio_gaussian` 扰动、结果记录和测试均已
> 实现；20-demo A0/A1 正式结果已在第 9 节登记。当前实际保留的训练产物只有 020000 checkpoint，
> 没有保留这两次训练的 000000 checkpoint 和完整 loss 曲线。

## 2. 固定数据和配置

| 项目 | 固定值 |
|---|---|
| LIBERO suite | `libero_object` |
| Task ID | `8` |
| Dataset task index | `29` |
| Language | `pick up the chocolate pudding and place it in the basket` |
| Dataset root | `datasets/lerobot/lerobot/libero` |
| 数据量 | 20 episodes，3233 frames |
| Observation | 两路 256×256 RGB + 8D state |
| Action | 7D relative EEF action |
| Control frequency | 10 Hz |
| Batch size | 8 |
| Updates | 20,000 |
| Seed | 1000 |
| VAE | false |

固定 episodes：

```text
823, 826, 834, 840, 841,
845, 847, 869, 875, 878,
910, 918, 920, 929, 944,
948, 949, 953, 966, 972
```

A0 和 A1 的 checkpoint 配置只改变 `chunk_size` 与 `n_action_steps`。训练目标实际由
`chunk_size` 改变；`n_action_steps` 只改变 checkpoint 的默认闭环执行协议。视觉骨干、
Transformer、optimizer 和数据处理保持一致。

`chunk_size` 会改变 ACT query embedding 数量，所以 A0 和 A1 的参数量不会完全相同，也不能
共享完整的 step-0 state dict。应使用相同 seed，并记录两者参数量。

## 3. 已实现的文件

```text
src/robot_practice/experiments/act_initialization.py
scripts/experiments/act/action_chunking.sh
scripts/experiments/act/a0_single_step.sh
scripts/experiments/act/a1_chunking.sh
tests/test_act_experiment_launchers.py
```

共享 noisy evaluation 涉及：

```text
src/robot_practice/evaluation/libero_act.py
src/robot_practice/evaluation/rerun_visualization.py
scripts/experiments/act/eval_libero.sh
tests/test_evaluation_interfaces.py
tests/test_evaluation_rerun.py
```

### 3.1 通用 step-0 initializer

LeRobot 0.4.4 的训练循环在 `steps=0` 时不会进入保存 checkpoint 的分支，因此不能只执行
`lerobot-train --steps=0`。

参考现有的：

```text
src/robot_practice/experiments/act_a_sanity.py
```

通用 initializer 使用固定实验变体：

```text
--variant=a0|a1
--output-dir
--dataset-root
```

它需要：

1. 使用 `set_seed(1000)`；
2. 加载固定的 20 条 episodes；
3. 创建 ACT policy 和 pre/postprocessor；
4. 保存到 `checkpoints/000000/pretrained_model`；
5. 使用 `TrainPipelineConfig.save_pretrained()` 保存 `train_config.json`；
6. 验证保存前后的 state dict 完全一致；
7. 在初始化 manifest 中保存参数量和模型 SHA256；
8. 拒绝覆盖已有输出目录。

A0 和 A1 的 initializer 分别传入：

```text
A0: chunk_size=1,  n_action_steps=1
A1: chunk_size=20, n_action_steps=5
```

这里显式传入 `n_action_steps` 是为了得到合法、可复现的 checkpoint 推理默认值，不表示训练
loss 只监督这么多步。当前 LeRobot ACT 默认 `n_action_steps=100`；当 `chunk_size` 改为 1
或 20 时，如果不同时提供合法的 `n_action_steps`，会违反
`n_action_steps <= chunk_size` 的配置约束。

### 3.2 一个公开入口和两个便捷别名

公开入口直接包含完整的 `conda run ... lerobot-train` 命令：

```
bash scripts/experiments/act/action_chunking.sh --variant=a0 --steps=20000
bash scripts/experiments/act/action_chunking.sh --variant=a1 --steps=20000
```

接口为：

```text
--variant=a0|a1
--steps=N
--policy-path=PATH
--print-command
-h, --help
```

`a0_single_step.sh` 和 `a1_chunking.sh` 只是向公开入口固定注入 variant 的便捷别名，
不包含另一套训练逻辑。

固定输出目录：

```text
outputs/act/a0_single_step/<timestamp>
outputs/act/a1_chunking/<timestamp>
```

固定训练参数应出现在 `--print-command` 输出中：

```text
--dataset.repo_id=lerobot/libero
--dataset.root=datasets/lerobot/lerobot/libero
--dataset.episodes=[823,826,834,840,841,845,847,869,875,878,910,918,920,929,944,948,949,953,966,972]
--policy.use_vae=false
--batch_size=8
--steps=20000
--seed=1000
--eval_freq=0
--wandb.enable=false
```

`--variant=a0` 固定：

```text
--policy.chunk_size=1
--policy.n_action_steps=1
```

训练 loss horizon 为 1；checkpoint 默认每执行 1 步重新规划。

`--variant=a1` 固定：

```text
--policy.chunk_size=20
--policy.n_action_steps=5
```

训练 loss horizon 为 20；checkpoint 默认每执行 5 步重新规划。`n_action_steps=5` 不会把
训练监督截断为 5 步。

不要允许调用者通过额外参数悄悄改变固定数据或核心配置。

### 3.3 Proprioception 扰动

为 closed-loop evaluator 增加：

```text
--perturbation=proprio_gaussian
--proprio-noise-std=0.01
```

默认仍为：

```text
--perturbation=none
```

噪声应加在 checkpoint preprocessor 输出的归一化 state 上：

```python
processed_observation = loaded.preprocessor(prepared.policy_observation)
processed_observation = dict(processed_observation)

state = processed_observation["observation.state"].clone()
noise = make_deterministic_noise(state.shape, episode_seed, step_index)
processed_observation["observation.state"] = state + noise
```

不要修改 `prepared.state`、相机图像、模拟器状态或传给 `env.step()` 的环境本身。Rerun 和
Parquet 应明确区分：

- simulator measured state；
- clean normalized policy state；
- noisy normalized policy state。

相同 seed、episode 和 step 必须产生相同噪声。

## 4. 最小测试

先运行：

```
bash -n scripts/experiments/act/a0_single_step.sh
bash -n scripts/experiments/act/a1_chunking.sh
bash -n scripts/experiments/act/action_chunking.sh

bash scripts/experiments/act/action_chunking.sh \
  --variant=a0 --steps=20000 --print-command
bash scripts/experiments/act/action_chunking.sh \
  --variant=a1 --steps=20000 --print-command
```

测试至少覆盖：

- 两个脚本使用完全相同的 20 条 episodes；
- A0 只产生 `chunk_size=1, n_action_steps=1`；
- A1 只产生 `chunk_size=20, n_action_steps=5`；
- `--steps=0` 保存完整 checkpoint；
- 重复参数、未知参数和已有输出目录被拒绝；
- `perturbation=none` 不改变现有 evaluator 行为；
- `proprio_gaussian` 只改变 policy input state；
- 相同 seed 的噪声可复现。

运行相关测试：

```
pytest -q \
  tests/test_act_experiment_launchers.py \
  tests/test_evaluation_interfaces.py \
  tests/test_evaluation_rerun.py
```

## 5. 训练

以下命令均从 repo 根目录执行。

### 5.1 创建 A0 和 A1 的 step-0 checkpoint

```
bash scripts/experiments/act/action_chunking.sh --variant=a0 --steps=0
bash scripts/experiments/act/action_chunking.sh --variant=a1 --steps=0
```

把两个命令打印出的路径保存下来：

```
a0_b0="outputs/act/a0_single_step/<timestamp>/checkpoints/000000/pretrained_model"
a1_b0="outputs/act/a1_chunking/<timestamp>/checkpoints/000000/pretrained_model"
```

确认配置：

```
jq '{type, chunk_size, n_action_steps, use_vae}' "${a0_b0}/config.json"
jq '{type, chunk_size, n_action_steps, use_vae}' "${a1_b0}/config.json"
```

从两个初始化 manifest 中记录 `total_parameters`。A0/A1 的参数量允许因 query embedding 数量
不同而有小幅差异，其余结构配置必须一致。

### 5.2 分别训练 20,000 updates

launcher 启动时会分别打印：

```text
Training supervision horizon: chunk_size=<N> GT actions per sample
Checkpoint inference default: n_action_steps=<N> (not used by training loss)
```

```
bash scripts/experiments/act/action_chunking.sh \
  --variant=a0 \
  --steps=20000 \
  --policy-path="${a0_b0}"
```

```
bash scripts/experiments/act/action_chunking.sh \
  --variant=a1 \
  --steps=20000 \
  --policy-path="${a1_b0}"
```

记录两个训练命令打印出的 run 目录：

```
a0_policy="outputs/act/a0_single_step/<timestamp>/checkpoints/020000/pretrained_model"
a1_policy="outputs/act/a1_chunking/<timestamp>/checkpoints/020000/pretrained_model"
```

## 6. Offline 评测

```
episodes=(
  823 826 834 840 841 845 847 869 875 878
  910 918 920 929 944 948 949 953 966 972
)
```

```
bash scripts/experiments/act/eval_offline.sh \
  --policy-path="${a0_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --agentview-image-convention=lerobot_180 \
  --episodes "${episodes[@]}" \
  --batch-size=1 \
  --num-visualized-episodes=1 \
  --output-dir=outputs/act/a0_single_step/offline_20
```

```
bash scripts/experiments/act/eval_offline.sh \
  --policy-path="${a1_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --agentview-image-convention=lerobot_180 \
  --episodes "${episodes[@]}" \
  --batch-size=1 \
  --num-visualized-episodes=1 \
  --output-dir=outputs/act/a1_chunking/offline_20
```

A0 的 `chunk_size=1`，所以 `delta_action_mae` 应为 `null`，这是正常结果。

## 7. Closed-loop 评测

`--n-episodes=20` 运行 20 个独立 rollout；`--num-visualized-episodes=M` 将前 M 个
rollout 分别写成独立 Rerun Recording，再合并进同一个 `rollout.rrd`。Viewer 左侧按
episode、seed、init state 和 success/failure 名称点击切换。若要查看全部 20 个，
把命令中的 `--num-visualized-episodes=1` 改为 `20`。

### 7.1 Clean

```
bash scripts/experiments/act/eval_libero.sh \
  --policy-path="${a0_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a0_single_step/closed_clean
```

```
bash scripts/experiments/act/eval_libero.sh \
  --policy-path="${a1_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a1_chunking/closed_clean
```

### 7.2 用同一 A1 权重测试每步重规划

`--n-action-steps` 只覆盖本次评测的执行 horizon，不修改 checkpoint。下面的命令仍由
A1 一次预测 20 步，但每次只执行第一步，然后根据新的 simulator observation 重新规划：

```
bash scripts/experiments/act/eval_libero.sh \
  --policy-path="${a1_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-action-steps=1 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a1_chunking/closed_replan_1
```

`run_config.json` 同时记录 checkpoint 原始值和本次实际生效值。

### 7.3 Normalized proprioception noise 0.01

```
bash scripts/experiments/act/eval_libero.sh \
  --policy-path="${a0_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=proprio_gaussian \
  --proprio-noise-std=0.01 \
  --seed=1000 \
  --output-dir=outputs/act/a0_single_step/closed_noisy
```

```
bash scripts/experiments/act/eval_libero.sh \
  --policy-path="${a1_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=proprio_gaussian \
  --proprio-noise-std=0.01 \
  --seed=1000 \
  --output-dir=outputs/act/a1_chunking/closed_noisy
```

正式评测不要传 `--max-steps`。

## 8. 查看结果

同时打开 A0/A1 两个独立 Viewer 时必须使用不同端口。Offline 对比：

```
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9876 \
  outputs/act/a0_single_step/offline_20/prediction.rrd

conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9877 \
  outputs/act/a1_chunking/offline_20/prediction.rrd
```

如果保留两个 Offline Viewer，再同时打开两个 closed-loop Viewer，继续使用空闲端口：

```
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9878 \
  outputs/act/a0_single_step/closed_clean/rollout.rrd

conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9879 \
  outputs/act/a1_chunking/closed_clean/rollout.rrd
```

`--detach-process` 只分离 Viewer 进程，不会自动更换端口。若两个进程都使用默认 `9876`，
第二份数据会进入第一个 Viewer。把两个 RRD 路径放在同一条命令中也只会打开一个 Viewer。
`--persist-state` 用于保存调整后的布局；同一 application ID 的并行窗口共享持久化布局，
最后关闭的窗口可能覆盖布局状态。

指标文件：

```text
<output-dir>/metrics.md
<output-dir>/metrics.json
```

重点比较：

| 配置 | Clean SR | Noisy SR | First MAE | Action TV | Action Jerk | Gripper toggles | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 (`n_action_steps=1`) | 9/20（45%） | 未运行 | 0.03096 | 0.01495 | 0.01786 | 3.45 | 10.38 ms |
| A1 matched replan (`n_action_steps=1`) | 4/20（20%） | 未运行 | 0.02564 | 0.01879 | 0.02264 | 1.80 | 9.24 ms |
| A1 default (`n_action_steps=5`) | 3/20（15%） | 未运行 | 0.02564 | 0.02615 | 0.03017 | 14.80 | 20.53 ms |

支持 Action Chunking 假设的 development 门槛：

- A1 clean success rate 相比 A0 下降不超过 5 个百分点；
- A1 closed-loop action jerk 至少降低 15%；
- A1 noisy success rate 不低于 A0。

`action_jerk` 是执行动作前六维的平均绝对二阶差分，只是固定 10 Hz 下的抖动代理，不是带
物理单位的真实 jerk。

## 9. 已完成结果（2026-07-24—25）

A0 和 A1 都完成了 20,000 updates、20 条 demonstration 的 Offline 和 Formal Clean
Closed-loop 评测。两次训练使用相同的 task、20 条训练 episode、batch size 8、seed 1000、
optimizer、视觉骨干和主要网络配置；策略差异为 A0 `chunk_size=1`、A1 `chunk_size=20`。
由于 `chunk_size` 会改变 ACT query embedding 数量，两个配置不是逐张量相同的模型。

Offline 结果：A0 的 first-action MAE 为 0.03096，teacher-forced Action Jerk 为 0.01661；
A1 分别为 0.02564（下降 17.2%）和 0.01349（下降 18.8%）。这说明 A1 在 demonstration
上的动作拟合更好，但不代表闭环成功。

正式闭环的关键对照是 A0 与 A1 都使用 `n_action_steps=1` 的每步重规划：A0 成功 9/20，
A1 成功 4/20；A1 的执行 Action Jerk 为 0.02264，相比 A0 的 0.01786 上升 26.7%。
A1 使用 checkpoint 默认的 `n_action_steps=5` 时成功 3/20、Action Jerk 0.03017，这是
额外的默认执行结果，不能与 A0 的单步配置直接混为同一个因果对照。

因此，本实验的当前结论是：在这个 task、seed 和训练预算下，action chunking 提高了
teacher-forced offline 拟合，但没有保持闭环成功率，也没有改善闭环平滑性；不能支持原先
“A1 在不降低成功率的同时更连续”的预设假设。证据仍限于单任务、单 seed，且没有 noisy
closed-loop 结果。

结果落点：

- `outputs/act/a0_single_step/offline_20/metrics.json`
- `outputs/act/a0_single_step/closed_clean/metrics.json`
- `outputs/act/a1_chunking/offline_20/metrics.json`
- `outputs/act/a1_chunking/closed_clean/metrics.json`（A1 matched replan）
- `outputs/act/a1_chunking/closed_clean_n5/metrics.json`（A1 default execution）

最后将 task、数据量、observation、action、horizon、frequency、train loss、offline error、
closed-loop success、latency 和 failure cases 写入 `reports/experiment_log.md`。
