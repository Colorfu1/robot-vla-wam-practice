# ACT 实验一：单步预测与 Action Chunking

## 1. 实验要回答什么

比较以下两种 ACT：

| 配置 | `chunk_size` | `n_action_steps` | 含义 |
|---|---:|---:|---|
| A0 | 1 | 1 | 每次只预测并执行一个动作 |
| A1 | 20 | 5 | 每次预测 20 个动作，执行前 5 个后重新规划 |

实验问题：

> 在相同数据、训练步数和主要网络规模下，A1 是否能在不明显降低任务成功率的前提下，让动作比
> A0 更连续？

当前 A-Sanity 已经证明 ACT 能在五条轨迹上学习这个任务。本实验不再验证“能不能训练”，而是
用 20 条轨迹比较单步预测和 action chunking。

> 当前状态：实验启动脚本、step-0 initializer、`proprio_gaussian` 扰动、结果记录和测试均已
> 实现，可以按第 5～7 节直接执行。

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

A0 和 A1 只改变 `chunk_size` 与 `n_action_steps`。视觉骨干、Transformer、optimizer 和数据
处理保持一致。

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

### 3.2 一个公开入口和两个便捷别名

公开入口直接包含完整的 `conda run ... lerobot-train` 命令：

```bash
scripts/experiments/act/action_chunking.sh --variant=a0 --steps=20000
scripts/experiments/act/action_chunking.sh --variant=a1 --steps=20000
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

`--variant=a1` 固定：

```text
--policy.chunk_size=20
--policy.n_action_steps=5
```

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

```bash
bash -n scripts/experiments/act/a0_single_step.sh
bash -n scripts/experiments/act/a1_chunking.sh
bash -n scripts/experiments/act/action_chunking.sh

scripts/experiments/act/action_chunking.sh \
  --variant=a0 --steps=20000 --print-command
scripts/experiments/act/action_chunking.sh \
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

```bash
pytest -q \
  tests/test_act_experiment_launchers.py \
  tests/test_evaluation_interfaces.py \
  tests/test_evaluation_rerun.py
```

## 5. 训练

以下命令均从 repo 根目录执行。

### 5.1 创建 A0 和 A1 的 step-0 checkpoint

```bash
scripts/experiments/act/action_chunking.sh --variant=a0 --steps=0
scripts/experiments/act/action_chunking.sh --variant=a1 --steps=0
```

把两个命令打印出的路径保存下来：

```bash
a0_b0="outputs/act/a0_single_step/<timestamp>/checkpoints/000000/pretrained_model"
a1_b0="outputs/act/a1_chunking/<timestamp>/checkpoints/000000/pretrained_model"
```

确认配置：

```bash
jq '{type, chunk_size, n_action_steps, use_vae}' "${a0_b0}/config.json"
jq '{type, chunk_size, n_action_steps, use_vae}' "${a1_b0}/config.json"
```

从两个初始化 manifest 中记录 `total_parameters`。A0/A1 的参数量允许因 query embedding 数量
不同而有小幅差异，其余结构配置必须一致。

### 5.2 分别训练 20,000 updates

```bash
scripts/experiments/act/action_chunking.sh \
  --variant=a0 \
  --steps=20000 \
  --policy-path="${a0_b0}"
```

```bash
scripts/experiments/act/action_chunking.sh \
  --variant=a1 \
  --steps=20000 \
  --policy-path="${a1_b0}"
```

记录两个训练命令打印出的 run 目录：

```bash
a0_policy="outputs/act/a0_single_step/<timestamp>/checkpoints/020000/pretrained_model"
a1_policy="outputs/act/a1_chunking/<timestamp>/checkpoints/020000/pretrained_model"
```

## 6. Offline 评测

```bash
episodes=(
  823 826 834 840 841 845 847 869 875 878
  910 918 920 929 944 948 949 953 966 972
)
```

```bash
scripts/experiments/act/eval_offline.sh \
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

```bash
scripts/experiments/act/eval_offline.sh \
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

### 7.1 Clean

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${a0_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a0_single_step/closed_clean
```

```bash
scripts/experiments/act/eval_libero.sh \
  --policy-path="${a1_policy}" \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=1 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a1_chunking/closed_clean
```

### 7.2 Normalized proprioception noise 0.01

```bash
scripts/experiments/act/eval_libero.sh \
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

```bash
scripts/experiments/act/eval_libero.sh \
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

```bash
rerun outputs/act/a0_single_step/closed_clean/rollout.rrd
rerun outputs/act/a1_chunking/closed_clean/rollout.rrd
```

指标文件：

```text
<output-dir>/metrics.json
```

重点比较：

| 配置 | Clean SR | Noisy SR | First MAE | Action TV | Action Jerk | Gripper toggles | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | | | | | | | |
| A1 | | | | | | | |

支持 Action Chunking 假设的 development 门槛：

- A1 clean success rate 相比 A0 下降不超过 5 个百分点；
- A1 closed-loop action jerk 至少降低 15%；
- A1 noisy success rate 不低于 A0。

`action_jerk` 是执行动作前六维的平均绝对二阶差分，只是固定 10 Hz 下的抖动代理，不是带
物理单位的真实 jerk。

最后将 task、数据量、observation、action、horizon、frequency、train loss、offline error、
closed-loop success、latency 和 failure cases 写入 `reports/experiment_log.md`。
