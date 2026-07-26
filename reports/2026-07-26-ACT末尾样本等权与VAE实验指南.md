# ACT 末尾样本等权与 VAE 实验指南

## 1. 实验目标

在现有 A1（ACT Action Chunking）基础上分别验证两个假设：

1. **A1-ObsEqual**：让每个随机 observation 的总 L1 loss 等权，是否能改善接近
   episode 末尾的释放和释放后保持张开行为。
2. **A1-VAE**：保持原始 ACT loss，只打开标准 VAE，是否能改善未来动作存在不确定性时的
   闭环成功率和夹爪稳定性。

两项都是待验证假设，不能在正式评测完成前写成结论。

## 2. 实验思路

- 唯一基线是已经完成的 A1。
- 三组都使用相同的 20 条 demonstration、`chunk_size=20`、batch、seed 和训练步数。
- A1-ObsEqual 只改变 padding 后的 L1 归一化方式，不使用 VAE 和 delta loss。
- A1-VAE 使用 LeRobot 原始 ACT loss，只设置 `use_vae=true`。
- 首轮 Closed-loop 固定 `n_action_steps=5`，不启用 Temporal Ensemble 和扰动。
- Offline 用于检查动作拟合；LIBERO Closed-loop 才用于判断任务成功。

## 3. 固定实验配置

| 项目 | 值 |
|---|---|
| Task suite | `libero_object` |
| Task ID | `8`（拿起 chocolate pudding 并放入 basket） |
| Dataset | `lerobot/libero`，20 episodes / 3233 frames |
| Episodes | `823, 826, 834, 840, 841, 845, 847, 869, 875, 878, 910, 918, 920, 929, 944, 948, 949, 953, 966, 972` |
| Observation | 双 256×256 RGB 相机 + 8D state，当前帧输入 |
| Action | 7D relative EEF action，前 6 维机械臂、最后 1 维夹爪 |
| Training action horizon | `chunk_size=20` |
| Default execution horizon | `n_action_steps=5` |
| Control frequency | 10 Hz |
| Batch size | 8 |
| Training updates | 20,000 |
| Seed | 1000 |
| VAE defaults under test | `latent_dim=32`, `kl_weight=10.0` |

三组控制关系：

| 配置 | Policy type | L1 归一化 | VAE | Delta loss |
|---|---|---|---|---|
| A1 基线 | `act` | 固定 20 步张量 mean | false | 无 |
| A1-ObsEqual | `act_obs_equal` | 每个 observation 内按有效 action mean | false | 无 |
| A1-VAE | `act` | 固定 20 步张量 mean | true | 无 |

A1-ObsEqual 的 loss 是：

```python
valid = ~action_is_pad
per_action_l1 = abs(pred - gt).mean(dim=-1)
per_observation_l1 = (
    (per_action_l1 * valid).sum(dim=1)
    / valid.sum(dim=1)
)
loss = per_observation_l1.mean()
```

这会让每个 observation 样本等权。只剩 1 个有效 action 时，该 action 会承担整个样本的
loss；这是本实验刻意测试的变化。

## 4. 训练

以下命令使用固定目录，训练结束后的 checkpoint 路径可以直接用于后续命令。

### 4.1 A1-ObsEqual

```bash
bash scripts/experiments/act/a1_observation_equal.sh \
  --steps=20000 \
  --output-dir=outputs/act/a1_observation_equal/run_20000_01
```

训练命令会使用本地 `act_obs_equal` policy。它只允许 `use_vae=false`。

### 4.2 A1-VAE

```bash
bash scripts/experiments/act/a1_vae.sh \
  --steps=20000 \
  --output-dir=outputs/act/a1_vae/run_20000_01
```

该命令使用标准 `policy.type=act`，并显式传入：

```text
use_vae=true
latent_dim=32
kl_weight=10.0
```

### 4.3 只检查实际训练命令

下面两条命令不会创建输出目录，也不会启动训练：

```bash
bash scripts/experiments/act/a1_observation_equal.sh \
  --steps=20000 \
  --output-dir=outputs/act/a1_observation_equal/run_20000_01 \
  --print-command
```

```bash
bash scripts/experiments/act/a1_vae.sh \
  --steps=20000 \
  --output-dir=outputs/act/a1_vae/run_20000_01 \
  --print-command
```

训练产物：

```text
outputs/act/a1_observation_equal/run_20000_01/checkpoints/020000/pretrained_model/
outputs/act/a1_vae/run_20000_01/checkpoints/020000/pretrained_model/
```

原始 A1 对照 checkpoint：

```text
outputs/act/a1_chunking/20260724_121730/checkpoints/020000/pretrained_model/
```

注意：三种 loss 的绝对数值不能直接横向比较。A1-ObsEqual 改变了归一化分母；
A1-VAE 的总 loss 还包含 `10.0 × KLD`。

## 5. Offline 正式评测

Offline 是 teacher-forced prediction，不会执行预测动作，也不产生模拟器成功率。

```bash
bash scripts/experiments/act/eval_offline.sh \
  --policy-path=outputs/act/a1_observation_equal/run_20000_01/checkpoints/020000/pretrained_model \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --agentview-image-convention=lerobot_180 \
  --episodes 823 826 834 840 841 845 847 869 875 878 910 918 920 929 944 948 949 953 966 972 \
  --batch-size=1 \
  --num-visualized-episodes=1 \
  --output-dir=outputs/act/a1_observation_equal/run_20000_01/offline_20 \
  --overwrite
```

```bash
bash scripts/experiments/act/eval_offline.sh \
  --policy-path=outputs/act/a1_vae/run_20000_01/checkpoints/020000/pretrained_model \
  --suite=libero_object \
  --task-id=8 \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --agentview-image-convention=lerobot_180 \
  --episodes 823 826 834 840 841 845 847 869 875 878 910 918 920 929 944 948 949 953 966 972 \
  --batch-size=1 \
  --num-visualized-episodes=1 \
  --output-dir=outputs/act/a1_vae/run_20000_01/offline_20 \
  --overwrite
```

已有 A1 Offline 对照：

```text
outputs/act/a1_chunking/offline_20/
```

重点查看：

- `first_action_mae`
- gripper 维 first-action MAE
- `delta_action_mae`
- teacher-forced Action TV / Jerk

## 6. LIBERO Closed-loop 正式评测

下面每组都运行相同的 20 个 `init_state_id=0..19`。正式比较不要添加
`--max-steps`，也不要启用 Temporal Ensemble。

### 6.1 A1-ObsEqual

```bash
bash scripts/experiments/act/eval_libero.sh \
  --policy-path=outputs/act/a1_observation_equal/run_20000_01/checkpoints/020000/pretrained_model \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=20 \
  --n-action-steps=5 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a1_observation_equal/run_20000_01/closed_clean_n5 \
  --overwrite
```

### 6.2 A1-VAE

```bash
bash scripts/experiments/act/eval_libero.sh \
  --policy-path=outputs/act/a1_vae/run_20000_01/checkpoints/020000/pretrained_model \
  --suite=libero_object \
  --task-id=8 \
  --n-episodes=20 \
  --num-visualized-episodes=20 \
  --n-action-steps=5 \
  --perturbation=none \
  --seed=1000 \
  --output-dir=outputs/act/a1_vae/run_20000_01/closed_clean_n5 \
  --overwrite
```

已有 A1 Closed-loop 对照：

```text
outputs/act/a1_chunking/closed_clean_n5/
```

## 7. 查看结果

每个评测目录都会生成可直接阅读的指标表：

```text
<eval-output>/metrics.md
<eval-output>/metrics.json
```

Offline 可视化：

```text
<offline-output>/prediction.rrd
```

Closed-loop 可视化和模拟器视频：

```text
<closed-output>/rollout.rrd
<closed-output>/videos/
```

同时打开两个 Closed-loop Viewer 时使用不同端口：

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9876 \
  outputs/act/a1_observation_equal/run_20000_01/closed_clean_n5/rollout.rrd
```

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9877 \
  outputs/act/a1_vae/run_20000_01/closed_clean_n5/rollout.rrd
```

## 8. 判断标准

主要指标是 Formal Clean Closed-loop success rate 和释放阶段失败数量，次要指标是 Action
Jerk、gripper toggles、Offline gripper MAE 和延迟。

| 配置 | Offline first MAE | Gripper MAE | Clean SR | Release failures | Action Jerk | Gripper toggles | p95 latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.02564 | 0.04787 | 3/20（15%） | 待逐帧标注 | 0.03017 | 14.80 | 20.53 ms |
| A1-ObsEqual | 0.02488 | 0.04537 | 9/20（45%） | 未结构化标注 | 0.02590 | 13.50 | 14.06 ms |
| A1-VAE | 未执行 | 未执行 | 19/20（95%） | 1 次 time limit | 0.02205 | 11.40 | 12.39 ms |

A1-ObsEqual 假设获得支持需要同时满足：

- clean success rate 相比 A1 至少提高 10 个百分点，或明确的释放阶段失败至少减少 2 次；
- Action Jerk 相比 A1 不恶化超过 15%；
- 逐帧检查确认提升发生在释放或释放后保持阶段，而不只是其他阶段。

A1-VAE 假设获得支持需要同时满足：

- clean success rate 相比 A1 至少提高 10 个百分点；
- Action Jerk 相比 A1 不恶化超过 15%；
- 失败案例没有出现明显的随机动作多模态退化。

实验完成后，将 task、数据量、observation、action、horizon、frequency、train loss、
Offline error、Closed-loop success、latency 和 failure cases 写入
`reports/experiment_log.md`。

## 9. 已完成实验与结论

### 9.1 原始猜想

本轮验证两个独立猜想：

1. 原始 A1 在 episode 尾部使用固定 20 步分母计算 masked L1。越接近末尾，一个
   observation 拥有的有效 action 越少，因此该 observation 的总训练权重越低。我们猜测
   这会削弱接近 basket 时的释放和释放后保持行为。
2. 同一个当前 observation 可能对应多个合理的未来 action chunk。我们猜测 ACT 的 VAE
   可以比确定性 A1 更好地建模这种未来不确定性。

### 9.2 Formal Closed-loop 对照条件

三组都使用 `libero_object` task 8、相同 20 条 demonstration、20,000 updates、
seed 1000、`chunk_size=20` 和相同 `init_state_id=0..19`。正式评测统一为：

```text
n_action_steps=5
temporal_ensemble_coeff=None
perturbation=none
control_frequency=10 Hz
episode_length=280
```

因此 A1 与 A1-ObsEqual 的主要训练差异是 L1 归一化方式；A1 与 A1-VAE 的主要训练差异是
是否启用 VAE。

### 9.3 Formal Closed-loop 结果

| 配置 | Success rate | Action TV | Action Jerk | Mean gripper toggles |
|---|---:|---:|---:|---:|
| A1 | 3/20（15%） | 0.02615 | 0.03017 | 14.80 |
| A1-ObsEqual | 9/20（45%） | 0.02442 | 0.02590 | 13.50 |
| A1-VAE | 19/20（95%） | 0.02307 | 0.02205 | 11.40 |

A1-ObsEqual 相比 A1：

- success rate 提高 30 个百分点；
- Action TV 降低 6.6%；
- Action Jerk 降低 14.1%；
- mean gripper toggles 降低 8.8%。

A1-VAE 相比 A1：

- success rate 提高 80 个百分点；
- Action TV 降低 11.8%；
- Action Jerk 降低 26.9%；
- mean gripper toggles 降低 23.0%。

A1-VAE 唯一失败的是 `init_state_id=17`：运行到 280 步 time limit，原始 gripper
command 符号切换 34 次。该失败仍表现出明显夹爪不稳定，值得单独查看 rollout。

### 9.4 Offline 结果边界

A1-ObsEqual 已完成 20 条 demonstration 的 Offline 评测：

```text
first_action_mae=0.02488
gripper_mae=0.04537
delta_action_mae=0.01392
teacher_forced_action_jerk=0.01353
```

A1-VAE 尚未生成 `offline_20/metrics.json`，因此当前 VAE 结论只来自 Formal
Closed-loop，不能补写或推断 VAE 的 Offline MAE。

### 9.5 当前结论

本轮单 seed、固定 20 个 init states 的对照实验支持两个猜想：

- 每个 observation 等权后，成功率从 15% 提高到 45%，且动作连续性指标没有退化。
  这支持“episode 末尾 observation 总监督过弱会损害闭环行为”的判断。
- 打开标准 VAE 后，成功率达到 95%，同时 TV、Jerk 和夹爪切换均优于 A1。
  A1-VAE 是当前 A1 系列中表现最好的已测配置。

证据边界：

- 当前只有一个训练 seed，尚不能声称结果能够跨随机初始化稳定复现；
- A1-ObsEqual 的失败阶段尚未逐 episode 结构化标注，不能把全部提升都归因于释放阶段；
- 三组成功 episode 长度不同，TV/Jerk 的闭环轨迹分布也不同；
- `gripper_toggles` 是原始命令过零次数，不等于机械夹爪完整开合次数。

结果落点：

```text
outputs/act/a1_chunking/closed_clean_n5/
outputs/act/a1_observation_equal/run_20000_01/offline_20/
outputs/act/a1_observation_equal/run_20000_01/closed_clean_n5/
outputs/act/a1_vae/run_20000_01/closed_clean_n5/
```
