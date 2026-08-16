# 官方 Diffusion Policy StackCube 18D/4D 对照实验

## 目的

这组实验把当前 M1-W 的训练协议固定为 `RGB + state18(qpos+qvel) + 4D pd_ee_delta_pos`，只改变数据来源，定位当前低成功率主要来自数据还是观测/动作协议。

两条线使用相同的 787 条训练 episode、50k iterations、global batch 256、每 10k 保存与监控。
官方数据线使用显式的、无重叠的 CPU development seed 区间（默认从 20000 开始），而不是 M1-W
的 state bank；历史 M1-W 投影线保留其原有 GPU state-bank evaluator，避免把两种数据协议混为一谈。

| 线 | 数据 | 4D 标签含义 | 结论边界 |
|---|---|---|---|
| official | 官方 motion-planning RGB H5 | 原生 `pd_ee_delta_pos` | 干净的 state18 消融基线 |
| m1w（历史） | M1-W GPU live-RGB H5 | 7D `pd_ee_delta_pose` 取 `[dx,dy,dz,gripper]` | 非原生 4D 动作投影，只能作动作消融 |

M1-W 4D 线不能单独证明数据质量：它的 4D 标签从未在 4D controller 下执行过。若官方 18D/4D 明显正常而 M1-W 投影线明显差，说明应优先回查 M1-W 数据分布；若两条都差，优先回查 18D 观测/评测协议或 StackCube 的视觉泛化，而不是直接判定 M1-W 数据损坏。

### 一个 episode 的离线 action 对照

没有启动 replay，只读取已有 H5：M1-W `d_core/traj_0` 的
`source_group_id=40` 对应原始 joint trajectory，官方原生 4D H5 的 `traj_40`
来自同一 source group。将 M1-W 7D 取 `[0,1,2,6]` 后，与官方原生 4D 的前 105 步比较：

- 长度为 107 vs 105；
- 每通道 MAE 为 `[0.044, 0.178, 0.071, 0.190]`；
- 只有 23.1% 的标量值在 `1e-6` 内相等；
- M1-W 的三个旋转通道在约 87.9% 帧非零。

因此“投影后数值一定等于原生 4D”不成立。差异来自目标 controller/转换路径（以及此处 CPU/GPU backend）不同，而不是简单的数组维度变化。

## 远端启动

两条任务分别使用独立产物目录，可以先后或在资源允许时并行启动。命令只使用 0–3 卡；训练和正式 exact-state 评测都使用这四张卡。

```bash
bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=official \
  --phase=all \
  --gpus=0,1,2,3 \
  --num-demos=787 \
  --total-iters=50000 \
  --eval-freq=10000 \
  --save-freq=10000 \
  --dataloader-workers=4 \
  --output-root=outputs/maniskill/official_dp_stackcube_18d4_official
```

```bash
bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=m1w \
  --phase=all \
  --gpus=0,1,2,3 \
  --num-demos=787 \
  --total-iters=50000 \
  --eval-freq=10000 \
  --save-freq=10000 \
  --dataloader-workers=4 \
  --output-root=outputs/maniskill/official_dp_stackcube_18d4_m1w_projected
```

训练阶段仍保留 vendored 官方脚本的 CPU 内评测：多卡训练时每个 DDP rank 在自己的 GPU
上推理 action，再把 action 交给 CPU rollout；`--eval-freq=10000` 控制它的频率。训练内评测
使用固定 development seed，并按 rank 切成不重叠区间。训练结束后的 `--phase=eval` 再对每个
数值 checkpoint 独立执行同一类 CPU rollout，便于断点续跑和正式选模，不依赖训练内的随机
评测文件。每个 checkpoint 的正式结果写入
`20_eval/development/checkpoint_<step>/metrics.json`，已完成的目录会自动跳过，最后按
`success_once`、`success_at_end`、最早 step 选择，并汇总到 `20_eval/development/metrics.json`。

如果当前这次旧训练中断，可以从已有的数值 checkpoint 做 warm resume。例如已有
`20000.pt` 时：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=official --phase=train --gpus=0,1,2,3 \
  --num-demos=787 --total-iters=50000 --save-freq=10000 \
  --resume-from=outputs/maniskill/official_dp_stackcube_18d4_official/10_train/runs/official_dp_stackcube_18d4_official_50000/checkpoints/20000.pt \
  --output-root=outputs/maniskill/official_dp_stackcube_18d4_official

bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=official --phase=eval --gpus=0,1,2,3 \
  --eval-episodes=200 --eval-steps=200 --eval-seed-start=20000 \
  --output-root=outputs/maniskill/official_dp_stackcube_18d4_official
```

旧的 `20000.pt` 是修复前格式，只含模型权重，因此可以继续训练，但优化器、scheduler 和 EMA
内部计数会重新初始化；这是 warm resume，不是逐位等价的断点续训。修复后的新 checkpoint
会保存这些状态，之后用同一个 `--resume-from` 才是完整 resume。将 `official` 替换为 `m1w`
即可运行另一条线；runner 会拒绝覆盖已有训练目录，但 eval 会跳过已经有 `metrics.json` 的
checkpoint 目录。

## 产物与判读

每条线的产物均分为：

```text
00_prepare/adapter_manifest.json
00_prepare/train_episode_ids.json
logs/10_train.log
10_train/runs/<exp-name>/checkpoints/
20_eval/development/checkpoint_<step>/metrics.json
20_eval/development/selection.json
20_eval/development/metrics.json          # selected checkpoint 的副本
```

## 官方数据评测设备边界

仓库现在有统一的运行时 rollout contract：配置必须声明 `data_provenance`，并由
`src/robot_practice/experiments/maniskill/rollout_contract.py` 在创建评测环境前校验。
`official` 数据只能使用 `physx_cpu` rollout；`m1w` 数据只能使用 `physx_cuda` rollout。
结果的 `metrics.json`/`run_manifest.json` 会记录 `data_provenance`、
`required_rollout_backend` 和 `actual_rollout_backend`，因此不能只看启动命令推断实际口径。

这条规则不约束 M1-W 为了生成中间 4D 标签而运行的 CPU converter；那是转换诊断阶段，不能
当作 M1-W 正式 rollout。M1-W 的正式 state-bank evaluation 和 live-RGB data replay 仍必须走 GPU。

对 `dataset-kind=official`，只允许以下设备布局：

- policy inference：每个 torchrun DDP rank 使用自己的 CUDA device；
- rollout：每个 rank 的官方 CPU vector worker 使用 `physx_cpu`；
- boundary：`action`、`truncated/terminated` 和 episode metrics 在环境/NumPy 汇总边界转到 CPU；
- merge：rank 0 合并所有 rank 的 per-episode metrics，再执行官方 success 指标与 checkpoint 逻辑。

这条规则只改变评测执行位置，不改变官方 Diffusion Policy 的网络、loss、数据读取、优化器、
训练采样或 checkpoint 格式。`dataset-kind=m1w` 是历史投影对照，不继承 official 数据线的
CPU backend 强制项。

主指标是 `20_eval/development/metrics.json` 的 `evaluation.success_once.mean`（state-bank
路线兼容读取 `rate`）。训练内评测和 official 正式评测的身份都是显式 environment seed；M1-W/native4d state-bank
路线的身份仍是 `start_state_fingerprint`。训练期间不再用随机内评测做 checkpoint 选择。

## 实现说明

- 适配器：`src/robot_practice/experiments/maniskill/official_stackcube_18d4_adapter.py`。
- 训练入口仍是 vendored 官方 `train_rgbd.py`，只增加了两个显式 `dataset_format`。
- H5 按窗口懒读，每个 DataLoader 进程独立打开只读句柄，不把完整 RGB 数据复制到显存；
  `--dataloader-workers=4` 表示每个 torchrun rank 使用 4 个数据读取 worker。
- 上面的 `m1w` 投影线仅保留作历史对照，不能作为“原生 4D M1-W 数据”结论。当前要从
  joint action 重新生成原生 4D 数据时，使用独立路线：

  ```bash
  bash scripts/experiments/maniskill/run_m1_wam_native_4d.sh \
    --phase=all --num-envs=100
  ```

  该路线从原始 `pd_joint_pos` H5 重新执行 ManiSkill 官方 CPU 转换到
  `pd_ee_delta_pos`（4D），再做 GPU baseline replay、失败样本的 4D position-only online EE
  repair，最后在 GPU `env.step` 中同步保存 RGB、4D action、18D proprioception、env state 和
  success。它写入 `outputs/maniskill/m1_wam_native_4d/StackCube-v1/`，不会覆盖冻结的 7D
  M1-W 数据；各阶段可单独重跑：`convert`、`gpu-replay`、`gpu-repair`、`merge`、`capture`。

### 断点与评测实现

- 新 checkpoint 保存 `agent`、`ema_agent`、optimizer、LR scheduler、EMA shadow state 和训练 step。
- 训练内评测由所有 DDP rank 共同完成：GPU 只负责 policy inference，CPU 负责 rollout；正式评测按 checkpoint 独立落盘，因此某个 checkpoint 失败后只需重跑 `--phase=eval`，不会重做训练。
- `official` 评测由 `eval_official_stackcube_rgb_multi_gpu` 启动“一卡一模型、CPU 一环境”的
  worker，seed 区间按 worker 连续切分；`m1w/native4d` 仍使用 state-bank 的旧 exact evaluator。
