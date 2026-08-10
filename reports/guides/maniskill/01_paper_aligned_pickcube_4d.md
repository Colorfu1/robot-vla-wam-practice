# ManiSkill PickCube 4D 论文参照基线指南

## 重要进展

- 这是一条独立的论文参照基线，不修改 M0 的 7D `pd_ee_delta_pose` 主线。
- 数据源固定为 motion-planning，replay 固定为 `pd_ee_delta_pos`、RGB、`physx_cpu`。
- 训练、官方协议选模和固定 holdout test 使用独立目录；总入口会复用已经完成的产物。
- 官方协议的 `official_best` 用于和论文 Table III 并列；`holdout_once` 只用于仓库内部对照。
- 本地不执行 replay、rollout 或训练；下面命令全部在远端 ManiSkill 机器执行。

## 实验范围

本基线只回答一个问题：在 motion-planning、4D action、CPU backend、horizon 100 和官方评测语义下，
当前 LeRobot diffusion 管路是否能达到 ManiSkill3 Table III 的 PickCube RGB 数量级。

有意保留的一处差异是模型实现：我们使用 LeRobot diffusion（ResNet18），不是 ManiSkill 官方 DP 实现。
因此结果可以定位“管路是否到同一量级”，不能声称复现官方 DP 的具体实现。

| 项目 | 固定值 |
| --- | --- |
| 源数据 | `datasets/raw/maniskill_demos/PickCube-v1/motionplanning/trajectory.h5` |
| replay | `--use-first-env-state -c pd_ee_delta_pos -o rgb --save-traj --num-envs 10 -b physx_cpu` |
| action | `pd_ee_delta_pos`，4D：平移 3 + 夹爪 1 |
| student state | `qpos(9) + qvel(9) + goal_pos(3)`，21D |
| policy | `n_obs_steps=2`、`horizon=16`、`n_action_steps=8` |
| train | 30,000 steps；8 卡时每卡 batch 32、有效 batch 256；每 2,500 steps 保存 checkpoint |
| official eval | 100 局、10 个 CPU 单环境 worker、horizon 100、不传 reset seed、成功不停机 |
| holdout | CPU 单环境、`reset(seed=[k])`、`k=3000..3399`、400 局、成功不停机 |

## 数据准备

总入口会在远端自动执行下面的官方 replay；如果目标 replay 文件已经存在，就会复用它，不会删除或覆盖。
如果只想单独准备数据，可以执行：

```bash
cd "$(git rev-parse --show-toplevel)"
conda run --no-capture-output -n robot-practice \
  python -m mani_skill.trajectory.replay_trajectory \
  --traj-path datasets/raw/maniskill_demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu

bash scripts/experiments/maniskill/convert_mp_4d.sh \
  --replay-h5=datasets/raw/maniskill_demos/PickCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5 \
  --source-h5=datasets/raw/maniskill_demos/PickCube-v1/motionplanning/trajectory.h5 \
  --output-dir=datasets/lerobot/maniskill/pickcube_mp_4d \
  --repo-id=local/maniskill_pickcube_mp_4d \
  --audit-output=outputs/maniskill/paper_aligned_4d/conversion_audit.json \
  --action-stats-output=outputs/maniskill/paper_aligned_4d/action_channel_stats.json \
  --survival-report=outputs/maniskill/paper_aligned_4d/replay_survival.json
```

转换器在训练前检查四个固定闸门：replay survival ≥95%、动作越界比例为 0、成功前帧占比 ≥85%、首次成功
p90 ≤90 步。任何一个失败都会停止，不会继续训练。

注意：官方 `--num-envs 10` 的 CPU replay 会把多个 worker 的轨迹合并，并可能重新编号 `traj_<id>`。
转换器会读取 replay 元数据里的 `episode_seed`，按源 HDF5 的 episode_id 恢复顺序，并把
`replay_episode_id → source_episode_id` 写入审计与 conversion manifest；不要手工按 replay 文件夹编号取前
100 条。

官方 `convert_to_lerobot` 已在 M0 smoke 中验证能生成可训练数据，但它只写 9D `qpos`；本基线需要
21D `qpos+qvel+goal_pos`，所以最终数据由 `convert_mp_4d.sh` 写入，并保留相同的 LeRobot v3 契约。

远端数据盘如果拒绝临时目录到正式目录的原子 rename（`Invalid cross-device link`），转换器会自动退回为
复制发布；复制成功后才清理临时目录。重跑总入口时会复用已经生成的 replay H5，不会重新 replay。
评测结果目录也使用同样的复制回退，因此这个错误不会要求重新训练；只需重跑入口，已完成的评测会复用，缺失的
checkpoint 评测会继续执行。

## 一条命令跑完训练、model select、test

远端仓库根目录执行这一条即可。默认也会补齐缺失的数据准备阶段；已经存在的阶段会跳过。

```bash
(mkdir -p outputs/maniskill/paper_aligned_4d && \
  exec nohup bash scripts/experiments/maniskill/run_paper_aligned_4d.sh \
    --num-gpus=8 --eval-device=cpu --eval-parallelism=8 \
    > outputs/maniskill/paper_aligned_4d/nohup.log 2>&1) &
echo $!
```

`--eval-parallelism=8` 表示同时启动 8 个 checkpoint 评测进程；每个进程内部仍使用 10 个 CPU
环境 worker。每个 checkpoint 的独立日志 `outputs/maniskill/paper_aligned_4d/logs/20_*.log` 中都有一个
自己的 `tqdm` episode 进度条。若使用 GPU 做策略前向，把 `--eval-device=cpu` 改回 `cuda`，但不要在显存
不足时把并发数开得过大。

如果数据已经准备好，只想从训练开始，追加 `--skip-data`。如果中途出错，修复原因后重复执行同一条命令；
训练会从 `checkpoints/last` 恢复，已完成的 checkpoint rollout、selected copy 和 holdout 结果会复用，
不会从头重跑完整流程。

如果泄漏审计发现重复状态，入口会在 holdout 前停止；必须先修复划分，不能绕过这个闸门。

### 三段产物

训练的两个 arm 分开保存：

```text
outputs/maniskill/paper_aligned_4d/train_1000/
outputs/maniskill/paper_aligned_4d/train_100/
```

官方协议对每个 checkpoint 都单独保存：

```text
outputs/maniskill/paper_aligned_4d/official/1000/checkpoint_002500/metrics.json
outputs/maniskill/paper_aligned_4d/official/1000/official_best.json
outputs/maniskill/paper_aligned_4d/official/1000/selected/pretrained_model/
outputs/maniskill/paper_aligned_4d/official/100/...
```

固定 test 和参照系统保存为：

```text
outputs/maniskill/paper_aligned_4d/leakage_report.json
outputs/maniskill/paper_aligned_4d/holdout/diffusion_1000/
outputs/maniskill/paper_aligned_4d/holdout/diffusion_100/
outputs/maniskill/paper_aligned_4d/holdout/random/
outputs/maniskill/paper_aligned_4d/holdout/ppo_privileged/
outputs/maniskill/paper_aligned_4d/comparison.json
```

所有完整评测目录都包含 `metrics.json`、`episodes.parquet` 和 `run_manifest.json`。选中的视觉 checkpoint
是实际复制出来的目录，不是 symlink；后续启动 holdout 时使用 `selected/pretrained_model`。

## 两种成功率不能混用

`official_best` 是官方不传 seed 的随机 reset 协议下，所有 checkpoint 的 `success_once` 最大值，并额外对
选中 checkpoint 重复 5 次报告均值和极差。它和论文的 max-over-checkpoints 数字并列。

`holdout_once` 是固定 `3000..3399` seed、选中 checkpoint 只跑一次的仓库口径；它用于内部比较，不能拿去
直接和论文的 `1.00/0.76` 比。官方协议的成功不会让 episode 停止，因此 `success_once` 与
`success_at_end` 可能不同；这也不能和 M0 的成功即停结果直接相减。

## 监控与恢复

```bash
tail -f outputs/maniskill/paper_aligned_4d/pipeline.log
tail -f outputs/maniskill/paper_aligned_4d/logs/20_official_1000_002500.log
find outputs/maniskill/paper_aligned_4d/logs -maxdepth 1 -type f -printf '%f\n' | sort
```

训练有效 batch 必须在日志中看到 `Effective batch size: 32 x 8 = 256`。单独检查某个 checkpoint 时，
使用 `eval_official_protocol.sh`；不要直接把 M0 的 GPU/50-step `eval_closed_loop.sh` 当作这条基线的结果。
