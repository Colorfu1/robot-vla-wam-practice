# ManiSkill PickCube 4D 论文参照基线指南

## 重要进展

- 这是一条已完成的独立论文参照基线：不回写历史 7D `pd_ee_delta_pose` 产物，但其结果改变后续 ManiSkill 的任务选择。
- 数据源固定为 motion-planning，replay 固定为 `pd_ee_delta_pos`、RGB、`physx_cpu`。
- 全量 replay/转换闸门通过：1000/1000 条、77,976 帧，动作越界为 0，成功前帧占比 93.14%，首次成功 p90 为 85 步。
- 1000-demo 与 100-demo 两套训练、12 个 checkpoint 官方评测、5 次重复和 400 条固定 CPU holdout 均已完成。
- 1000-demo 选中 `022500`，官方 `success_once` 为 100/100，五次重复均值 99.8%；固定 holdout 为 397/400（99.25%）。
- 100-demo 选中 `020000`，官方 `success_once` 为 73/100，五次重复均值 57.8%；固定 holdout 为 206/400（51.5%）。
- 官方协议的 `official_best` 用于和论文 Table III 并列；`holdout_once` 只用于仓库内部隔离比较。
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
| holdout | 固定 `reset(seed=[k])`、`k=3000..3399`、400 局、成功不停机；默认 8 个固定 seed 分片并行，每片 50 局，单片仍是 1 个 CPU 环境 |

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
    --num-gpus=8 --eval-device=cuda \
    --eval-gpus=0,1,2,3,4,5,6,7 --eval-parallelism=8 \
    --holdout-parallelism=8 \
    > outputs/maniskill/paper_aligned_4d/nohup.log 2>&1) &
echo $!
```

`--eval-parallelism=8` 表示同时启动 8 个 checkpoint 评测进程，按顺序分别绑定 GPU 0..7；每个进程内部
仍使用 10 个 CPU 环境 worker 和 CPU RGB 渲染。每个 checkpoint 的独立日志
`outputs/maniskill/paper_aligned_4d/logs/20_*.log` 中都有自己的 `tqdm` episode 进度条。
如果只想使用 CPU 推理，可改为 `--eval-device=cpu`；此时不需要 `--eval-gpus`。
固定 holdout 默认拆成 8 片并行，每片使用连续且不重叠的 50 个 seed；视觉系统在 CUDA 模式下每片绑定一张 GPU，环境仍在
CPU 上运行。每片完成后写入 `holdout/*/shards/parallel_8/worker_XX/`，最后合并回上面的 canonical 目录，因而最终仍只看
`holdout/*/metrics.json`、`episodes.parquet` 和 `run_manifest.json`。可用 `--holdout-parallelism=N` 调整并发数；CUDA 模式下
`N` 不能超过 `--eval-gpus` 的数量，CPU 推理模式则只受 CPU 资源约束。

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

## 实测结果

### 数据准备与隔离（Closed-loop / Data preparation）

| 项目 | 结果 | 判定 |
| --- | --- | --- |
| 官方 CPU replay | 1000/1000 条，77,976 帧，存活率 100% | 通过 ≥95% |
| 转换动作越界 | 0.00% | 通过 |
| 成功前帧占比 | 93.14% | 通过 ≥85% |
| 首次成功步数 p90 | 85 | 通过 ≤90 |
| 训练/holdout 状态隔离 | 1000 个训练状态与 400 个 holdout 状态无 exact duplicate | 通过 |

产物审计见 `conversion_audit.json`、`action_channel_stats.json` 和 `leakage_report.json`。

### 官方协议 checkpoint 评测（Formal / Closed-loop）

每个训练规模评测 12 个 checkpoint，每个 checkpoint 100 局；按 `success_once` 选模，成功后继续
执行至 horizon 100。

| 训练 demos | 选中 checkpoint | `success_once` | `success_at_end` | 5 次重复 `success_once` |
| ---: | ---: | ---: | ---: | ---: |
| 1000 | `022500` | 100/100（100.0%） | 85/100（85.0%） | 均值 99.8%，范围 99.0%–100.0% |
| 100 | `020000` | 73/100（73.0%） | 36/100（36.0%） | 均值 57.8%，范围 48.0%–67.0% |

选中的实体 checkpoint 位于：

```text
outputs/maniskill/paper_aligned_4d/official/1000/selected/pretrained_model
outputs/maniskill/paper_aligned_4d/official/100/selected/pretrained_model
```

### 固定 CPU holdout（Formal / Closed-loop）

固定使用 `reset(seed=k)`，`k=3000..3399`，400 局，horizon 100；成功后不停机，因此两个成功率都保留。

| 系统 | `success_once` | `success_at_end` | 作用 |
| --- | ---: | ---: | --- |
| diffusion / 1000 demos | 397/400（99.25%） | 313/400（78.25%） | 主视觉结果 |
| diffusion / 100 demos | 206/400（51.5%） | 92/400（23.0%） | 数据量对照 |
| privileged PPO | 300/400（75.0%） | 296/400（74.0%） | 特权参照 |
| random | 0/400（0%） | 0/400（0%） | 随机基线 |

1000-demo 策略有 3/400 条未曾成功；另有部分轨迹曾成功但在剩余 horizon 内离开成功状态，
所以 `success_at_end` 明显低于 `success_once`。

### 与论文的关系

论文 Table III 报告官方 DP RGB PickCube 为 1000 demos `1.00`、100 demos `0.76`。本实验使用
LeRobot diffusion 而非官方 DP 实现，且论文数字是多轮评测中的最大值，因此不声称逐项复现。
本实验的 1000-demo 结果达到同一量级；100-demo 结果明显低于论文参照，说明数据量和实现细节仍有显著影响。

## 指标完整性与限制

- Train loss：训练过程 stdout 曾记录 loss，但未随当前结果目录保留完整、可比较的 loss 序列。
- Open-loop action error：未测量，不作为本实验判据。
- Latency：1000-demo holdout 的 policy forward p50 约 588.6 ms，environment step p50 约 6.2 ms；
  这是 CPU 环境、CUDA 推理的 holdout 口径，不与 7D/GPU/50-step 结果横向比较。
- PPO 使用 privileged state，不是视觉策略的同输入上限，不能把它与 diffusion 的差值直接解释为纯感知代价。
- 4D 结果不能与历史 7D 结果直接做动作维度因果比较，因为专家来源、有效帧比例、控制模式、后端、horizon
  和终止语义均同时变化。

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
tail -f outputs/maniskill/paper_aligned_4d/logs/31_holdout_diffusion_1000_worker_00.log
find outputs/maniskill/paper_aligned_4d/holdout/diffusion_1000/shards/parallel_8 \
  -name metrics.json -printf '%h\n' | sort
find outputs/maniskill/paper_aligned_4d/logs -maxdepth 1 -type f -printf '%f\n' | sort
```

worker 日志中的 `tqdm` 是该片的局部进度（默认 0/50 到 50/50）；总进度可通过已完成的
`worker_XX/metrics.json` 数量和各 worker 日志查看。某个 worker 失败时，重跑同一条总命令会复用已经完整的片，
只补跑缺失片，再重新合并，不会重新做整个 400 局 holdout。

训练有效 batch 必须在日志中看到 `Effective batch size: 32 x 8 = 256`。单独检查某个 checkpoint 时，
使用 `eval_official_protocol.sh`；不要直接把 M0 的 GPU/50-step `eval_closed_loop.sh` 当作这条基线的结果。
