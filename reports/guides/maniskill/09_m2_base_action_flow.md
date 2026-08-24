# M2 base Action-only Flow 4D/7D 适配指南

## 状态

M2 base 的离线数据适配与单卡 2-step smoke 已完成；M1-W GPU 7D、projected 4D 和原生 4D
路线均已有独立训练产物，其中 native 4D 路线的 100k、4 卡训练、development 选模和
post-selection test 已完成。本入口与已经冻结的 M2-W 7D 正式 runner 分开：这里复用同一
Action-only Flow trunk，允许数据 action 为 4D 或 7D。

## 官方 anchor 的日志在哪里

官方 StackCube RGB anchor 的终端输出由 runner 写入：

```text
outputs/maniskill/official_stackcube_rgb_anchor/logs/10_train.log
```

如果使用过 retry 输出目录，则对应为：

```text
outputs/maniskill/official_stackcube_rgb_anchor_retry/logs/10_train.log
```

官方训练脚本同时写 TensorBoard event 文件，位置在：

```text
outputs/maniskill/official_stackcube_rgb_anchor/10_train/runs/<experiment_name>/events.out.tfevents.*
```

所以官方不是没有 log：`logs/10_train.log` 是文本日志，`events.out.tfevents.*` 是可用
TensorBoard 读取的标量日志。训练失败时先看文本日志末尾；不能把未完成的官方训练当成结果。

## 数据和模型合同

- 训练数据默认是官方 ManiSkill motionplanning replay 生成的 RGB H5：
  `datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5`。
- H5 中的官方 action 是 4D `pd_ee_delta_pos`；当前本地审计为 990 条完整 replay episode，训练切片取前 100 条，development 取 100 条，test 790 条。
- 输入是两路 `128×128` RGB、最近 2 帧和 18D `agent.qpos(9)+agent.qvel(9)`；不读取目标位姿、成功标志或其他 privileged state。
- 输出为 16 步 action chunk；Flow 训练是随机 `t`、随机高斯源噪声，预测 `x1-x0` velocity，推理固定 Euler/10 steps。
- `ActionFlowConfig` 与 checkpoint normalization 支持 4D 和 7D。4D 对应 `pd_ee_delta_pos`，7D 对应 `pd_ee_delta_pose`。
- 7D 示例配置为 `configs/experiments/maniskill/m2_base_m1w_rgb_7d.json`，读取 M1-W 的 D_core；它不改变 M2-W 冻结 runner 的 7D gate。

### 本次 M1-W GPU 7D 训练

本次 7D 训练直接参考已经完成的
`outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k`，使用
M2 base runner，不使用 M2-W 的 8-GPU formal runner。现成的 7D config 已固定：

- source：M1-W frozen GPU live-RGB replay，`pd_ee_delta_pose` 7D，987 条 episode；
- split：train/development/test = `787/100/100`，与 D_core split manifest 一致；
- 训练：4 GPU、每卡 batch 64、global batch 256、50,000 optimizer steps；
- normalization：只从 train split 统计 image/state/action 的 mean/std，action 为 7 个 channel
  分别统计，std floor 为 `1e-6`；
- 训练阶段只做 offline training；训练完成后再用 M2 base 的 GPU selector 做 development
  选模和 test 评测。

先做远端 4 卡输入检查：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=validate-config \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_7d.json \
  --num-gpus=4
```

启动 7D 训练，输出使用独立目录：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_7d.json \
  --num-gpus=4 \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/formal_seed1000
```

训练完成后，用同一套 M2 base GPU selector 评估 `5 checkpoints × {n_action_steps=4,8}`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_7d.json \
  --train-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/formal_seed1000 \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/development_gpu \
  --num-gpus=4 \
  --num-envs=64 \
  --max-steps=200 \
  --n-action-steps=4,8
```

development 选模通过后，再执行独立 test 评测：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=evaluate-test \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_7d.json \
  --selection-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/development_gpu \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/test_gpu \
  --num-gpus=4 \
  --num-envs=64 \
  --max-steps=200
```

### M1-W GPU 7D 已完成结果

这条 M2 Base 7D route 使用 987 条 D_core，train/development/test 为 `787/100/100`，4-GPU
DDP、global batch `256`、50k steps。development 选模选择 step 30k、`n_action_steps=8`：
`18/100` success-once、`18/100` success-at-end；`ever_grasped=95%`、`ever_on_cube=28%`。
训练 loss 的 last 值为 `0.07552`，最后 1,000 steps 均值为 `0.07153`。当前这条 route 只报告
development，未把 test 访问写入正式结果。

产物：

```text
outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/formal_seed1000/loss_history.jsonl
outputs/maniskill/m2_base_action_flow/m1w_rgb_7d/development_gpu/selected_checkpoint.json
```

### M1-W GPU 7D projected 4D success-at-end 路线

这条路线读取同一份 M1-W 7D H5，但监督 action 只取 `[0,1,2,6]`，使用 4D
`pd_ee_delta_pos` model/config/pipeline；它不是重新生成的 native 4D 数据。为了只保留原始
`success_at_end` source，episode index 为 train/development/test = `668/100/100`。训练与
评测仍使用 4 GPU、global batch `256`、50k steps 和 `physx_cuda`，不访问 native-4D 数据目录。

配置检查与训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=validate-config \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_4d_success_at_end.json \
  --num-gpus=4

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_4d_success_at_end.json \
  --num-gpus=4 \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/formal_seed1000
```

development 选模和 held-out test：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_4d_success_at_end.json \
  --train-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/formal_seed1000 \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/development_gpu \
  --num-gpus=4 \
  --num-envs=64 \
  --max-steps=200 \
  --n-action-steps=4,8

CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=evaluate-test \
  --config=configs/experiments/maniskill/m2_base_m1w_rgb_4d_success_at_end.json \
  --selection-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/development_gpu \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/test_gpu \
  --num-gpus=4 \
  --num-envs=64 \
  --max-steps=200
```

已完成结果：selected step 50k、`n_action_steps=8`；development 为 `22/100 / 22/100`，
held-out test 为 `21/100 / 21/100`。训练 loss 的 last 值为 `0.04619`，最后 1,000 steps
均值为 `0.04240`；development/test 的 truncation 分别为 78/100、79/100。结果产物为：

```text
outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/formal_seed1000/loss_history.jsonl
outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/development_gpu/selected_checkpoint.json
outputs/maniskill/m2_base_action_flow/m1w_rgb_4d_success_at_end/test_gpu/metrics.json
```

### M1-W 7D→4D 全量 replay 诊断

在不做模型推理的情况下，对 D_core 987 条 source episode 只取 action `[0,1,2,6]`，使用
`pd_ee_delta_pos`、`physx_cuda`、50 个并行 environments 做完整 replay。结果为
`911/987` success-once（92.30%）和 `844/987` success-at-end（85.51%）；其中 67 条曾经
success 但没有在 episode 结束时保持。development/test/train 分别为 `95/100 / 92/100`、
`90/100 / 84/100`、`726/787 / 668/787`。这是 replay/protocol 诊断，不是 learned-policy
结果；它说明 action projection 大体可执行，不能说明对应 RGB/state 下的 policy 已经学会该映射。

完整记录：

```text
outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/action_projection_0126_full/metrics.json
```

### M1-W 原生 4D 路线

`configs/experiments/maniskill/m2_base_m1w_native_rgb_4d.json` 使用
`outputs/maniskill/m1_wam_native_4d/StackCube-v1/06_final_live_rgb/replay.h5`，不是把 7D
标签投影成 4D。数据是 M1-W native `pd_ee_delta_pos` action、双 RGB、18D
`qpos+qvel`；最终 H5 有 946 条成功 episode。训练沿用原 M1-W 的 source split，并过滤
native-4D 重放后不存在的 episode，实际划分为 train/development/test = 748/95/96。

这条路线复用同一个 `ActionOnlyFlowModel` 和线性 Flow Matching 目标，但不修改冻结的
M2-W 7D 配置。先做输入检查：

```bash
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=validate-config \
  --config=configs/experiments/maniskill/m2_base_m1w_native_rgb_4d.json
```

远端 4 卡正式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_base_m1w_native_rgb_4d.json \
  --num-gpus=4 \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000
```

该命令只训练，不做 rollout、选模或 test 评测；global batch 为 256，每卡 64，训练
50,000 steps，每 10,000 steps 保存 checkpoint。训练日志和 loss 在 output root 下的
`training.log` 与 `loss_history.jsonl`。

训练器按窗口懒读 H5，不把整个 H5 放进内存；正式每个 rank 使用 4 个 persistent DataLoader
worker，worker 内复用只读 H5 handle。训练只读 train split，脚本不会 rollout、选 checkpoint
或访问 test split。

## 本地验证

本地先做配置和输入审计：

```bash
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=validate-config \
  --config=configs/experiments/maniskill/m2_base_official_rgb_4d.json
```

单卡 smoke 只验证读取、归一化、4D head、checkpoint 写入和训练日志，不是实验指标：

```bash
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=smoke \
  --config=configs/experiments/maniskill/m2_base_official_rgb_4d.json \
  --num-gpus=1 \
  --output-root=outputs/maniskill/m2_base_action_flow/official_rgb_4d/smoke
```

## 远端 4 卡正式训练

先确认远端已经同步官方 RGB H5、sidecar JSON 和本指南中的两个 config。正式训练使用 4 卡、
每卡 batch 64，合计 global batch 256；每 10,000 steps 保存一个实体 checkpoint，共 50,000
steps。它只训练，不做 rollout 和选模；中途失败时已写出的 checkpoint 和 loss 不会被后续
评测覆盖。当前入口先不宣称自动 resume，恢复训练需要显式增加恢复流程：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_base_official_rgb_4d.json \
  --num-gpus=4 \
  --output-root=outputs/maniskill/m2_base_action_flow/official_rgb_4d/formal_seed1000
```

训练日志和 loss：

```text
outputs/maniskill/m2_base_action_flow/official_rgb_4d/formal_seed1000/training.log
outputs/maniskill/m2_base_action_flow/official_rgb_4d/formal_seed1000/loss_history.jsonl
outputs/maniskill/m2_base_action_flow/official_rgb_4d/formal_seed1000/checkpoint_step_00010000.pt
...
outputs/maniskill/m2_base_action_flow/official_rgb_4d/formal_seed1000/checkpoint_step_00050000.pt
```

## development 选模

M2 base 有独立的 development selector，不复用 M2-W 的 7D selector。它读取配置中
development 的初始 `env_states`，不使用 test split，评估每个 checkpoint ×
`n_action_steps={4,8}`，排序为：`success_at_end`、`success_once`、on-cube rate、首次成功
步数、checkpoint step、action steps。

评测后端由 config 的 `protocol.backend` 强制决定，避免数据和 rollout 后端错配：

- 官方 motionplanning RGB 数据：`physx_cpu`，使用 `--num-workers` 并行 CPU 进程；
- M1-W native RGB 数据：`physx_cuda`，使用 `--num-gpus` 启动每卡一个 GPU worker，
  每个 worker 内以 `--num-envs` 批量运行。`CUDA_VISIBLE_DEVICES` 中的 GPU 数量必须不少于
  `--num-gpus`。每个 spawned worker 会进一步收窄为一张物理卡，在 worker 内统一使用逻辑
  `cuda:0`；这是为了避免 ManiSkill reset 内部的 `torch.random.fork_rng()` 误触碰父进程可见的
  其他 GPU。

训练完成后执行：

```bash
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_base_official_rgb_4d.json \
  --train-root=outputs/maniskill/m2_base_action_flow/official_rgb_4d/formal_seed1000 \
  --output-root=outputs/maniskill/m2_base_action_flow/official_rgb_4d/development \
  --num-workers=4 \
  --max-steps=200
```

M1-W native 4D 的 100k 训练结果使用 GPU PhysX 评测：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
bash scripts/experiments/maniskill/run_m2_base_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_base_m1w_native_rgb_4d_100k.json \
  --train-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k \
  --output-root=outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/development_gpu \
  --num-gpus=4 \
  --num-envs=64 \
  --max-steps=200 \
  --n-action-steps=4,8
```

### 已完成的 M1-W 原生 4D 100k 结果

- 训练：4 GPU、global batch `256`、100k steps；source 为 946 条 native-4D live-RGB episode，
  split 为 train/development/test = 748/95/96。
- 选模：step 100k、`n_action_steps=4`；GPU development 为 `83/95` success-once、
  `83/95` success-at-end（87.4% / 87.4%）。
- post-selection test：`82/96` success-once、`82/96` success-at-end（85.4% / 85.4%）。
- train mean tail loss 为 `0.02532`；test 评测使用 4 GPU、64 environments、`physx_cuda`，不是
  official CPU rollout。

结果产物：

```text
outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/run_manifest.json
outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/development_gpu/selected_checkpoint.json
outputs/maniskill/m2_base_action_flow/m1w_native_rgb_4d/formal_seed1000_100k/test_gpu/metrics.json
```

选模产物：

```text
outputs/.../development/selected_checkpoint.json
outputs/.../development/selected_checkpoint.pt
outputs/.../development/candidate_metrics.json
outputs/.../development/candidates/step_*/metrics.json
```

`--policy-device=auto`（默认）会跟随 `protocol.backend` 选择 CPU 或 CUDA；显式指定冲突的
设备会直接报错。旧 selector 把所有配置硬编码成 CPU，因此此前对 M1-W native 数据得到的
CPU rollout 只能作为诊断，不能作为正式 GPU 结果。当前 4D 训练结果不能拿 M2-W 的 7D
evaluator 推理，动作维度和控制协议必须一致。GPU selector 会在每个 candidate 内按
仿真步数和已完成 episode 输出进度，避免长时间只有 candidate 完成时才有日志。
