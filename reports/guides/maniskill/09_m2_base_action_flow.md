# M2 base Action-only Flow 4D/7D 适配指南

## 状态

M2 base 的离线数据适配与单卡 2-step smoke 已完成；正式 4 卡训练尚未在本地启动。
本入口与已经冻结的 M2-W 分开：M2-W 仍严格使用自己的 7D GPU 协议，本入口只是复用同一
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

M2 base 有独立的 CPU development selector，不复用 M2-W 的 GPU state-bank selector。它读取
官方 RGB H5 中 development 的 100 条初始 `env_states`，在 ManiSkill3 CPU 环境中重新渲染
RGB，评估 5 个 checkpoint × `n_action_steps={4,8}`，不使用 test split。排序为：
`success_at_end`、`success_once`、on-cube rate、首次成功步数、checkpoint step、action steps。

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

选模产物：

```text
outputs/.../development/selected_checkpoint.json
outputs/.../development/selected_checkpoint.pt
outputs/.../development/candidate_metrics.json
outputs/.../development/candidates/step_*/metrics.json
```

selector 默认把策略推理也放在 CPU，和官方 CPU attribution protocol 一致；`--num-workers`
是 CPU 评测进程数，不是 GPU 数。当前 4D 训练结果不能拿 M2-W 的 7D evaluator 推理，动作
维度和控制协议必须一致。
