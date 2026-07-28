# PushT Diffusion 多模态与采样速度 Pilot

## 实验目标

本实验回答三个问题：

1. 完全相同的两帧 observation 下，仅改变 Diffusion 初始 noise，是否会产生不同的
   8-step action chunk？
2. 完全相同的 PushT 初始状态下，不同 policy sampling seed 是否能产生多种成功闭环轨迹？
3. 减少 DDIM 去噪步数后，成功率与 action-chunk 推理延迟如何变化？

主对照是确定性的 `BCChunkPolicy`。公开的 `lerobot/diffusion_pusht` 只作为外部兼容性
参考，不与本地固定 split 做公平模型对比。

## 训练配置与入口

模型配置是明确的 Python factory，不从 Bash 或隐藏 JSON 中拼装：

```text
BCChunk:
src/robot_practice/experiments/pusht_diffusion/model_configs/bc_chunk.py

Diffusion:
src/robot_practice/experiments/pusht_diffusion/model_configs/diffusion.py
```

两个模型训练 worker 分别是：

```text
robot_practice.experiments.pusht_diffusion.train_bc_chunk
robot_practice.experiments.pusht_diffusion.train_diffusion
```

这两个文件是唯一实际执行模型训练的入口；二者都直接构造 LeRobot 官方 `DatasetConfig` 和
`TrainPipelineConfig`，并在各自的 `main()` 中明确调用
`lerobot.scripts.lerobot_train.train(cfg)`。共享的参数解析、路径解析和输入校验位于
`training_common.py`，该文件没有 `main()`，不能作为训练入口启动。dataset、policy、
optimizer、scheduler、training loop 和 checkpoint 仍全部使用 LeRobot 原始 pipeline。

`train_bc_chunk.sh` 和 `train_diffusion.sh` 只负责进入 `robot-practice` conda 环境并把
参数转交给对应 Python module。使用 `--print-command` 可以在不创建输出、不加载数据、
不训练的情况下查看最终命令和 resolved configuration。

单机多卡使用两个独立 launcher：

```text
scripts/experiments/pusht/train_bc_chunk_multi_gpu.sh
scripts/experiments/pusht/train_diffusion_multi_gpu.sh
```

它们只使用 Accelerate 启动多个 worker，最终仍分别进入上面的
`train_bc_chunk` / `train_diffusion`，不会改变模型配置、checkpoint 格式或后续评测入口。
多卡命令额外要求：

```text
--num-gpus=N            必填且 N >= 2；使用当前可见 GPU 中的前 N 张
```

两个入口使用相同参数：

```text
--mode=smoke|pilot       必填
--steps=N                可选；默认 smoke=1000、pilot=200000
--train-seed=N           默认 1000
--split-file=PATH        默认使用本实验固定 split
--output-dir=PATH        可选；默认生成带时间戳的新目录
--batch-size=N           可选；默认 smoke=8、pilot=64
--num-workers=N          默认 4；VSCode Debug 使用 0
--device=auto|cpu|cuda|mps
--print-command          只打印命令和 resolved configuration
```

`--batch-size` 在单卡和多卡下都表示每个训练进程的 batch size。多卡时：

```text
global batch size = --batch-size × --num-gpus
```

例如双卡 `--batch-size=32` 的全局 batch 是 64；双卡
`--batch-size=64` 的全局 batch 是 128。脚本不会自动缩放。现有单卡命令和所有
E1/E2/E4 推理命令保持不变。

## 固定实验约束

- 数据共 206 条 episode：165 train / 20 development / 21 held-out audit。
- Smoke 只使用 train 中固定的 8 条 episode，仅验证 pipeline。
- `n_obs_steps=2`，内部 action horizon 为 16，执行切片为 `[1:9]`，每次执行 8 步。
- BCChunk 与 Diffusion 使用相同输入、训练样本统计和逐 tensor 一致的 step-zero 初始化。
- PushT action 是二维绝对目标位置，不是速度。
- 当前 Pilot 不恢复 dataset 中间帧，也不实现完整 Pymunk snapshot。

## 术语

- `num_inference_steps`：生成一条 action chunk 时，Diffusion 内部执行的去噪次数。
- `num_action_samples`：对同一个固定 observation 独立生成的候选 chunk 数量。
- `num_rollouts`：分别 reset 环境并真正执行的完整 episode 数量。

## 执行顺序

必须按下面的依赖顺序执行：

1. **环境检查**：执行 [Smoke 指南](smoke.md) 的接口检查，确认 LeRobot、PushT、
   Pymunk、显式 noise 和环境 reset 可以复现。
2. **数据与 step-zero 准备**：执行 `prepare_dataset.sh`。这一步下载/校验数据、创建
   Smoke/Pilot 数据视图，并使用上述两个 Python model factory 创建成对初始化权重；
   不进行训练。
3. **E0-lite 数据检查**：执行下面的必需命令，生成后续 Smoke E1 和正式 E1 共用的
   `candidates.json`：

   ```bash
   bash scripts/experiments/pusht/inspect_demonstrations.sh \
     --num-episodes=50 \
     --num-candidates=10 \
     --output-dir=outputs/pusht_diffusion/data_inspection
   ```

   具体输出和检查内容见
   [E0 数据与 demonstration 检查](e0_data_inspection.md)。未生成该文件时，
   `smoke_eval.sh` 会在加载模型前退出并打印上述恢复命令。
4. **Smoke 小训练与评测**：回到 [Smoke 指南](smoke.md)，分别训练 BCChunk 和
   Diffusion 1,000 updates，再执行缩小版 E1/E2。Smoke 只验证 pipeline。
5. **可选外部兼容性参考**：按
   [正式训练与 benchmark](benchmark.md) 运行公开预训练模型；该结果不进入本地 split
   的公平比较。
6. **正式训练**：从 Pilot step-zero 权重分别训练 BCChunk 和 Diffusion 200,000 updates。
   不得续训 Smoke checkpoint。
7. **Development checkpoint 选择**：只使用 20 个 development environment seed，
   按 [benchmark 指南](benchmark.md) 选出两个正式 checkpoint。
8. **正式实验项**：依次执行
   [E1 固定 observation 多 noise 采样](e1_fixed_observation.md)、
   [E2 相同初始环境多 seed rollout](e2_multi_seed_rollout.md) 和
   [E4 去噪步数与延迟](e4_sampling_speed.md)。
9. **生成总报告**：E0/E1/E2/E4 全部完成后，运行 `make_report.sh` 汇总正式结果。

依赖关系可以简写为：

```text
环境检查
  → 数据与 step-zero 准备
  → E0 candidates
  → Smoke 训练与评测
  → Pilot 正式训练
  → Development checkpoint 选择
  → E1 / E2 / E4
  → 正式总报告
```

## 统一实验字段

- Task suite：PushT
- Dataset size：206 episodes；165 train / 20 development / 21 audit
- Observation keys：96×96 RGB、二维 agent position；历史长度 2
- Action space：二维绝对 target position
- Action horizon：内部 16，执行 `[1:9]` 共 8 步
- Control frequency：10 Hz
- Train loss：正式训练后记录
- Open-loop action error/diversity：checkpoint selection 与 E1 后记录
- Closed-loop success rate：E2/E4 后记录
- Latency：E4 后记录
- Failure cases：闭环 rollout 后按 environment seed、policy seed 和轨迹记录

## 当前状态

代码、配置和启动入口已经准备完成，尚未运行 Smoke 或正式训练，因此当前没有可登记的
PushT 实验结论。首次执行从 [Smoke 指南](smoke.md) 第 1 步开始。
