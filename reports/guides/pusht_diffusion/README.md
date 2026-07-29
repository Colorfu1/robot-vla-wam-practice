# PushT Diffusion 多模态与采样速度 Pilot

## 实验目标

本实验回答四个问题：

1. 完全相同的两帧 observation 下，仅改变 Diffusion 初始 noise，是否会产生不同的
   8-step action chunk？
2. 完全相同的 PushT 初始状态下，不同 policy sampling seed 是否能产生多种成功闭环轨迹？
3. 减少 DDIM 去噪步数后，成功率与 action-chunk 推理延迟如何变化？
4. 从现有 demonstrations 提取的全局 trajectory prototype，能否作为离散 condition
   显式控制动作和完整闭环行为；NULL 分支是否覆盖多个成功 prototype？

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
E1/E2/E3 推理不使用这个训练 batch-size 语义；E2/E3 通过各自的 `--num-gpus`
独立控制闭环评测 worker 数量。

## 固定实验约束

- 数据共 206 条 episode：165 train / 20 development / 21 held-out audit。
- Smoke 只使用 train 中固定的 8 条 episode，仅验证 pipeline。
- `n_obs_steps=2`，内部 action horizon 为 16，执行切片为 `[1:9]`，每次执行 8 步。
- BCChunk 与 Diffusion 使用相同输入和训练样本统计；两者分别由 `train_seed` 直接随机
  初始化，不要求初始参数逐 tensor 一致。
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
2. **数据准备**：执行 `prepare_dataset.sh`。这一步下载/校验数据，并创建
   Smoke/Pilot 数据视图和 padding statistics；不生成模型权重。
3. **E0-lite 数据检查**：执行下面的必需命令，生成后续 Smoke E1 和正式 E1 共用的
   `candidates.json`：

   ```bash
   bash scripts/experiments/pusht/inspect_demonstrations.sh \
     --num-episodes=206 \
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
6. **正式训练**：BCChunk 和 Diffusion 分别从 `train_seed` 控制的随机权重训练
   200,000 updates，不读取 step-zero 或 Smoke checkpoint。
7. **Development checkpoint 选择**：只使用 20 个 development environment seed，
   按 [benchmark 指南](benchmark.md) 选出两个正式 checkpoint。入口默认单卡，
   可通过 `--num-gpus=N` 将不同 checkpoint 分发给 N 张可见 CUDA GPU 并行评测。
8. **Development 逐 seed 对比**：按
   [Development checkpoint 对比](development_comparison.md) 在同一张目标 GPU 上
   复跑选中的两个模型，保存四类 seed、Parquet、视频，以及每个模型各 20 个独立的
   `env_<seed>.rrd`。目录启动器会把同一模型的 RRD 一次加载到一个 Viewer，并通过
   稳定的 `/rollout/*` Blueprint 切换。
9. **正式实验项**：依次执行
   [E1 固定 observation 多 noise 采样](e1_fixed_observation.md)、
   [E2 相同初始环境多 seed rollout](e2_multi_seed_rollout.md) 和
   [E3 去噪步数与延迟](e3_sampling_speed.md)。
10. **生成总报告**：E0/E1/E2/E3 全部完成后，运行 `make_report.sh` 汇总正式结果。
11. **E4 同中心纯旋转场景（Smoke）**：按
    [E4 同中心纯旋转场景](e4_rotation_only.md) 验证初始 T 与 180° 目标 T 共用几何中心，
    并确认原始重合度 success 判定正确。现有 DP-175000 已完成 5 条 OOD 诊断 rollout，
    但仍不含 controller、配对数据或 E4 训练。
12. **E5 Prototype-ID 条件控制**：按
    [E5 Prototype-ID 条件控制](e5_mode_conditioned.md) 从现有 train demonstrations
    提取 K-medoids prototypes，训练 Mode-BCChunk/Mode-Diffusion，并执行固定
    observation 条件切换与完整闭环 Prototype/NULL 网格。实现与 Smoke 已完成，正式
    训练和 880 条闭环 rollout 尚未执行。

依赖关系可以简写为：

```text
环境检查
  → 数据准备
  → E0 candidates
  → Smoke 训练与评测
  → Pilot 正式训练
  → Development checkpoint 选择
  → Development 逐 seed 对比
  → E1 / E2 / E3
  → 正式总报告
  → E4 同中心纯旋转场景 Smoke
  → E5 Prototype-ID 条件控制
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
- Closed-loop success rate：E2/E3 后记录
- Latency：E3 后记录
- Failure cases：闭环 rollout 后按 environment seed、policy seed 和轨迹记录

## 当前状态

E1 已以 `Offline / Formal / Stage complete` 结束：完整 DDPM sampling stream 会稳定
改变 action chunk，但平均轨迹差异约 1.946 px。E2a/E2b 已从空目录完成正式闭环执行；
E2b 使用 20 个环境 × 10 个 policy seed，并把每条完整 rollout 保存为可独立切换的
Rerun Recording。主阈值 0.05 下有 14/20 个环境出现至少两个成功 mode。E3
采样步数与延迟实验也已完成；E4 已完成场景 Smoke 和 5 条 DP OOD 诊断。E5 已完成
K=3 prototype fit、条件模型与评测入口，并通过 1-step/一帧 Smoke；尚无正式训练或
能力结论。
E3 已支持通过 `--num-gpus=N` 按 environment seed 并行执行；每个场景输出一个包含六个
同步采样条件的 active-Blueprint Recording。E3 的 300/300 条正式闭环 rollout 和
latency benchmark 均已完成：DDIM-5 以 58% 成功率和 47.7 ms end-to-end p95 获得
相对 DDPM-100 的 12.57× 加速，是当前最佳 success-latency 配置。
