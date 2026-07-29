# 2026-07-27 PushT BCChunk 与 Diffusion 模型规划和训练

## 重要进展

- 确定 PushT Pilot 的主问题：比较确定性 BCChunk 与随机 Diffusion 的 action chunk，
  并研究固定 observation 多样性、同初始状态闭环多样性和采样步数权衡。
- 固定 206 条 episode 的数据拆分：165 train / 20 development / 21 held-out audit；
  Smoke 只使用 train 中固定的 8 条 episode。
- 统一两种模型的输入与 action 语义：2 帧 observation history、16 步内部 horizon、
  执行切片 `[1:9]`，每次输出 8 个二维绝对目标位置。
- 完成与 Diffusion 容量匹配的确定性 BCChunk：使用相同视觉编码器和 temporal U-Net，
  但不加噪、不去噪，单次前向回归完整 clean-action horizon。
- 将 BCChunk 与 Diffusion 的完整模型参数收敛到两个明确的 Python model factory，
  两者继续使用 LeRobot 原始 dataset、optimizer、scheduler、training loop 和 checkpoint。
- 完成两组 1,000-step Smoke 训练：BCChunk 记录 loss 从 0.110 降至 0.018，
  Diffusion 记录 loss 从 0.750 降至 0.064。
- 启动正式 BCChunk 训练并推进到 76,782 step；已保存 75,000-step checkpoint，
  本轮训练随后中断，尚不能登记为完整 200,000-step 正式模型。

## 实验问题与分层

本轮规划围绕三个问题展开：

1. 相同两帧 observation 下，不同 Diffusion 初始 noise 是否产生不同的 action chunks？
2. 相同 PushT 初始状态下，不同 sampling seed 是否产生多种成功完整轨迹？
3. 减少 DDIM denoising steps 后，成功率与 action-chunk 推理延迟如何变化？

实验按以下顺序组织：

- `Smoke`：用 8 条 episode 验证 BCChunk 和 Diffusion 的训练 pipeline；
- `Formal`：用 165 条 train episode 分别训练 200,000 updates；
- development：只用 20 个 environment seed 选择 checkpoint；
- held-out audit：保留 21 条 episode，用于正式分析，不参与训练和模型选择；
- E1/E2/E3：分别回答固定输入多 noise、相同初始状态多 rollout、去噪步数与延迟问题。

Smoke 只证明代码和数据流可以运行，不能作为模型性能或多模态结论。

## 统一模型设置

| 字段 | 设置 |
|---|---|
| Task suite | PushT |
| Observation | 96×96 RGB + 2D agent position |
| Observation history | `n_obs_steps=2` |
| Action space | 2D absolute target position |
| Internal action horizon | `horizon=16` |
| Public action chunk | `[1:9]`，共 8 步 |
| Control frequency | 10 Hz |
| Visual backbone | ResNet-18 |
| Temporal model | 1D U-Net，`down_dims=(512,1024,2048)` |
| Normalization | image mean/std；state/action min-max |
| Episode tail | `drop_n_last_frames=7`，不 mask padding loss |
| Optimizer LR | `1e-4` |
| Scheduler | cosine，500-step warm-up |
| Train seed | 1000 |

BCChunk 和 Diffusion 使用相同数据、预处理、网络容量、action horizon 和 step-zero
共享参数。二者的核心区别是训练目标：

- BCChunk：固定零 trajectory query 和 timestep 0，单次前向预测 clean action，
  使用 clean-action MSE；
- Diffusion：训练时对 action 加 Gaussian noise，预测 epsilon，使用
  epsilon-prediction MSE；推理时执行多步反向去噪。

因此两种 loss 的数值含义不同，不能直接用绝对大小判断哪个模型更好。

## 数据拆分

固定数据版本包含 206 条 episode，按固定 seed 生成：

| Split | Episode 数量 | 用途 |
|---|---:|---|
| Train | 165 | 正式训练 |
| Development | 20 | checkpoint 选择 |
| Held-out audit | 21 | 正式分析，不参与训练或选模 |
| Smoke subset | 8 | 仅验证 pipeline，属于 Train 子集 |

Smoke 子集包含 793 frames；正式 Train 包含 20,610 frames。两个模型都从同一组
成对 step-zero 初始化开始，Smoke checkpoint 不会用于正式训练续跑。

## Smoke 训练结果

模式：`Smoke`。

| 模型 | Episodes / Frames | Updates | Batch size | Step 100 loss | Step 1,000 loss |
|---|---:|---:|---:|---:|---:|
| BCChunk | 8 / 793 | 1,000 | 8 | 0.110 | 0.018 |
| Diffusion | 8 / 793 | 1,000 | 8 | 0.750 | 0.064 |

两组训练都生成了 1,000-step checkpoint，说明两套模型配置可以经过同一个 LeRobot
训练 pipeline 完成反向传播、优化和保存。由于数据和 updates 都很小，这里不比较任务
成功率，也不把 loss 下降解释为模型已经学会 PushT。

## 正式训练进展

模式：`Formal`，未完成。

正式 BCChunk 使用 165 条 episode、20,610 frames 和 200,000-step 目标启动。训练推进至
76,782 step 后中断，最近记录的 clean-action MSE 约为 `0.000–0.001`，已保存
75,000-step checkpoint。

该 checkpoint 只能作为中途训练状态：

- 还未达到预定 200,000 updates；
- 还未经过 development checkpoint selection；
- 仅凭接近零的训练 MSE 不能判断闭环泛化；
- 正式 Diffusion 在本报告统计截止点尚未形成可登记的完整训练结果。

## 指标状态

本报告仅记录模型规划与训练，因此评测字段状态如下：

| 字段 | 当前状态 |
|---|---|
| Train loss | 已记录 Smoke；Formal BCChunk 仅记录中途状态 |
| Open-loop action error | 本报告未评测 |
| Closed-loop success rate | 本报告未评测 |
| Latency | 本报告未评测 |
| Failure cases | Formal BCChunk 在 76,782 step 中断；未形成完整正式模型 |

## 当前结论

截至 2026-07-27，可以确认：

1. BCChunk 与 Diffusion 已形成结构、输入、容量和训练数据受控的主对照。
2. Smoke 证明两个模型都能完成最小训练并保存可加载 checkpoint。
3. 正式 BCChunk 已推进到首个可分析的中间 checkpoint，但正式训练和选模尚未完成。
4. 当前没有足够证据回答 Diffusion 是否学到多种有效行为，也不能比较两种模型的正式
   success、diversity 或 latency。

## 结果落点

- 固定 split：`configs/experiments/pusht_diffusion/split.json`
- BCChunk model factory：
  `src/robot_practice/experiments/pusht_diffusion/model_configs/bc_chunk.py`
- Diffusion model factory：
  `src/robot_practice/experiments/pusht_diffusion/model_configs/diffusion.py`
- Smoke BCChunk：`outputs/pusht_diffusion/smoke/bc_chunk/`
- Smoke Diffusion：`outputs/pusht_diffusion/smoke/diffusion/`
- Formal BCChunk 中途 checkpoint：
  `outputs/pusht_diffusion/pilot/bc_chunk/checkpoints/075000/`
- 实验指南：`reports/guides/pusht_diffusion/README.md`
