# SmolVLA LIBERO LoRA 与语言增强

## 当前状态

- 状态：`Phase 1 数据准备已完成`
- 环境：LIBERO、SmolVLA、PEFT 和 RTX 3090/BF16 基础能力已具备。
- 已完成：两步 LoRA 训练 Smoke；Hub、本地普通 checkpoint 和本地 PEFT checkpoint
  共用的单任务闭环评测/Rerun 入口；公共 SmolVLA-LIBERO 完整 episode Smoke。
- 已完成：完整 LIBERO-Goal source freeze、B0 background split、目标 action provenance、
  Train/Offline Validation demonstration split、20/30 closed-loop Validation/Test split、
  paraphrase 审核和正式 target split 发布。
- 已完成：B0 37任务 domain adaptation 训练入口、8卡 BF16 启动、逐 step 指标和
  checkpoint resume；本地真实数据两步 Smoke 及 step 2→3 中断续训均已通过，正式训练
  尚未启动。
- 已完成：C1 Default Non-PEFT 的目标 Train 白名单、task/episode-balanced sampler、
  selected-B0 provenance 校验、8卡 BF16 训练/续训入口和单步 VS Code debug 配置；正式训练
  尚未启动。
- 已完成：C2 Canonical LoRA 的 rank-32 PEFT 预检、同一目标采样协议、8卡 BF16
  训练/adapter 续训入口和单步 VS Code debug 配置；正式训练尚未启动。
- 已完成：C3 LoRA + Paraphrase 的 split-bound 25/75 instruction selector、逐 rank
  checkpoint 计数、adapter 续训入口、8卡 launcher 和单步 VS Code debug 配置；正式训练
  尚未启动。
- 未完成：C1/C2/C3 正式训练和多协议正式评测。
- 启动前动作：在训练环境确认 selected B0、其 provenance 和 8 张可见 GPU 均已就位，再运行
  对应的一键 launcher。

## 实验目标

在 LIBERO-Goal 的同构场景中回答：

1. 37 个非目标 LIBERO 任务能否把原始 SmolVLA 适配到 LIBERO 域？
2. LoRA 能否以较少可训练参数适配三个 held-out 目标任务？
3. Paraphrase augmentation 能否改善 held-out instruction 下的闭环表现？
4. 模型是否真正服从语言目标，而不是始终执行同一个放置策略？

## 核心对照

| 配置 | 数据 | 训练方式 | 作用 |
| --- | --- | --- | --- |
| C0 | 公共 checkpoint | 不训练 | 闭环接口与可视化 Smoke |
| B0 | 37 个非目标任务 | Default Non-PEFT | LIBERO 域适配 |
| Z0 | 目标 Validation states | 不训练 | B0 的任务迁移基线 |
| C1 | 目标 Train split | Default Non-PEFT | 普通适配对照 |
| C2 | 目标 Train split | LoRA，canonical | 参数高效适配 |
| C3 | 目标 Train split | LoRA + paraphrase | 语言增强 |

C1 保留 SmolVLA 默认的 `train_expert_only=true` 和
`freeze_vision_encoder=true`，不是 full-model fine-tuning。

## Guide 导航

按以下顺序执行：

1. [准备工作与 Smoke](00_preparation.md)
   - 环境审计、依赖、feature/stat contract、两步 LoRA 与 C0 闭环 Smoke
2. [数据、Split 与 Paraphrase 准备](01_data_preparation.md)
   - 三个任务、action provenance、demonstration split、20/30 closed-loop split 和语言表
3. [B0 Domain Adaptation](02_b0_domain_adaptation.md)
   - 使用 `background_split.json` 训练统一 B0 checkpoint；支持 Smoke、8卡正式训练和续训
4. [C1 Default Non-PEFT](03_c1_non_peft.md)
   - 从 B0 开始的普通适配对照
5. [C2 Canonical LoRA](04_c2_canonical_lora.md)
   - 从同一 B0 开始的 LoRA 对照
6. [C3 LoRA + Paraphrase](05_c3_paraphrase_lora.md)
   - 只改变训练 instruction
7. [闭环评测与判断标准](06_evaluation.md)
   - Canonical、Held-out Paraphrase、Prompt Swap、结果表和结论边界

## 固定公共约束

- Task suite：LIBERO-Goal，task IDs 1/4/8
- B0 initialization：本地 materialized `smolvla_base` 实体副本
- C1/C2/C3 initialization：同一个 selected B0 checkpoint
- C0 Smoke policy：`HuggingFaceVLA/smolvla_libero`
- Observation：两路 256×256 RGB + 8 维 state
- Action：7 维相对末端动作
- Control frequency：10 Hz
- Prediction horizon：50 steps
- Closed-loop execution horizon：10 steps，每 1 秒重新规划
- Observation horizon：`n_obs_steps=1`
- Train seed：1000
- Validation policy seed：1500
- Test policy seed：2000
- C1/C2/C3 正式训练：单机 8 卡、每卡一个 Accelerate process、每卡 batch 1、BF16
- Smoke/debug：可用 `--num-processes=N` 改为单卡或单机多卡
- Resume：恢复原 run 的 world size，续训时不能改变 GPU 数；当前不支持跨节点训练
- 每个预注册 Test protocol 在 frozen Test states 上只运行一次；查看结果后不得追加新
  prompt、配置或 rollout

## 执行顺序

```text
准备环境与两步 LoRA Smoke
  → C0 单条闭环 Smoke
  → B0/Target splits 与 paraphrase 审计
  → B0 37任务 domain adaptation
  → background validation checkpoint selection
  → Z0 目标任务迁移基线
  → C1 / C2 / C3 从同一 B0 checkpoint 训练
  → 各自 validation checkpoint selection
  → Canonical / Held-out Paraphrase / Prompt Swap Pilot
  → experiment_log.md 登记结果
```

`lerobot/smolvla_base` 只初始化 B0；C1/C2/C3 不得绕过 B0 或使用不同 B0 checkpoint。
它自身的 state/action processor 与 LIBERO 不匹配，因此不作为闭环 C0。Phase 0/1
未通过时，不得启动正式训练。尚不存在的 checkpoint selection 和正式评测脚本均在
对应 guide 中标为
`Planned / CLI 未实现 / 当前不可运行`。

实际结果统一登记到 [实验运行记录](../../experiment_log.md)，通用指标定义见
[指标说明](../../metrics.md)。
