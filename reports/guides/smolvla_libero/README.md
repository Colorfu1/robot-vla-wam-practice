# SmolVLA LIBERO LoRA 与语言增强

## 当前状态

- 状态：`Planned`
- 环境：LIBERO、SmolVLA 和 RTX 3090/BF16 基础能力已具备；PEFT 依赖和正式实验入口未就绪。
- 结果：尚未训练或评测，本目录只记录实验协议。
- 首个动作：完成环境与两步 LoRA Smoke。

## 实验目标

在 LIBERO-Goal 的同构场景中回答：

1. LoRA 能否以较少可训练参数完成 SmolVLA 任务适配？
2. Paraphrase augmentation 能否改善 held-out instruction 下的闭环表现？
3. 模型是否真正服从语言目标，而不是始终执行同一个放置策略？

## 核心对照

| 配置 | 训练方式 | Instruction | 作用 |
| --- | --- | --- | --- |
| C0 | 不训练 | Canonical | Base contract Smoke |
| C1 | Default Non-PEFT | Canonical | 非 PEFT 性能对照 |
| C2 | LoRA，rank 32 | Canonical | LoRA 任务适配 |
| C3 | LoRA，rank 32 | Canonical + train paraphrases | 语言增强 |

C1 保留 SmolVLA 默认的 `train_expert_only=true` 和
`freeze_vision_encoder=true`，不是 full-model fine-tuning。

## Guide 导航

按以下顺序执行：

1. [准备工作与 Smoke](00_preparation.md)
   - 环境审计、依赖、feature/stat contract、两步 LoRA 与 C0 闭环 Smoke
2. [数据、Split 与 Paraphrase 准备](01_data_preparation.md)
   - 三个任务、provenance、30/10/10 split 和语言表
3. [C2 Canonical LoRA](02_c2_canonical_lora.md)
   - 第一个正式训练配置
4. [C3 LoRA + Paraphrase](03_c3_paraphrase_lora.md)
   - 只改变训练 instruction
5. [C1 Default Non-PEFT](04_c1_non_peft.md)
   - 资源允许时补充的非 PEFT 对照
6. [闭环评测与判断标准](05_evaluation.md)
   - Canonical、Held-out Paraphrase、Prompt Swap、结果表和结论边界

## 固定公共约束

- Task suite：LIBERO-Goal，task IDs 1/4/8
- Base checkpoint：`lerobot/smolvla_base`
- Observation：两路 256×256 RGB + 8 维 state
- Action：7 维相对末端动作
- Control frequency：10 Hz
- Prediction horizon：50 steps
- Closed-loop execution horizon：10 steps，每 1 秒重新规划
- Observation horizon：`n_obs_steps=1`
- Train seed：1000
- Validation policy seed：1500
- Test policy seed：2000
- Test init states 只运行一次；查看结果后不得追加同一 test 的配置或 rollout

## 执行顺序

```text
准备环境与两步 LoRA Smoke
  → C0 单条闭环 Smoke
  → provenance-aware split 与 paraphrase 审计
  → C2 Canonical LoRA
  → validation checkpoint selection
  → C3 LoRA + Paraphrase
  → validation checkpoint selection
  → C1 Default Non-PEFT（资源允许）
  → Canonical / Held-out Paraphrase / Prompt Swap Pilot
  → experiment_log.md 登记结果
```

Phase 0/1 未通过时，不得启动 30,000-step 训练。尚不存在的脚本均在对应 guide 中标为
`Planned / CLI 未实现 / 当前不可运行`。

实际结果统一登记到 [实验运行记录](../../experiment_log.md)，通用指标定义见
[指标说明](../../metrics.md)。
