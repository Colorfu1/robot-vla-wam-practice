# SmolVLA LIBERO LoRA 与语言增强

## 当前状态

- 状态：`Z0/B0、C1、C2、C3 目标协议和 C3 background retention 均已完成正式评测`
- 环境：LIBERO、SmolVLA、PEFT 和 RTX 3090/BF16 基础能力已具备。
- 已完成：两步 LoRA 训练 Smoke；Hub、本地普通 checkpoint 和本地 PEFT checkpoint
  共用的单任务闭环评测/Rerun 入口；公共 SmolVLA-LIBERO 完整 episode Smoke。
- 已完成：完整 LIBERO-Goal source freeze、B0 background split、目标 action provenance、
  Train/Offline Validation demonstration split、20/30 closed-loop Validation/Test split、
  paraphrase 审核和正式 target split 发布。
- 已完成：B0 37任务 domain adaptation 正式训练、严格 checkpoint selection、冻结
  `025000` 和 background held-out closed-loop test。
- 已完成：C1 Default Non-PEFT 的目标 Train 白名单、task/episode-balanced sampler、
  selected-B0 provenance 校验、8卡 BF16 训练/续训入口和正式训练；统一正式评测已完成。
- 已完成：C1 遗忘诊断的 A/B 学习率曲线与 `k=4/8` background replay 探针；Replay selected
  checkpoint 达到目标 59/60、背景 149/210。`k=2/8` 与 `k=6/8` 仍待执行，结果不进入正式 C1。
- 已完成：C2 Canonical LoRA 的 rank-32 PEFT 预检、同一目标采样协议、8卡 BF16
  训练/adapter 续训入口、正式训练和统一正式评测。
- 已完成：C3 LoRA + Paraphrase 的 split-bound 25/75 instruction selector、逐 rank
  checkpoint 计数、adapter 续训入口、8卡 launcher、正式训练、checkpoint selection 和
  210 条目标协议评测；selected checkpoint 为 `015000`。
- 已完成：B0/C1/C2/C3 共用的 21-job / 210-rollout target protocol evaluator，覆盖
  Canonical、Held-out Paraphrase、Prompt Swap、多 GPU、断点续跑和冻结协议校验；修复后的
  evaluator 已用于 C3 正式评测。
- 当前结果：C1 canonical 87/90（96.7%）、Prompt Swap 首次达成 29/30（96.7%，终态保持
  27/30）；C2 canonical 87/90（96.7%）、Prompt Swap 首次/终态均 28/30（93.3%）。
  详细 Prompt Swap 口径见 [修复后合并结果](../../2026-08-06-PromptSwap修复后合并结果.md)。
- 当前结果：C3 canonical 88/90（97.8%）、held-out paraphrase 67/90（74.4%）、Prompt Swap
  首次达成 30/30（100.0%，终态保持 29/30）。C3 相比 C2 的 held-out paraphrase 提升 53.3 个百分点，
  三个任务均改善。完整结论见 [C3 正式评测结论](../../daily/2026-08-07-C3改写增强LoRA正式评测结论.md)。
- 当前结果：C3 selected `015000` 的 background retention 为 0/210（0.0%），7 个背景任务均为 0/30，
  210 条 rollout 全部以 `time_limit` 结束；B0 基线为 183/210（87.1%）。旧的同名 `c3_selected 0/210`
  产物不计入结论，本次结果来自重跑后的当前 selected adapter。详细结果见
  [C3 background regression 结论](../../daily/2026-08-10-C3背景回归正式评测结论.md)。

## 结果类别标签与数据分工

报告里的标签是三个维度拼起来的：`Offline`/`Closed-loop` 说评测方式，`Smoke`/`Formal` 说
完整程度，`Validation`/`Test` 说用的是允许反复看的选模集还是只看一次的最终集。第三段不在
`reports/AGENTS.md` 的四个基础标签里，是本族的扩展；缺了它读者无法判断一个数字是否被选模
过程看过。

数据分三条互不相干的轴，跨轴不能求交集：

| 轴 | 是什么 | B0（背景 37 任务） | C 家族 + Z0（目标 3 任务） |
| --- | --- | --- | --- |
| 数据侧 demonstration | 录好的轨迹，逐帧含观测与动作。用于训练与开环动作误差 | 训练 1,412 条 / 236,470 帧；离线 validation 138 条 / 23,004 帧（**无入口使用**） | 训练 113 条 / 11,022 帧；离线 validation 30 条 / 2,969 帧（选模第二判据） |
| 仿真侧 init state | LIBERO 每任务 50 个摆法的**编号**，不是数据。闭环按编号摆好物体让模型自己跑，不需要参考轨迹 | 7 任务 × 20 validation，六个 checkpoint 各跑一遍 = `6 × 7 × 20 = 840` 条；test 只跑冻结 winner = `7 × 30 = 210` 条 | 3 任务 × 20 validation，六个 checkpoint = `6 × 3 × 20 = 360` 条；test 由三套协议组成，合计 210 条（见下） |
| 语言侧 instruction | 指令文本 | 只用 canonical | 每任务 1 canonical + 8 train paraphrases（仅 C3 训练用）+ 5 held-out paraphrases（仅评测用） |

四条硬约束：

- 离线 validation 不进训练，train 与 validation episode 有交集会直接报错。
- 每任务的 validation 与 test 初始状态互不重叠，合并恰好覆盖官方 `0..49`。
- test 初始状态与 held-out paraphrases 都不得进入任何选模过程。
- held-out 只针对**初始状态**，不针对任务：闭环评测的任务都在自己的训练任务内。跨任务迁移
  由目标三任务的 Z0 baseline 回答。

关于"未见过"的实测边界：官方 50 个初始状态与 demonstration 的起始状态是同一摆放分布下的
两批独立抽样（三个任务实测均无精确匹配），所以不存在字面意义的状态泄漏；但物体摆放跨度只有
约 3 cm，每个 eval 初始状态到最近训练起点约 2 cm。因此 test 衡量的是同分布、未被使用过的
摆放，不是远距离泛化。

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
| Z0 | B0 在冻结目标 Test states 上执行 | 不训练 | B0 的任务迁移 baseline 结果 |
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
  → 冻结 B0 target baseline workload（正式 Test 与 C1/C2/C3 一起发布结果）
  → C1 / C2 / C3 从同一 B0 checkpoint 训练
  → 各自 validation checkpoint selection
  → Canonical / Held-out Paraphrase / Prompt Swap Pilot
  → experiment_log.md 登记结果
```

`lerobot/smolvla_base` 只初始化 B0；C1/C2/C3 不得绕过 B0 或使用不同 B0 checkpoint。
它自身的 state/action processor 与 LIBERO 不匹配，因此不作为闭环 C0。Phase 0/1
未通过时，不得启动正式训练。统一正式评测入口已经实现；C3 selected checkpoint 已经过
validation selection，不能用 Test 结果补选。当前目标协议和该 winner 的 background retention
都已有正式结果。

实际结果统一登记到 [实验运行记录](../../experiment_log.md)，通用指标定义见
[指标说明](../../metrics.md)。
