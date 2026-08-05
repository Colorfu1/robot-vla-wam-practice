# 2026-08-05 C2 Canonical LoRA 正式评测结论

## 重要进展

- C2（Canonical LoRA，selected `020000`）完成训练、选模与两类正式评测：目标协议 210 条与
  背景回退 210 条，均 `status=complete`，协议 contract 与 C1/Z0 除策略字段外逐项相同。
- **目标任务与 C1 打平**：canonical 都是 87/90 = 96.7%，Prompt Swap 依从率都是 18/30，
  混淆矩阵逐格相同。
- **语言泛化反而更差**：held-out paraphrase 19/90 = 21.1%，低于 C1 的 30/90 = 33.3%；
  `para_gap` 从 C1 的 +0.6333 升到 +0.7556，两者 bootstrap 95% CI 不重叠
  （C1 `[0.5222, 0.7333]`，C2 `[0.6667, 0.8444]`）。
- **LoRA 没有防住灾难性遗忘**：背景任务 8/210 = 3.8%，相对 B0 记录的 183/210 保留率仅 4.4%；
  七个背景任务中四个仍为 0/30。C1 是 0/210，C2 只是从"全零"回到"接近全零"。
- 因此 C1 guide 里 TODO-2（改用 LoRA 缓解遗忘）的假设**已验证为负**：低秩增量加主干冻结不足以
  保住旧能力。
- C2 的推理成本反而更高：模型查询 p50 376.8 ms 对 C1 的 285.3 ms，动作 jerk 0.0588 对 0.0285，
  平均 gripper 切换 10.52 对 3.76。

## 核心结论

### 1. 目标任务上 C2 与 C1 无差别

| 协议 | Z0（B0 零样本） | C1（Non-PEFT） | C2（Canonical LoRA） |
|---|---:|---:|---:|
| canonical | 0/90 = 0.0% | 87/90 = **96.7%** | 87/90 = **96.7%** |
| held-out paraphrase | 2/90 = 2.2% | 30/90 = 33.3% | 19/90 = **21.1%** |
| prompt compliance | 0/30 = 0.0% | 18/30 = 60.0% | 18/30 = 60.0% |

canonical 完全打平，逐任务分布略有差异：C1 为 29/30、29/30、29/30，C2 为 29/30、30/30、28/30。

选模阶段 C2 的 validation 峰值 0.9667（58/60）略低于 C1 的 0.9833（59/60），开环归一化 MSE
0.3717 高于 C1 的 0.3097；但 test 上的 canonical 成绩一致。另外 C2 在 **step 5000 就达到
0.9667**（C1 同期只有 0.8833），LoRA 收敛更快。

### 2. LoRA 的语言泛化更差，不是更好

C2 换用训练时未见的改写句后掉到 21.1%，比 C1 的 33.3% 低 12 个百分点。逐任务为
7/30、8/30、4/30。

`para_gap`（canonical 减 held-out paraphrase）：

| | 点估计 | bootstrap 95% CI |
|---|---:|---|
| C1 | +0.6333 | `[0.5222, 0.7333]` |
| C2 | +0.7556 | `[0.6667, 0.8444]` |

两个区间不重叠，掉幅差异不是噪声。C2 与 C1 的训练指令完全相同（都只用 canonical），因此这不是
数据差异，而是低秩参数化在同样的单一指令上更容易贴死字面形式。

### 3. LoRA 没有防住灾难性遗忘

用同一份背景 test 协议（7 个 libero_goal 背景任务 × 30 个 test 初始状态 = 210 条）：

| task | 指令 | B0（记录） | C1 | C2 |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 30/30 | 0/30 | **3/30** |
| 2 | put the wine bottle on top of the cabinet | 27/30 | 0/30 | 0/30 |
| 3 | open the top drawer and put the bowl inside | 15/30 | 0/30 | 0/30 |
| 5 | push the plate to the front of the stove | 27/30 | 0/30 | **1/30** |
| 6 | put the cream cheese in the bowl | 26/30 | 0/30 | **2/30** |
| 7 | turn on the stove | 30/30 | 0/30 | **2/30** |
| 9 | put the wine bottle on the rack | 28/30 | 0/30 | 0/30 |
| **合计** | | **183/210** | **0/210** | **8/210** |

C2 的 8/210 = 3.8%，相对 B0 的 183/210 保留率 4.4%。七个任务中四个仍为零；成绩最好的 task 0
从 30/30 掉到 3/30。终止原因：C2 为 8 条 success + 202 条 `time_limit`，C1 为 210 条全部
`time_limit`。

**结论是 LoRA 并未提供实用意义上的保护。** 8 条对 0 条在方向上确实更好，但在"B0 原本能做
183/210"的对照下，两者都属于旧能力基本丧失。

### 4. 遗忘的根因不在参数化方式

C2 与 C1 的差别只有参数化：C2 冻结主干、只训练 rank-32 的低秩增量（target 覆盖 lm_expert 的
q/v 投影与 state/action 相关投影），C1 训练完整的 action expert 与 state projection。两者的
数据、LR、步数、batch、seed 完全相同。

在这个对照下，遗忘程度只从 0/210 变到 8/210——说明根因不是"训练了多少参数"，而是**训练数据里
没有背景任务，损失函数里不存在保留旧行为的项**。换参数化方式改不了这一点。可能有效的方向是
改数据（背景回放）或缩短优化路径（早停、降 LR），后者由
[C1 遗忘权衡曲线诊断](../plans/2026-08-05-C1遗忘权衡曲线诊断.md) 量化。

### 5. C2 的推理成本高于 C1

| 指标 | C1 | C2 |
|---|---:|---:|
| 模型查询延迟 p50 | 285.3 ms | 376.8 ms |
| 模型查询延迟 p95 | 443.2 ms | 541.4 ms |
| action TV | 0.0268 | 0.0436 |
| action jerk | 0.0285 | 0.0588 |
| 平均 gripper 切换 | 3.76 | 10.52 |

延迟升高与 PEFT 前向增加的低秩分支一致。动作平滑度与 gripper 切换次数变差，但两者的 canonical
成功率相同，说明 C2 的成功路径更"抖"。LoRA 的成本优势在 trainable parameters、optimizer state
与 adapter checkpoint 体积上，不在推理端。

## Checkpoint selection

结果类别：`Formal / Closed-loop / Validation`。6 个 checkpoint × 3 任务 × 20 个 validation
初始状态 = 360 条闭环，外加每个 checkpoint 在 30 条离线 validation demonstration
（2,969 帧）上的开环动作误差。

| Rank | Step | Macro SR | Offline norm MSE | Successes | 生效判据 |
|---:|---:|---:|---:|---:|---|
| 1 | **020000** | **0.9667** | 0.371719 | 58/60 | winner |
| 2 | 025000 | 0.9667 | 0.376928 | 58/60 | level 2 |
| 3 | 005000 | 0.9667 | 0.402103 | 58/60 | level 2 |
| 4 | 030000 | 0.9500 | 0.374399 | 57/60 | level 1 |
| 5 | 015000 | 0.9333 | 0.375955 | 56/60 | level 1 |
| 6 | 010000 | 0.9333 | 0.385587 | 56/60 | level 2 |

三个 checkpoint 在主判据上并列 0.9667，由第二判据（macro 归一化开环 MSE）分出 `020000`。
这是开环判据在 C 家族第二次真正决定 winner（C1 同样如此）。

注意 C2 的最优 step 是 20000 而非终点 30000，且 030000 反而掉到 0.9500——与 C1（030000 为
winner）不同。

## 协议一致性核对

C2 与 C1、Z0 三份 `run_manifest.json` 的 `contract` 仅以下字段不同：`model_label`、
`policy_path`、`policy_config_sha256`、`jobs`（后者只差 `output_dir` 与 `log_path`）。相同项：

- `split_sha256` `f323688693cba919…`、`paraphrases_sha256` `4f4891da2570f3bf…`、
  `protocol_config_sha256` `366069c1f40de11e…`
- `env_seed=1000`、`policy_seed=2000`、`n_action_steps=10`、`total_rollouts=210`

背景 test 侧：`split_sha256` `ec4b1d633cf3776d…`、`env_seed=1000`、`policy_seed=2000`、
`n_action_steps=10`、7 任务 × 30 状态，与 C1 一致。

## 必填实验字段（C2）

- Task suite：LIBERO-Goal 三个目标任务（本地编号 1 / 4 / 8）
- Dataset size：训练 113 条 demonstration / 11,022 帧；离线 validation 30 条 / 2,969 帧
- Observation keys：两路 256×256 RGB + 8 维 `observation.state`
- Action space：7 维相对末端动作
- Action horizon：`chunk_size=50`，执行 `n_action_steps=10`
- Control frequency：10 Hz
- Policy：从 selected B0（step 25000，`background_validation`）出发；LoRA rank 32、
  alpha 8、dropout 0，target 覆盖 `lm_expert` 的 q/v 投影与
  `state_proj`/`action_in_proj`/`action_out_proj`/`action_time_mlp_{in,out}`；
  `data_parallel_layout=8x1`（全局 batch 8）、BF16、30,000 步、每 5,000 步保存；
  AdamW LR 1e-4 / betas (0.9,0.95) / eps 1e-8 / wd 1e-10 / clip 10；
  warmup 1,000 + cosine 30,000 → 2.5e-6；train seed 1000
- Train loss：30,000 updates，首步 0.1568，末 1,000 步均值 0.06798（C1 同项为 0.01739）；
  grad norm 峰值 0.64（C1 为 9.42）；8 个 rank 状态均为 `complete`
- Open-loop action error：选中 checkpoint 归一化 0.371719（C1 为 0.309668）
- Closed-loop success rate：validation macro 0.9667（58/60）；test canonical 0.9667（87/90）、
  held-out paraphrase 0.2111（19/90）、prompt compliance 0.6000（18/30）；
  背景任务 0.0381（8/210）
- Latency：模型查询 p50 376.8 ms、p95 541.4 ms（21 个 job 均值）
- Failure cases：canonical 3 条 `time_limit`；held-out paraphrase 71 条 `time_limit`；
  prompt swap 12 条 `no_goal_reached`（stove 指令 10 条全失败）；背景任务 202 条 `time_limit`

## Pretrain / Provenance

- C2 起点为 selected B0 `025000`（`provenance_sha256` `036d2af3568ed6d9…`，
  `selection_set=background_validation`），与 C1 同一起点；`verified=true`。
- C2 winner 为 `020000`，`selection_set=target_validation`、`status=complete`；
  评测使用 `checkpoints/selected/pretrained_model`（adapter 形态）；`verified=true`。
- split / paraphrases / 协议配置的 sha256 与 C1、Z0 一致；`verified=true`。
- B0 背景基线 183/210 取自文档记录（`reports/experiment_log.md`、
  `reports/guides/smolvla_libero/02_b0_domain_adaptation.md`），原始产物已丢失、未重跑；
  `verified=false`。
- 未记录评测机器的源码 commit；code provenance `verified=false`。

## 结果位置

```text
outputs/smolvla_libero/pilot/c2_canonical_lora/
outputs/smolvla_libero/eval/c2_canonical_lora/
outputs/smolvla_libero/eval/background_regression/c2_selected/
```

## 结论边界

本次实验支持：

- C2 与 C1 在目标任务 canonical 上无差别（同为 87/90）；
- C2 的语言泛化差于 C1（21.1% 对 33.3%，`para_gap` 区间不重叠）；
- LoRA 未防住灾难性遗忘（背景 8/210 对 C1 的 0/210，B0 为 183/210）；
- 在数据、LR、步数、batch、seed 全部相同的对照下，遗忘程度对参数化方式不敏感；
- C2 的推理延迟与动作抖动均高于 C1。

本次实验不支持：

- "LoRA 在本设置下有实用价值"——它省的是训练成本，不是能力，也不是推理成本；
- 任何关于 C2 与 C1 canonical 成绩差异的陈述——两者相同；
- "8/210 说明 LoRA 部分保住了旧能力"——相对 183/210 的保留率只有 4.4%，四个任务仍为零；
- 关于 C3（LoRA + Paraphrase）的任何推断——尚未训练；
- 背景回退差异的配对显著性——B0 侧无逐 episode 记录，只能做非配对比较。

## 计划 / 待验证

以下均为**未执行**项：

- C3（LoRA + Paraphrase）的训练、选模、目标协议评测与背景回退评测。C2 的
  held-out paraphrase 21.1% 是 C3 要改善的直接目标。
- [C1 遗忘权衡曲线诊断](../plans/2026-08-05-C1遗忘权衡曲线诊断.md)：早停与 LR 按 batch 缩放
  能挽回多少背景能力。
- 背景数据回放（C1 guide 的 TODO-1）。C2 的负结果强化了它的必要性——参数化方式解决不了，
  只能改数据或优化路径。
- stove 目标不可达的成因诊断。C1 与 C2 的 Prompt Swap 混淆矩阵逐格相同（cabinet 10/10、
  plate 8/10、stove 0/10），说明这不是某个训练方案的问题。
