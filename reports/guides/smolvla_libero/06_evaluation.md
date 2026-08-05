# SmolVLA LIBERO：闭环评测与判断标准

## 实验目标

在互斥 test init states 上比较 B0/C1/C2/C3 的 canonical performance、held-out
language 泛化和 Prompt Swap 语言服从性。B0 `025000` 是只见过 background tasks 的
domain-adapted zero-shot baseline；四个模型必须共用同一评测入口和协议文件。

结果类别：`Closed-loop / Pilot`。

## 实验思路

B0/C1/C2/C3 使用相同 test states、prompt 分配、policy seed 和 full 300-step episode。
Canonical 与 Paraphrase rollout 必须恢复相同的 flow-noise 随机流。Prompt Swap 在同一个
reference environment 中恢复完全相同的 simulator state，只改变 prompt，并用独立三目标
分类器判断 bowl 最终落点。Test 只运行一次，不能根据结果追加同一批状态的实验。

## 评测前冻结

第一次查看 test 结果前必须冻结：

- test init-state IDs
- 每个 state 使用的 held-out paraphrase
- policy seed；本 Pilot 固定为 2000
- 三目标 simulator predicate 分类器及 Prompt Swap reference states
- cluster bootstrap 实现和随机种子
- B0 `025000` 和三个 selected C1/C2/C3 checkpoints

所有 checkpoint 只能由 validation 选择。Test、held-out paraphrases 和 Prompt Swap
不能用于选模。

## 随机流配对

Canonical/Paraphrase 的每对 rollout 使用相同 `(task_id, init_state_id, policy_seed)`。
Evaluator 必须在每个 rollout 开始前重置：

- Python RNG
- NumPy RNG
- PyTorch CPU RNG
- PyTorch CUDA RNG
- policy action queue

这样每对 rollout 的 flow sampling noise 从相同随机流开始。不能只在评测进程启动时设置
一次 seed。Prompt Swap 的三个 prompt 也必须分别恢复同一 simulator state 和同一 policy
RNG 初态。

## 评测命令

现有 `eval_libero.sh` 已支持单个 task 的 canonical 闭环 rollout，可用于 checkpoint
Smoke、Rerun 检查和单任务诊断：

```bash
bash scripts/experiments/smolvla_libero/eval_libero.sh \
  --policy-path=outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoints/selected/pretrained_model \
  --suite=libero_goal \
  --task-id=1 \
  --n-episodes=1 \
  --num-visualized-episodes=1 \
  --n-action-steps=10 \
  --seed=2000 \
  --policy-seed=2000 \
  --init-state-ids=0 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/eval/c2_task1_diagnostic \
  --no-open-viewer
```

该入口输出 `rollouts.parquet`、`rollout.rrd`、MP4、`metrics.json` 和
`run_config.json`。它支持显式 `--init-state-ids`，并在每个 rollout 前按
`--policy-seed` 恢复 policy RNG；但它尚不直接读取冻结 split/paraphrase 文件，也不实现
Prompt Swap，因此不能替代以下正式多协议入口。B0 严格选模由
`select_b0_checkpoint.sh` 读取冻结 split 并调用该底层评测器。

正式多协议入口为 `eval_target_protocols.sh`。它从
`target_eval_protocol.json` 读取冻结的 21-job / 210-rollout 协议；B0/C1/C2/C3 只改变
`--policy-path`、`--model-label` 和 `--output-dir`。先用 B0 检查远端 workload，不启动
rollout：

```bash
bash scripts/experiments/smolvla_libero/eval_target_protocols.sh \
  --policy-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/025000/pretrained_model \
  --model-label=b0_domain_adapted_zero_shot \
  --num-gpus=8 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/eval/b0_domain_adapted_zero_shot \
  --print-command
```

确认输出为 `21 jobs / 210 rollouts` 后，正式运行 B0：

```bash
bash scripts/experiments/smolvla_libero/eval_target_protocols.sh \
  --policy-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/025000/pretrained_model \
  --model-label=b0_domain_adapted_zero_shot \
  --num-gpus=8 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/eval/b0_domain_adapted_zero_shot
```

中断后原命令追加 `--resume`。C1/C2/C3 selected checkpoints 冻结后分别运行：

```bash
bash scripts/experiments/smolvla_libero/eval_target_protocols.sh \
  --policy-path=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected/pretrained_model \
  --model-label=c1_default_non_peft \
  --num-gpus=8 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/eval/c1_default_non_peft

bash scripts/experiments/smolvla_libero/eval_target_protocols.sh \
  --policy-path=outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoints/selected/pretrained_model \
  --model-label=c2_canonical_lora \
  --num-gpus=8 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/eval/c2_canonical_lora

bash scripts/experiments/smolvla_libero/eval_target_protocols.sh \
  --policy-path=outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoints/selected/pretrained_model \
  --model-label=c3_paraphrase_lora \
  --num-gpus=8 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/eval/c3_paraphrase_lora
```

默认不保存逐 episode 视频，避免 210 条正式 rollout 产生大量媒体；每个 job 仍保存一个
summary `rollout.rrd`。需要诊断媒体时用 `eval_libero.sh` 对已识别的 case 单独复跑，不能
把诊断复跑计入正式指标。如果 C1 因显存限制未训练，其结果只能显式记录为 `N/A: OOM`。

## Protocol A：Canonical Prompt

- 每任务使用冻结 split 中全部 30 个 test init states
- 共 90 rollouts/模型
- 使用任务 canonical prompt
- 输出 per-task SR 和三任务 macro SR

该结果同时是 unseen-pose SR；不要把同一批 rollout 重新命名成独立的 Unseen Initial
State 实验。

## Protocol B：Held-out Paraphrase

- 与 Canonical 使用相同 init states 和 policy seed
- 每个 state 固定分配一条 held-out paraphrase
- 五条 held-out 表达在每个任务的 30 个 states 上循环平衡，每条正好分配 6 个 states
- 共 90 rollouts/模型

定义有符号退化：

```text
Para Gap = SR_canonical - SR_paraphrase
```

报告百分点差，并对 `(task_id, init_state_id)` 做 10,000 次 paired cluster bootstrap，
输出 95% CI。同一 state 内保留完整的 Canonical/Paraphrase 配对。未来若预注册多个
policy seeds，也不能把同一 state 的重复样本当成独立初始状态。

## Protocol C：Prompt Swap

选择 10 个 reference test states。每个 state：

1. 恢复完全相同的 simulator state。
2. 恢复完全相同的 policy RNG。
3. 分别输入 stove、cabinet、plate canonical prompt。
4. 禁止 reference task 的内部 success 提前终止 rollout。
5. Episode 结束后独立判断 bowl 与三个目标的空间关系。

不能分别启动三个 task，再把相同 `init_state_id` 当作同一场景；三个任务的官方 init-state
数组并不相同。

输出 3×4 confusion matrix：

| Input Prompt | Stove | Cabinet | Plate | Failed |
| --- | ---: | ---: | ---: | ---: |
| Stove | | | | |
| Cabinet | | | | |
| Plate | | | | |

`Failed` 包括：

- 没有到达任何目标
- bowl 掉落
- 超时
- 多目标判定冲突

```text
Prompt Compliance =
  最终目标与输入 prompt 一致的 rollout 数 / 全部 Prompt-Swap rollout 数
```

Failed 必须留在分母。

## 三目标分类器

冻结分类器为 `libero_bowl_three_goal_predicate_v1`。它不读取 reference task 的单目标
`check_success()`；而是在 episode 终点对同一个 simulator state 同时计算：

```text
on(bowl, stove_cook_region)
on(bowl, cabinet_top_side)
on(bowl, plate)
```

正式 test 前已通过真实 1-step Prompt Swap Smoke，验证 classifier 能在 reference task
success termination 被禁用后读取终局 simulator predicates。正式结果还必须满足：

- 已知 stove/cabinet/plate 成功状态分别只命中一个目标
- 已知未完成状态进入 Failed
- 三种成功判定互斥
- 多目标冲突进入 Failed

命中零个目标进入 `Failed/no_goal_reached`；同时命中多个目标进入
`Failed/multi_goal_conflict`。不能用 environment 的单目标 `check_success()` 代替该分类器。

## 可选 Post-grasp Test

该诊断不阻塞 Pilot。只有 state save/restore 满足以下条件后执行：

- 恢复前后图像、object pose 和 robot pose 在容差内一致
- 没有接触爆炸或物体漂移
- 三条 prompt 从完全相同的 bowl-grasped 状态开始

Post-grasp 结果必须单独标为诊断，不与完整闭环 SR 混合。

## 结果落点

```text
outputs/smolvla_libero/eval/
├── b0_domain_adapted_zero_shot/
├── c1_default_non_peft/
├── c2_canonical_lora/
└── c3_paraphrase_lora/
```

每个目录至少保存：

- `run_manifest.json`
- `metrics.json`
- `metrics.md`
- `episodes.parquet`
- `prompt_goal_confusion.csv`
- `jobs/canonical/**`
- `jobs/held_out_paraphrase/**`
- `jobs/prompt_swap/**`
- `logs/**`

各 job 目录继续保存底层 evaluator 的 `run_config.json`、`metrics.json`、
`rollouts.parquet` 和 `rollout.rrd`。`target_eval_protocol.json` 保存 split/paraphrase SHA-256、
Prompt Swap reference states、分类器、seeds 和总 rollout 数；`--resume` 会拒绝协议不一致的
已有输出。

训练目录另外提供：

- trainable/total parameters
- adapter/checkpoint 大小
- peak VRAM
- 训练时间
- selected checkpoint step

## 结果表

| 模型 | Canonical SR | Paraphrase SR | Para Gap ↓ | Prompt Compliance ↑ | Trainable Params | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 Domain-adapted zero-shot | 待运行 | 待运行 | 待运行 | 待运行 | N/A | 已记录于 B0 |
| C1 Default Non-PEFT | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| C2 Canonical LoRA | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |
| C3 LoRA + Para | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 |

同时记录项目要求的统一字段：

- Task suite
- Dataset unique states、episodes 和 frames
- Observation keys
- Action space
- Prediction/execution horizon
- Control frequency
- Train loss
- Validation open-loop action error
- Closed-loop success rate
- Policy latency
- Failure cases

指标解释优先链接 [通用指标说明](../../metrics.md)，`metrics.json` 保持机器可读。

## 判断标准

### LoRA Pilot 晋级信号

- C2 canonical macro SR 距 C1 不超过 10 个百分点
- C2 trainable parameters 和 adapter checkpoint 明显小于 C1
- 相同 batch 下 C2 peak VRAM 不高于 C1，但不强制要求大幅下降

如果 C1 因 OOM 未运行，只能结论为“LoRA 能完成任务适配”，不能声称接近非 PEFT 上限。
B0 `025000` 提供“已适配 LIBERO 域、未使用目标任务训练数据”的 zero-shot baseline。
C0 仍只有一条 contract Smoke；若要分离 B0 background fine-tuning 本身的收益，还需要另行
预注册原始 SmolVLA base 的完整闭环 baseline。

### Paraphrase Pilot 晋级信号

- C3 held-out paraphrase macro SR 比 C2 至少高 10 个百分点
- Paired cluster bootstrap 的 95% CI 下界高于 0
- C3 canonical macro SR 比 C2 下降不超过 5 个百分点
- 至少两个任务的提升方向一致

90 条 paired rollout 的单条分辨率是 1.11 个百分点。提升只有 5～10 个百分点或 CI 跨 0
时，结论为“不确定”，不能追加同一 test 改变结论。

### 语言服从 Pilot 晋级信号

- Prompt Swap 对角线高于各行最大非对角项
- Prompt Compliance 点估计至少 60%
- 按 reference state 做 cluster bootstrap，95% CI 下界高于固定执行单目标的 1/3 基线
- 视频中 prompt 改变后目标阶段行为随之改变

只提高 canonical SR、但 Prompt Swap 始终落到同一目标，不能解释为语言增强有效。

## 结论边界与停止条件

- 本 Pilot 只使用 training seed 1000，只能提供晋级信号。
- 正式跨训练随机性结论至少需要预注册 3 个 training seeds 和一批未查看过的新初始状态。
- Validation 发现 flow 方差过大时，必须在 test 前冻结 replicate policy-seed 数。
- Test 已开始后保持原协议，不追加 states、prompts、policy seeds 或模型配置。
- C2/C3 差异小或 CI 跨 0 时，结论写为“不确定”。
- Prompt Swap 分类器有冲突时，先修分类器，不用单目标 success 替代。

实际结果统一登记到 [实验运行记录](../../experiment_log.md)。
