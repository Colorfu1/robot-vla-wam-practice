# SmolVLA LIBERO：闭环评测与判断标准

## 实验目标

在互斥 test init states 上比较 C1/C2/C3 的 canonical performance、held-out language
泛化和 Prompt Swap 语言服从性。

结果类别：`Closed-loop / Pilot`。

## 实验思路

三个模型使用相同 test states、prompt 分配、policy seed 和 full 300-step episode。
Canonical 与 Paraphrase rollout 必须恢复相同的 flow-noise 随机流。Prompt Swap 在同一个
reference environment 中恢复完全相同的 simulator state，只改变 prompt，并用独立三目标
分类器判断 bowl 最终落点。Test 只运行一次，不能根据结果追加同一批状态的实验。

## 评测前冻结

第一次查看 test 结果前必须冻结：

- test init-state IDs
- 每个 state 使用的 held-out paraphrase
- policy seed；本 Pilot 固定为 2000
- Prompt Swap reference states
- 三目标分类器几何阈值
- cluster bootstrap 实现和随机种子
- 三个 selected checkpoints

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

以下 evaluator 尚未实现，属于 `Planned / CLI 未实现 / 当前不可运行`。实现后可直接运行
三个 selected checkpoints：

```bash
C1_POLICY="outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected/pretrained_model"
C1_NA="outputs/smolvla_libero/pilot/c1_default_non_peft/N_A.json"
MODELS=(c2_canonical_lora c3_paraphrase_lora)

if [[ -d "${C1_POLICY}" ]]; then
  MODELS=(c1_default_non_peft "${MODELS[@]}")
elif [[ -f "${C1_NA}" ]]; then
  echo "Skipping C1 according to ${C1_NA}"
else
  echo "error: C1 checkpoint and explicit N_A.json are both missing" >&2
  exit 1
fi

for MODEL in "${MODELS[@]}"; do
  case "${MODEL}" in
    c1_default_non_peft)
      POLICY_PATH="${C1_POLICY}"
      ;;
    c2_canonical_lora)
      POLICY_PATH="outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoints/selected/pretrained_model"
      ;;
    c3_paraphrase_lora)
      POLICY_PATH="outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoints/selected/pretrained_model"
      ;;
  esac

  bash scripts/experiments/smolvla_libero/eval.sh \
    --policy-path="${POLICY_PATH}" \
    --suite=libero_goal \
    --task-ids=1,4,8 \
    --split-file=configs/experiments/smolvla_libero/split.json \
    --paraphrases=configs/experiments/smolvla_libero/paraphrases.json \
    --protocols=canonical,held_out_paraphrase,prompt_swap \
    --policy-seed=2000 \
    --episode-length=300 \
    --num-visualized-episodes=30 \
    --output-dir="outputs/smolvla_libero/eval/${MODEL}"
done
```

如果 C1 因显存限制未训练，命令只在检测到显式 `N_A.json` 后跳过 C1，并在结果中标为
`N/A: OOM`；未知原因的 checkpoint 缺失会直接报错。

## Protocol A：Canonical Prompt

- 每任务 10 个 test init states
- 共 30 rollouts/模型
- 使用任务 canonical prompt
- 输出 per-task SR 和三任务 macro SR

该结果同时是 unseen-pose SR；不要把同一批 rollout 重新命名成独立的 Unseen Initial
State 实验。

## Protocol B：Held-out Paraphrase

- 与 Canonical 使用相同 init states 和 policy seed
- 每个 state 固定分配一条 held-out paraphrase
- 五条 held-out 表达在每个任务的 10 个 states 上循环平衡
- 共 30 rollouts/模型

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

分类器阈值必须在 validation 或手工构造状态上冻结，正式 test 前验证：

- 已知 stove/cabinet/plate 成功状态分别只命中一个目标
- 已知未完成状态进入 Failed
- 三种成功判定互斥
- 多目标冲突进入 Failed

不能用 environment 的单目标 `check_success()` 代替三目标终局分类。

## 可选 Post-grasp Test

该诊断不阻塞 Pilot。只有 state save/restore 满足以下条件后执行：

- 恢复前后图像、object pose 和 robot pose 在容差内一致
- 没有接触爆炸或物体漂移
- 三条 prompt 从完全相同的 bowl-grasped 状态开始

Post-grasp 结果必须单独标为诊断，不与完整闭环 SR 混合。

## 结果落点

```text
outputs/smolvla_libero/eval/
├── c1_default_non_peft/
├── c2_canonical_lora/
└── c3_paraphrase_lora/
```

每个目录至少保存：

- `run_config.json`
- `metrics.json`
- `metrics.md`
- `episodes.parquet`
- `prompt_goal_confusion.csv`
- `goal_classifier.json`
- `videos/`
- `recordings/*.rrd`
- `failures/`

训练目录另外提供：

- trainable/total parameters
- adapter/checkpoint 大小
- peak VRAM
- 训练时间
- selected checkpoint step

## 结果表

| 模型 | Canonical SR | Paraphrase SR | Para Gap ↓ | Prompt Compliance ↑ | Trainable Params | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
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
C0 只有一条 contract Smoke，不提供可比较的 base success rate；若要量化相对未适配
SmolVLA 的提升，必须另行预注册完整的 base closed-loop baseline。

### Paraphrase Pilot 晋级信号

- C3 held-out paraphrase macro SR 比 C2 至少高 10 个百分点
- Paired cluster bootstrap 的 95% CI 下界高于 0
- C3 canonical macro SR 比 C2 下降不超过 5 个百分点
- 至少两个任务的提升方向一致

30 条 Pilot rollout 的单条分辨率是 3.33 个百分点。提升只有 5～10 个百分点或 CI 跨 0
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
