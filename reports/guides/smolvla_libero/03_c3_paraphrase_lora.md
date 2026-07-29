# SmolVLA LIBERO：C3 LoRA + Paraphrase

## 实验目标

验证训练时加入语义等价 instruction 改写，是否能改善 SmolVLA 对 held-out paraphrase 的
行为泛化，同时保持 canonical task performance。

结果类别：训练为 `Offline`，checkpoint selection 为 `Closed-loop`。

## 实验思路

C3 与 [C2](02_c2_canonical_lora.md) 使用相同 base、demonstrations、LoRA target、
rank、LR、batch、steps 和 seed。唯一变化是训练 instruction：25% 使用 canonical，
75% 从该任务的 train paraphrases 中抽样。Held-out test paraphrases 不得参与训练或选模。

前置条件：

- C2 训练与 validation 已通过
- [数据准备](01_data_preparation.md) 已冻结 `paraphrases.json`

## 固定配置

```text
base_policy = lerobot/smolvla_base
method_type = LORA
r = 32
optimizer_lr = 1e-4
steps = 30000
train_seed = 1000
canonical_probability = 0.25
paraphrase_probability = 0.75
mixed_precision = bf16
n_obs_steps = 1
```

每个任务使用：

- 1 条 canonical instruction
- 5～10 条 train paraphrases
- 5 条完全隔离的 test paraphrases

Instruction 抽样必须可复现。随机数由 train seed、epoch/step 和 sample index 派生，
不得在多 worker `__getitem__` 中直接调用未受控的全局 `random.random()`。

## 训练命令

以下 launcher 尚未实现，属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
bash scripts/experiments/smolvla_libero/train.sh \
  --config=paraphrase_lora \
  --base-policy=lerobot/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/split.json \
  --paraphrases=configs/experiments/smolvla_libero/paraphrases.json \
  --canonical-probability=0.25 \
  --n-obs-steps=1 \
  --steps=30000 \
  --batch-size=1 \
  --learning-rate=0.0001 \
  --lora-rank=32 \
  --train-seed=1000 \
  --mixed-precision=bf16 \
  --output-dir=outputs/smolvla_libero/pilot/c3_paraphrase_lora
```

除 task text 外，launcher 解析出的模型、optimizer、scheduler、dataset episodes 和训练
seed 必须与 C2 一致。

## Checkpoint Selection

沿用 C2 的 checkpoint cadence 和选择协议：

1. 只使用 validation canonical closed-loop SR。
2. 成功率相同时比较 validation open-loop action error。
3. 仍相同时选择更早 checkpoint。

Held-out test paraphrases 不允许参与 checkpoint selection。若需要语言诊断，必须单独准备
不属于 test 的 validation paraphrases，并在查看 test 前冻结。

```bash
bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/c3_paraphrase_lora \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --split-file=configs/experiments/smolvla_libero/split.json \
  --checkpoint-steps=5000,10000,15000,20000,25000,30000 \
  --policy-seed=1500 \
  --output-dir=outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoint_selection \
  --selected-dir=outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoints/selected
```

该 selection 入口属于 `Planned / CLI 未实现 / 当前不可运行`。

## 结果落点

```text
outputs/smolvla_libero/pilot/c3_paraphrase_lora/
├── run_config.json
├── train_metrics.jsonl
├── instruction_counts.json
├── checkpoint_selection/
└── checkpoints/
    └── selected/
```

除 C2 的训练字段外，额外记录：

- 每个 task/instruction 的采样次数
- canonical 实际采样比例
- train paraphrase 覆盖率
- C2/C3 resolved config diff

## 判断标准

C3 训练通过：

- canonical 实际比例接近 25%
- 三个任务采样量平衡
- 所有 train paraphrases 都被采样
- resolved config diff 只包含 instruction augmentation 相关项
- validation canonical SR 没有灾难性下降

语言增强是否有效不能只由训练 loss 判断，必须执行
[闭环评测](05_evaluation.md)。C2 成功而 C3 validation canonical 明显退化时，先检查
paraphrase 语义和抽样实现，不得直接查看 test。

下一步：资源允许时补充 [C1 Default Non-PEFT](04_c1_non_peft.md)，然后统一评测。
