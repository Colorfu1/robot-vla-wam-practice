# SmolVLA LIBERO：C1 Default Non-PEFT

## 实验目标

建立与 LoRA 配置共享数据、训练步数和优化设置的非 PEFT 对照，判断 C2 是否接近 SmolVLA
默认任务适配性能。

结果类别：训练为 `Offline`，checkpoint selection 为 `Closed-loop`。

## 实验思路

C1 从同一个 `lerobot/smolvla_base` 开始，不使用 PEFT，保留 SmolVLA 默认冻结边界。
它训练 action expert 和相关 projections，但仍冻结 VLM 与 vision encoder。只有在单卡
BF16 batch 1 Smoke 通过后才运行；不能缩小模型后继续称为同一个上限对照。

## 配置边界

```text
use_peft = false
train_expert_only = true
freeze_vision_encoder = true
optimizer_lr = 1e-4
steps = 30000
train_seed = 1000
mixed_precision = bf16
n_obs_steps = 1
```

C1 不是 full-model fine-tuning。实验报告统一称为 `Default Non-PEFT Adaptation`。

C1/C2/C3 必须共享：

- train/validation split
- canonical instructions
- base checkpoint
- batch size
- LR、steps、seed
- checkpoint cadence 与 selection protocol

## 训练命令

以下 launcher 尚未实现，属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
bash scripts/experiments/smolvla_libero/train.sh \
  --config=default_non_peft \
  --base-policy=lerobot/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/split.json \
  --n-obs-steps=1 \
  --steps=30000 \
  --batch-size=1 \
  --learning-rate=0.0001 \
  --train-seed=1000 \
  --mixed-precision=bf16 \
  --output-dir=outputs/smolvla_libero/pilot/c1_default_non_peft
```

## Checkpoint Selection

使用与 C2/C3 完全相同的 validation protocol：

1. 三任务 macro canonical closed-loop SR。
2. 平局时比较固定 flow-noise 的 validation open-loop action error。
3. 仍平局时选择更早 checkpoint。

不得因为 C1 训练成本较高而减少 validation states，否则 C1/C2 的 checkpoint selection
不再公平。

```bash
bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/c1_default_non_peft \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --split-file=configs/experiments/smolvla_libero/split.json \
  --checkpoint-steps=5000,10000,15000,20000,25000,30000 \
  --policy-seed=1500 \
  --output-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoint_selection \
  --selected-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected
```

该 selection 入口属于 `Planned / CLI 未实现 / 当前不可运行`。

## 结果落点

```text
outputs/smolvla_libero/pilot/c1_default_non_peft/
├── run_config.json
├── train_metrics.jsonl
├── checkpoint_selection/
├── checkpoints/
│   └── selected/
└── N_A.json
```

至少记录：

- trainable/total parameters
- checkpoint 大小
- optimizer state 大小
- peak VRAM
- 训练时间与 step latency
- train loss、validation error 和 validation SR

成功时不生成 `N_A.json`。如果 C1 因 OOM 无法训练，launcher 必须写入 `N_A.json`，
记录失败阶段、batch、mixed precision、peak VRAM 和 `reason="oom"`，供统一评测安全跳过。

## 判断标准

C1 通过：

- batch 1 BF16 下没有 OOM
- resolved config 确认 VLM/vision 仍冻结
- validation 至少出现成功 rollout
- selected checkpoint 只使用 validation

如果 batch 1 BF16 仍 OOM：

- 记录为资源限制
- 保存 `N_A.json`
- 不通过缩小 C1 结构制造新对照
- 主结论缩减为 C2/C3 的语言增强比较
- 不能声称 C2 已接近非 PEFT 上限

LoRA 仍需执行冻结主干前向并保存部分 activation，因此 C2 peak VRAM 不一定大幅低于 C1。
LoRA 的主要成本优势应同时看 trainable parameters、optimizer state 和 adapter checkpoint。

下一步：[闭环评测与判断标准](05_evaluation.md)。
