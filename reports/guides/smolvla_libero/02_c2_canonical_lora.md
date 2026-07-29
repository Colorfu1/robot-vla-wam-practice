# SmolVLA LIBERO：C2 Canonical LoRA

## 实验目标

验证 SmolVLA 使用 LoRA 后，能否只依靠三个任务的 canonical instruction 完成 LIBERO-Goal
任务适配。

结果类别：训练为 `Offline`，checkpoint selection 为 `Closed-loop`。

## 实验思路

C2 是首个正式训练配置。固定 `lerobot/smolvla_base`、30 个 train states/任务、
LoRA rank 32、BF16、30,000 steps 和 train seed 1000。只允许使用 validation split
选择 checkpoint；test 和 held-out paraphrases 在训练与选模阶段不可见。

前置条件：

- [准备工作与 Smoke](00_preparation.md) 已通过
- [数据准备](01_data_preparation.md) 已冻结 split 与 provenance

## 配置

```text
base_policy = lerobot/smolvla_base
method_type = LORA
r = 32
optimizer_lr = 1e-4
steps = 30000
train_seed = 1000
mixed_precision = bf16
chunk_size = 50
n_action_steps = 10
n_obs_steps = 1
```

LeRobot 0.4.4 的通用 `PeftConfig` 只有 `target_modules`、`full_training_modules`、
`method_type`、`init_type` 和 `r`，没有 `lora_alpha`。因此
`--peft.lora_alpha=32` 是无效参数。C2 使用实际生成的 PEFT adapter config，并把它保存
到 `run_config.json`；如果必须单独控制 alpha，需要先扩展本地 CLI。

SmolVLA 默认 LoRA target 包括：

- action expert 的 Q/V projections
- state projection
- action input/output projections
- action-time projections

它不微调 VLM language encoder。本实验验证的是 action expert 能否利用冻结 VLM 产生的
文本 embedding，不应表述成“语言模型已经被 LoRA 微调”。

## 训练命令

以下 launcher 尚未实现，属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
bash scripts/experiments/smolvla_libero/train.sh \
  --config=canonical_lora \
  --base-policy=lerobot/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/split.json \
  --n-obs-steps=1 \
  --steps=30000 \
  --batch-size=1 \
  --learning-rate=0.0001 \
  --lora-rank=32 \
  --train-seed=1000 \
  --mixed-precision=bf16 \
  --output-dir=outputs/smolvla_libero/pilot/c2_canonical_lora
```

`batch-size=1` 是安全起点。Smoke 后可以依次测试 2、4、8，但 C1/C2/C3 最终必须使用
相同实际 batch。当前 LeRobot 训练入口没有可配置 gradient accumulation，不得只对某个
配置改变 effective batch。

## Checkpoint Selection

每 5,000 steps 保存 checkpoint。只能使用 validation split：

1. 比较三个任务的 macro canonical closed-loop SR。
2. 成功率相同时，比较 validation open-loop action error。
3. 仍相同时选择更早 checkpoint。

Open-loop error 使用相同 frame、固定 flow-noise seed 和相同 action horizon；先按任务求
均值，再计算 macro average。不得使用 test split 或 held-out paraphrase 选择 checkpoint。

以下 selection 入口属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/c2_canonical_lora \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --split-file=configs/experiments/smolvla_libero/split.json \
  --checkpoint-steps=5000,10000,15000,20000,25000,30000 \
  --policy-seed=1500 \
  --output-dir=outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoint_selection \
  --selected-dir=outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoints/selected
```

`selected-dir` 必须保存自包含的 checkpoint 副本，不能只创建软链接。

## 结果落点

```text
outputs/smolvla_libero/pilot/c2_canonical_lora/
├── run_config.json
├── train_metrics.jsonl
├── checkpoint_selection/
└── checkpoints/
    ├── 005000/pretrained_model/
    ├── ...
    ├── 030000/pretrained_model/
    └── selected/
```

至少记录：

- train loss
- validation open-loop action error
- validation canonical closed-loop SR
- trainable/total parameter 数量
- adapter config 和 checkpoint 大小
- peak allocated/reserved VRAM
- 训练时间与单 step latency
- checkpoint selection 依据

## 判断标准

C2 训练通过：

- 训练 loss 有限且总体下降
- LoRA trainable parameters 非零，target 与 Smoke 一致
- validation 至少出现成功 rollout
- selected checkpoint 完全由 validation 选择
- test 和 held-out paraphrase 没有参与调参

如果 C2 canonical validation SR 仍接近 0，先排查 action/state normalization、执行
horizon 和 base checkpoint 适配；不得直接启动 C3。

下一步：[C3 LoRA + Paraphrase](03_c3_paraphrase_lora.md)。
