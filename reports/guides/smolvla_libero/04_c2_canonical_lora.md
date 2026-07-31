# SmolVLA LIBERO：C2 Canonical LoRA

## 实验目标

验证 SmolVLA 使用 LoRA 后，能否只依靠三个目标任务的 canonical instruction 完成
LIBERO-Goal 任务适配，并与共享数据和训练预算的 C1 Default Non-PEFT 公平比较。

结果类别：训练为 `Offline`，checkpoint selection 为 `Closed-loop`。

## 实验思路

C2 从与 C1 相同的 selected B0 checkpoint 开始，只把可训练边界改为 rank-32 LoRA。
它严格消费 113 条 Target Train demonstrations，并使用
`uniform_task_then_uniform_episode`；30 条 Offline Validation demonstrations 不进入训练。
正式模式固定 8 卡、每卡 batch 1、BF16、30,000 steps、LR `1e-4` 和 seed 1000。
launcher 在分配训练显存前严格加载 selected B0 并实际包裹 LoRA，核验 adapter 配置与可训练
参数边界。

前置条件：

- [准备工作与 Smoke](00_preparation.md) 已通过
- [数据准备](01_data_preparation.md) 已冻结 split 与 provenance
- B0 checkpoint selection 已生成自包含的 `selected/pretrained_model`

## 配置边界

```text
base_policy = outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model
method_type = LORA
r = 32
optimizer_lr = 1e-4
steps = 30000
train_seed = 1000
mixed_precision = bf16
world_size = 8
batch_size_per_gpu = 1
global_batch_size = 8
chunk_size = 50
n_action_steps = 10
n_obs_steps = 1
```

LeRobot 0.4.4 的通用 `PeftConfig` 只有 `target_modules`、`full_training_modules`、
`method_type`、`init_type` 和 `r`，没有 `lora_alpha` CLI 字段。C2 不伪造
`--peft.lora_alpha`；实际生成的 adapter config 会写入 `run_config.json`。当前依赖版本产生的
effective `lora_alpha` 为 8。

SmolVLA 默认 LoRA target 为：

- action expert 的 Q/V projections
- state projection
- action input/output projections
- action-time projections

`modules_to_save=[]`，所有可训练参数名都必须属于 `lora_*`。VLM language encoder 和
vision encoder 不参与微调。本实验验证的是 action expert 能否利用冻结 VLM 产生的文本
embedding，不能表述成“语言模型已经被 LoRA 微调”。

## 训练命令

正式配置已经冻结在 launcher 中。以下命令可直接启动 8 卡训练：

```bash
bash scripts/experiments/smolvla_libero/train_c2.sh
```

它固定使用：

- `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model`
- `datasets/lerobot/lerobot/libero`
- `configs/experiments/smolvla_libero/target_split.json`
- rank-32 canonical LoRA
- 8 个 Accelerate processes、每卡 batch 1、全局 batch 8
- 30,000 steps、每 5,000 steps 保存、BF16、seed 1000、LR `1e-4`
- 全离线 Hugging Face/Transformers/Datasets 模式

正式模式读取 selected checkpoint 旁的 `selected_checkpoint.json`，确认它由 B0 background
validation selection 产生。缺失或 provenance 不匹配时 fail closed，不会退回 `last` 或任意
step。启动前还会严格加载本地权重，并确认 effective LoRA rank、target、
`modules_to_save` 和 trainable parameter names。

启动前只检查完整命令、不运行训练：

```bash
bash scripts/experiments/smolvla_libero/train_c2.sh --print-command
```

### 多 GPU 启动语义

- 无参数正式入口是单机 8 卡：Accelerate 使用 8 个 process、`--gpu_ids=all` 和
  `--multi_gpu`，每张卡对应一个训练进程。
- C2 正式协议冻结为 world size 8、每卡 batch 1、全局 batch 8；少于 8 卡不能继续标记为
  同一份正式 C2 结果。
- Smoke/debug 可通过 `--num-processes=N` 使用单卡或多卡；`CUDA_VISIBLE_DEVICES` 暴露的
  GPU 数必须不少于 `N`。
- 续训从原始 `run_config.json` 恢复 world size 和 BF16，必须提供与原 run 相同数量的 GPU，
  不能在 resume 时改变卡数。
- 当前 launcher 支持单机多卡，不包含跨节点 rendezvous 配置。

单卡两步 Smoke：

```bash
bash scripts/experiments/smolvla_libero/train_c2.sh \
  --mode=smoke \
  --base-policy=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/target_split.json \
  --num-processes=1 \
  --batch-size=1 \
  --steps=2 \
  --save-freq=2 \
  --learning-rate=0.0001 \
  --lora-rank=32 \
  --train-seed=1000 \
  --num-workers=0 \
  --mixed-precision=bf16 \
  --output-dir=outputs/smolvla_libero/debug/c2_smoke
```

从最新完整 checkpoint 续训：

```bash
bash scripts/experiments/smolvla_libero/train_c2.sh \
  --resume-from=outputs/smolvla_libero/pilot/c2_canonical_lora/checkpoints/005000
```

续训会恢复原 run 的 GPU 数、BF16、optimizer、scheduler、RNG、target sampler 和 LoRA
adapter；adapter 以 `is_trainable=true` 加载，并跳过 LeRobot 默认的第二次 PEFT 包裹。
只允许从该 run 的最新完整 checkpoint 恢复。

VS Code 可选择 `SmolVLA LIBERO C2: train (1-step offline LoRA debug)` 做单卡单步调试；每次
必须填写新的 output directory。

## Checkpoint Selection

每 5,000 steps 保存 checkpoint。只能使用 validation split：

1. 比较三个任务的 macro canonical closed-loop SR。
2. 成功率相同时，比较 validation open-loop action error。
3. 仍相同时选择更早 checkpoint。

Open-loop error 使用相同 frame、固定 flow-noise seed 和相同 action horizon；先按任务求
均值，再计算 macro average。不得使用 test split 或 held-out paraphrases 选择 checkpoint。

以下 selection 入口仍属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/c2_canonical_lora \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --split-file=configs/experiments/smolvla_libero/target_split.json \
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
├── train.log
├── runtime_attempt_000_rank_*.json
├── N_A.json                         # 仅正式训练 OOM 时生成
└── checkpoints/
    ├── 005000/
    │   ├── pretrained_model/
    │   │   ├── adapter_config.json
    │   │   ├── adapter_model.safetensors
    │   │   ├── config.json
    │   │   └── train_config.json
    │   └── training_state/
    ├── ...
    └── 030000/
```

至少记录：

- train loss、gradient norm、LR 与 step latency
- trainable/total parameter 数量和完整 effective adapter config
- adapter checkpoint 与 optimizer state 大小
- peak allocated/reserved VRAM
- validation open-loop action error
- validation canonical closed-loop SR
- checkpoint selection 依据

实现阶段的 CPU 离线预检在本地 B0 step-30000 权重上测得 1,485,312 个 trainable
parameters / 451,531,488 个 total parameters，且 74 个 trainable parameter names 全部为
LoRA 参数；实际 SmolVLA adapter 的保存、trainable reload 和防重复包裹也已通过 CPU
离线验证。该结果标记为 `Smoke`，只证明加载、冻结和续训契约正确，不是正式 C2 训练结果。

## 判断标准

C2 训练通过：

- 训练 loss 有限且总体下降
- LoRA trainable parameters 非零，rank、target 和 `modules_to_save` 与预注册一致
- resolved config 证明只使用 113 条 Train demonstrations
- validation 至少出现成功 rollout
- selected checkpoint 完全由 validation 选择
- test 和 held-out paraphrase 没有参与调参

正式训练 OOM 时生成 `N_A.json`，记录 batch、mixed precision、rank、peak VRAM 和
`reason="oom"`。如果 C2 canonical validation SR 仍接近 0，先排查 action/state
normalization、执行 horizon 和 base checkpoint 适配；不得直接启动 C3。

下一步：[C3 LoRA + Paraphrase](05_c3_paraphrase_lora.md)。
