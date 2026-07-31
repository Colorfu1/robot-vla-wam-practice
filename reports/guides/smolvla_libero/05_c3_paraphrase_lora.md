# SmolVLA LIBERO：C3 LoRA + Paraphrase

## 实验目标

验证训练时加入语义等价 instruction 改写，是否能改善 SmolVLA 对 held-out paraphrase 的
行为泛化，同时保持 canonical task performance，并与 C2 Canonical LoRA 构成单变量对照。

结果类别：训练为 `Offline`，checkpoint selection 为 `Closed-loop`。

## 实验思路

C3 与 [C2](04_c2_canonical_lora.md) 使用相同 selected B0、113 条 Train
demonstrations、task/episode-balanced sampler、LoRA target、rank、LR、batch、steps 和
seed。唯一训练差异是 tokenizer 前的 task text：25% 使用 canonical，75% 从对应任务的
8 条 reviewed train paraphrases 中均匀选择。5 条 held-out paraphrases 只做绑定和泄漏审计，
正文不会进入训练 selector。30 条 Offline Validation demonstrations 仍不进入训练。

前置条件：

- [数据准备](01_data_preparation.md) 已冻结 `target_split.json` 与 `paraphrases.json`
- B0 checkpoint selection 已生成自包含的 `selected/pretrained_model`
- 正式解释 C3 结果前，C2 必须完成同一 validation protocol

## 固定配置

```text
base_policy = outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model
method_type = LORA
r = 32
optimizer_lr = 1e-4
steps = 30000
train_seed = 1000
canonical_probability = 0.25
paraphrase_probability = 0.75
world_size = 8
batch_size_per_gpu = 1
global_batch_size = 8
mixed_precision = bf16
chunk_size = 50
n_action_steps = 10
n_obs_steps = 1
```

每个任务使用 1 条 canonical instruction 和 8 条 train paraphrases。训练代码完成完整
`paraphrases.json` semantic review 与 split SHA-256 绑定检查后，只把 canonical、train
paraphrases 和 held-out hashes 放入训练 spec；held-out 正文不能被 selector 访问。

Instruction 选择发生在 DataLoader/worker 之后、tokenizer 之前。无状态选择键为：

```text
blake2b(train_seed, global_step, rank, batch_index, dataset_index, task_id)
```

因此 worker 数量、预取时序和 Python 全局 RNG 不会改变 task text。每次 optimizer update
成功后才提交计数；每个 checkpoint step 保存各 rank 的累计 instruction snapshot。续训从
snapshot 恢复累计计数，并从 checkpoint global step 继续语言随机流。

## 训练命令

正式配置已经冻结在 launcher 中。以下命令可直接启动 8 卡训练：

```bash
bash scripts/experiments/smolvla_libero/train_c3.sh
```

它固定使用：

- `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model`
- `datasets/lerobot/lerobot/libero`
- `configs/experiments/smolvla_libero/target_split.json`
- `configs/experiments/smolvla_libero/paraphrases.json`
- 25% canonical / 75% train paraphrase 和 rank-32 LoRA
- 8 个 Accelerate processes、每卡 batch 1、全局 batch 8
- 30,000 steps、每 5,000 steps 保存、BF16、seed 1000、LR `1e-4`
- 全离线 Hugging Face/Transformers/Datasets 模式

正式模式会严格检查 selected-B0 provenance、paraphrase semantic review、split 绑定哈希、
effective LoRA config 和 trainable parameter names。任一项漂移都会 fail closed。

启动前只检查完整命令、不运行训练：

```bash
bash scripts/experiments/smolvla_libero/train_c3.sh --print-command
```

### 多 GPU 启动语义

- 无参数正式入口是单机 8 卡：Accelerate 使用 8 个 process、`--gpu_ids=all` 和
  `--multi_gpu`，每张卡对应一个训练进程。
- C3 正式协议冻结为 world size 8、每卡 batch 1、全局 batch 8；少于 8 卡不能继续标记为
  同一份正式 C3 结果。
- Smoke/debug 可通过 `--num-processes=N` 使用单卡或多卡；`CUDA_VISIBLE_DEVICES` 暴露的
  GPU 数必须不少于 `N`。
- 续训从原始 `run_config.json` 恢复 world size 和 BF16，并要求同一 step 的所有 rank
  instruction snapshots 完整，因此不能在 resume 时改变卡数。
- 当前 launcher 支持单机多卡，不包含跨节点 rendezvous 配置。

单卡两步 Smoke：

```bash
bash scripts/experiments/smolvla_libero/train_c3.sh \
  --mode=smoke \
  --base-policy=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/target_split.json \
  --paraphrases=configs/experiments/smolvla_libero/paraphrases.json \
  --canonical-probability=0.25 \
  --num-processes=1 \
  --batch-size=1 \
  --steps=2 \
  --save-freq=2 \
  --learning-rate=0.0001 \
  --lora-rank=32 \
  --train-seed=1000 \
  --num-workers=0 \
  --mixed-precision=bf16 \
  --output-dir=outputs/smolvla_libero/debug/c3_smoke
```

从最新完整 checkpoint 续训：

```bash
bash scripts/experiments/smolvla_libero/train_c3.sh \
  --resume-from=outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoints/005000
```

续训除了恢复 C2 的 adapter、optimizer、scheduler 和 RNG 状态，还要求 checkpoint step
对应的所有 rank instruction snapshots 完整且 provenance 一致。不能从较早 checkpoint
回退，也不能在续训时替换 paraphrase 文件。

VS Code 可选择 `SmolVLA LIBERO C3: train (1-step offline paraphrase debug)` 做单卡单步
调试；每次必须填写新的 output directory。

## C1/C2 兼容性边界

实现依赖方向固定为 `C3 → C2 → C1`：C1 不导入 C2/C3，C2 不导入 C3。C3 没有修改
C1/C2 的 parser、launcher、train config 或 sampler；C2/C3 的 resolved LeRobot train
config 除 `job_name` 外逐字段相同，C1/C2 的既有等价性测试保持不变。

C2 的 PEFT hook 和 C3 的 instruction hook 只在各自训练调用的 context manager 内临时
生效，正常退出和异常退出都会恢复 LeRobot 原函数。单纯 import C3 不会修改
`make_policy`、`make_pre_post_processors`、`update_policy` 或 SmolVLA PEFT validation。
因此新增 C3 不会改变 C1/C2 旧入口的运行行为，C2 也不会改变 C1 入口。

## Checkpoint Selection

沿用 C2 的 checkpoint cadence 和选择协议：

1. 只使用 validation canonical closed-loop SR。
2. 成功率相同时比较 validation open-loop action error。
3. 仍相同时选择更早 checkpoint。

Held-out test paraphrases 不允许参与 checkpoint selection。若需要语言诊断，必须单独准备
不属于 test 的 validation paraphrases，并在查看 test 前冻结。

以下 selection 入口仍属于 `Planned / CLI 未实现 / 当前不可运行`：

```bash
bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/c3_paraphrase_lora \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --split-file=configs/experiments/smolvla_libero/target_split.json \
  --checkpoint-steps=5000,10000,15000,20000,25000,30000 \
  --policy-seed=1500 \
  --output-dir=outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoint_selection \
  --selected-dir=outputs/smolvla_libero/pilot/c3_paraphrase_lora/checkpoints/selected
```

## 结果落点

```text
outputs/smolvla_libero/pilot/c3_paraphrase_lora/
├── run_config.json
├── train_metrics.jsonl
├── train.log
├── instruction_counts.json
├── instruction_counts_step_005000_rank_0.json
├── instruction_counts_step_005000_rank_1.json
├── ...
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

`instruction_counts.json` 汇总：

- 每个 task 的 canonical count
- 每条 train paraphrase 的 count
- canonical 实际比例
- train paraphrase 覆盖率
- 累计 step、world size、batch 和所有 rank snapshot provenance

`run_config.json` 额外保存 paraphrase 文件路径、SHA-256、sampling contract、概率、semantic
review 状态和 held-out hashes。C2/C3 的 resolved LeRobot train config 除 `job_name` 外必须
完全一致；instruction augmentation 作为 C3 独立 runtime contract 记录。

## 判断标准

C3 训练通过：

- canonical 实际比例接近 25%
- 三个任务采样量平衡
- 所有 24 条 train paraphrases 都被采样
- held-out paraphrase 正文没有进入 selector 或 tokenizer
- C2/C3 resolved train config diff 只包含 `job_name`
- validation canonical SR 没有灾难性下降
- selected checkpoint 完全由 canonical validation 选择

正式训练 OOM 时生成 `N_A.json`。语言增强是否有效不能只由训练 loss 判断，必须执行
[闭环评测](06_evaluation.md)。C2 成功而 C3 validation canonical 明显退化时，先检查
paraphrase 语义和抽样实现，不得直接查看 test。

下一步：[统一闭环评测](06_evaluation.md)。
