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

## 本实验用到的数据

三个目标任务是 `libero_goal` 本地编号 1 / 4 / 8，被完整排除在 B0 训练之外。

- 训练：113 条 demonstration / 11,022 帧（38 + 36 + 39），离线 validation 不进训练。
- 选模主判据（`Validation`）：每任务 20 个初始状态 = 60 个 case，六个 checkpoint 各跑一遍，
  `6 × 3 × 20 = 360` 条闭环 rollout。
- 选模第二判据：30 条离线 validation demonstration / 2,969 帧上的开环动作误差。
- 最终成绩（`Test`）：只跑冻结的 winner，用每任务 30 个此前保留的初始状态。目标协议评测由
  三套协议组成，共 21 个 job / 210 条 rollout：canonical `3 × 30 = 90`；held-out paraphrase
  每任务 5 条改写句、按状态序号模 5 各分 6 个状态，`3 × 5 × 6 = 90`；prompt swap 3 个目标
  各 10 个参考状态，`3 × 10 = 30`。
- 指令：训练按 25/75 抽 canonical 与同任务 8 条 train paraphrases；5 条 held-out paraphrases 训练时不出现，只作评测期 Held-out Paraphrase 协议的指令。

test 初始状态与 held-out paraphrases 都不得进入选模。held-out 只针对初始状态——这 3 个任务
正是本实验训练的任务，且摆放随机化范围约 3 cm，所以 test 测的是同任务同分布、未被使用过的
摆放。完整清单见[实验族总览](README.md#结果类别标签与数据分工)。

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
- 默认 8 个 Accelerate processes、每卡 batch 1、全局 batch 8（合法布局：
  8×1、4×2、2×4、1×8）
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
- C3 正式协议冻结的是**全局 batch 8**（同时冻结 30,000 steps、每 5,000 保存、
  LR `1e-4`、seed 1000、BF16、LoRA rank 32、25/75 instruction 契约）。卡数可配置为
  8×1、4×2、2×4、1×8 任一合法布局；`--num-processes=N` 必须满足
  `N × --batch-size = 8`，且不能超过可见 GPU 数。
- 不同卡数的 run **协议等价但不是逐位复现**：sampler 分片与数据顺序随 world size
  改变。报告必须写明 `data_parallel_layout`（如 `4x2`）。
- **C1/C2/C3 三方对照必须使用同一个 `data_parallel_layout`**：全局 batch 相同的不同布局
  数据顺序不同，混用布局会把布局差异算进模型差异里。
- **C3 的 instruction 序列随布局改变**：选择哈希包含 `rank` 与 `batch_index`，换布局必然
  换序列；25/75 只在期望意义上成立。聚合 instruction 总数仍是 `steps × 8`，快照数量等于
  world size。报告需写明布局与实际 canonical 占比。
- 少于 8 卡时每卡 batch 上升（1 卡需 batch 8），LoRA 仍需前向整个冻结主干并保存部分
  activation，显存不一定比 C1 低；OOM 时按 `N_A.json` 契约记录，不缩小模型、不降全局
  batch。
- `--num-processes` 超过可见 GPU 数会立即失败并提示用 `CUDA_VISIBLE_DEVICES` 选卡
  （launcher 固定 `--gpu_ids=all`）。
- 续训从原始 `run_config.json` 恢复 world size、每卡 batch 与 BF16，并要求同一 step 的
  所有 rank instruction snapshots 完整，因此不能在 resume 时改变卡数
  （`world_size × per_gpu_batch_size` 仍必须等于 8）。
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
调试；它使用固定 output 目录并带 `--overwrite`，每次重跑直接覆盖上一次 debug 结果。

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

该入口已实现（`scripts/experiments/smolvla_libero/select_checkpoint.sh`），判据与固定条件：

1. 三任务 macro canonical closed-loop SR（高优先）；
2. 平局时比较 macro **normalized** offline action error（低优先；denormalized 同时记录，
   只作参考）；
3. 仍平局时选择更早 checkpoint。

Open-loop error 使用相同 frame（30 条 offline validation demonstration 的全部
2,969 帧）、固定 flow-noise seed（按 task/episode/frame 派生，跨 checkpoint 相同）和
相同 action horizon（chunk 50、取前 10 步）；先按任务求均值，再 macro average。闭环
规模为 `6 × 3 × 20 = 360` 条 rollout；`ranking_audit` 在
`checkpoint_selection/ranking_audit.json`。

开环误差按 checkpoint 并行评测：每个 checkpoint 分派到一张 GPU（并行度 =
`min(num_gpus, checkpoint 数)`），6 卡下预计约 12 分钟（单卡串行约 69 分钟）。子进程日志
在 `checkpoint_selection/logs/open_loop_checkpoint_<step>.log`（首行为完整命令），中间结果
在 `checkpoint_selection/jobs/open_loop/checkpoint_<step>.json`；`--resume` 会复用
`checkpoint_id` / `split_sha256` / `max_frames_per_task` 一致的已完成结果。并行调度不改
抽帧、噪声派生与误差公式，数值与串行逐值相同。

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
