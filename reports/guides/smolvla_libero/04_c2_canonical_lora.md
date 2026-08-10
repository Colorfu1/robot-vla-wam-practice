# SmolVLA LIBERO：C2 Canonical LoRA

<!-- PROMPT_SWAP_MERGE_NOTICE:BEGIN -->
## Prompt Swap 修复后当前结果

**Z0/C1/C2 Prompt Swap 已合并；本文旧 Prompt Swap 数值仅为历史快照。**

以下是当前有效的结果；本文其余位置中修复前的 Prompt Swap 数值、混淆矩阵和由此得到的结论，均只保留作历史审计，不得继续引用。

| Model | First-goal compliance | Terminal-goal compliance | Full horizon |
| --- | ---: | ---: | ---: |
| z0_b0_zero_shot | 0/30 = 0.0% | 0/30 = 0.0% | 30/30 |
| c1_default_non_peft | 29/30 = 96.7% | 27/30 = 90.0% | 30/30 |
| c2_canonical_lora | 28/30 = 93.3% | 28/30 = 93.3% | 30/30 |

主指标是每一步 simulator predicate 的 `first_goal_reached`；`terminal_goal` 只表示达成后的保持性诊断。
这次合并只替换 Prompt Swap 的 job/rollout/episode 与汇总结果，canonical、held-out paraphrase、训练和 checkpoint selection 结论不变。
详细合并报告：`reports/2026-08-06-PromptSwap修复后合并结果.md`。
<!-- PROMPT_SWAP_MERGE_NOTICE:END -->

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
- 指令：只用 canonical；8 条 train paraphrases 与 5 条 held-out paraphrases 都不参与 C2。

test 初始状态与 held-out paraphrases 都不得进入选模。held-out 只针对初始状态——这 3 个任务
正是本实验训练的任务，且摆放随机化范围约 3 cm，所以 test 测的是同任务同分布、未被使用过的
摆放。完整清单见[实验族总览](README.md#结果类别标签与数据分工)。

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
- 默认 8 个 Accelerate processes、每卡 batch 1、全局 batch 8（合法布局：
  8×1、4×2、2×4、1×8）
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
- C2 正式协议冻结的是**全局 batch 8**（同时冻结 30,000 steps、每 5,000 保存、
  LR `1e-4`、seed 1000、BF16、LoRA rank 32）。卡数可配置为 8×1、4×2、2×4、1×8 任一
  合法布局；`--num-processes=N` 必须满足 `N × --batch-size = 8`，且不能超过可见 GPU 数。
- 不同卡数的 run **协议等价但不是逐位复现**：sampler 分片与数据顺序随 world size
  改变。报告必须写明 `data_parallel_layout`（如 `4x2`）。
- **C1/C2/C3 三方对照必须使用同一个 `data_parallel_layout`**：全局 batch 相同的不同布局
  数据顺序不同，混用布局会把布局差异算进模型差异里。
- 少于 8 卡时每卡 batch 上升（1 卡需 batch 8），LoRA 仍需前向整个冻结主干并保存部分
  activation，显存不一定比 C1 低；OOM 时按 `N_A.json` 契约记录，不缩小模型、不降全局
  batch。
- `--num-processes` 超过可见 GPU 数会立即失败并提示用 `CUDA_VISIBLE_DEVICES` 选卡
  （launcher 固定 `--gpu_ids=all`）。
- 续训从原始 `run_config.json` 恢复 world size、每卡 batch 与 BF16，必须提供与原 run
  相同数量的 GPU（`world_size × per_gpu_batch_size` 仍必须等于 8），不能在 resume 时
  改变卡数。
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

VS Code 可选择 `SmolVLA LIBERO C2: train (1-step offline LoRA debug)` 做单卡单步调试；
它使用固定 output 目录并带 `--overwrite`，每次重跑直接覆盖上一次 debug 结果。

## Checkpoint Selection

每 5,000 steps 保存 checkpoint。只能使用 validation split：

1. 比较三个任务的 macro canonical closed-loop SR。
2. 成功率相同时，比较 validation open-loop action error。
3. 仍相同时选择更早 checkpoint。

Open-loop error 使用相同 frame、固定 flow-noise seed 和相同 action horizon；先按任务求
均值，再计算 macro average。不得使用 test split 或 held-out paraphrases 选择 checkpoint。

该入口已实现（`scripts/experiments/smolvla_libero/select_checkpoint.sh`），判据与固定条件：

1. 三任务 macro canonical closed-loop SR（高优先）；
2. 平局时比较 macro **normalized** offline action error（低优先；denormalized 同时记录，
   只作参考）；
3. 仍平局时选择更早 checkpoint。

Open-loop error 使用相同 frame（30 条 offline validation demonstration 的全部
2,969 帧）、固定 flow-noise seed（按 task/episode/frame 派生，跨 checkpoint 相同）和
相同 action horizon（chunk 50、取前 10 步）；先按任务求均值，再 macro average。闭环
规模为 `6 × 3 × 20 = 360` 条 rollout；`ranking_audit` 在
`checkpoint_selection/ranking_audit.json`。不得使用 test split 或 held-out paraphrases。

开环误差按 checkpoint 并行评测：每个 checkpoint 分派到一张 GPU（并行度 =
`min(num_gpus, checkpoint 数)`），6 卡下预计约 12 分钟（单卡串行约 69 分钟）。子进程日志
在 `checkpoint_selection/logs/open_loop_checkpoint_<step>.log`（首行为完整命令），中间结果
在 `checkpoint_selection/jobs/open_loop/checkpoint_<step>.json`；`--resume` 会复用
`checkpoint_id` / `split_sha256` / `max_frames_per_task` 一致的已完成结果。并行调度不改
抽帧、噪声派生与误差公式，数值与串行逐值相同。

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

## 已完成的正式结果

训练、选模、目标协议评测与背景回退评测都已跑完。完整对比与结论边界见
[C2 Canonical LoRA 正式评测结论](../../2026-08-05-C2%20Canonical%20LoRA%20正式评测结论.md)。

### 训练（Formal / Offline）

一次 attempt 完成 30,000 updates，`data_parallel_layout=8x1`，8 个 rank 状态均为 `complete`。
LoRA rank 32 / alpha 8 / dropout 0，target 覆盖 `lm_expert` 的 q/v 投影与 `state_proj`、
`action_in_proj`、`action_out_proj`、`action_time_mlp_{in,out}`。loss 首步 0.1568、末 1,000 步
均值 **0.06798**（C1 同项 0.01739），grad norm 峰值 **0.64**（C1 为 9.42）——低秩增量的梯度
规模小一个量级，拟合也更浅。

### Checkpoint selection（Formal / Closed-loop / Validation）

| Rank | Step | Macro SR | Offline norm MSE | Successes | 生效判据 |
|---:|---:|---:|---:|---:|---|
| 1 | **020000** | **0.9667** | 0.371719 | 58/60 | winner |
| 2 | 025000 | 0.9667 | 0.376928 | 58/60 | level 2 |
| 3 | 005000 | 0.9667 | 0.402103 | 58/60 | level 2 |
| 4 | 030000 | 0.9500 | 0.374399 | 57/60 | level 1 |
| 5 | 015000 | 0.9333 | 0.375955 | 56/60 | level 1 |
| 6 | 010000 | 0.9333 | 0.385587 | 56/60 | level 2 |

三个 checkpoint 在主判据上并列 0.9667，由第二判据分出 `020000`。两点与 C1 不同：C2 的最优 step
是 **20000 而非终点**（030000 反而掉到 0.9500），且 **005000 就已达 0.9667**（C1 同期 0.8833），
LoRA 收敛更快。

### 目标协议 test（Formal / Closed-loop / Test）

| 协议 | Z0（B0 零样本） | C1（Non-PEFT） | C2（LoRA） |
|---|---:|---:|---:|
| canonical | 0/90 = 0.0% | 87/90 = 96.7% | 87/90 = **96.7%** |
| held-out paraphrase | 2/90 = 2.2% | 30/90 = 33.3% | 19/90 = **21.1%** |
| prompt compliance | 0/30 = 0.0% | 29/30 = 96.7% | 28/30 = 93.3% |

canonical 与 C1 完全打平（逐任务 C2 为 29/30、30/30、28/30）。Prompt Swap 修复后，C2 的首次
达成率为 28/30（93.3%），stove 10/10、cabinet 10/10、plate 8/10；终态保持率同为 28/30。
C1 的首次达成率为 29/30（stove 9/10、cabinet 10/10、plate 10/10）。修复前的 stove 0/10 与
18/30 汇总值来自参考场景目标成功后的 reset，不能再说明 stove 不可达或训练方案无关。

**语言泛化差于 C1**：`para_gap` 点估计 +0.7556，bootstrap 95% CI `[0.6667, 0.8444]`，与 C1 的
`[0.5222, 0.7333]` 不重叠。C1 与 C2 的训练指令相同（都只用 canonical），所以这不是数据差异，
而是低秩参数化在单一指令上更容易贴死字面形式。

### 背景任务回退（Formal / Closed-loop / Test）

| task | 指令 | B0（记录） | C1 | C2 |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 30/30 | 0/30 | **3/30** |
| 2 | put the wine bottle on top of the cabinet | 27/30 | 0/30 | 0/30 |
| 3 | open the top drawer and put the bowl inside | 15/30 | 0/30 | 0/30 |
| 5 | push the plate to the front of the stove | 27/30 | 0/30 | **1/30** |
| 6 | put the cream cheese in the bowl | 26/30 | 0/30 | **2/30** |
| 7 | turn on the stove | 30/30 | 0/30 | **2/30** |
| 9 | put the wine bottle on the rack | 28/30 | 0/30 | 0/30 |
| **合计** | | **183/210** | **0/210** | **8/210 = 3.8%** |

**LoRA 没有防住灾难性遗忘。** 相对 B0 的保留率只有 4.4%，七个任务中四个仍为零，成绩最好的
task 0 从 30/30 掉到 3/30。C1 与 C2 的数据、LR、步数、batch、seed 完全相同，唯一差别是参数化
方式；在这个对照下遗忘程度几乎不变，说明根因是**训练数据里没有背景任务**，不是"训练了多少
参数"。

B0 的 183/210 是文档记录值（原始产物已丢失、未重跑），因此只能做非配对比较，无法给出配对
显著性。

### 推理成本高于 C1

| 指标 | C1 | C2 |
|---|---:|---:|
| 模型查询延迟 p50 | 285.3 ms | 376.8 ms |
| 模型查询延迟 p95 | 443.2 ms | 541.4 ms |
| action TV | 0.0268 | 0.0436 |
| action jerk | 0.0285 | 0.0588 |
| 平均 gripper 切换 | 3.76 | 10.52 |

两者 canonical 成功率相同，但 C2 的成功路径更抖。**LoRA 的成本优势在 trainable parameters、
optimizer state 与 adapter checkpoint 体积上，不在推理端。**

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
