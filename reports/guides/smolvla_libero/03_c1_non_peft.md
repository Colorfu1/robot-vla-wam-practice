# SmolVLA LIBERO：C1 Default Non-PEFT

<!-- PROMPT_SWAP_MERGE_NOTICE:BEGIN -->
## Prompt Swap 修复后当前结果

**C1/Z0 Prompt Swap 已合并；本文旧 Prompt Swap 数值仅为历史快照。**

以下是当前有效的结果；本文其余位置中修复前的 Prompt Swap 数值、混淆矩阵和由此得到的结论，均只保留作历史审计，不得继续引用。

| Model | First-goal compliance | Terminal-goal compliance | Full horizon |
| --- | ---: | ---: | ---: |
| z0_b0_zero_shot | 0/30 = 0.0% | 0/30 = 0.0% | 30/30 |
| c1_default_non_peft | 29/30 = 96.7% | 27/30 = 90.0% | 30/30 |

主指标是每一步 simulator predicate 的 `first_goal_reached`；`terminal_goal` 只表示达成后的保持性诊断。
这次合并只替换 Prompt Swap 的 job/rollout/episode 与汇总结果，canonical、held-out paraphrase、训练和 checkpoint selection 结论不变。
详细合并报告：`reports/2026-08-06-PromptSwap修复后合并结果.md`。
<!-- PROMPT_SWAP_MERGE_NOTICE:END -->

## 实验目标

建立与 LoRA 配置共享数据、训练步数和优化设置的非 PEFT 对照，判断 C2 是否接近 SmolVLA
默认任务适配性能。

结果类别：训练为 `Offline`，checkpoint selection 为 `Closed-loop`。

## 实验思路

C1 从统一 selected B0 checkpoint 开始，不使用 PEFT，保留 SmolVLA 默认冻结边界。
它训练 action expert 和相关 projections，但仍冻结 VLM 与 vision encoder。只有在单卡
BF16 batch 1 Smoke 通过后才运行；不能缩小模型后继续称为同一个上限对照。

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
- 指令：只用 canonical；8 条 train paraphrases 与 5 条 held-out paraphrases 都不参与 C1。

test 初始状态与 held-out paraphrases 都不得进入选模。held-out 只针对初始状态——这 3 个任务
正是本实验训练的任务，且摆放随机化范围约 3 cm，所以 test 测的是同任务同分布、未被使用过的
摆放。完整清单见[实验族总览](README.md#结果类别标签与数据分工)。

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

训练入口严格消费 `target_split.json` 中 113 条 Train demonstrations，并使用
`uniform_task_then_uniform_episode`：先均匀采样任务，再均匀采样该任务内 episode，最后
均匀采样 episode 内帧。30 条 Offline Validation demonstrations 不会进入训练。

## 训练命令

正式配置已经冻结在 launcher 中。B0 checkpoint selection 完成并生成
`selected/pretrained_model` 后，以下命令可直接启动 8 卡训练：

```bash
bash scripts/experiments/smolvla_libero/train_c1.sh \
  --mode=formal \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/target_split.json \
  --output-dir=outputs/smolvla_libero/pilot/c1_default_non_peft
```

`--mode`、`--dataset-root`、`--split-file`、`--output-dir` 是 fresh 启动的必填参数，
不允许裸跑。该命令固定使用：

- `outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model`
- `datasets/lerobot/lerobot/libero`
- `configs/experiments/smolvla_libero/target_split.json`
- 默认 8 个 Accelerate processes、每卡 batch 1、全局 batch 8（合法布局：
  8×1、4×2、2×4、1×8）
- 30,000 steps、每 5,000 steps 保存、BF16、seed 1000、LR `1e-4`
- 全离线 Hugging Face/Transformers/Datasets 模式

正式模式会读取 selected checkpoint 旁的 `selected_checkpoint.json`，并确认它确实由 B0
background validation 闭环 selection 产生。缺失 provenance 时 fail closed，不会退回
`last` 或任意 step。

启动前只检查完整命令、不运行训练：

```bash
bash scripts/experiments/smolvla_libero/train_c1.sh --print-command
```

### 多 GPU 启动语义

- 无参数正式入口是单机 8 卡：Accelerate 使用 8 个 process、`--gpu_ids=all` 和
  `--multi_gpu`，每张卡对应一个训练进程。
- C1 正式协议冻结的是**全局 batch 8**（同时冻结 30,000 steps、每 5,000 保存、
  LR `1e-4`、seed 1000、BF16）。卡数可配置为 8×1、4×2、2×4、1×8 任一合法布局；
  `--num-processes=N` 必须满足 `N × --batch-size = 8`，且不能超过可见 GPU 数。
- 不同卡数的 run **协议等价但不是逐位复现**：sampler 分片与数据读取顺序随 world size
  改变，loss 曲线不会重合。报告必须写明 `data_parallel_layout`（如 `4x2`），不能把
  不同布局当成同一次运行的重复。
- 少于 8 卡时每卡 batch 会上升（1 卡需 batch 8），显存风险随之上升；OOM 时按下面的
  `N_A.json` 契约记录，不缩小模型、不降全局 batch。
- `--num-processes` 超过可见 GPU 数会立即失败并提示用 `CUDA_VISIBLE_DEVICES` 选卡
  （launcher 固定 `--gpu_ids=all`，选特定卡只能通过环境变量）。
- 续训从原始 `run_config.json` 恢复 world size、每卡 batch 与 BF16，必须提供与原 run
  相同数量的 GPU（`world_size × per_gpu_batch_size` 仍必须等于 8），不能在 resume 时
  改变卡数。
- 当前 launcher 支持单机多卡，不包含跨节点 rendezvous 配置。

单卡 Smoke：

```bash
bash scripts/experiments/smolvla_libero/train_c1.sh \
  --mode=smoke \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/target_split.json \
  --base-policy=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected/pretrained_model \
  --steps=2 \
  --save-freq=2 \
  --num-workers=0 \
  --output-dir=outputs/smolvla_libero/debug/c1_smoke
```

从最新完整 checkpoint 续训：

```bash
bash scripts/experiments/smolvla_libero/train_c1.sh \
  --resume-from=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/005000
```

VS Code 可选择 `SmolVLA LIBERO C1: train (1-step offline debug)` 做单卡单步调试；它使用
固定 output 目录并带 `--overwrite`，每次重跑直接覆盖上一次 debug 结果。

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
  --split-file=configs/experiments/smolvla_libero/target_split.json \
  --checkpoint-steps=5000,10000,15000,20000,25000,30000 \
  --policy-seed=1500 \
  --output-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoint_selection \
  --selected-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected
```

该入口已实现（`scripts/experiments/smolvla_libero/select_checkpoint.sh`）。判据顺序：

1. 三任务 macro canonical closed-loop SR（高优先）；
2. 平局时比较 macro **normalized** offline action error（低优先；denormalized 误差同时
   记录但只作参考）；
3. 仍平局时选择更早 checkpoint。

Open-loop error 的固定条件：同一批 frame（30 条 offline validation demonstration 的
全部 2,969 帧，顺序由 split 固定）、固定 flow-noise seed（按 task/episode/frame 从
split seed 派生，同一 frame 在所有 checkpoint 上使用同一份噪声）、同一 action horizon
（chunk 50、只比较前 10 步）；先按任务求均值，再对三个任务取 macro average。

闭环规模为 `6 × 3 × 20 = 360` 条 300-step rollout；`ranking_audit` 记录实际生效的判据
级别，位于 `checkpoint_selection/ranking_audit.json`。**不得使用 test split 或
held-out paraphrases 选模**。

开环误差按 checkpoint 并行评测：每个 checkpoint 分派到一张 GPU（并行度 =
`min(num_gpus, checkpoint 数)`），6 卡下预计约 12 分钟（单卡串行约 69 分钟）。子进程日志
在 `checkpoint_selection/logs/open_loop_checkpoint_<step>.log`（首行为完整命令），中间结果
在 `checkpoint_selection/jobs/open_loop/checkpoint_<step>.json`；`--resume` 会复用
`checkpoint_id` / `split_sha256` / `max_frames_per_task` 一致的已完成结果。并行调度不改
抽帧、噪声派生与误差公式，数值与串行逐值相同。

## 已完成的正式结果

训练、选模与 210 条目标协议评测都已跑完。完整对比、协议一致性核对与结论边界见
[2026-08-04 Z0 基线与 C1 目标适配正式评测结论](../../2026-08-04-Z0基线与C1目标适配正式评测结论.md)。

### 训练（Formal / Offline）

一次 attempt 完成 30,000 updates，`data_parallel_layout=8x1`，8 个 rank 状态均为 `complete`。
loss 首步 0.1599、末 1,000 步均值 0.01739，grad norm 峰值 9.42，LR 峰值 9.99e-5、终值 2.5e-6，
全程无 NaN/Inf。

### Checkpoint selection（Formal / Closed-loop / Validation）

360 条闭环 rollout + 每个 checkpoint 在 2,969 帧上的开环误差。

| Rank | Step | Macro SR | Offline norm MSE | Successes | 生效判据 |
|---:|---:|---:|---:|---:|---|
| 1 | **030000** | **0.9833** | 0.309668 | 59/60 | winner |
| 2 | 025000 | 0.9833 | 0.311054 | 59/60 | level 2 |
| 3 | 015000 | 0.9833 | 0.324043 | 59/60 | level 2 |
| 4 | 020000 | 0.9667 | 0.310969 | 58/60 | level 1 |
| 5 | 010000 | 0.9500 | 0.335584 | 57/60 | level 1 |
| 6 | 005000 | 0.8833 | 0.389176 | 53/60 | level 1 |

三个 checkpoint 在主判据上并列，由第二判据分出 `030000`。macro 成功率粒度只有 1/60，
并列是常态，所以开环判据在 C 家族会真正生效。**0.9833 不是无偏成绩**——同一批 60 个 case
被六个 checkpoint 各看过一遍。

### 目标协议 test（Formal / Closed-loop / Test）

与 Z0（selected B0 原样零样本）在同一份冻结协议上各跑 210 条：

| 协议 | Z0（B0 零样本） | C1（目标适配） |
|---|---:|---:|
| canonical | 0/90 = **0.0%** | 87/90 = **96.7%** |
| held-out paraphrase | 2/90 = 2.2% | 30/90 = 33.3% |
| prompt compliance | 0/30 = 0.0% | 29/30 = 96.7% |

逐任务 canonical：C1 为 29/30、29/30、29/30；Z0 三任务全 0/30，90 条全部 `time_limit`。

模型查询延迟 p50 / p95：Z0 310.1 / 471.3 ms，C1 285.3 / 443.2 ms。动作平滑度 C1 明显更好
（action TV 0.0268 对 0.0483，jerk 0.0285 对 0.0706，平均 gripper 切换 3.76 对 18.62）。

三点读数纪律：

- **113 条示范就足以学会这三个任务，而 B0 完全不会做。** Z0 canonical 0/90 说明本设置下
  SmolVLA 没有跨任务零样本迁移能力，任务适配是必需的。
- **C1 学到的是指令与动作的固定绑定，不是语言理解。** canonical 96.7% 掉到改写句 33.3%，
  `para_gap` 点估计 +0.6333、bootstrap 95% CI `[0.5222, 0.7333]`，不含 0。这正是 C3 要针对的
  问题，本次为它提供了基线。
- **Prompt Swap 修复后，C1 首次达成率为 29/30（96.7%）。** 首次达成混淆矩阵为 stove 9/10、
  cabinet 10/10、plate 10/10；终态保持率为 27/30（90.0%），其中 2 条 plate episode 在首次
  达成后没有保持到第 300 步。
- 修复前的 18/30 与 stove 0/10 是参考场景原生目标成功后 reset 导致的判定错误，不能再用来说明
  stove 不可达或模型“不听指令”。当前 Prompt Swap 主指标使用 `first_goal_reached`，
  `terminal_goal` 只作保持性诊断。

最后，test 的 30 个初始状态与训练摆放同属约 3 cm 范围的同一分布，所以 96.7% 衡量的是同任务、
未被使用过的摆放，不是远距离泛化。

### 背景任务回退（Formal / Closed-loop / Test）

**目标任务的 96.7% 是以全部背景能力为代价换来的。** 用同一份背景 test 协议
（7 个 libero_goal 背景任务 × 30 个 test 初始状态 = 210 条）评测 C1 的 selected `030000`：

| task | 指令 | B0（记录） | C1（实测） | 差 |
|---:|---|---:|---:|---:|
| 0 | open the middle drawer of the cabinet | 30/30 | **0/30** | −30 |
| 2 | put the wine bottle on top of the cabinet | 27/30 | **0/30** | −27 |
| 3 | open the top drawer and put the bowl inside | 15/30 | **0/30** | −15 |
| 5 | push the plate to the front of the stove | 27/30 | **0/30** | −27 |
| 6 | put the cream cheese in the bowl | 26/30 | **0/30** | −26 |
| 7 | turn on the stove | 30/30 | **0/30** | −30 |
| 9 | put the wine bottle on the rack | 28/30 | **0/30** | −28 |
| **合计** | | **183/210** | **0/210** | **−183** |

210 条全部 `time_limit`，每条跑满 300 步，七个任务无一例外。这是彻底的灾难性遗忘，不是
"泛化变差"。冻结边界（`freeze_vision_encoder=True` + `train_expert_only=True`）**没有**拦住
它——action expert 正是决定动作的部分。

两点必须一起读：

- **在目标任务上 C1 并没有过拟合。** 选模曲线从 step 15000 起持平 0.9833、开环误差单调下降到
  30000、validation 0.9833 与 test 0.9667 只差 1.7 个点。问题不在目标任务侧。
- **B0 的 183/210 是文档记录值，不是本次实测。** 原始产物已丢失且未重跑，因此没有逐 episode
  配对，无法做 McNemar。不过在 0 对 183 的量级下这不影响结论。

诊断中发现两处配置不对称：`FORMAL_LEARNING_RATE = 1e-4` 定义在 `b0_training.py` 并被 C1 直接
导入，但 B0 的 global batch 是 64、C1 是 8，LR 没有随之缩放；同时目标任务在 step 15000 就
饱和，后 15000 步对目标任务毫无贡献却继续漂离背景能力。这两条由
[C1 遗忘权衡曲线诊断](../../../plans/2026-08-05-C1遗忘权衡曲线诊断.md) 定量回答。

## 遗忘缓解诊断与后续方向

早停与按 batch 缩放 LR 已完成 A/B 诊断；背景回放的 `k=4/8` 探针也已完成。完整对比见
[C1 遗忘诊断与背景数据回放结论](../../2026-08-06-C1遗忘诊断与背景数据回放结论.md)。这些结果都属于
diagnostic scope，不改变正式 C1 的 target protocol 结果，也不产生可直接发布的 checkpoint。

### Diagnostic 训练命令

以下命令只允许写入 `outputs/smolvla_libero/diagnostic/`，不能覆盖正式 C1 目录。

同配置复跑 A：

```bash
bash scripts/experiments/smolvla_libero/train_c1.sh \
  --mode=diagnostic \
  --experiment-tag=a_same_config \
  --num-processes=8 --batch-size=1 \
  --steps=30000 --save-freq=5000 \
  --learning-rate=0.0001 \
  --output-dir=outputs/smolvla_libero/diagnostic/c1_a_same_config
```

按 global batch 线性缩放 LR 的 B：

```bash
bash scripts/experiments/smolvla_libero/train_c1.sh \
  --mode=diagnostic \
  --experiment-tag=b_scaled_lr \
  --num-processes=8 --batch-size=1 \
  --steps=30000 --save-freq=5000 \
  --learning-rate=0.0000125 \
  --output-dir=outputs/smolvla_libero/diagnostic/c1_b_scaled_lr
```

混入一半 background 数据的 Replay `k=4/8`：

```bash
bash scripts/experiments/smolvla_libero/train_c1.sh \
  --mode=diagnostic \
  --experiment-tag=replay_k4 \
  --background-replay=4 \
  --num-processes=8 --batch-size=1 \
  --steps=30000 --save-freq=5000 \
  --learning-rate=0.0001 \
  --output-dir=outputs/smolvla_libero/diagnostic/c1_replay_k4
```

A/B 的六 checkpoint 目标选模与背景曲线命令见
[C1 遗忘权衡曲线诊断计划](../../../plans/2026-08-05-C1遗忘权衡曲线诊断.md)；Replay 的选模与
回放比例扩展命令见[背景数据回放计划](../../../plans/2026-08-05-背景数据回放缓解遗忘.md)。

当前已完成的 Replay selected checkpoint background test 可复现为：

```bash
bash scripts/experiments/smolvla_libero/eval_b0_test.sh \
  --run-dir=outputs/smolvla_libero/diagnostic/c1_replay_k4 \
  --policy-path=outputs/smolvla_libero/diagnostic/c1_replay_k4/checkpoints/selected/pretrained_model \
  --model-label=c1_replay_k4_selected \
  --split-file=configs/experiments/smolvla_libero/background_closed_loop_split.json \
  --num-gpus=8 \
  --env-seed=1000 \
  --policy-seed=2000 \
  --n-action-steps=10 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/diagnostic/c1_replay_k4/background/selected
```

**背景数据回放（`k=4/8`，已完成一条探针）。** 在 LR `1e-4` 不变的情况下，将每个 global
batch 的一半样本取自 B0 background train pool。selected step 20,000 的 target validation 为
59/60（98.3%），background test 为 149/210（71.0%）；相同 target-only 配置 A 为 0/210，按
batch 缩放 LR 的 B selected 为 55/210。由于 replay 训练已经见过 background demonstration，
149/210 是混合训练后的 background 能力/保持结果，不能与 A/B 的纯遗忘读数完全等同。

按回放计划的预注册规则，`k=4/8` 达到继续条件，因此 `k=2/8` 与 `k=6/8` 仍待执行；当前没有
完整的 replay step 曲线。

**后续设计方向。** Replay 是目前最有希望的缓解方向，但新增了「目标 : 背景」采样比例这个
超参，背景侧有 236,470 帧（目标侧只有 11,022 帧），比例怎么定需要单独扫；且混入后目标任务
的收敛速度与饱和点会改变，选模协议需要重新校准。若要把某个 replay 配置变成正式模型，必须
另起计划、重训，并重新执行完整 target protocol 与 background test。

**~~TODO-2：改用 LoRA（即 C2/C3）~~ → 已验证，结果为负。** C2（Canonical LoRA，rank 32）已完成
训练、选模与两类正式评测，结论是 **LoRA 没有防住灾难性遗忘**：

| | 目标 canonical | held-out paraphrase | 背景任务 |
|---|---:|---:|---:|
| B0（记录） | 0/90 | 2/90 | 183/210 |
| C1（Non-PEFT） | 87/90 = 96.7% | 30/90 = 33.3% | **0/210** |
| C2（LoRA rank 32） | 87/90 = 96.7% | 19/90 = 21.1% | **8/210 = 3.8%** |

目标任务完全打平（都 87/90），背景任务只从 0/210 回到 8/210——相对 B0 的保留率 4.4%，七个背景
任务中四个仍为 0/30。语言泛化还更差（21.1% 对 33.3%，`para_gap` 的 bootstrap 95% CI
`[0.667, 0.844]` 与 C1 的 `[0.522, 0.733]` 不重叠）。推理成本也更高（查询 p50 376.8 ms 对
285.3 ms，jerk 0.0588 对 0.0285）。

C1 与 C2 的数据、LR、步数、batch、seed 完全相同，唯一差别是参数化方式。在这个对照下遗忘程度
几乎不变，说明**根因不是"训练了多少参数"，而是训练数据里没有背景任务**——损失函数里不存在
保留旧行为的项。换参数化方式改不了这一点。详见
[C2 Canonical LoRA 正式评测结论](../../2026-08-05-C2%20Canonical%20LoRA%20正式评测结论.md)。

因此可行方向收窄为两条：改数据（背景回放）或缩短优化路径（早停、降 LR，见
[C1 遗忘权衡曲线诊断](../../../plans/2026-08-05-C1遗忘权衡曲线诊断.md)）。C3 仍需跑同一份背景
test，但先验上不应期待它比 C2 好——它与 C2 的差别只在训练指令，不在参数化。

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

Diagnostic 结果不写入上述正式目录，分别落在：

```text
outputs/smolvla_libero/diagnostic/
├── c1_a_same_config/
├── c1_b_scaled_lr/
└── c1_replay_k4/
    ├── checkpoint_selection/
    └── background/selected/
```

完整诊断结果与限制见[C1 遗忘诊断与背景数据回放结论](../../2026-08-06-C1遗忘诊断与背景数据回放结论.md)。

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

下一步：[C2 Canonical LoRA](04_c2_canonical_lora.md)。
