# SmolVLA LIBERO：B0 37 任务域适配

## 实验目标

从 `lerobot/smolvla_base` 出发，仅使用
`configs/experiments/smolvla_libero/background_split.json` 中 37 个非目标任务的
Train episodes，训练供 Z0/C1/C2/C3 共用的 LIBERO 域适配 checkpoint。

结果类别：正式训练为 `Formal / Offline`；两步链路检查为 `Smoke / Offline`。

B0 三个阶段各用不同的 held-out 集合，标签也因此不同。这三段不可混着读：

| 阶段 | 结果类别 | 用哪个集合 | 规模 | 这个数字能回答什么 |
| --- | --- | --- | --- | --- |
| 正式训练 | `Formal / Offline` | 只用 37 任务的 train episodes | 1,412 episodes / 236,470 frames | 只是训练本身完成了。train loss 不能推断任何成功率 |
| checkpoint selection | `Formal / Closed-loop / Validation` | 7 个背景任务各 20 个 validation init states | 840 条 300-step rollout | 六个 checkpoint 里哪个最好。这个集合允许被反复看，因此它的成功率**不是**无偏成绩 |
| held-out test | `Formal / Closed-loop / Test` | 同 7 个任务各 30 个 test init states，此前完全保留 | 210 条 300-step rollout | 冻结 winner 的无偏成绩。看过之后不得加 seed、改 horizon 或重选 checkpoint |

标签第三段（`Validation` / `Test`）是本实验族的扩展，`reports/AGENTS.md` 的四个基础标签里
没有它。缺了这一段，读者无法判断一个成功率是否被选模过程看过。各集合的完整清单与相互约束见
[实验族总览](README.md#结果类别标签与数据分工)。

### B0 用到的数据

- 训练：37 个背景任务的 1,412 条 demonstration / 236,470 帧。
- 选模（`Validation`）：7 个 libero_goal 背景任务各 20 个初始状态 = 140 个 case，六个
  checkpoint 各跑一遍，`6 × 7 × 20 = 840` 条闭环 rollout。同一批 case 被看了六遍，成绩偏高，
  不是无偏结果。
- 最终成绩（`Test`）：只跑冻结的 winner，同 7 个任务各 30 个此前保留的初始状态，
  `7 × 30 = 210` 条闭环 rollout。看过之后不得加 seed、改 horizon 或重选 checkpoint。
- **未使用**：另有 138 条离线 validation demonstration / 23,004 帧从训练里划出，但没有任何
  入口消费它。因此 B0 的"开环动作误差"字段应写"未评测"，不能留空或由 train loss 推断。

初始状态是 LIBERO 每任务 50 个摆法的编号，不是数据；闭环按编号摆好物体让模型自己跑，不需要
参考轨迹。所以数据侧没有 test 并不妨碍跑 210 条 test rollout。

两条边界：这 7 个任务全在 B0 的 37 个训练任务内，所以 test 测的是同任务、未被使用过的摆放，
不是跨任务泛化；且摆放随机化范围本身只有约 3 cm，也不是远距离泛化。各集合的完整清单见
[实验族总览](README.md#结果类别标签与数据分工)。

## 冻结配置

| 字段 | 值 |
| --- | --- |
| Task suite | LIBERO-Goal 背景 37 任务 |
| Dataset size | 1,412 episodes / 236,470 frames |
| Observation keys | 两路 256×256 RGB + 8 维 `observation.state` |
| Action space | 7 维相对末端动作 |
| Action horizon | 50-step prediction / 10-step execution |
| Control frequency | 10 Hz |
| Initialization | `policies/smolvla_or_pi0/smolvla_base` 的原始 base 实体副本 |
| PEFT | 关闭 |
| Freeze boundary | vision encoder frozen；仅训练 expert 与 state projection |
| Batch | 8/GPU × 8 GPUs = 64 |
| Precision | BF16 autocast；参数/optimizer state 保持 FP32 |
| Optimizer | AdamW，LR `1e-4`，betas `(0.9,0.95)`，eps `1e-8` |
| Regularization | weight decay `1e-10`，global grad clip `10` |
| Scheduler | warmup 1,000；cosine decay 30,000；final LR `2.5e-6` |
| Checkpoints | 5k、10k、15k、20k、25k、30k |
| Train seed | 1000 |

## 本地模型准备

正式训练机器不需要 Hugging Face 网络。先在有原始 base cache 的工作站执行一次：

```bash
bash scripts/experiments/smolvla_libero/materialize_base_policy.sh
```

该命令只复制已有 cache，不会下载；目标目录是 Git 忽略的本地 ML artifact。制品包含：

- 外层 `model.safetensors`，其中已有 SmolVLA 和嵌套 SmolVLM 的完整参数；
- `vlm_assets/` 下约 5 MB 的 SmolVLM config、tokenizer 和 processor 文件；
- policy preprocessor/postprocessor。

`vlm_assets/` 不重复保存 SmolVLM 的约 1.9 GB `model.safetensors`。训练时先从
`vlm_assets/` 仅构造模型结构和文本/图像 processor，再由外层 checkpoint 一次性加载
全部参数。LeRobot 会独立加载 `policy_preprocessor.json` 中的 tokenizer step；B0
训练适配层也会把该 step 的 `tokenizer_name` 覆盖为同一个本地 `vlm_assets/`，避免
模型已经离线、processor 却仍访问 Hub 的双重加载问题。

将整个 `policies/smolvla_or_pi0/smolvla_base/` 同步到训练机器相同的仓库相对位置后，
B0 默认从此目录读取。启动入口固定设置 `HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，拒绝 Hub ID，并在创建数据集前
执行一次 `local_files_only=True, strict=True` 的完整权重加载检查。缺少任一离线文件
会立即失败，不会尝试联网。不要使用任何 B0 Smoke checkpoint 代替原始 base。

## 实验命令

### 两步单卡 Smoke

该命令只检查真实数据读取、base 权重加载、forward/backward、BF16、checkpoint 和运行记录：

```bash
bash scripts/experiments/smolvla_libero/train_b0.sh \
  --mode=smoke \
  --base-policy=policies/smolvla_or_pi0/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/background_split.json \
  --num-processes=1 \
  --batch-size=1 \
  --steps=2 \
  --save-freq=2 \
  --learning-rate=0.0001 \
  --train-seed=1000 \
  --num-workers=0 \
  --mixed-precision=bf16 \
  --output-dir=outputs/smolvla_libero/smoke/b0_2step
```

### 8 卡正式训练

```bash
bash scripts/experiments/smolvla_libero/train_b0.sh \
  --mode=formal \
  --base-policy=policies/smolvla_or_pi0/smolvla_base \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --split-file=configs/experiments/smolvla_libero/background_split.json \
  --num-processes=8 \
  --batch-size=8 \
  --steps=30000 \
  --save-freq=5000 \
  --learning-rate=0.0001 \
  --train-seed=1000 \
  --num-workers=4 \
  --mixed-precision=bf16 \
  --output-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline
```

只验证参数并查看最终 Accelerate 命令时，在任一 fresh 命令末尾添加：

```bash
--print-command
```

### 从最新完整 checkpoint 续训

`--resume-from` 必须精确指向该 run 中 step 最大的完整 checkpoint。续训不接受
batch、LR、seed、steps、输出目录或 GPU 数覆盖；启动脚本会从原 `run_config.json`
恢复 world size 和 BF16。

```bash
bash scripts/experiments/smolvla_libero/train_b0.sh \
  --resume-from=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/005000
```

如果已有更晚的完整 checkpoint，必须把上述路径替换为该 checkpoint；脚本不会自动
退回较旧状态。

## 严格 checkpoint selection

结果类别：`Formal / Closed-loop / Validation`。

训练完成后，使用
`configs/experiments/smolvla_libero/background_closed_loop_split.json` 冻结的七个
非目标 LIBERO-Goal 任务进行选模。每个任务固定 20 个 validation init states，另外
30 个 test init states 保留到选模结束后；两组互斥并覆盖官方 `0..49`。六个 5k 间隔
checkpoint 共执行 `6 × 7 × 20 = 840` 条完整 300-step 闭环 rollout。

所有 rollout 在开始前都把 Python、NumPy、PyTorch CPU/CUDA policy RNG 恢复为
`policy_seed=2000`，并清空 action queue。环境 seed 固定按 episode index 配对。主排序
指标是七任务 macro success rate；并列时依次比较 mean max reward（高优先）、成功
episode 平均步数（低优先）和 checkpoint step（早优先）。validation 以外的数据和
train loss 均不参与选择。

在远端 8 卡机器执行：

```bash
bash scripts/experiments/smolvla_libero/select_b0_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline \
  --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/005000/pretrained_model \
  --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/010000/pretrained_model \
  --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/015000/pretrained_model \
  --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/020000/pretrained_model \
  --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/025000/pretrained_model \
  --checkpoint-path=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/030000/pretrained_model \
  --split-file=configs/experiments/smolvla_libero/background_closed_loop_split.json \
  --num-gpus=8 \
  --env-seed=1000 \
  --policy-seed=2000 \
  --libero-assets-dir=datasets/raw/libero_assets \
  --output-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoint_selection \
  --selected-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected
```

六个 `--checkpoint-path` 必须是仓库相对路径，并分别指向完整的 `pretrained_model`
目录；入口要求它们恰好覆盖冻结的 `005000..030000` 候选集合，缺少、重复、跨 run 或使用
绝对路径都会在启动 GPU job 前失败。省略这些参数时仍兼容从 `--run-dir/checkpoints`
自动发现，但正式命令使用显式路径，便于远端和本地复现完全相同的候选集合。

`datasets/raw/libero_assets` 必须是完整离线资产树，并至少包含
`scenes/libero_tabletop_base_style.xml`。入口会在创建 GPU 子进程前统一校验资产，绝不在
评测过程中回退到 Hugging Face 下载。每完成一个 episode 就更新总进度，并每 15 秒刷新
active-job 心跳；episode 进度通过独立于 logging level 的 line-buffered stdout 协议强制
flush，任何 task job 失败都会立即终止其他子进程并指出对应日志。

若进程中断，原命令末尾增加 `--resume`。入口会校验并复用已完成且 init-state 协议完全
匹配的 task job，只重跑缺失或不完整项。启动前只检查解析后的工作量而不执行时，末尾
增加 `--print-command`。

### 只发布已选 checkpoint（不跑 rollout）

若 selection 已完成但 `checkpoints/selected/` 的实体副本缺失或损坏，不需要重跑任何
validation rollout，直接用发布入口把 `selected_checkpoint.json` 指向的源 checkpoint
实体复制到目标目录：

```bash
bash scripts/experiments/smolvla_libero/publish_selected_b0.sh \
  --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline \
  --selection-file=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoint_selection/selected_checkpoint.json \
  --selected-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/checkpoints/selected
```

发布前必须校验：`status=complete`、`selection_set=background_validation`、
`selected_checkpoint_step` 与 `source_checkpoint` 目录名一致，且若目标
`selected/pretrained_model/model.safetensors` 已存在，其 sha256 必须与源 checkpoint
相同；任一不符则拒绝发布。目标目录已存在时需要 `--overwrite`。先加
`--print-command` 只打印命令。

选中的 checkpoint 会被实体复制，而不是创建软链接：

```text
outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/
├── checkpoint_selection/
│   ├── checkpoint_results.json
│   ├── selected_checkpoint.json
│   ├── selection_manifest.json
│   ├── metrics.md
│   ├── jobs/
│   └── logs/
└── checkpoints/
    └── selected/
        ├── pretrained_model/
        ├── training_state/
        └── selected_checkpoint.json
```

这里的 validation/test 指选模协议中的官方闭环初始状态划分；当前数据转换没有提供
demonstration 与官方 init-state ID 的一一对应关系，因此不能宣称这些状态必然从未在
B0 demonstrations 中出现。它们可以保证 test 不参与 checkpoint selection，但不能替代
未来可追溯 demonstration-state mapping 的严格训练状态去重。

## Selected B0 held-out test

结果类别：`Formal / Closed-loop / Test`。

checkpoint selection 完成后，只评测 `selected_checkpoint.json` 已经冻结的 winner。入口
直接校验 selection 的 split hash、排名第一项和源 checkpoint，不根据 test 结果重新选择
模型。七个 background tasks 各使用此前完全保留的 30 个 test init states，共执行
`7 × 30 = 210` 条完整 300-step 闭环 rollout。

一键入口会先复用已经完成的 validation job，对 `020000`、`025000`、`030000` 在相同
140 个 task/init-state case 上进行配对分析。它不会启动新 simulator rollout，输出包括：

- `20k → 25k`、`25k → 30k` 和 `20k → 30k` 的 gain/regression case；
- 逐任务成功迁移；
- exact McNemar 配对检验；
- 共同成功 case 上的 step 差异；
- 原始排名与注册 tie-breaker 的一致性审计。

该 validation diagnostic 只能解释 winner 的稳定性，不允许据此更改已经冻结的 winner。
分析完成后，同一入口才会启动 `025000` 的 held-out test。

在远端 8 卡机器一键执行：

```bash
bash scripts/experiments/smolvla_libero/eval_b0_test.sh \
  --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline \
  --num-gpus=8 \
  --libero-assets-dir=datasets/raw/libero_assets
```

默认读取和写入：

```text
outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/
├── checkpoint_selection/
│   ├── checkpoint_results.json
│   ├── selected_checkpoint.json
│   ├── validation_paired_analysis.json
│   └── validation_paired_analysis.md
└── background_test/
    ├── test_results.json
    ├── test_manifest.json
    ├── metrics.md
    ├── jobs/task_*/metrics.json
    └── logs/task_*.log
```

该入口从 selection artifact 的 `source_checkpoint` 读取 winner，因此即使
`checkpoints/selected` 的实体复制曾因远端文件系统不支持目录原子 rename 而失败，也不
需要重新执行 840 条 validation rollout。selection 发布入口已彻底移除目录
rename/replace：先清理旧目录，再从源 checkpoint 直接 `copytree` 为实体副本。

如果只想先生成 validation 配对诊断而不启动 test rollout，可以执行：

```bash
bash scripts/experiments/smolvla_libero/analyze_b0_validation.sh \
  --run-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation_offline
```

运行中断后在完全相同命令末尾添加 `--resume`；入口只复用 policy path、environment
seeds、policy seed、init states 和 `n_action_steps` 全部匹配的完整 task job。只检查冻结
工作量而不启动 GPU 时添加 `--print-command`，应显示 selected step、七个 task 和 210
条 test rollout。

test 结果一旦查看，不得使用 `--overwrite` 增加 seed、修改执行 horizon 或重新选择
checkpoint。`test_results.json` 保存总体与逐任务成功率、成功耗时、失败终止原因、动作
平滑度和 latency 汇总；`metrics.md` 是供阅读的正式结果入口。

## 已完成的正式结果

训练、选模和 held-out test 都已跑完。以下数字由 `checkpoint_results.json`、
`test_results.json` 及各 job 的 `metrics.json` 复算核对一致。

### 训练（Formal / Offline）

一次 attempt 完成 30,000 updates，8 个 rank 状态全为 `complete`。loss、grad norm、LR
全程有限，无 NaN/Inf。LR 在 step 999 达到峰值 `9.990e-5`，随后 cosine 衰减，step 30000
正好落在 `2.5e-6`。末 1,000 步 loss 均值 `0.064`（首步 `0.607`）。`run_config.json` 记录的
1,412 episodes / 236,470 frames 与 `background_split.json` 的 aggregate 一致，split
sha256 与当前文件一致。

### Background validation checkpoint selection（Formal / Closed-loop / Validation）

6 个 checkpoint × 7 任务 × 20 validation init states = 840 条 300-step rollout。
每个 job 使用的 init state 集合与 split 中的 `validation_init_state_ids` 完全一致。

| checkpoint | macro success | 成功数 | 成功 episode 平均步数 |
|---:|---:|---:|---:|
| 5000 | 0.3286 | 46/140 | 141.74 |
| 10000 | 0.7143 | 100/140 | 121.81 |
| 15000 | 0.7929 | 111/140 | 126.96 |
| 20000 | 0.7929 | 111/140 | 123.06 |
| **25000** | **0.8643** | **121/140** | **120.85** |
| 30000 | 0.8643 | 121/140 | 122.21 |

25000 与 30000 在主指标上并列，按注册顺序由成功 episode 平均步数分出 25000
（120.85 < 122.21）。`validation_paired_analysis.json` 的 `ranking_audit` 为
`matches_registered_tie_breakers: true`；配对诊断中 25k→30k 的 exact McNemar
p = 1.0、净增益 0，没有改动已冻结的 winner。

注意第一个 tie-breaker `mean_max_reward_desc` 在 LIBERO 上不起作用：奖励是稀疏 0/1，
六个 checkpoint 的 `mean_max_reward` 全部等于对应的成功率，因此它永远无法打破平局，
实际生效的链条是成功步数、然后 checkpoint step。这一项保留在协议里只为记录完整，
不要当作独立判据。

### Selected B0 held-out test（Formal / Closed-loop / Test）

冻结 winner `025000`，7 任务 × 30 test init states = 210 条 300-step rollout，
全部使用此前保留、未参与选模的 30 个 init state。

| task | 指令 | 成功率 | 成功 episode 平均步数 |
|---:|---|---:|---:|
| 0 | open the middle drawer of the cabinet | 30/30 | 137.53 |
| 2 | put the wine bottle on top of the cabinet | 27/30 | 104.89 |
| 3 | open the top drawer and put the bowl inside | 15/30 | 185.00 |
| 5 | push the plate to the front of the stove | 27/30 | 137.93 |
| 6 | put the cream cheese in the bowl | 26/30 | 94.69 |
| 7 | turn on the stove | 30/30 | 71.77 |
| 9 | put the wine bottle on the rack | 28/30 | 153.11 |
| **合计** | macro success | **183/210 = 0.8714** | 122.18 |

失败全部是 `time_limit`（27 条），没有其他终止原因。动作平滑度 macro action TV
`0.0345`、macro action jerk `0.0466`、macro 平均 gripper 切换 `2.69` 次。模型查询延迟
macro p50 `253.2 ms`、p95 `348.9 ms`。最弱任务是 task 3（open the top drawer and put the
bowl inside，0.50），也是选模阶段最弱的一项（10/20）。

**这里的 held-out 只针对 init state。** 这 7 个任务本身都在 B0 的 37 个训练任务里，
因此该结果衡量的是同任务、未见初始状态下的表现，不是跨任务泛化。跨任务迁移由目标
三任务的 Z0 baseline 回答，不能用本节数字代替。

### 任务编号的两套命名空间

`background_split.json` 的 key 是 `lerobot/libero` 的全局 `task_index`（数据侧，0..39），
而 `background_closed_loop_split.json` 与 `target_split.json` 的 key 是 LIBERO 官方
libero_goal 本地编号（仿真侧，0..9）。同一个整数在两处含义不同：例如闭环 split 的
`0` 是 "open the middle drawer of the cabinet"，而全局 `task_index=0` 是一条 LIBERO-10
任务。两边靠各自文件里的 `canonical` 文本交叉校验；49 个 job 的 `task.language` 与 split
canonical 逐条一致。被排除的目标任务在数据侧是全局 `10/17/18`，对应仿真侧本地
`8/1/4`，即 "put the bowl on the plate" / "on the stove" / "on top of the cabinet"。
改动任一 split 时必须同时核对 canonical 文本，不要只看数字。

## 结果落点

```text
outputs/smolvla_libero/pilot/b0_domain_adaptation_offline/
├── run_config.json
├── train_metrics.jsonl
├── train.log
├── runtime_attempt_<attempt>_rank_<rank>.json
└── checkpoints/
    ├── 005000/
    ├── 010000/
    ├── 015000/
    ├── 020000/
    ├── 025000/
    └── 030000/
```

- `run_config.json`：split/hash、base revision/hash、dataset、sampler、normalization、
  world size、完整 optimizer/scheduler 和 resolved train config。
- `train_metrics.jsonl`：每个成功 optimizer update 的 loss、grad norm、LR、读取/更新
  耗时及 `attempt_id`。
- `runtime_attempt_*.json`：每个 rank 的耗时、峰值 allocated/reserved VRAM 和完成状态。
- checkpoint：`pretrained_model/` 与 `training_state/`，可恢复 model、optimizer、
  scheduler、RNG 和 step。

本训练入口不生成闭环 `metrics.json`、视频或 `rollout*.rrd`。B0 checkpoint selection
和 Z0 闭环评测是后续步骤。

## 判断标准

Smoke 通过条件：

- `checkpoints/000002/` 同时包含 `pretrained_model/` 和 `training_state/`；
- `train_metrics.jsonl` 包含 step 1、2，loss/grad norm/LR 为有限值；
- rank 0 runtime 状态为 `complete`，日志中无 CUDA OOM 或数据 feature 错误。

Formal 训练通过条件：

- 8 个 rank 均以 BF16 完成，global batch 明确为 64；
- 30,000 updates 和六个预定 checkpoint 全部存在；
- loss、grad norm、LR 无 NaN/Inf，LR 符合 warmup/cosine 轨迹；
- 最终 train loss、开放环 action error、闭环成功率、延迟和典型失败案例分别记录；
  尚未评测的字段必须写为“未评测”，不能由 train loss 推断。

B0 的最终 checkpoint 不能仅按训练 loss 直接指定。只有上述 background closed-loop
Validation selection 完成、`checkpoints/selected/` 成为实体副本后，才运行
Z0/C1/C2/C3。
