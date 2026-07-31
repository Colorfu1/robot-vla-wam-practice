# SmolVLA LIBERO：B0 37 任务域适配

## 实验目标

从 `lerobot/smolvla_base` 出发，仅使用
`configs/experiments/smolvla_libero/background_split.json` 中 37 个非目标任务的
Train episodes，训练供 Z0/C1/C2/C3 共用的 LIBERO 域适配 checkpoint。

结果类别：正式训练为 `Formal / Offline`；两步链路检查为 `Smoke / Offline`。

## 实验思路

B0 排除目标 task indices `10,17,18`，使用 1,412 episodes、236,470 frames。
训练保持 LeRobot 原生全局按帧 shuffle，不做 task/episode balancing。
图像使用 ImageNet stats，state/action 使用完整 `lerobot/libero` metadata stats。
正式训练采用 8 卡、每卡 batch 8、BF16、30,000 updates，每 5,000 steps 保存一次。
训练沿用 SmolVLA Non-PEFT 默认冻结边界及完整 AdamW/cosine preset。
本阶段不做目标任务训练，也不把 train loss 当作闭环成功率。

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
  --output-dir=outputs/smolvla_libero/pilot/b0_domain_adaptation
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
  --resume-from=outputs/smolvla_libero/pilot/b0_domain_adaptation/checkpoints/005000
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

## 结果落点

```text
outputs/smolvla_libero/pilot/b0_domain_adaptation/
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
