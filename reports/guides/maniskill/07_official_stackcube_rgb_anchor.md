# ManiSkill 官方 StackCube RGB Golden Anchor 指南

## 重要进展

- 已新增独立官方复现 runner，和 M1/M2 的数据、模型、评测代码隔离。
- 固定 `StackCube-v1`、官方 motionplanning demos、`pd_ee_delta_pos` 四维 action、RGB、`physx_cpu`。
- 固定官方当前 RGB 配方：100 demos、100,000 iterations、`max_episode_steps=200`。
- 训练内评测和最终验证都使用官方 `ignore_terminations=True` / `success_once` 协议。
- 当前只完成代码和静态检查，远端训练与 rollout 尚未执行；不要把本指南当成已有实验结果。

## 目的

这是定位 M2-W / LeRobot 管线问题的第一个 golden anchor。它不使用本仓库的 LeRobot 转换器、M2-W
模型、M2 Diffusion 模型、GPU simulator 或自定义 evaluator，只调用当前 ManiSkill checkout 中的
官方 Diffusion Policy 实现。

官方当前 `baselines.sh` 的 StackCube RGB 配方就是：100 demos、`pd_ee_delta_pos`、`physx_cpu`、
`max_episode_steps=200`、`total_iters=100000`。

仓库内置 baseline 的 evaluator 额外包含一个 Gymnasium 1.x info 容器兼容层：同时支持官方旧式
`final_info` 和当前 ManiSkill CPU wrapper 的 `episode`/`_episode`，不改变 success 指标计算。

官方资料：

- [ManiSkill Diffusion Policy baselines.sh](https://raw.githubusercontent.com/haosulab/ManiSkill/main/examples/baselines/diffusion_policy/baselines.sh)
- [ManiSkill Diffusion Policy train_rgbd.py](https://raw.githubusercontent.com/haosulab/ManiSkill/main/examples/baselines/diffusion_policy/train_rgbd.py)
- [ManiSkill Learning from Demonstrations setup](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html)

## 固定协议

| 项目 | 固定值 |
| --- | --- |
| 任务 | `StackCube-v1` |
| 源数据 | 官方 motionplanning `trajectory.h5` |
| replay | 官方 `mani_skill.trajectory.replay_trajectory` |
| observation | `rgb`，128×128 |
| action | `pd_ee_delta_pos`，4D |
| replay backend | `physx_cpu` |
| 训练 demos | 官方 H5 按 episode id 排序后的前 100 条 |
| train iterations | 100,000 |
| batch size | 256 |
| observation horizon | 2 |
| action horizon | 8 |
| prediction horizon | 16 |
| episode horizon | 200 |
| training-time eval | 5,000 steps 一次，100 episodes，10 CPU workers；DDP 时各 rank 分片推理 |
| checkpoint | 每 10,000 steps 保存 |
| primary metric | `success_once` |
| termination | `ignore_terminations=True`，成功后不提前 reset |
| reconfiguration | `reconfiguration_freq=1` |

`physx_cpu` 指模拟器 backend。官方数据任务的 action inference 与 rollout 是两条边界：DDP
各 rank 在自己的 CUDA 卡上推理，action、termination 和 metrics 在进入环境/汇总边界时回到
CPU；物理 rollout 始终由 `physx_cpu` 完成。这不等于使用 GPU simulator。如果需要完全 CPU，
可在 runner 中传 `--model-device=cpu`，但默认保留官方训练的 CUDA policy device。

仓库现在额外支持 `--gpus=4,5,6,7` 的 DDP 训练变体：全局 batch 仍为 256，每卡 batch 为
64；各 rank 共同执行 action inference 和内部 CPU 评测，rank 0 负责合并 metrics 并写 checkpoint。该变体用于加速 990-demo 对照实验，不属于
官方原始单卡 anchor，结果必须使用独立 output root 单独报告。

## 执行入口

远端仓库根目录执行。runner 默认使用仓库内置的
`third_party/maniskill_official`，不需要访问 GitHub；如果需要使用另一个已经准备好的官方 checkout，
可显式传 `--official-repo=/path/to/ManiSkill`。

```bash
(mkdir -p outputs/maniskill/official_stackcube_rgb_anchor && \
  exec nohup bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
    --phase=all \
    --gpu=0 \
    --model-device=cuda \
    > outputs/maniskill/official_stackcube_rgb_anchor/nohup.log 2>&1) &
echo $!
```

只准备数据：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=prepare --gpu=0 --model-device=cuda
```

只从已通过 replay audit 的数据开始训练：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=train --gpu=0 --model-device=cuda
```

训练完成后单独验证官方 `best_eval_success_once.pt`：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=verify --gpu=0 --model-device=cuda
```

## 产物

```text
outputs/maniskill/official_stackcube_rgb_anchor/
├── anchor_manifest.json
├── official_repo/（仅在显式传入外部 checkout 时使用）
├── 00_replay/
│   └── audit.json
├── 10_train/
│   ├── train_complete.json
│   └── runs/official_stackcube_rgb_100/
│       ├── events.out.tfevents.*
│       ├── checkpoints/0.pt, 10000.pt, ...
│       └── checkpoints/best_eval_success_once.pt
└── 20_verify/
    └── metrics.json
```

训练日志：

```text
outputs/maniskill/official_stackcube_rgb_anchor/logs/00_replay.log
outputs/maniskill/official_stackcube_rgb_anchor/logs/01_audit.log
outputs/maniskill/official_stackcube_rgb_anchor/logs/10_train.log
outputs/maniskill/official_stackcube_rgb_anchor/logs/20_verify.log
```

官方训练脚本本身在每 5,000 steps 使用官方 evaluator 记录 `success_once` 和 `success_at_end`，并
自动保存 `best_eval_success_once.pt`。`20_verify/metrics.json` 是对这个 checkpoint 的独立官方 evaluator
复核，不是本仓库自定义成功判定。

## 结果解释

- 官方 anchor 成功率明显恢复：优先归因到我们之前的转换、LeRobot/M2 模型或 GPU/50-step evaluator。
- 官方 anchor 也低：问题不应继续归咎于 M2 head，应检查 ManiSkill 版本、RGB 数据、任务难度和官方 recipe。
- 官方 CPU 4D 成功而 GPU/7D 失败：说明 backend、控制协议或 action conversion 是主要差异。

在远端产物完成前，不填写成功率结论。
