# ManiSkill 官方 StackCube RGB Golden Anchor 指南

## 重要进展

- 已新增独立官方复现 runner，和 M1/M2 的数据、模型、评测代码隔离。
- 固定 `StackCube-v1`、官方 motionplanning demos、`pd_ee_delta_pos` 四维 action、RGB、`physx_cpu`。
- 已完成主复现：990 demos 的官方 4D/默认 57D state 路线，分别训练 50k 和 100k steps。
- 两个主 run 都完成官方 CPU verification：50k 为 `0.340 / 0.330`，100k 为 `0.520 / 0.470`。
- 另完成 990-demo、100k steps、4-GPU DDP 工程变体，独立 verification 为 `0.480 / 0.450`；它不替代单卡 official anchor。
- 训练内评测和最终验证都使用官方 `ignore_terminations=True` / `success_once` 协议。

## 目的

这是 M1-W 数据协议之后的上游模型复现阶段，也是定位 M2-W / LeRobot 管线问题的 golden anchor。
它不使用本仓库的 LeRobot 转换器、M2-W 模型、M2 Diffusion 模型、GPU simulator 或自定义
evaluator，只调用当前 ManiSkill checkout 中的官方 Diffusion Policy 实现。

主表使用 990 demos 的官方 motion-planning replay，并用 50k/100k 两个训练预算建立曲线。旧的
100-demo/100k 配方仍由兼容 runner 保留，但不作为当前五条模型对照表的 official 主结果。

注意：命令中的 `obs-mode=rgb` 只决定视觉输入；官方默认 state extractor 还会拼接
`agent + extra`，当前 checkpoint 的 state 输入实际为 57D。因此本 anchor 是“RGB + 57D state +
4D action”，不是纯视觉 baseline。

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
| 训练 demos | 官方 H5 按 episode id 排序后的前 990 条 |
| train iterations | 主结果分别为 50,000 / 100,000 |
| batch size | 256 |
| observation horizon | 2 |
| action horizon | 8 |
| prediction horizon | 16 |
| episode horizon | 200 |
| training-time eval | 10,000 steps 一次，100 episodes，10 CPU workers；DDP 时各 rank 分片推理 |
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

## 已完成的 official 模型复现

| run | 训练结果 | 独立 verification | 证据角色 |
| --- | --- | --- | --- |
| 990 demos · 50k steps · 单卡 | final train loss `0.00227`；最后一次训练内 eval `0.400 / 0.370` | `physx_cpu`、100 episodes：`34/100` once、`33/100` end；[manifest](../../../outputs/maniskill/official_stackcube_rgb_990_50k/anchor_manifest.json) · [metrics](../../../outputs/maniskill/official_stackcube_rgb_990_50k/20_verify/metrics.json) | official 4D/57D 主 anchor 第一预算点 |
| 990 demos · 100k steps · 单卡 | final train loss `0.001`；最后一次训练内 eval `0.550 / 0.540` | `physx_cpu`、100 episodes：`52/100` once、`47/100` end；[manifest](../../../outputs/maniskill/official_stackcube_rgb_990_100k/anchor_manifest.json) · [metrics](../../../outputs/maniskill/official_stackcube_rgb_990_100k/20_verify/metrics.json) | 与 50k 同协议的主 anchor 终点 |
| 同一 990-demo 官方 motion-planning 4D replay · 100k steps · 4-GPU DDP | global batch `256`；仅训练并行方式不同 | `physx_cpu`、100 episodes：`48/100` once、`45/100` end；[manifest](../../../outputs/maniskill/official_stackcube_rgb_990_100k_4gpu4567/anchor_manifest.json) · [metrics](../../../outputs/maniskill/official_stackcube_rgb_990_100k_4gpu4567/20_verify/metrics.json) | 同一官方数据、state/action/eval contract 的工程并行对照，不替代单卡主结果 |

因此，official 模型复现阶段已经完成。主 anchor 的 50k→100k 提升说明这条官方 pipeline 在
当前任务上仍有明显训练 headroom；后面的 M2-W、LeRobot DP 和 18D/4D 对照应以这两个单卡结果
作为上游参照。

## 执行入口

以下主表命令在远端仓库根目录执行。runner 默认使用仓库内置的
`third_party/maniskill_official`，不需要访问 GitHub；如果需要使用另一个已经准备好的官方 checkout，
可显式传 `--official-repo=/path/to/ManiSkill`。

990-demo、50k steps：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_990_50k.sh \
  --phase=all --gpu=0 --model-device=cuda
```

990-demo、100k steps：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=all \
  --output-root=outputs/maniskill/official_stackcube_rgb_990_100k \
  --num-demos=990 --total-iters=100000 \
  --eval-freq=10000 --save-freq=10000 \
  --exp-name=official_stackcube_rgb_990_100k \
  --gpu=0 --model-device=cuda
```

主表的分阶段命令、990-demo 外部 checkpoint sweep 和 4-GPU DDP 变体见
[官方执行命令](08_official_stackcube_rgb_anchor_commands.md)。

兼容的旧 100-demo/100k runner（不进入当前 official 主表）：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=all --gpu=0 --model-device=cuda
```

如果只从已通过 replay audit 的数据开始训练，或者只验证已生成的
`best_eval_success_once.pt`，仍可分别使用：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=train --gpu=0 --model-device=cuda

bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=verify --gpu=0 --model-device=cuda
```

## 产物

```text
outputs/maniskill/official_stackcube_rgb_990_50k/
├── anchor_manifest.json
├── 00_replay/
│   └── audit.json
├── 10_train/
│   ├── train_complete.json
│   └── runs/official_stackcube_rgb_990_50k/
│       ├── events.out.tfevents.*
│       ├── checkpoints/0.pt, 10000.pt, ...
│       └── checkpoints/best_eval_success_once.pt
└── 20_verify/
    └── metrics.json
```

100k 主 anchor 使用同样的目录结构，根目录为
`outputs/maniskill/official_stackcube_rgb_990_100k/`；4-GPU DDP 结果独立放在
`outputs/maniskill/official_stackcube_rgb_990_100k_4gpu4567/`。

训练日志：

```text
outputs/maniskill/official_stackcube_rgb_990_50k/logs/00_replay.log
outputs/maniskill/official_stackcube_rgb_990_50k/logs/01_audit.log
outputs/maniskill/official_stackcube_rgb_990_50k/logs/10_train.log
outputs/maniskill/official_stackcube_rgb_990_50k/logs/20_verify.log
```

本次 990-demo 训练在每 10,000 steps 使用官方 evaluator 记录 `success_once` 和 `success_at_end`，并
自动保存 `best_eval_success_once.pt`。`20_verify/metrics.json` 是对这个 checkpoint 的独立官方 evaluator
复核，不是本仓库自定义成功判定。

## 结果解释

- 50k 的独立 CPU verification 为 `0.340 / 0.330`，100k 为 `0.520 / 0.470`；这证明官方模型、官方
  数据和 CPU rollout 组合在当前环境中可以形成可复现的 learned-policy baseline。
- 100k 相比 50k 的提升说明不能拿 50k 作为 official pipeline 的上限；后续比较必须明确写出训练 steps。
- 4-GPU DDP 的 `0.480 / 0.450` 与单卡 100k 不完全相同，因此只作为工程并行变体，不能覆盖单卡主结果。
- 官方 checkpoint 的 state 输入为默认 `agent + extra` 的 57D；与显式 18D state、M1-W GPU 7D
  数据和 M2-W 的比较必须同时标注 state、action、backend、episode 数和 horizon。
