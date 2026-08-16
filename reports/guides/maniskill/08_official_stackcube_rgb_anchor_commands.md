# 官方 StackCube RGB Golden Anchor：远端同步与执行命令

这份文档只负责“同步什么、按什么顺序执行”。实验协议和结果解释见
[官方 StackCube RGB Golden Anchor 指南](07_official_stackcube_rgb_anchor.md)。

## 1. 需要同步的数据

当前本地已有的官方源数据位于：

```text
datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5
datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.json
```

其中 `trajectory.h5` 约 37 MiB，`trajectory.json` 是元数据 sidecar。`sample.mp4` 不是本实验的输入，
不需要同步。

下面的命令从本地仓库根目录执行，同步源数据、runner、审计/验证代码，以及仓库内置的官方 baseline：

```bash
cd /path/to/practice

{
  printf '%s\n' \
    datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5 \
    datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.json \
    scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
    src/robot_practice/experiments/maniskill/official_stackcube_rgb_anchor.py \
    src/robot_practice/experiments/maniskill/official_stackcube_rgb_verify.py \
    reports/guides/maniskill/07_official_stackcube_rgb_anchor.md \
    reports/guides/maniskill/08_official_stackcube_rgb_anchor_commands.md
  find third_party/maniskill_official -type f -print
} | rsync -avP \
  -e 'ssh -p 3333' \
  --files-from=- \
  ./ root@localhost:/path/to/remote/practice/
```

上面命令中的文件清单等价于：

```text
datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5
datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.json
scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh
src/robot_practice/experiments/maniskill/official_stackcube_rgb_anchor.py
src/robot_practice/experiments/maniskill/official_stackcube_rgb_verify.py
reports/guides/maniskill/07_official_stackcube_rgb_anchor.md
reports/guides/maniskill/08_official_stackcube_rgb_anchor_commands.md
third_party/maniskill_official/*
```

远端同步后检查文件和校验和：

```bash
cd /path/to/remote/practice

test -s datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5
test -s datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.json
test -s third_party/maniskill_official/examples/baselines/diffusion_policy/train_rgbd.py
test -s third_party/maniskill_official/examples/baselines/diffusion_policy/diffusion_policy/make_env.py

sha256sum \
  datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.h5 \
  datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.json
```

期望校验和：

```text
61d07df7b52f303d2088e977df5f863be7d2274348ff05791f3a815359815523  trajectory.h5
48cb1119c9f5a8f0d7187fc83674a2c401b41466cb1930b2a5c26b706ba3bd81  trajectory.json
```

## 2. 分阶段执行

不要直接执行 `--phase=all`。下面三个阶段分别产生独立产物，某一步失败时可以从该阶段继续。

### 阶段 A：官方 CPU replay、生成 RGB/4D 数据并审计

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=prepare \
  --gpu=0 \
  --model-device=cuda
```

另开一个远端终端观察详细 replay 和审计日志：

```bash
cd /path/to/remote/practice
tail -f outputs/maniskill/official_stackcube_rgb_anchor/logs/00_replay.log
```

阶段 A 成功后，远端才会出现：

```text
datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5
datasets/raw/maniskill_demos/StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pos.physx_cpu.json
outputs/maniskill/official_stackcube_rgb_anchor/00_replay/audit.json
```

检查 gate：

```bash
grep -E '"gate_pass"|"num_episodes"|"success_once"|"success_at_end"' \
  outputs/maniskill/official_stackcube_rgb_anchor/00_replay/audit.json
```

只有 `"gate_pass": true` 才继续训练。

### 阶段 B：官方 100-demo、100k iteration 训练

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=train \
  --gpu=0 \
  --model-device=cuda
```

训练过程中观察官方训练日志：

```bash
cd /path/to/remote/practice
tail -f outputs/maniskill/official_stackcube_rgb_anchor/logs/10_train.log
```

训练完成标志：

```text
outputs/maniskill/official_stackcube_rgb_anchor/10_train/train_complete.json
outputs/maniskill/official_stackcube_rgb_anchor/10_train/runs/official_stackcube_rgb_100/checkpoints/best_eval_success_once.pt
```

### 阶段 B2：990-demo、50k iteration 对照实验

这是独立于上面 100-demo anchor 的实验。它复用已经生成的官方 RGB/4D CPU replay，
训练 990 条 demo，训练 50,000 iterations，每 10,000 iterations 做一次官方
`success_once` evaluation，并每 10,000 iterations 保存 checkpoint。

建议使用新的输出目录：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_990_50k.sh \
  --phase=prepare --gpu=0 --model-device=cuda

bash scripts/experiments/maniskill/run_official_stackcube_rgb_990_50k.sh \
  --phase=train --gpu=0 --model-device=cuda

bash scripts/experiments/maniskill/run_official_stackcube_rgb_990_50k.sh \
  --phase=verify --gpu=0 --model-device=cuda
```

对应产物为：

```text
outputs/maniskill/official_stackcube_rgb_990_50k/anchor_manifest.json
outputs/maniskill/official_stackcube_rgb_990_50k/00_replay/audit.json
outputs/maniskill/official_stackcube_rgb_990_50k/logs/10_train.log
outputs/maniskill/official_stackcube_rgb_990_50k/10_train/loss_history.jsonl
outputs/maniskill/official_stackcube_rgb_990_50k/10_train/runs/official_stackcube_rgb_990_50k/
outputs/maniskill/official_stackcube_rgb_990_50k/20_verify/metrics.json
```

训练日志中的评测点应为 10k、20k、30k、40k、50k；不要只看最终 checkpoint，需同时记录
训练过程中的最大 `success_once` 与 50k final 值。

训练完成后，如需用多张卡对所有 numeric checkpoint 做更稳定的外部评测：

```bash
bash scripts/experiments/maniskill/eval_official_stackcube_rgb_checkpoints_multi_gpu.sh \
  --run-dir=outputs/maniskill/official_stackcube_rgb_990_50k/10_train/runs/official_stackcube_rgb_990_50k \
  --output-root=outputs/maniskill/official_stackcube_rgb_990_50k/30_multigpu_eval \
  --gpus=0,1,2,3,4,5,6,7 \
  --num-episodes=400 \
  --max-episode-steps=200 \
  --seed-start=3000
```

这条命令把每个 checkpoint 的 400 条 episode 分配给 8 个独立进程，每个进程使用一张
GPU 和一个 CPU simulator；它是外部 paired evaluation，不替换官方训练日志中每 10k 的单进程
evaluation。`--seed-start` 现在是环境 reset seed 的起点，而不是仅用于 Python/NumPy/Torch
随机数：8 个 worker 会按 episode 数量接收连续、互不重叠的 seed 区间。上例中各卡依次使用
`3000..3049`、`3050..3099`、……，实际区间会写入合并结果的
`parallelism.environment_seed_ranges`，worker seed 不匹配时合并会直接失败。结果分别写入
`30_multigpu_eval/checkpoint_<step>/metrics.json`。

这里的固定区间使用仓库 ManiSkill test split 的 `3000` 起点，避开当前官方 990-demo RGB
元数据实际出现的 `0..1025` seed 范围，适合未见 test seed 评测。它不等价于“精确重放训练集”：
官方 990-demo RGB 元数据中的 `episode_seed` 不是完整的 `0..989` 连续区间；若要严格评测训练集，
需另用按 episode seed 列表 reset 的 evaluator。

### 阶段 B3：990-demo、100k iteration、4 卡 DDP 变体

这是 990-demo/100k 的多卡训练变体，不再是官方原始的单进程单卡 anchor。使用
`--gpus=4,5,6,7` 时，全局 batch 仍为 256，每张卡的 local batch 为 64；内部 CPU
各 rank 分片执行 action inference 和内部 CPU rollout，rank 0 负责合并 metrics 与 checkpoint
写入，避免重复保存产物。

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=all \
  --output-root=outputs/maniskill/official_stackcube_rgb_990_100k_4gpu4567 \
  --num-demos=990 \
  --total-iters=100000 \
  --eval-freq=10000 \
  --save-freq=10000 \
  --gpus=4,5,6,7 \
  --model-device=cuda \
  --exp-name=official_stackcube_rgb_990_100k_4gpu4567
```

它仍然使用 `physx_cpu`，但训练由 `torchrun` 启动 DDP。训练、内部评测和最终验证的 action
inference 都按 GPU worker 并行，rollout 仍在 CPU simulator 中完成；产物
与单卡 990/50k 完全隔离在新的 output root 下。

### 阶段 C：独立官方 success_once 验证

```bash
bash scripts/experiments/maniskill/run_official_stackcube_rgb_anchor.sh \
  --phase=verify \
  --gpu=0 \
  --model-device=cuda
```

观察验证日志：

```bash
tail -f outputs/maniskill/official_stackcube_rgb_anchor/logs/20_verify.log
```

最终结果：

```text
outputs/maniskill/official_stackcube_rgb_anchor/20_verify/metrics.json
```

这里的 `physx_cpu` 是模拟器 backend，不能把它理解成 GPU simulator。单卡命令默认只使用 GPU 0；
多卡命令则由每个 GPU worker 做 action inference，最终把 action 交给 CPU simulator。整个流程
的 replay、训练时评测和最终验证都保持官方 CPU simulator 协议。

## 3. 官方 ManiSkill baseline 依赖

runner 默认直接使用仓库内置的官方 baseline：

```text
third_party/maniskill_official/examples/baselines/diffusion_policy/
```

它已经包含 `train_rgbd.py`、`make_env.py`、`evaluate.py`、Diffusion Policy 网络和官方
`baselines.sh`，因此远端执行不需要 clone 或访问 GitHub。若要使用另一个已经准备好的官方 checkout，
再在每条命令后追加：

```text
--official-repo=/absolute/path/to/ManiSkill
```
