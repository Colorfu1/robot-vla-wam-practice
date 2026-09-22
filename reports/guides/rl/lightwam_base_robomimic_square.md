# Light-WAM base：robomimic Square pixel

## 目标

使用现有 `square-img/train.npz` 训练独立的 Light-WAM base policy：

- 输入：单路 `agentview_image`、9D proprio、固定任务文本。
- 输出：4 步 7D action chunk。
- 主干：Wan2.1 T2V 1.3B + LoRA/WAM adapter + state-fusion action expert。
- 不连接 DPPO 或 Residual RL，也不能加载 DPPO `state_4000.pt`。
- 不修改或重建 Docker；默认禁止下载任何模型。

## 运行位置

在已经启动的 LOCAL Docker 或 REMOTE GPU worker 中执行。源码仓库必须挂载或同步完成；
REMOTE worker 内不要再运行 Docker。

先进入仓库根目录一次：

```bash
cd "$(git rev-parse --show-toplevel)"
```

## 1. 环境与资产检查

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/experiments/rl/run_lightwam_base_robomimic.sh \
  --phase=check --device=cuda:0
```

这一条命令会检查 CUDA、Light-WAM/Accelerate/DeepSpeed、官方源码、Square NPZ、Wan 和 VAE。
若语言 context 不存在，会读取本地 T5 自动生成约 1 MB 的 context cache。命令不会下载资产；
缺少大模型时直接失败，看到 `LIGHTWAM_BASE_CHECK=PASS` 才继续。

## 2. 真实短训练 smoke 与耗时估算

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LIGHTWAM_NUM_PROCESSES=8 \
LIGHTWAM_MAIN_PROCESS_PORT=29517 \
  bash scripts/experiments/rl/run_lightwam_base_robomimic.sh \
  --phase=smoke --device=cuda \
  --output-root=outputs/lightwam_base_robomimic_square/smoke_8gpu_b32
```

这一条命令使用真实 Square 数据和正式 Accelerate + DeepSpeed 训练路径，执行 10 step
warmup 和 50 个计时 optimizer step，包含 forward、backward 和 optimizer update，但不保存
大 checkpoint。结果写入 `training_speed_benchmark.json` 和 `smoke_summary.json`；后者给出
峰值显存、step/s，以及 420,000 step 正式训练的纯计算耗时估算。默认每卡 batch 32，
8 卡的 effective global batch 为 256。看到
`LIGHTWAM_BASE_SMOKE=PASS` 才表示 smoke 完成。建议每种 GPU 配置分别跑一次；该估算不包含
模型初始化、checkpoint 写盘、评测和共享存储抖动。

## 3. 正式训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LIGHTWAM_NUM_PROCESSES=8 \
LIGHTWAM_MAIN_PROCESS_PORT=29518 \
  bash scripts/experiments/rl/run_lightwam_base_robomimic.sh \
  --phase=train --device=cuda \
  --output-root=outputs/lightwam_base_robomimic_square/train_seed42
```

训练自动使用官方 Accelerate + DeepSpeed ZeRO-1。持续进度写入
`outputs/lightwam_base_robomimic_square/train_seed42/run.log`，权重写入
`checkpoints/weights/`，完整续训状态写入 `checkpoints/state/`。

训练目录必须尚不存在或为空；需要重跑时换一个 `--output-root`。配置对齐 DPPO 的 4000 个
dataset epoch 和 effective global batch 256。Light-WAM 有 26,814 个窗口，每 epoch 105 update，
因此共 420,000 optimizer step，并每 52,500 step（500 epoch）保存一次。单个 Light-WAM 完整
checkpoint 很大，8 个保存点需预留约 100 GiB；当前阶段只训练 base，不产生闭环成功率结论。

### 3.1 分段训练：先跑 26,350 optimizer step 取中间 checkpoint

需要先拿到一个中期 checkpoint 做评测时，用 `--max-steps` 保持 420,000 step 的完整 scheduler
horizon，同时用 `--stop-after-steps` 在指定 step 保存完整状态并退出：

```bash
run_root=outputs/lightwam_base_robomimic_square/train_seed42_stage26350

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LIGHTWAM_NUM_PROCESSES=8 \
LIGHTWAM_MAIN_PROCESS_PORT=29518 \
  nohup bash scripts/experiments/rl/run_lightwam_base_robomimic.sh \
  --phase=train \
  --config=configs/experiments/rl/lightwam_base_robomimic_square.yaml \
  --max-steps=420000 --stop-after-steps=26350 \
  --output-root="$run_root" \
  > "${run_root}.launch.log" 2>&1 &

echo "PID=$!"
```

因为 26,350 不是 `save_every=52500` 的整数倍，trainer 的 stop 分支会自己落一次 checkpoint，
权重文件名为 6 位补零的 `step_026350`：

- 权重：`"${run_root}"/checkpoints/weights/step_026350.pt`
- 完整续训状态：`"${run_root}"/checkpoints/state/step_026350/`
- 训练日志：`"${run_root}"/run.log`，launcher 日志：`"${run_root}".launch.log`
- 解析后的配置：`"${run_root}"/config.yaml`（其中保留 `max_steps=420000`）

`run.log` 中出现 `[done] stop_after_steps reached step=26350` 表示本段训练结束。此时 checkpoint
只完成 26,350 / 420,000 step，正式评测仍会标记 `training_budget_complete=false`、
`formal_result=false`，仅代表一次中期检查，不代表训练预算完成。

### 3.2 从分段 checkpoint 续训到 52,500 step

对 3.1 产出的 checkpoint 续训时，`--checkpoint` 指向 `checkpoints/state/step_026350` 目录
（包含 optimizer/scheduler/sampler 与 trainer state），并复用同一个 `--output-root`。
`--max-steps=420000` 必须保持不变，这样 cosine scheduler 仍以 420,000 step 为 horizon；
`--stop-after-steps=52500` 在第一个周期保存点停下：

```bash
run_root=outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LIGHTWAM_NUM_PROCESSES=8 \
LIGHTWAM_MAIN_PROCESS_PORT=29518 \
  nohup bash scripts/experiments/rl/run_lightwam_base_robomimic.sh \
  --phase=train \
  --config=configs/experiments/rl/lightwam_base_robomimic_square.yaml \
  --checkpoint="$run_root/checkpoints/state/step_026350" \
  --max-steps=420000 --stop-after-steps=52500 \
  --output-root="$run_root" \
  > "${run_root}.resume52500.launch.log" 2>&1 &

echo "PID=$!"
```

注意：`--output-root` 已非空时，不传 `--checkpoint` 会直接拒绝启动；传入 state 目录才允许
原地续训。`run.log` 会继续追加；`step=52500` 是 `save_every=52500` 的整数倍，会按周期
落盘权重 `checkpoints/weights/step_052500.pt`。续训完成后按第 4 节流程对该 checkpoint 重跑
`3 seeds × 50` 评测，用来和 26,350 step 的读数做趋势对比。

### 3.3 续训到 105,000 step（25% 预算决策门）

52,500 step 的 `3 × 50` 正式评测只有 10.0%（15/150），而训练 action loss 已经极低、
动作分布疑似停滞，因此先续训到 105,000 step（总预算的 25%）作为决策门，而不是直接烧满
420,000 step。`--checkpoint` 指向 `checkpoints/state/step_052500`，仍复用同一个
`--output-root`；`--max-steps=420000` 保持不变以维持 cosine scheduler 的 horizon：

```bash
run_root=outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LIGHTWAM_NUM_PROCESSES=8 \
LIGHTWAM_MAIN_PROCESS_PORT=29518 \
  nohup bash scripts/experiments/rl/run_lightwam_base_robomimic.sh \
  --phase=train \
  --config=configs/experiments/rl/lightwam_base_robomimic_square.yaml \
  --checkpoint="$run_root/checkpoints/state/step_052500" \
  --max-steps=420000 --stop-after-steps=105000 \
  --output-root="$run_root" \
  > "${run_root}.resume105000.launch.log" 2>&1 &

echo "PID=$!"
```

`run.log` 会继续追加；`step=105000` 是 `save_every=52500` 的整数倍，会按周期落盘
`checkpoints/weights/step_105000.pt`。完成后先跑第 5 节离线诊断，再按 4.1 做正式评测。

## 与官方 DPPO base 配置的差异

| 项目 | DPPO v0.6 `state_4000.pt` recipe | 当前 Light-WAM |
|---|---:|---:|
| optimizer update | 424,000 | 420,000（窗口数不同） |
| epoch / 每 epoch update | 4000 / 106 | 4000 / 105 |
| local/global batch | 单卡 256 / 256 | 每卡 32×8 / 256 |
| 梯度累积 | 1 | 1 |
| 4000 epoch 遍历的唯一窗口数 | 107,656,000 | 107,256,000 |
| optimizer / 初始 LR | AdamW / `1e-4` | AdamW / `1e-4` |
| AdamW betas | PyTorch 默认 `(0.9, 0.999)` | `(0.9, 0.95)` |
| weight decay | `1e-6` | `0` |
| LR schedule | cosine-restart；warmup 100 epoch（约 10,600 update）；min `1e-5` | cosine；warmup 5%（21,000 step）；min `1e-6` |
| 梯度裁剪 / 精度 | 未配置 / FP32 | norm `1.0` / BF16 |
| 输入 | 1×96×96 RGB + 9D state | 5×224×224 RGB + 9D state + 文本 context |
| 训练参数 | 小型 diffusion policy 全参数 + EMA | 冻结 Wan backbone；训练 LoRA/WAM/action expert，无 EMA |
| action 学习 | 100-step epsilon diffusion | state-fusion action regression；视频侧 flow matching |
| 数据增强 | 开启 | 未开启 |

两者都使用 seed 42、4×7 action chunk、9D state、AdamW 和 `1e-4` 初始学习率。当前严格对齐
的是 4000 个 dataset epoch 和 effective global batch 256。Light-WAM 为监督 4 个 action 需要
5 帧，因而每条轨迹比 DPPO 少一个窗口；总 update 相差约 0.94%，唯一窗口曝光量相差约 0.37%。

## 4. 用 Light-WAM 权重做 Square 闭环评测

评测入口不会修改 DPPO 或 Light-WAM 的第三方代码，也不会下载资产。Light-WAM 模型运行在
`lightwam` Conda 环境；robomimic Square 环境运行在隔离的 `dppo` Conda 子进程，两边只交换
观测和动作。评测合同与 DPPO Square 内置评测一致：

- `agentview_image` 96×96 RGB + 9D 归一化 state；
- 归一化 7D action、每次执行 4 个 action、每 episode 最多 400 个原始环境 step；
- `reward_shaping=false`；
- 成功判定为 `max(chunk_reward) / 4 >= 1`；
- 正式设置为 seeds `1,2,3`，每个 seed 启动 50 个异步环境并完成 50 episodes。

先用同一个真实 checkpoint 跑一个两次 action-chunk 的闭环 smoke。它只验证模型、进程桥、
EGL 环境和动作合同，不能作为成功率结果：

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash scripts/experiments/rl/run_lightwam_base_robomimic.sh \
  --phase=eval \
  --checkpoint=outputs/lightwam_base_robomimic_square/pilot_4000step_b32x8_seed42/checkpoints/weights/step_004000.pt \
  --eval-seeds=1 --episodes=1 --max-episode-steps=8 \
  --output-root=outputs/lightwam_base_robomimic_square/eval_smoke_step004000
```

由于 smoke 只给环境 8 个原始 step，`success_rate=0` 是正常现象，不能据此判断模型质量。
新版指标会明确写入 `evaluation_mode=pipeline_smoke`、
`success_rate_interpretable=false`，并记录 action/state/RGB 的范围、标准差和 action 饱和率；
它们用于排除输出常量、非有限值和归一化饱和问题。模型质量只能看下面 400-step 的完整评测。

smoke 通过后，用三张 GPU 各跑一个 seed，启动完整的 `3 × 50 = 150` episodes 评测。每个
seed 内的 50 个 Square 环境按官方路径异步运行；所有 worker 完成后脚本自动合并指标：

```bash
eval_root=outputs/lightwam_base_robomimic_square/eval_step004000
mkdir -p "$eval_root"

nohup bash scripts/experiments/rl/run_lightwam_base_robomimic_eval_multigpu.sh \
  --gpus=0,1,2 --seeds=1,2,3 \
  --checkpoint=outputs/lightwam_base_robomimic_square/pilot_4000step_b32x8_seed42/checkpoints/weights/step_004000.pt \
  --output-root="$eval_root" \
  > "$eval_root/launch.log" 2>&1 &

echo "PID=$!"
```

查看实时进度和结果：

```bash
tail -f \
  outputs/lightwam_base_robomimic_square/eval_step004000/launch.log \
  outputs/lightwam_base_robomimic_square/eval_step004000/seed_*/run.log

jq '{episodes, successes, success_rate, per_seed, formal_eval_protocol,
     training_budget_complete, formal_result}' \
  outputs/lightwam_base_robomimic_square/eval_step004000/metrics.json
```

每个 seed 的逐 episode 结果写入 `seed_<N>/episodes.jsonl`，环境侧警告和异常写入
`seed_<N>/env_worker.log`，汇总结果写入 `metrics.json`。若进程中断，原命令增加 `--resume`：

```bash
bash scripts/experiments/rl/run_lightwam_base_robomimic_eval_multigpu.sh \
  --resume --gpus=0,1,2 --seeds=1,2,3 \
  --checkpoint=outputs/lightwam_base_robomimic_square/pilot_4000step_b32x8_seed42/checkpoints/weights/step_004000.pt \
  --output-root=outputs/lightwam_base_robomimic_square/eval_step004000
```

续评会跳过已经完整落盘的 seed；被中断的 seed 会从头重跑。`step_004000.pt` 只完成了
4,000 optimizer steps，而计划的正式训练预算是 420,000 steps，因此其完整 150-episode
闭环结果仍会标记 `formal_eval_protocol=true`，但 `training_budget_complete=false`、
`formal_result=false`。这用于区分“评测协议完整”与“模型训练预算完整”。

### 4.1 105,000 step 决策门正式评测

```bash
eval_root=outputs/lightwam_base_robomimic_square/eval_step105000
mkdir -p "$eval_root"

nohup bash scripts/experiments/rl/run_lightwam_base_robomimic_eval_multigpu.sh \
  --gpus=0,1,2 --seeds=1,2,3 \
  --checkpoint=outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42/checkpoints/weights/step_105000.pt \
  --output-root="$eval_root" \
  > "$eval_root/launch.log" 2>&1 &

echo "PID=$!"
```

汇总结果读取方式与第 4 节相同。决策门沿用 26,350=6.0%、52,500=10.0% 的延续标准：

- 成功率 ≥20% 且相对前两个读数趋势明确上升：继续续训。
- 成功率 ≤12% 且离线诊断仍显示动作输出停滞：停止当前配方，先排查 action expert。

## 5. 便宜离线 action-error 诊断（对照）

闭环成功率低需要区分“策略动作贴不贴训练数据”和“动作本身有没有信息量”。下面的单卡脚本
在 `train.npz` 上用与闭环评测一致的 conditioning（当前帧 `t` → 后续 4 个 action）做开放环
前向，并同时给两个对照：随机初始化模型，以及恒零/恒均值 action 基线。它只度量开放环动作
保真度，不产生闭环成功率结论。

注意：Square 目前只有 `train.npz`，没有正式 held-out 集；本诊断在同一训练数据上做描述性
对照，不等价于泛化测试。采样在全部 26,814 个窗口上均匀取点以覆盖每条轨迹。

```bash
cd "$(git rev-parse --show-toplevel)"

CUDA_VISIBLE_DEVICES=0 \
PYTHONNOUSERSITE=1 \
PYTHONPATH=src \
  nohup /opt/conda/envs/lightwam/bin/python -m \
  robot_practice.experiments.rl.lightwam_base_open_loop_diagnostic \
  --config=configs/experiments/rl/lightwam_base_robomimic_square.yaml \
  --checkpoints=outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42/checkpoints/weights/step_026350.pt,outputs/lightwam_base_robomimic_square/formal_420k_gate_026350_b32x8_seed42/checkpoints/weights/step_052500.pt \
  --include-untrained \
  --num-windows=512 \
  --device=cuda \
  --output-root=outputs/lightwam_base_robomimic_square/offline_diag_26350_52500 \
  > outputs/lightwam_base_robomimic_square/offline_diag_26350_52500.launch.log 2>&1 &

echo "PID=$!"
```

单卡上约 20–30 分钟（三次模型加载 + 512 窗口前向）。查看进度：

```bash
tail -f outputs/lightwam_base_robomimic_square/offline_diag_26350_52500/run.log
```

结果写入 `offline_diag_26350_52500/metrics.json`，每个模型一份
`predictions_<model>.npz`。重点看每个 `models.<name>`：

- `overall.mse_clipped_over_mean_baseline`：小于 1 才说明动作头比“恒预测训练均值”更有信息量。
- `per_dim.*.pred_std_ratio` 与 `per_dim.*.clip_rate`：比值接近 0 或大量被 clip，说明该维度
  冻结或饱和。
- 与 `untrained_random_init` 对比：训练后 MSE、相关性和分布是否真的移动。
- `per_step_mse`：4 步 chunk 内误差是否随步数漂移。

## 6. 同数据小型 Diffusion Policy 对照（2026-09-18）

目的：区分训练预算不足与策略设计差异。使用相同 100 条 / 27,214 帧 NPZ，
从随机初始化训练上游 DPPO ViT + MLP diffusion；不是 PPO，也不加载发布的预训练权重。
保留上游 epsilon loss、数据增强、AdamW、EMA；单卡 batch 256。
学习率周期为 4000 epoch，默认先到 1000 epoch，每 250 epoch 保存并评测。
每 epoch 106 updates，默认约 106,000 updates。checkpoint 编号是 epoch，不是 update。

闭环使用 EMA、DDIM 5、单帧 agentview 96×96 + 9D proprio、7D 动作、horizon 4、20 Hz，
OSMesa/MSAA4、spawn、串行环境 step、8 个 seeds 10001–10008、每回合 400 原始步。
这对齐最近 Light-WAM OSMesa 诊断，不等于旧 EGL / 150 回合协议。
不提供训练场景恢复，也不代表原始 Diffusion Policy 双相机绝对动作配置。

在已安装 DPPO 的容器内、仓库根目录执行：

```bash
# Smoke：两次真实梯度更新 + 一个完整 400 步回合
GPU=0 bash scripts/experiments/rl/run_dppo_square_bc_control.sh --smoke --envs=1

# 阶段性对照训练；新建独立输出目录，不覆盖历史结果
GPU=0 bash scripts/experiments/rl/run_dppo_square_bc_control.sh \
  --stop-after=1000 --eval-every=250 --envs=8
```

结果：`outputs/dppo_square_bc_control_*/run.log` 提供训练与评测 completed/total；
`train_metrics.jsonl` 保存 epoch、实际更新数和训练 loss；`metrics.json` 汇总成功率曲线；
`eval_epoch_*/metrics.json` 保存逐回合结果；`manifest.json` 保存数据与上游源码校验值。
`training_state.pt` 保存最新训练状态，但当前入口不支持自动 resume，失败后不要直接复用目录。

验证情况：本地 Docker Smoke 已完成两次更新及一个完整回合；未运行正式长训。
判断：比较 26.5k、53k、79.5k、106k updates 的闭环曲线与对应 Light-WAM 结果；
只有单训练 seed、8 个评测初始状态，改善需扩大样本复核。低训练 loss 不能替代成功率。

## 7. 锁定官方配置复现（2026-09-18）

入口 `scripts/experiments/rl/run_dppo_square_official.sh` 独立下载上游至
`third_party/dppo_official` 并锁定 `cc7234ad7ff39a8f32de3af903606723a16f0648`。
拒绝脏的 tracked 源码或不同 commit，不覆盖已有 DPPO runtime。
沿用现有发布版 Square NPZ，不重复下载；保存输入文件 SHA256。

直接调用官方 TrainDiffusionAgent.run：2000 epochs、batch 256、LR 1e-4、
cycle 2000、warmup 100、EMA 0.995，其余沿用该 revision YAML。
训练结束后独立评测 epochs 500/1000/1500/2000；直接调用官方 EvalImgDiffusionAgent.run，
20 environments、400 policy iterations、400 environment steps/episode、seed 42、DDIM 5。
官方加载器选择 `model` 而非 `ema`，入口不更改这一行为；不额外 clip action。
没有语言输入：单相机图像加 9D proprio。

显式兼容差异：修正官方 YAML 已失效的 diffusion_eval_ft 模块引用；使用 OSMesa、
spawn、保留共享缓冲区以避开已观察到的 EGL abort 和 semaphore 生命周期问题。
不声称已复现历史论文成绩或原版 Diffusion Policy 的约 90% 成功率。

```bash
# 先测试完整一轮训练 + 单环境 400 步；仅缩短预算，不改变 batch 或 LR schedule。
GPU=0 bash scripts/experiments/rl/run_dppo_square_official.sh --smoke

# 完整训练，然后评测四个保存节点；输出目录必须是新目录。
export OUTPUT_ROOT="outputs/dppo_square_official_$(date +%Y%m%d_%H%M%S)"
GPU=0 nohup bash scripts/experiments/rl/run_dppo_square_official.sh \
  > "${OUTPUT_ROOT}.launch.log" 2>&1 &
echo "PID=$! RESULT_DIR=$OUTPUT_ROOT"
```

训练日志与解析配置在 `train/run.log`、`train/config.yaml`；评测结果为
`eval_epoch_*/result.npz`，不是旧对照入口的 metrics.json。每个子任务记录 manifest.json。
训练和评测持续写 completed/total。当前不支持续训；不要把 smoke checkpoint 当成正式结果。

## 8. 已发布 checkpoint 与训练 checkpoint 的 EMA 对照

`scripts/experiments/rl/run_dppo_square_ema_controls.sh` 接收两个 checkpoint 路径：
官方发布 state_4000 和当前训练 state_2000。仅运行评测，按官方 EMA、当前 EMA 顺序执行。
默认 GPU 0；默认协议与第 7 节一致（20 envs、400 policy iterations、seed 42、DDIM 5、OSMesa）。
保留每项 `result.npz`、`run.log`、config/manifest，额外生成 `summary.json` 及源 checkpoint SHA256。
源 checkpoint 不修改；EMA 提取副本只写入本次新输出目录。

```bash
export OUTPUT_ROOT="outputs/dppo_square_ema_controls_$(date +%Y%m%d_%H%M%S)"
GPU=0 nohup bash scripts/experiments/rl/run_dppo_square_ema_controls.sh \
  "$OFFICIAL_CKPT" "$TRAINED_CKPT" > "${OUTPUT_ROOT}.launch.log" 2>&1 &
echo "PID=$! RESULT_DIR=$OUTPUT_ROOT"
```

远端无网，先从本地同步脚本及官方 checkpoint。正式入口校验两份权重 epoch 分别为 4000/2000，
EMA 必须存在，并验证发布版 normalization 校验值。若仅验证链路可追加 `--smoke`，
它将每项缩短到单环境完整一回合，并允许第二份使用本地 smoke checkpoint；结果明确标记 smoke。
当前原始 model 的 24/80 基线与新 EMA 结果分开报告，不以烟测成绩替代正式对照。
