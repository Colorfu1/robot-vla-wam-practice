# robomimic + DPPO pixel：官方 DPPO 基准与 Residual RL

## 实验目标

在 robomimic `Square` pixel 工作点固定同一个 DPPO checkpoint，严格按以下顺序完成实验：

1. 先评估 `Frozen Base`，确认 `state_4000.pt` 与当前 worker/environment 合同一致。
2. 使用上游 DPPO 原始 `ft_ppo_diffusion_mlp_img.yaml` 完成官方 pixel 全参数微调，评测该结果作为主基准。
3. 回到同一个 `state_4000.pt`，再进行 `Zero Residual`、`Learned Bounded Residual (PPO)` 和
   `Random Bounded Residual` 对照。

正式评测统一使用 3 个 seed，每个 seed 50 个初始状态。

这里的“全参数”是官方 DPPO pixel recipe 的含义：`actor_ft` 的全部参数都由 PPO 更新；官方配置仍只
微调最后 `ft_denoising_steps=5` 个 DDIM step，前面的 base actor 保持冻结。这不是把 100 个 denoising
step 全部改成另一种训练，也不是 residual adapter。

## 实验思路

- LOCAL 只构建并发布镜像、下载并校验资产、传输代码和资产到 REMOTE。
- REMOTE 使用平台启动的 DPPO worker；进入 worker 后直接运行实验命令，不执行嵌套
  `docker load` 或 `docker run`。
- 官方 DPPO 微调阶段不使用本地 residual 代码；Residual 阶段的 base 始终冻结，residual 只读
  `agentview_image` 和机器人自身 9D 状态。
- residual 在 normalized action space 使用固定 `±0.1` bound。
- Frozen/Zero/Learned/Random 四个 residual 组共用同一 manifest 和配对的 base diffusion RNG seed；
  官方 DPPO full fine-tuning 使用上游 evaluator，单独记录，不塞进 residual 四组汇总器。
- `run_residual_robomimic_dppo.sh` 在当前 DPPO worker 内直接运行一个 phase。
- `run_official_dppo_baseline.sh` 只做官方 DPPO 的编排：一次完成 full fine-tuning、上游 eval 和
  `metrics.json` 汇总，不替换任何上游训练/评测入口。
- 多卡脚本默认把 seed 调度到用户选择的 GPU，并等待本阶段全部完成后再继续；只有在真正的
  Docker 宿主机上才使用 `--launch-mode=docker`。
- 不从 demonstrations 重新 pretrain，因此官方微调和 residual 当前都不需要 `train.npz`；只需要上游
  `state_4000.pt`、对应 `normalization.npz` 和在线 MuJoCo 环境。

## 1. 构建镜像

执行位置：**LOCAL 主机，仓库根目录**。

目的：构建固定 DPPO、robomimic、robosuite、MuJoCo/EGL 和 CUDA 依赖的环境镜像；数据和
checkpoint 不写入镜像。

```bash
set -euo pipefail
docker build \
  -f docker/training/Dockerfile.robomimic-dppo-pixel \
  -t practice:robomimic-dppo-pixel .
```

## 2. 下载、校验资产

执行位置：**LOCAL 主机**；下载和结构校验程序运行在 **LOCAL Docker** 内，产物通过 bind mount
写回本地仓库。

目的：准备 frozen base 与官方 DPPO full fine-tuning 共同需要的 `normalization.npz` 和
`state_4000.pt`，验证文件结构并记录 checksum；镜像单独在第 3.1 节发布。当前路线不需要下载或
重新渲染 demonstrations。

```bash
set -euo pipefail
export IMAGE=practice:robomimic-dppo-pixel
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
# Keep these repository-relative and identical on LOCAL and REMOTE; do not
# inherit an absolute DPPO_* value from another experiment.
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"
export NORM="$DPPO_DATA_DIR/robomimic/square-img/normalization.npz"
export CKPT="$DPPO_LOG_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt"
mkdir -p "$EXP_ROOT/transfer"

docker run --rm -i \
  -v "$PWD:$PWD" \
  -w "$PWD" \
  "$IMAGE" \
  conda run --no-capture-output -n dppo \
  python scripts/experiments/rl/prepare_robomimic_dppo_assets.py \
  --normalization="$NORM" \
  --checkpoint="$CKPT" \
  --log="$EXP_ROOT/transfer/asset_prepare.log"

sha256sum "$NORM" "$CKPT" > "$EXP_ROOT/transfer/assets.sha256"
```

## 3. 发布镜像并传到远端

### 3.1 发布 DPPO worker 镜像

执行位置：**LOCAL 主机**。远端平台必须用这个镜像启动 worker；不要在 worker 内执行
`docker load`。`REMOTE_IMAGE` 使用一个固定 tag，避免实验中途被 `latest` 覆盖。

```bash
set -euo pipefail
export IMAGE=practice:robomimic-dppo-pixel
export REMOTE_IMAGE="${REMOTE_IMAGE:?export REMOTE_IMAGE as registry/image:tag}"

docker tag "$IMAGE" "$REMOTE_IMAGE"
docker push "$REMOTE_IMAGE"
```

在远端任务/容器配置中选择同一个 `REMOTE_IMAGE`，分配 GPU，并把仓库共享目录挂载到远端仓库
路径。下面第 4 节开始的命令，全部在这个已经启动的 DPPO worker 容器内部执行。

### 3.2 同步代码和资产

执行位置：**LOCAL 主机**。

目的：把两个资产及本轮运行代码同步到 REMOTE 仓库根目录。镜像通过上面的 registry/平台配置
提供，不再把 Docker tar 包同步到 worker。`REMOTE_REPO` 只在当前 shell 设置。

```bash
set -euo pipefail
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"
export NORM="$DPPO_DATA_DIR/robomimic/square-img/normalization.npz"
export CKPT="$DPPO_LOG_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt"
# LOCAL 终端执行；按你的 SSH 转发/主机填写。
# 示例：export REMOTE_REPO='root@localhost:/path/to/practice'
: "${REMOTE_REPO:?请先在 LOCAL 终端设置 REMOTE_REPO}"

rsync -avP --relative -e 'ssh -p 3333' \
  configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh \
  scripts/experiments/rl/record_official_dppo_visualization.sh \
  scripts/experiments/rl/play_dppo_base_failure_rl_success.sh \
  src/robot_practice/experiments/rl/robomimic_dppo_residual.py \
  src/robot_practice/experiments/rl/robomimic_dppo_visualization.py \
  "$NORM" "$CKPT" \
  "$EXP_ROOT/transfer/assets.sha256" \
  "${REMOTE_REPO%/}/"
```

## 4. 在 DPPO worker 内做 preflight

执行位置：**REMOTE DPPO worker 容器，仓库根目录**。此处不运行任何 Docker 命令。

本节**不会下载或生成资产**。如果下面第一条 `sha256sum` 报
`transfer/assets.sha256: No such file`，说明第 2 节或第 3 节没有在 LOCAL 主机成功完成；先回到
LOCAL 执行资产准备和 rsync，不能在 worker 内用默认路径继续跑。若 checksum 存在但两个 anchor 文件
缺失，说明 rsync 的目标仓库根目录不对，先检查相对路径是否仍为
`datasets/raw/dppo/...` 与 `outputs/rl_residual_robomimic_20260901/dppo_logs/...`。

目的：验证传输 checksum、确认当前 worker 中的 CUDA/EGL 与 DPPO 环境可用，并再次解析
normalization 与 checkpoint。这里选择一张可用卡即可。

```bash
set -euo pipefail
export GPU=0
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"

test -d /opt/robot-practice/dppo
test -x /opt/conda/envs/dppo/bin/python
test -s "$DPPO_DATA_DIR/robomimic/square-img/normalization.npz"
test -s "$DPPO_LOG_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt"
sha256sum -c "$EXP_ROOT/transfer/assets.sha256"

CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n dppo \
  env MUJOCO_GL=egl PYOPENGL_PLATFORM=egl SDL_VIDEODRIVER=dummy \
  python /usr/local/lib/robot-practice/verify_robomimic_dppo_env.py \
  --source=/opt/robot-practice/dppo --render-smoke

CUDA_VISIBLE_DEVICES="$GPU" ROBOT_PRACTICE_DPPO_GPU="$GPU" \
  bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=preflight \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root="$EXP_ROOT/preflight"

jq -e \
  '.contract_ok == true and (.missing_assets | length) == 0 and (.asset_errors | length) == 0' \
  "$EXP_ROOT/preflight/gates.json"
```

必须看到 `render_smoke=True`、`cuda_available=True` 和最后的 `true`。这是 `Smoke`，不是实验结果。

`run_residual_robomimic_dppo_docker.sh` 仅用于有 Docker daemon 的宿主机启动一个新容器；当前
REMOTE worker 没有 `/var/run/docker.sock` 时不要调用它。Residual 阶段使用
`run_residual_robomimic_dppo.sh`，多卡脚本统一使用 `--launch-mode=direct`；官方 DPPO 基准阶段则
直接调用 worker 中的上游 `/opt/robot-practice/dppo/script/run.py`。

## 5. 官方 DPPO pixel 全参数微调基准（先完成）

执行位置：**REMOTE DPPO worker 容器**；多卡脚本为每个 seed 启动一个当前容器内的
**单 GPU 进程**，不会启动新的 Docker 容器。

`GPUS` 由用户选择：`0` 表示三个 seed 串行，`0,1` 表示两个并行后再跑第三个，`0,1,2` 表示三个
同时运行。每个 stage 都会等待全部 seed 完成。

### 5.1 Frozen Base

目的：用上游 evaluator 并行评测三个 seed，然后校验并合并成 150-episode base 指标和统一 manifest。

```bash
set -euo pipefail
export GPUS=0,1,2
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"

bash scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh \
  --stage=base --gpus="$GPUS" --launch-mode=direct

jq -e \
  '.episodes == 150 and .success_rate >= 0.10 and .success_rate <= 0.90' \
  "$EXP_ROOT/base/metrics.json"
```

headroom gate 不通过就停止：该 checkpoint/任务不适合验证 residual 增益。

### 5.2 官方 DPPO 全参数微调 + eval（一条命令）

这一步只做一个独立的官方 DPPO 对照：从同一个 `state_4000.pt` 出发，按上游
`ft_ppo_diffusion_mlp_img.yaml` 做全参数 pixel PPO 微调，全部 seed 训练完成后再用上游
`eval_diffusion_mlp_img.yaml` 统一评测。底层训练和评测入口没有改动；本地脚本只负责
多 seed 调度、日志、checkpoint 检查和最终汇总。

正式 run 固定为：3 个 seed（1/2/3）、每个 seed 一个 GPU、`n_train_itr=301`、
`env.n_envs=50`、`train.n_steps=400`、每个 eval 50 episodes、`ft_denoising_steps=5`。
3 张卡会并行运行；GPU 少于 seed 数时自动分批。wrapper 会把每个 seed 的 `itr/301`、rollout
进度、已用时间和估计剩余秒数写入顶层 `run.log` 及对应的 `finetune/seed_*/run.log`；上游 DPPO
源码不改。

先做 smoke，再做正式 run。Smoke 只跑 1 个 seed、1 个训练 iteration、1 个 eval episode，
只验证官方训练/评测入口、checkpoint 落盘和汇总链路，不作为实验结果。由于上游 scheduler
的 warmup 默认是 10，而 smoke 只有 1 个 iteration，脚本仅在 smoke 路径把 actor/critic
warmup 设为 0；正式 run 完全使用上游默认值：

```bash
nohup bash scripts/experiments/rl/run_official_dppo_baseline.sh \
  --smoke --gpus=0 --seeds=1 \
  --experiment-root=outputs/rl_residual_robomimic_20260901 \
  > outputs/rl_residual_robomimic_20260901/official_dppo_baseline_smoke.launch.log 2>&1 &
echo "PID=$!"
```

Smoke 必须先通过：

```bash
jq -e '.smoke == true and .episodes == 1 and (.per_seed | length) == 1' \
  outputs/rl_residual_robomimic_20260901/official_dppo_baseline_smoke/metrics.json
```

Smoke 通过后，再启动正式 3-seed train+eval：

执行位置：**REMOTE DPPO worker 容器**。确保 5.1 已通过，并且当前目录是仓库根目录：

```bash
set -euo pipefail
nohup bash scripts/experiments/rl/run_official_dppo_baseline.sh \
  --gpus=0,1,2 \
  --seeds=1,2,3 \
  --experiment-root=outputs/rl_residual_robomimic_20260901 \
  > outputs/rl_residual_robomimic_20260901/official_dppo_baseline.launch.log 2>&1 &
echo "PID=$!"
```

脚本默认使用：

- base：`outputs/rl_residual_robomimic_20260901/dppo_logs/.../state_4000.pt`；
- normalization：`datasets/raw/dppo/robomimic/square-img/normalization.npz`；
- 上游 DPPO checkout：`/opt/robot-practice/dppo`。

如路径不同，再显式加 `--base-checkpoint=...`、`--normalization=...`、
`--dppo-root=...` 或 `--data-root=...`，不需要手工拼接训练和评测命令。

完成判据：

```bash
jq '{episodes, successes, success_rate, per_seed}' \
  outputs/rl_residual_robomimic_20260901/official_dppo_baseline/metrics.json
tail -n 5 outputs/rl_residual_robomimic_20260901/official_dppo_baseline/run.log
```

应看到 `episodes=150`、3 个 `per_seed`，每个 seed 50 episodes，以及
`DONE official DPPO fine-tuning + eval ... failures=0`。最终 checkpoint 位于
`official_dppo_baseline/finetune/seed_*/checkpoint/state_300.pt`（smoke 使用 `state_0.pt`），评测原始结果位于
`official_dppo_baseline/eval/seed_*/result.npz`。只有这一步完成后，才进入 Residual 阶段；
官方 `state_300.pt` 只作为独立 DPPO 基准，不替换 Residual 使用的 frozen `state_4000.pt`。

如果正式训练中断，可以在同一个 output root 上 resume；脚本会为每个 seed 找到最大的非空
`checkpoint/state_N.pt`，从 `N+1` 继续，已经存在 `state_300.pt` 的 seed 会跳过训练，之后统一重跑
eval（中断写出的空 checkpoint 会被跳过）：

```bash
nohup bash scripts/experiments/rl/run_official_dppo_baseline.sh \
  --resume --gpus=0,1,2 --seeds=1,2,3 \
  --experiment-root=outputs/rl_residual_robomimic_20260901 \
  > outputs/rl_residual_robomimic_20260901/official_dppo_baseline.resume.launch.log 2>&1 &
echo "PID=$!"
```

官方 `state_N.pt` 本身只保存模型和 iteration，不保存 Adam 动量、随机数状态等；因此 resume 能接上
模型和 iteration，并对齐官方 learning-rate scheduler，但不承诺 bitwise identical 的连续运行。若 output
root 不存在，或不加 `--resume` 而目录已存在，脚本都会拒绝启动以避免误覆盖。

### 5.3 测试集审计与 DPPO Rerun 可视化

这里的“测试集”不是离线 HDF5 split，而是官方 evaluator 生成的闭环评测集合：3 个 policy seed、每个
seed 50 个 vector-environment episode，共 150 个 episode。`result.npz` 只保存成功率等汇总指标，不能
直接回放；可视化脚本会使用各 policy seed 对应的 `state_300.pt`、同一个 seed 规则和同一个
50 环境/100 policy-step 合同重新 rollout。Frozen Base 则使用同一个 `state_4000.pt` 复用于三个
policy seed。

先在已经同步了正式 eval 结果的本地目录审计：

```bash
set -euo pipefail
export EXP_ROOT=outputs/rl_residual_robomimic_20260901

bash scripts/experiments/rl/audit_official_dppo_eval.sh \
  --eval-root="$EXP_ROOT/official_dppo_baseline/eval" \
  --seeds=1,2,3 --episodes-per-seed=50
```

录制必须在有 DPPO/robomimic 环境的 worker 或本地 DPPO Docker 中执行；它不需要 Rerun SDK，只会写
轻量的 raw NPZ。Base 和 RL 必须使用相同的 `seeds=1,2,3`、`env_count=50` 和
`policy_seed + env_index` reset 规则，但写入不同目录。

RL / DPPO 全参数微调模型（每个 seed 使用对应的 `state_300.pt`）：

```bash
set -euo pipefail
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export VIS_ROOT="$EXP_ROOT/official_dppo_visualization"

bash scripts/experiments/rl/record_official_dppo_visualization.sh \
  --dppo-root=/opt/robot-practice/dppo \
  --checkpoint-root="$EXP_ROOT/official_dppo_baseline/finetune" \
  --normalization=datasets/raw/dppo/robomimic/square-img/normalization.npz \
  --output-root="$VIS_ROOT/raw" \
  --seeds=1,2,3 --gpu=0 \
  --ft-denoising-steps=5 \
  --policy-label=dppo_finetuned \
  --max-record-episodes=50
```

Frozen Base 模型（同一个 `state_4000.pt` 复用于三个 policy seed；Base 必须使用
`ft_denoising_steps=0`）：

```bash
set -euo pipefail
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export BASE_VIS_ROOT="$EXP_ROOT/official_dppo_visualization/base"
export BASE_CKPT="$EXP_ROOT/dppo_logs/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt"

bash scripts/experiments/rl/record_official_dppo_visualization.sh \
  --dppo-root=/opt/robot-practice/dppo \
  --checkpoint="$BASE_CKPT" \
  --normalization=datasets/raw/dppo/robomimic/square-img/normalization.npz \
  --output-root="$BASE_VIS_ROOT/raw" \
  --seeds=1,2,3 --gpu=0 \
  --ft-denoising-steps=0 \
  --policy-label=frozen_base \
  --max-record-episodes=50
```

只想先检查可视化链路时，把 `--max-record-episodes=50` 改成 `12`；这仍会运行完整的 50 个环境，
只是每个 seed 只保存前 12 个 episode。若在远端录制，先将两个 raw 目录同步回本地，再执行：

```bash
set -euo pipefail
export REMOTE_REPO=/path/to/remote/practice
export REMOTE_HOST=root@localhost

# 先把正式 eval 汇总同步回来，审计的 150 是“实际落盘”的结果；
# raw rollout 录制完成后，再同步第二条路径。
rsync -avP --relative -e 'ssh -p 3333' \
  "${REMOTE_HOST}:${REMOTE_REPO%/}/./outputs/rl_residual_robomimic_20260901/official_dppo_baseline/eval/" \
  ./

rsync -avP --relative -e 'ssh -p 3333' \
  "${REMOTE_HOST}:${REMOTE_REPO%/}/./outputs/rl_residual_robomimic_20260901/official_dppo_visualization/raw/" \
  ./

rsync -avP --relative -e 'ssh -p 3333' \
  "${REMOTE_HOST}:${REMOTE_REPO%/}/./outputs/rl_residual_robomimic_20260901/official_dppo_visualization/base/raw/" \
  ./
```

同步后先重新运行上面的 audit；应看到 `expected.total_episodes=150`，并在正式 eval 已完成时看到
`materialized_episode_total=150`。raw 目录的 `manifest.json` 再额外确认
`recorded_episode_count=150`。如果只录了 12 个/seed，则该字段应为 36，不能把它当作完整测试集。

```bash
set -euo pipefail
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export VIS_ROOT="$EXP_ROOT/official_dppo_visualization"

bash scripts/experiments/rl/visualize_official_dppo.sh \
  --raw-root="$VIS_ROOT/raw" \
  --output-dir="$VIS_ROOT/rerun_rl" \
  --write-video

bash scripts/experiments/rl/visualize_official_dppo.sh \
  --raw-root="$VIS_ROOT/base/raw" \
  --output-dir="$VIS_ROOT/rerun_base" \
  --write-video

if command -v robot-rerun >/dev/null 2>&1; then
  robot-rerun "$VIS_ROOT/rerun_rl/robomimic_dppo_visualization.rrd"
else
  rerun "$VIS_ROOT/rerun_rl/robomimic_dppo_visualization.rrd"
fi
```

再打开 Base archive（建议在另一个终端执行）：

```bash
if command -v robot-rerun >/dev/null 2>&1; then
  robot-rerun "$VIS_ROOT/rerun_base/robomimic_dppo_visualization.rrd"
else
  rerun "$VIS_ROOT/rerun_base/robomimic_dppo_visualization.rrd"
fi
```

Rerun 每个 episode 一个 Recording，并额外生成可切换的合并 archive；视图包括 agentview RGB、末端
位置轨迹、9D normalized state、首个 action 和完整 action chunk、reward/success 曲线。录制的是 100 个
policy-level frame（每帧对应 4 个低层动作），默认按 5 Hz 播放；不是 400 帧的低层渲染视频。DPPO
当前只有 `agentview_image` 一个相机输入，不会伪造 wrist camera 或 object privileged state。

### 5.4 自动配对 Base 失败、DPPO 成功的 rollout

在本地完成两个 `visualize` 命令后，可以按相同的 `policy_seed + episode_id` 自动配对，只筛选
Frozen Base 失败而 DPPO full fine-tuning 成功的 episode。脚本的默认 pair mode 会在
`base_failure_rl_success/{base,rl}/` 创建轻量 symlink 文件夹；每个文件夹只包含对应的独立 `.rrd`
recording。随后用同一个脚本的 folder mode 分别打开两个文件夹；每个 Rerun 窗口左侧的 recording
selector 可以切换场景，两个窗口的候选顺序一致。需要本机安装 `rerun` 或 `robot-rerun`。

先筛选并生成两个 folder（不启动 viewer）：

```bash
bash scripts/experiments/rl/play_dppo_base_failure_rl_success.sh \
  --prepare-only
```

然后在两个终端分别执行同一个 folder 播放脚本：

```bash
bash scripts/experiments/rl/play_dppo_base_failure_rl_success.sh \
  --folder=outputs/rl_residual_robomimic_20260901/official_dppo_visualization/base_failure_rl_success/base \
  --port=9877
```

```bash
bash scripts/experiments/rl/play_dppo_base_failure_rl_success.sh \
  --folder=outputs/rl_residual_robomimic_20260901/official_dppo_visualization/base_failure_rl_success/rl \
  --port=9878
```

两个终端必须使用不同端口。Rerun 默认使用 `9876`；如果已有 Viewer 正在监听默认端口，新的
Rerun CLI 会把 recording 转发给旧 Viewer 后退出。这里显式使用 `9877`/`9878`，因此可以保留旧
Viewer，同时打开独立的 Base 和 RL 窗口。若端口被占用，换成其他两个不同的空闲端口。

只查看候选，不创建筛选 folder：

```bash
bash scripts/experiments/rl/play_dppo_base_failure_rl_success.sh \
  --list-only
```

只加载前 12 个候选进行快速检查：

```bash
bash scripts/experiments/rl/play_dppo_base_failure_rl_success.sh \
  --prepare-only --max-pairs=12
```

如果筛选 folder 已经存在且需要重新生成，只删除脚本创建的 symlink：

```bash
bash scripts/experiments/rl/play_dppo_base_failure_rl_success.sh \
  --prepare-only --replace-selection
```

如果只想筛选某一个候选，可以加 `--pair-index=2`。folder mode 默认加载指定 folder 下的全部
独立 `.rrd`，不会加载同目录的合并 archive；如确实需要合并 archive，再加 `--include-merged`。
脚本不会复制 `.rrd`/MP4，只创建指向已有 recording 的 symlink。


## 6. Residual：Zero Residual 与后续训练

执行位置：**REMOTE DPPO worker 容器**。Residual 仍从同一个 frozen `state_4000.pt` 开始，不把
官方 `state_300.pt` 直接当成 residual 的 base；否则会同时改变 base 参数和训练方法，无法回答
“Residual 相对同一 frozen base 的增益”这个问题。若要做“DPPO full fine-tuning 后再叠加 residual”，应
另开实验合同。

### 6.1 Zero Residual

目的：在相同 manifest 和 base diffusion RNG 下并行运行 Zero Residual，确认本地 wrapper 没有改变
base 行为。

```bash
set -euo pipefail
export GPUS=0,1,2
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"

bash scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh \
  --stage=zero --gpus="$GPUS" --launch-mode=direct

for seed in 1 2 3; do
  base_rate="$(jq -r --argjson seed "$seed" \
    '.per_seed[] | select(.seed == $seed) | .success_rate' \
    "$EXP_ROOT/base/metrics.json")"
  zero_metrics="$EXP_ROOT/zero_eval/seed_${seed}/metrics.json"
  test "$base_rate" = "$(jq -r '.success_rate' "$zero_metrics")"
  jq -e '.base_hash_unchanged == true and .zero_residual_action_equal == true' \
    "$zero_metrics"
done
```

任一 seed 不一致就停止并修接线。Frozen Base 和 Zero Residual 都是 `Formal Closed-loop`。

## 7. 单 iteration 训练探针

执行位置：**REMOTE DPPO worker 容器**；按 `GPUS` 将三个 seed 的探针调度到当前容器内的
独立 **单 GPU 进程**。

目的：用每个 seed 的一个完整 iteration 测量 base 推理、residual 推理、环境和 PPO update 时间，并验证
checkpoint、loss、base 冻结和进度记录。

```bash
set -euo pipefail
export GPUS=0,1,2
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"

bash scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh \
  --stage=timing --gpus="$GPUS" --launch-mode=direct

for seed in 1 2 3; do
  jq -e \
    '.iterations == 1 and .base_hash_unchanged == true and
     .train_loss != null and .stage_seconds.update > 0' \
    "$EXP_ROOT/timing/seed_${seed}/metrics.json"
done
```

### 7.1 读出耗时分解与正式预算

目的：把三个探针的单轮耗时换算成正式训练预算，并定位瓶颈阶段。`n_train_itr` 从 config 读取，
不在这里写死。

```bash
set -euo pipefail
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export CONFIG=configs/experiments/rl/residual_robomimic_dppo_square.yaml

for seed in 1 2 3; do
  jq -r --arg seed "$seed" '
    "seed=\($seed)  一轮=\(.elapsed_seconds | . * 10 | round / 10)s" +
    "  base推理=\(.stage_seconds.base_inference | . * 10 | round / 10)s" +
    "  residual推理=\(.stage_seconds.residual_inference | . * 10 | round / 10)s" +
    "  环境step=\(.stage_seconds.env_step | . * 10 | round / 10)s" +
    "  PPO更新=\(.stage_seconds.update | . * 10 | round / 10)s"
  ' "$EXP_ROOT/timing/seed_${seed}/metrics.json"
done

n_itr="$(awk '/^  n_train_itr:/ {print $2}' "$CONFIG")"
slowest="$(for seed in 1 2 3; do
  jq -r '.elapsed_seconds' "$EXP_ROOT/timing/seed_${seed}/metrics.json"
done | sort -g | tail -1)"
gpu_count="$(nvidia-smi -L | wc -l)"

python3 - "$slowest" "$n_itr" "$gpu_count" <<'PY'
import math, sys
per_itr, n_itr, gpus = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
one_seed_h = per_itr * n_itr / 3600
batches = math.ceil(3 / max(1, gpus))
print(f"最慢单轮        {per_itr:.1f} s")
print(f"正式 iteration  {n_itr}")
print(f"单 seed 全程    {one_seed_h:.1f} h")
print(f"3 seed / {gpus} 卡  {one_seed_h * batches:.1f} h  ({batches} 批)")
print(f"3 seed / 1 卡   {one_seed_h * 3:.1f} h")
PY
```

按最慢 seed 估算，因为同一批并行 worker 要等最慢的那个完成。三个 seed 用三张卡时是一批；
卡数少于 seed 数时按批次向上取整。

`stage_seconds` 的四项决定瓶颈归属，也决定加卡有没有用：

| 主导阶段 | 含义 | 加卡能否缓解 |
| --- | --- | --- |
| `env_step` | MuJoCo 仿真与 EGL 离屏渲染 | 只能靠 seed 级并行摊平总时长，单 seed 不变快 |
| `base_inference` | 冻结 base 的 DDIM 去噪 | 同上；降 `ddim_steps` 会偏离本实验固定的去噪合同 |
| `residual_inference` | residual actor 前向 | 同上；占比应远小于前两项 |
| `update` | PPO 反向与优化器 | 只有这一项主导时，单 seed 内的数据并行才有意义 |

这是 `Smoke`。预算不可接受时停止并重新规划，不缩减 `n_train_itr` 后声称复现。

## 8. 正式训练与配对评测

执行位置：**REMOTE DPPO worker 容器**；训练及每一种 eval 都按 `GPUS` 调度到当前容器内的
独立 **单 GPU 进程**。

这里是 seed 级并行，不是单个 seed 内的 DDP；因此不会改变每个 on-policy run 的 batch、优化器或
checkpoint 合同。

### 8.1 Learned Residual 训练

目的：并行训练三个独立 seed。每个 run 写 `STATUS`、`progress.jsonl`、`run.log` 和原子 checkpoint；
原命令重跑会从各自 checkpoint 续训。

```bash
set -euo pipefail
export GPUS=0,1,2
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"

bash scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh \
  --stage=train --gpus="$GPUS" --launch-mode=direct
```

### 8.2 Learned 与 Random Residual 评测

目的：分别在统一 manifest 上并行评测 learned 和 random residual；两个 stage 依次执行，stage 内按
seed 多卡并行。

```bash
set -euo pipefail
export GPUS=0,1,2
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"

bash scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh \
  --stage=learned --gpus="$GPUS" --launch-mode=direct
bash scripts/experiments/rl/run_residual_robomimic_dppo_multigpu.sh \
  --stage=random --gpus="$GPUS" --launch-mode=direct
```

### 8.3 汇总

目的：检查训练预算、episode 数、manifest、diffusion RNG、base hash 和四组指标是否完整，再计算实验
有效性及假设是否成立。汇总不并行，在当前 worker 内使用一张卡即可。

```bash
set -euo pipefail
export GPU=0
export EXP_ROOT=outputs/rl_residual_robomimic_20260901
export DPPO_DATA_DIR=datasets/raw/dppo
export DPPO_LOG_DIR="$EXP_ROOT/dppo_logs"

CUDA_VISIBLE_DEVICES="$GPU" ROBOT_PRACTICE_DPPO_GPU="$GPU" \
  bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=summarize \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root="$EXP_ROOT"

jq '{comparison, learned_gain, random_gain, contract_gates, hypothesis_gates,
     experiment_valid, hypothesis_supported}' "$EXP_ROOT/summary.json"
jq -e '.experiment_valid == true' "$EXP_ROOT/summary.json"
```

`hypothesis_supported=false` 是有效负结果，不得修改阈值后重算。

## 结果落点

- `Smoke`：`outputs/rl_residual_robomimic_20260901/preflight/gates.json`
- `Smoke`：`outputs/rl_residual_robomimic_20260901/timing/seed_*/{run_config,metrics}.json`
- `Formal Closed-loop`：`outputs/rl_residual_robomimic_20260901/base/metrics.json`
- `Smoke`：`outputs/rl_residual_robomimic_20260901/official_dppo_baseline_smoke/{run.log,metrics.json}`
- 官方 DPPO full fine-tuning：`outputs/rl_residual_robomimic_20260901/official_dppo_baseline/finetune/seed_*/checkpoint/state_300.pt`
- `Formal Closed-loop`：`outputs/rl_residual_robomimic_20260901/official_dppo_baseline/eval/seed_*/result.npz`
- 官方 DPPO 基准汇总：`outputs/rl_residual_robomimic_20260901/official_dppo_baseline/metrics.json`
- DPPO fine-tuned 可视化 raw rollout：`outputs/rl_residual_robomimic_20260901/official_dppo_visualization/raw/`
- Frozen Base 可视化 raw rollout：`outputs/rl_residual_robomimic_20260901/official_dppo_visualization/base/raw/`
- DPPO Base/RL Rerun/MP4 可视化：`outputs/rl_residual_robomimic_20260901/official_dppo_visualization/{rerun_base,rerun_rl}/`
- `Formal Closed-loop`：`outputs/rl_residual_robomimic_20260901/zero_eval/seed_*/metrics.json`
- 训练：`outputs/rl_residual_robomimic_20260901/residual/seed_*/{checkpoint.pt,progress.jsonl,run.log}`
- `Formal Closed-loop`：`outputs/rl_residual_robomimic_20260901/residual_eval/seed_*/metrics.json`
- `Formal Closed-loop`：`outputs/rl_residual_robomimic_20260901/random_residual/seed_*/metrics.json`
- Residual 四组对照：`outputs/rl_residual_robomimic_20260901/summary.json`

官方 DPPO full fine-tuning 的 `result.npz` / `metrics.json` 记录 task、seed、episode 数、成功率和平均
reward；Residual 的每个正式 eval `metrics.json` 另外记录 task suite、dataset size、observation keys、
action space/horizon、control frequency、train loss、open-loop action error、closed-loop success rate、
latency、failure count 和 base hash。

## 判断标准

| 检查 | 通过条件 | 失败处理 |
| --- | --- | --- |
| 资产/环境 | preflight 无缺失；EGL、CUDA、版本检查通过 | 停止 |
| 官方 DPPO 基准 | 3 个 full fine-tuning seed 均产出 `state_300.pt`，且每个 eval 完成 50 episodes | 不进入 residual |
| headroom | Frozen Base pooled success 在 `[10%, 90%]` | 停止本任务 |
| Zero 一致性 | 每个 seed 的成功率与 base 相同；zero 合成动作等于同次 base action | 修接线后重跑 |
| 配对合同 | manifest 与 base diffusion RNG seed 在四组间一致 | 实验无效 |
| residual base 冻结 | Residual 训练/评测 `base_hash_unchanged=true` | 实验无效 |
| residual 增益 | learned 相对 base `≥ +10pp`，且 3 个 seed 都提升 | 否则记负结果 |
| random 对照 | random gain 小于 learned gain 的一半 | 否则不能归因于学习 |
| residual 触界 | learned `saturation_fraction < 0.20` | bound 不合格，当前结论无效 |
