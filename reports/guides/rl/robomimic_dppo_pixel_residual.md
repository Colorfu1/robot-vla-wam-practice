# robomimic + DPPO pixel residual RL guide

这份 guide 对应计划 `plans/2026-09-01-Residual-RL下游改接robomimic-DPPO-pixel.md`。它把
DPPO 的 `square` pixel policy 当作 frozen base，在 Docker 中接一个 bounded residual actor/critic；
DPPO 上游本身仍只作为外部 pixel anchor。不会修改本机 `robot-practice` 环境，也不会在镜像构建或本
文档中的命令里自动下载数据、归一化文件或 checkpoint。

## 1. 构建只含环境的 Docker 层

在仓库根目录执行（需要 Docker 能访问基础镜像和公开 Git 仓库）：

```bash
docker build \
  -f docker/training/Dockerfile.robomimic-dppo-pixel \
  -t practice:robomimic-dppo-pixel .
```

该层基于 `practice:latest`，从基础镜像已有的 `lightwam` 环境克隆出独立的 `dppo` 环境（复用其
Python 3.10/CUDA Torch，避免重复下载 CUDA wheel；DPPO commit 为
`cc7234ad7ff39a8f32de3af903606723a16f0648`，robomimic/robosuite 1.4.1）。构建不包含任何训练数据和
模型文件。

启动容器并挂载当前仓库：

```bash
docker run --rm -it --gpus all \
  -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl -e SDL_VIDEODRIVER=dummy \
  -v "$PWD":/workspace/practice \
  -w /workspace/practice \
  practice:robomimic-dppo-pixel bash
```

容器内先检查环境（只读，不下载）：

```bash
conda run --no-capture-output -n dppo \
  python /usr/local/lib/robot-practice/verify_robomimic_dppo_env.py \
  --source=/opt/robot-practice/dppo
```

## 2. 由用户下载 DPPO 资产

以下是上游 `script/download_url.py` 对 Square pixel 工作点给出的地址。请在容器内、确认直连/代理
策略后由用户执行；本次实现不会替用户下载。目录可以换成有足够空间的挂载盘，但之后要保持环境变量
一致。

```bash
export DPPO_DATA_DIR=/workspace/practice/datasets/raw/dppo
export DPPO_LOG_DIR=/workspace/practice/outputs/rl_residual_robomimic_20260901/dppo_logs
mkdir -p "$DPPO_DATA_DIR/robomimic/square-img" \
         "$DPPO_LOG_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint"

# pixel square dataset folder (contains train.npz)
conda run --no-capture-output -n dppo python - <<'PY'
import gdown, os
gdown.download_folder(
    url="https://drive.google.com/drive/u/1/folders/1-aGqVeKLIzJCEst8p0ZTjfjkrXFfJLxa",
    output=os.path.join(os.environ["DPPO_DATA_DIR"], "robomimic/square-img"),
)
PY

# normalization.npz
conda run --no-capture-output -n dppo python - <<'PY'
import gdown, os
gdown.download(
    url="https://drive.google.com/file/d/1jq5atfHdu-ZMQ8YaFjwcctbJESBYDgP8/view?usp=drive_link",
    output=os.path.join(os.environ["DPPO_DATA_DIR"], "robomimic/square-img/normalization.npz"),
    fuzzy=True,
)
PY

# official square pixel base state_4000.pt
conda run --no-capture-output -n dppo python - <<'PY'
import gdown, os
gdown.download(
    url="https://drive.google.com/file/d/1miud5SX41xjPoW8yqClRhaFx0Q7EWl3V/view?usp=drive_link",
    output=os.path.join(
        os.environ["DPPO_LOG_DIR"],
        "robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt",
    ),
    fuzzy=True,
)
PY
```

下载后确认：

```bash
test -s "$DPPO_DATA_DIR/robomimic/square-img/train.npz"
test -s "$DPPO_DATA_DIR/robomimic/square-img/normalization.npz"
test -s "$DPPO_LOG_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt"
```

## 3. 按阶段运行

每一步都写 `run.log`、`progress.jsonl` 和 `metrics.json`；长任务会周期性输出
`STATUS completed=<n> total=<N>`，结束输出 `DONE`。

先跑不依赖资产的合同预检和 dummy smoke：

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=preflight --dummy \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/preflight

bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=train --dummy --smoke \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/smoke
```

smoke 的用途是验证 pixel 白名单、`7D × horizon=4` 接口、采样/更新/保存 checkpoint、zero residual
精确复现 base，以及 base 参数 hash 前后一致；它不是成功率结果。

资产就绪后，先让预检记录缺失项为空，再调用上游官方 eval 作为 anchor（默认 sparse reward）：

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/third_party/dppo/dppo${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=preflight \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/preflight

cd third_party/dppo/dppo
conda run --no-capture-output -n dppo python script/run.py \
  --config-name=eval_diffusion_mlp_img \
  --config-dir=cfg/robomimic/eval/square wandb=null \
  base_policy_path="$DPPO_LOG_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt" \
  normalization_path="$DPPO_DATA_DIR/robomimic/square-img/normalization.npz"
cd /workspace/practice
```

如果 anchor 与上游已发表数字相差超过 5 个百分点，停止，不进入 residual 训练。

本地 adapter 的 frozen-base/zero 对照、单 iteration 计时、正式 residual、random bounded residual 和
汇总入口如下：

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=base-eval --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/base

bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=train --max-iterations=1 --time-stages \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/timing

for seed in 1 2 3; do
  bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
    --phase=train --seed="$seed" \
    --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
    --output-root="outputs/rl_residual_robomimic_20260901/residual/seed_${seed}"
done

for seed in 1 2 3; do
  bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
    --phase=eval --seed="$seed" --residual-root="outputs/rl_residual_robomimic_20260901/residual/seed_${seed}" \
    --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
    --output-root="outputs/rl_residual_robomimic_20260901/residual_eval/seed_${seed}"
done

for seed in 1 2 3; do
  bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
    --phase=eval --seed="$seed" --residual-mode=random-bounded \
    --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
    --output-root="outputs/rl_residual_robomimic_20260901/random_residual/seed_${seed}"
done

bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=summarize --output-root=outputs/rl_residual_robomimic_20260901
```

正式训练前应使用 `n_envs=50, n_steps=400, n_train_itr=301`；如果 CPU 不足而降低 `n_envs`，必须按
比例增加 `n_steps`，并把 resolved 配置和实际耗时一起保存。判据、成功条件和产物路径以计划文件为准。

## 4. 结果检查

至少检查以下字段：

```bash
jq '{phase, observation_keys, action_dim, horizon_steps, base_hash_unchanged, zero_residual_action_equal}' \
  outputs/rl_residual_robomimic_20260901/smoke/metrics.json
```

正常 smoke 必须同时看到 `base_hash_unchanged: true`、`zero_residual_action_equal: true`、完整
`STATUS`/`DONE`。任何 object/目标真值 key 出现在 adapter observation 中，或 base hash 变化，都视为
接线缺陷；不把 smoke、timing 或上游 anchor 当作本地 residual 正式结果。
