# PushT Diffusion Smoke 指南

## 实验目标

验证环境复现、数据读取、BCChunk/Diffusion 随机初始化、最小训练、显式 noise、闭环
rollout、结果文件和 Rerun 可视化链路能够正常工作。

## 实验思路

- 先检查当前 Python、LeRobot、gym-pusht、Pymunk 和显式 noise 接口。
- 下载完整数据，但 Smoke 训练只使用固定的 8 条 train episode。
- 两个策略分别从 `train_seed=1000` 控制的随机权重训练 1,000 updates。
- 固定输入采样缩小为 3 个 noise；E2a/E2b 都执行 1 个环境 × 2 个 policy seed 的闭环
  检查。
- 所有输出明确标记为 `Smoke`，不得用作正式实验结论。

## 模式

`Smoke`。该文档中的运行不属于正式 benchmark。

## 可直接执行命令

### 1. 安装并检查接口

```bash
conda activate robot-practice
pip install -r requirements.txt

bash scripts/experiments/pusht/check_environment.sh
```

`gym-pusht==0.1.x` 需要 `pymunk<7`，项目依赖已固定为 `pymunk>=6.6,<7`。

### 2. 准备数据视图

```bash
bash scripts/experiments/pusht/prepare_dataset.sh
```

### 3. 导出 Smoke 需要的固定 observation

```bash
bash scripts/experiments/pusht/inspect_demonstrations.sh \
  --num-episodes=206 \
  --num-candidates=10 \
  --output-dir=outputs/pusht_diffusion/data_inspection
```

这一步生成 `candidates.json`，供 Smoke 的缩小版 E1 和后续正式 E1 共用。

### 4. 训练两个 Smoke checkpoint

下面两个 Bash 文件只是 conda/Python 转发层；模型配置分别位于
`model_configs/bc_chunk.py` 和 `model_configs/diffusion.py`。启动前可在任一命令末尾
添加 `--print-command` 查看最终 Python 入口和完整 resolved configuration。

```bash
bash scripts/experiments/pusht/train_bc_chunk.sh \
  --mode=smoke \
  --steps=1000 \
  --train-seed=1000 \
  --output-dir=outputs/pusht_diffusion/smoke/bc_chunk

bash scripts/experiments/pusht/train_diffusion.sh \
  --mode=smoke \
  --steps=1000 \
  --train-seed=1000 \
  --output-dir=outputs/pusht_diffusion/smoke/diffusion
```

两个入口都直接调用 LeRobot 的 `lerobot_train.train(cfg)`，不会使用本地重写的训练循环，
也不会读取 step-zero checkpoint。启动日志应显示
`initialization=random_from_train_seed`。

如果有至少两张可见 CUDA GPU，可以用下面的命令替代上面的两个单卡命令。不要同时
运行单卡和多卡命令写入同一个输出目录：

```bash
bash scripts/experiments/pusht/train_bc_chunk_multi_gpu.sh \
  --num-gpus=2 \
  --mode=smoke \
  --steps=1000 \
  --train-seed=1000 \
  --batch-size=4 \
  --output-dir=outputs/pusht_diffusion/smoke/bc_chunk

bash scripts/experiments/pusht/train_diffusion_multi_gpu.sh \
  --num-gpus=2 \
  --mode=smoke \
  --steps=1000 \
  --train-seed=1000 \
  --batch-size=4 \
  --output-dir=outputs/pusht_diffusion/smoke/diffusion
```

这里 `--batch-size=4` 是每卡 batch，双卡全局 batch 仍为 8，与原单卡 Smoke
保持一致。多卡输出的 checkpoint 路径和格式不变，因此第 5 步的 `smoke_eval.sh`
无需修改。

### 5. 执行缩小版 E1 和 E2

```bash
bash scripts/experiments/pusht/smoke_eval.sh \
  --bc-policy=outputs/pusht_diffusion/smoke/bc_chunk/checkpoints/001000/pretrained_model \
  --diffusion-policy=outputs/pusht_diffusion/smoke/diffusion/checkpoints/001000/pretrained_model \
  --candidates=outputs/pusht_diffusion/data_inspection/candidates.json \
  --output-dir=outputs/pusht_diffusion/smoke/evaluation
```

## 结果落点

- 接口检查：`outputs/pusht_diffusion/metadata/`
- 数据视图：`outputs/pusht_diffusion/dataset_views/{smoke,pilot}/`
- Smoke checkpoint：`outputs/pusht_diffusion/smoke/{bc_chunk,diffusion}/`
- Smoke 评测：`outputs/pusht_diffusion/smoke/evaluation/`

Smoke E1 目录包含 `metrics.json`、`metrics.md`、Parquet，以及
`recording_initial_noise_only.rrd` 和 `recording_full_sampling_seed.rrd`；Smoke
E2a/E2b 分别写入 `e2a_initial_noise_sensitivity/` 和
`e2b_full_ddpm_sampling/`，包含闭环视频与逐 rollout Rerun 文件。

## 判断标准

- 相同 reset seed 和相同 action sequence 的 PushT 轨迹可以复现。
- 相同 observation、noise 和 reverse seed 的 Diffusion 输出最大误差不超过 `1e-6`。
- 两个训练入口都从记录的 `train_seed` 随机初始化且不依赖初始化目录。
- 两种策略都能完成最小训练并被 evaluator 加载。
- Smoke E1/E2 能产生非空指标、视频和 Rerun 文件。

通过只表示 pipeline 可用，不表示模型性能或多模态假设已经得到验证。
