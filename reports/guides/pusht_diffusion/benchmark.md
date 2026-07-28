# PushT Diffusion 正式训练与 Benchmark 指南

## 实验目标

使用固定的 165 条 train episode 公平训练 BCChunk 与 Diffusion，仅通过 20 个
development environment seed 选择 checkpoint，并为 E1、E2、E4 提供冻结模型。

## 实验思路

- 正式训练必须从 Pilot 的成对随机 step-zero 权重开始，不能续训 Smoke checkpoint。
- 两个策略使用相同数据、batch size、训练步数和 checkpoint 保存频率。
- checkpoint 主排序使用 development closed-loop success、mean max reward、较早 step。
- held-out audit episode 和 final environment seed 不参与模型选择。
- 公开预训练模型只验证环境和评测兼容性，单独标记为外部参考。

## 模式

本地完整训练与 development checkpoint selection 标记为 `Formal`；公开预训练模型标记为
外部兼容性参考，不进入本地 split 的公平对照。

## 可直接执行命令

### 1. 前置准备

如果尚未运行 Smoke 数据准备，先执行：

```bash
bash scripts/experiments/pusht/prepare_dataset.sh \
  --train-seed=1000 \
  --device=cuda
```

### 2. 可选：公开预训练模型兼容性参考

```bash
bash scripts/experiments/pusht/benchmark_pretrained.sh \
  --n-episodes=10 \
  --output-dir=outputs/pusht_diffusion/benchmark_pretrained
```

该公开 checkpoint 使用旧版 LeRobot 格式，normalization statistics 保存在
`model.safetensors` 中而不是独立的 processor JSON。当前兼容加载器会按 LeRobot 官方
migration 语义恢复这些统计。启动时看到仅包含 `normalize_inputs`、
`normalize_targets` 和 `unnormalize_outputs` 的 8 个 `Unexpected key` 属于预期提示；
随后应出现 `recovered legacy normalization statistics` 并继续执行。

该结果必须标记为 `external pretrained reference — not split-comparable`。

### 3. 正式训练

模型结构与优化器配置来自两个明确的 Python model factory；Bash 不保存模型参数。
需要先审阅配置时，在命令末尾添加 `--print-command`，该操作无副作用。

```bash
bash scripts/experiments/pusht/train_bc_chunk.sh \
  --mode=pilot \
  --steps=200000 \
  --train-seed=1000 \
  --output-dir=outputs/pusht_diffusion/pilot/bc_chunk

bash scripts/experiments/pusht/train_diffusion.sh \
  --mode=pilot \
  --steps=200000 \
  --train-seed=1000 \
  --output-dir=outputs/pusht_diffusion/pilot/diffusion
```

如果有至少两张可见 CUDA GPU，可以用下面的命令替代上面的两个单卡命令：

```bash
bash scripts/experiments/pusht/train_bc_chunk_multi_gpu.sh \
  --num-gpus=2 \
  --mode=pilot \
  --steps=200000 \
  --train-seed=1000 \
  --batch-size=32 \
  --output-dir=outputs/pusht_diffusion/pilot/bc_chunk

bash scripts/experiments/pusht/train_diffusion_multi_gpu.sh \
  --num-gpus=2 \
  --mode=pilot \
  --steps=200000 \
  --train-seed=1000 \
  --batch-size=32 \
  --output-dir=outputs/pusht_diffusion/pilot/diffusion
```

多卡入口使用当前 `CUDA_VISIBLE_DEVICES` 中的前 N 张卡。`--batch-size` 表示每卡
batch；所以上面的双卡配置是 `32 × 2 = 64`，保持原正式实验的全局 batch 不变。如果
设置双卡 `--batch-size=64`，全局 batch 会变为 128。脚本不会自动缩放。

单卡和多卡只允许二选一，不能写入同一已有输出目录。二者生成相同结构的 LeRobot
checkpoint，后续 development checkpoint selection、E1、E2 和 E4 命令都不需要修改。

正式训练保持全局 batch size 64，每 25,000 step 保存 checkpoint。BCChunk 的
clean-action MSE 与 Diffusion 的 epsilon-prediction MSE 含义不同，不能直接比较数值
大小。

### 4. 仅使用 development 选择 checkpoint

```bash
bash scripts/experiments/pusht/select_checkpoints.sh \
  --bc-run=outputs/pusht_diffusion/pilot/bc_chunk \
  --diffusion-run=outputs/pusht_diffusion/pilot/diffusion \
  --num-development-seeds=20 \
  --output-dir=outputs/pusht_diffusion/pilot/checkpoint_selection
```

后续实验读取选中的路径：

```bash
BC_POLICY="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["bc_chunk"]["checkpoint_path"])')"
DIFFUSION_POLICY="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["diffusion"]["checkpoint_path"])')"
```

### 5. E1、E2、E4 完成后生成总报告

```bash
bash scripts/experiments/pusht/make_report.sh \
  --e0=outputs/pusht_diffusion/data_inspection \
  --e1=outputs/pusht_diffusion/pilot/e1_fixed_observation \
  --e2=outputs/pusht_diffusion/pilot/e2_rollouts \
  --e4=outputs/pusht_diffusion/pilot/e4_sampling_steps \
  --output-dir=outputs/pusht_diffusion/pilot/report
```

## 结果落点

- 外部参考：`outputs/pusht_diffusion/benchmark_pretrained/`
- 正式 checkpoint：`outputs/pusht_diffusion/pilot/{bc_chunk,diffusion}/`
- 训练日志：各训练目录下的 `train.log`
- checkpoint 选择：
  `outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json`
- 总报告：`outputs/pusht_diffusion/pilot/report/report.md`

## 判断标准

- 正式训练只读取 165 条 train episode。
- BCChunk 与 Diffusion 的初始共享参数 hash 一致。
- 20 个 development seed 的 closed-loop 结果完整且没有使用 audit/final seed。
- checkpoint 排序依据和最终选择写入机器可读结果。
- `report.md` 不接受 `Smoke` 输出冒充正式结果。

训练完成后，应把 train loss、open-loop action error、closed-loop success、latency 和
failure cases 登记到 [实验运行记录](../../experiment_log.md)。
