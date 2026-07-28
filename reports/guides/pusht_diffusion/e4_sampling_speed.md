# PushT E4：Diffusion 采样步数与延迟

## 实验目标

使用同一个 Diffusion checkpoint 比较 DDIM 5/10/20/50/100 与 DDPM 100 的闭环成功率和
推理延迟，获得 success-latency Pareto curve。

## 实验思路

- 所有条件使用相同 checkpoint、environment seeds、policy seed 和每次 replan 的 noise。
- DDIM 使用 `eta=0`，条件之间只改变 scheduler 和 `num_inference_steps`。
- Pilot 每个条件运行相同的 50 个 final environment seed。
- latency 使用 batch size 1、50 次 warm-up 和至少 1,000 次正式测量。
- 分开报告 model-only chunk、end-to-end policy 和每个实际执行 action 的摊销延迟。

## 模式

`Closed-loop / Formal`，并包含独立的 latency benchmark。

## 可直接执行命令

```bash
DIFFUSION_POLICY="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["diffusion"]["checkpoint_path"])')"

bash scripts/experiments/pusht/benchmark_sampling_steps.sh \
  --diffusion-policy="${DIFFUSION_POLICY}" \
  --schedulers=ddim,ddpm \
  --ddim-steps=5,10,20,50,100 \
  --ddpm-steps=100 \
  --n-episodes=50 \
  --latency-warmup-iterations=50 \
  --latency-iterations=1000 \
  --videos-per-condition=5 \
  --output-dir=outputs/pusht_diffusion/pilot/e4_sampling_steps
```

打开可视化：

```bash
conda run --no-capture-output -n robot-practice \
  rerun --persist-state outputs/pusht_diffusion/pilot/e4_sampling_steps/recording.rrd
```

## 结果落点

```text
outputs/pusht_diffusion/pilot/e4_sampling_steps/
├── rollouts.parquet
├── replans.parquet
├── latency.parquet
├── condition_metrics.parquet
├── videos/
├── metrics.json
├── metrics.md
└── recording.rrd
```

## 判断标准

- 六个条件使用相同 50 个 environment seed 和相同的初始 noise 派生规则。
- latency 记录 mean、p50、p90、p95、p99、std 和 peak memory。
- 成功率与 DDPM-100 做配对比较并报告置信区间。
- 主图使用 p95 end-to-end action-chunk latency 为横轴、success rate 为纵轴。
- 目标参考是 p95 至少加速 2 倍，同时 success drop 不超过 5 个百分点；这是评价标准，
  不是预设结论。
