# PushT E3：Diffusion 采样步数与延迟

## 实验目标

使用同一个 Diffusion checkpoint 比较 DDIM 5/10/20/50/100 与 DDPM 100 的闭环成功率和
推理延迟，获得 success-latency Pareto curve。

## 实验思路

- 所有条件使用相同 checkpoint、environment seeds、policy seed 和每次 replan 的 noise。
- DDIM 使用 `eta=0`，条件之间只改变 scheduler 和 `num_inference_steps`。
- Pilot 每个条件运行相同的 50 个 final environment seed。
- `--num-gpus=N` 按 environment seed 做 round-robin 分片；同一场景的六个条件始终由
  同一个 GPU worker 执行。
- latency 使用 batch size 1、每卡每条件 50 次 warm-up，以及每条件总计 200 次正式测量。
- rollout worker 全部退出后，每张卡都测量全部六个条件；200 次正式测量平均分摊到显式
  指定的 GPU 并汇总。条件顺序按 GPU 轮转，避免条件与单张卡或固定热状态绑定。
- latency 运行时显示 0–100% 进度条，并在 `latency_progress/` 中按 GPU、条件和
  latency kind 写入完成标记。
- 分开报告 model-only chunk、end-to-end policy 和每个实际执行 action 的摊销延迟。
- 不读取或恢复任何中间状态；任一 worker 失败时整次原子输出不发布。

## 模式

`Closed-loop / Formal`，并包含独立的 latency benchmark。

## 最终结论

**E3 支持 DDIM-5 作为当前 PushT Diffusion Policy 的最佳部署配置。对这个 checkpoint，
100-step sampling 明显冗余：增加 diffusion steps 没有带来可观测的闭环成功率收益，
只增加了推理延迟。**

正式结果包含 50 个相同场景上的 6 个条件，共 300 条闭环 rollout。本次已发布结果实际
保留了每个条件、每种 latency kind 各 1,000 个正式样本，由 8 张 NVIDIA H20 各测
125 个；这比当前启动命令默认的 200 个样本更充分，无需重跑。

| 条件 | 成功数 | 成功率（Wilson 95% CI） | end-to-end p95 | 相对 DDPM-100 | 达标 |
| --- | ---: | ---: | ---: | ---: | --- |
| **DDIM-5** | **29/50** | **58%（44.2%–70.6%）** | **47.7 ms** | **12.57×** | **是** |
| DDIM-10 | 28/50 | 56%（42.3%–68.8%） | 71.9 ms | 8.35× | 是 |
| DDIM-20 | 29/50 | 58%（44.2%–70.6%） | 121.4 ms | 4.94× | 是 |
| DDIM-50 | 27/50 | 54%（40.4%–67.0%） | 307.7 ms | 1.95× | 否 |
| DDIM-100 | 29/50 | 58%（44.2%–70.6%） | 601.1 ms | 1.00× | 否 |
| DDPM-100 | 30/50 | 60%（46.2%–72.4%） | 600.1 ms | reference | — |

结果支持以下判断：

- DDIM-5 将 diffusion steps 相对 100-step reference 减少 20 倍，end-to-end p95
  加速 12.57 倍，而成功率点估计仅下降 2 个百分点，满足预设的“至少 2×、成功率下降
  不超过 5 个百分点”标准。
- DDIM-5 相对 DDPM-100 的同场景翻转为 6 个新增成功、7 个新增失败，精确 McNemar
  `p=1.0`，没有检测到成功率差异。DDIM-5 与 DDIM-20 成功率相同，但前者又快约
  2.54 倍，因此最终选择 DDIM-5。
- DDIM-5/10/20/50/100 的成功率没有随 steps 增加而单调上升；同场景总体比较为
  Cochran's Q = 0.421、`p=0.981`。
- DDIM-100 与 DDPM-100 的成功率分别为 58% 和 60%，p95 分别为 601.1 ms 和
  600.1 ms。相同步数下仅更换 scheduler 没有产生可观测收益。
- diffusion steps 会改变具体闭环轨迹：50 个场景中有 31 个在五种 DDIM steps 下同时
  出现过成功和失败。但这些场景级变化没有转化为总体成功率提升。
- 短步数的汇总 p95 受到一张较慢 H20 的影响，因此是偏保守的估计。DDIM-5 各卡 p95
  约为 33.0–58.6 ms；即使使用最慢卡结果，相对同卡 DDPM-100 仍超过 10×，不改变
  DDIM-5 的选择。

E3 支持的能力表述是：

> 当前 Diffusion Policy 可以使用大幅截断的 DDIM 去噪过程，在基本保持闭环成功率的
> 同时显著降低推理延迟。

结论边界：

- 50 个场景的成功率置信区间仍然较宽；当前结果满足预设的点估计标准，但不能证明
  DDIM-5 与 DDPM-100 严格等价或完成正式非劣效检验。
- E3 研究 sampling steps、scheduler 与 success-latency trade-off，不验证固定状态下
  action distribution 的多模性；多模性结论仍由 E1/E2 的对应实验支撑。

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
  --latency-iterations=200 \
  --videos-per-condition=5 \
  --num-gpus=8 \
  --output-dir=outputs/pusht_diffusion/pilot/e3_sampling_steps \
  --overwrite
```

`--num-gpus` 必须是正整数，不能超过本次 environment seed 数量或当前可见 CUDA
device 数量。单卡执行时使用 `--num-gpus=1`。
`--overwrite` 会从空的原子临时目录重新执行，不复用旧 E3 输出。

打开 50 个场景的可切换同步可视化：

```bash
bash scripts/experiments/pusht/view_e3_rollouts.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/e3_sampling_steps/recordings \
  --port=9878
```

每个 `env_<seed>.rrd` 是一个独立 Recording。默认 Blueprint 同步展示
DDIM-5/10/20/50/100 和 DDPM-100 六个面板，并在底部汇总 reward。每个面板都包含
simulator、agent path、block path、橙色完整实际 action-target 序列和当前 action target。
Recording selector 只切换场景。

所有 50 个场景、六个条件都写入完整 simulator frames。`--videos-per-condition=5` 只限制
额外 MP4 文件数量，不会裁剪 Rerun Recording。

## 结果落点

```text
outputs/pusht_diffusion/pilot/e3_sampling_steps/
├── rollouts.parquet
├── replans.parquet
├── latency.parquet
├── condition_metrics.parquet
├── latency_progress/
├── videos/
├── recordings/
│   ├── env_2000.rrd
│   ├── env_2001.rrd
│   └── ...
├── metrics.json
├── metrics.md
└── run_manifest.json
```

## 判断标准

- 六个条件使用相同 50 个 environment seed 和相同的初始 noise 派生规则。
- 每个场景文件必须恰好包含六个同步条件，并使用 active/default Blueprint。
- latency 记录 mean、p50、p90、p95、p99、std 和 peak memory。
- `latency.parquet` 每个条件、每种 latency kind 的样本数必须与 `run_manifest.json`
  一致，并记录样本所属 GPU；所有参与测量的 GPU 型号必须一致。本次正式结果为每种
  1,000 条，当前脚本默认的新运行配置为每种 200 条。
- 成功率与 DDPM-100 做配对比较并报告置信区间。
- 主图使用 p95 end-to-end action-chunk latency 为横轴、success rate 为纵轴。
- 目标参考是 p95 至少加速 2 倍，同时 success drop 不超过 5 个百分点；这是评价标准，
  不是预设结论。
