# PushT E2：相同初始环境的多 Seed Rollout

## 实验目标

回答从完全相同的 PushT 初始状态出发，不同 Diffusion sampling seed 是否能产生不同且
成功的完整闭环轨迹。

## 实验思路

- Pilot 使用 10 个 environment seed，每个 seed 执行 5 个独立 Diffusion rollout。
- 每次 rollout 都重新 `env.reset(seed=相同值)` 并执行 `policy.reset()`。
- noise 由 environment seed、policy seed 和真正发生的 replan index 确定性派生。
- BCChunk 每个 environment seed 运行一次，作为 deterministic 对照。
- 分开报告全部、成功和失败 rollout 的 agent/block joint trajectory diversity。

## 模式

`Closed-loop / Formal`。

## 可直接执行命令

```bash
BC_POLICY="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["bc_chunk"]["checkpoint_path"])')"
DIFFUSION_POLICY="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["diffusion"]["checkpoint_path"])')"

bash scripts/experiments/pusht/eval_multi_seed_rollouts.sh \
  --bc-policy="${BC_POLICY}" \
  --diffusion-policy="${DIFFUSION_POLICY}" \
  --num-environment-seeds=10 \
  --num-policy-seeds=5 \
  --output-dir=outputs/pusht_diffusion/pilot/e2_rollouts
```

打开可视化：

```bash
conda run --no-capture-output -n robot-practice \
  rerun --persist-state outputs/pusht_diffusion/pilot/e2_rollouts/recording.rrd
```

## 结果落点

```text
outputs/pusht_diffusion/pilot/e2_rollouts/
├── rollouts.parquet
├── replans.parquet
├── videos/
├── metrics.json
├── metrics.md
└── recording.rrd
```

每条记录包含 model、checkpoint、train seed、environment seed、policy seed、replan index、
noise seed/hash、success、reward、episode length、轨迹和 action target。

## 判断标准

- 产生 10 条 BCChunk 与 50 条独立 Diffusion rollout。
- 相同 environment seed 的每个 policy seed 都从独立 reset 的环境开始。
- 记录 single-sample success、any@5 success 及其置信区间。
- success-conditioned diversity 与 failed-rollout diversity 分开报告。
- 只有不同 policy seed 产生明显不同且多个成功的轨迹时，才能声称存在多种成功闭环行为。
