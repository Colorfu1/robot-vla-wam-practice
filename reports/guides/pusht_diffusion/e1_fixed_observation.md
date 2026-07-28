# PushT E1：固定 Observation 的多 Noise Sampling

## 实验目标

回答完全相同的模型输入下，不同 Diffusion 初始 noise 是否产生不同的 8-step action
chunks。E1 只分析模型输出，不执行环境动作。

## 实验思路

- 固定 E0 导出的完整两帧 observation tensor、state、归一化信息和 observation queue。
- BCChunk 重复调用 3 次，确认 deterministic 输出一致。
- Diffusion 预先保存 20 个 noise tensor，每个 noise 完整去噪一次。
- 相同 noise 重复推理验证复现，不同 noise 计算轨迹和 endpoint diversity。
- 分析绝对 action target 相对当前 agent position 的方向，不把 action 当速度积分。

## 模式

`Offline / Formal`。只分析固定输入的模型输出，不执行环境。

## 可直接执行命令

先读取 development 选中的 checkpoint：

```bash
BC_POLICY="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["bc_chunk"]["checkpoint_path"])')"
DIFFUSION_POLICY="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["diffusion"]["checkpoint_path"])')"
```

执行 E1：

```bash
bash scripts/experiments/pusht/eval_fixed_observations.sh \
  --bc-policy="${BC_POLICY}" \
  --diffusion-policy="${DIFFUSION_POLICY}" \
  --candidates=outputs/pusht_diffusion/data_inspection/candidates.json \
  --num-action-samples=20 \
  --output-dir=outputs/pusht_diffusion/pilot/e1_fixed_observation
```

打开可视化：

```bash
conda run --no-capture-output -n robot-practice \
  rerun --persist-state outputs/pusht_diffusion/pilot/e1_fixed_observation/recording.rrd
```

## 结果落点

```text
outputs/pusht_diffusion/pilot/e1_fixed_observation/
├── observations/
├── noises/
├── action_chunks.parquet
├── diversity_metrics.parquet
├── fixed_observation_samples.png
├── metrics.json
├── metrics.md
└── recording.rrd
```

## 判断标准

- 20 次 sampling 使用 hash 一致的 processed observation tensor。
- 相同 observation、initial noise 和 reverse seed 的 action 最大误差不超过 `1e-6`。
- 不同 noise 的差异使用 trajectory diversity 和 endpoint diversity 量化。
- 输出能够区分归一化空间与真实 action 空间的指标。
- E1 只能证明存在不同 action patterns，不能单独声称这些模式有效或能够成功。
