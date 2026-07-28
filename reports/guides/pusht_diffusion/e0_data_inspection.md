# PushT E0-lite：数据与 Demonstration 检查

## 实验目标

检查 `lerobot/pusht` 中是否存在不同的接近、推动和恢复轨迹，并为 E1 选择 5–10 个
两帧 observation history。该阶段支持实验动机，但不证明严格条件多模态。

## 实验思路

- 可视化至少 50 条 demonstration。
- 保存 agent path、action target path、rendered frames 和可获得的 block path。
- 关注左右接近、下方推动、旋转后平移、不同接触点和失败恢复。
- 从 held-out audit episode 中选择固定 observation candidates。
- 保存模型需要的两帧历史、状态、未来 8 步 action 和后续图像。

## 模式

`Offline` 定性检查。

## 可直接执行命令

若数据尚未准备，先执行：

```bash
bash scripts/experiments/pusht/prepare_dataset.sh \
  --train-seed=1000 \
  --device=cuda
```

然后导出 demonstration 和候选 observation：

```bash
bash scripts/experiments/pusht/inspect_demonstrations.sh \
  --num-episodes=50 \
  --num-candidates=10 \
  --output-dir=outputs/pusht_diffusion/data_inspection
```

打开可视化：

```bash
conda run --no-capture-output -n robot-practice \
  rerun --persist-state outputs/pusht_diffusion/data_inspection/recording.rrd
```

## 结果落点

```text
outputs/pusht_diffusion/data_inspection/
├── demonstrations/
├── candidate_groups/
├── trajectories.parquet
├── candidates.json
├── metrics.json
├── metrics.md
└── recording.rrd
```

## 判断标准

- 至少成功读取并展示 50 条 demonstration。
- 每个 candidate 都保存完整两帧输入、状态、未来 8 步 action 和来源索引。
- 文档明确区分“任务理论上存在多解”“数据中观察到不同轨迹”和“尚未证明严格条件多模态”。
- E0 失败不能被静默忽略，但简单视觉轨迹提取不足不应阻塞正式训练。
