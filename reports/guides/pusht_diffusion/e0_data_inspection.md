# PushT E0：完整 Demonstration 审查

## 实验目标

从头到尾审查 `lerobot/pusht` 的全部 206 条 demonstration，了解数据中有哪些完整
接近、推动、旋转和重新接触行为，并为 E1 选择 10 个具有代表性的两帧 observation。

E0 只描述完整轨迹的全局多样性，不要求证明严格条件多模态，也不阻塞训练、E1、E2
或 E3。

## 实验做什么

- 逐帧读取全部 206 条 demonstration。
- 保存原始图像、agent path、absolute action-target path 和 dense reward。
- 从固定视角图像提取移动的灰蓝色 T-block 中心、朝向和完整轨迹。
- 根据接近方向和是否先旋转，对完整 demonstration 进行可解释的粗分类。
- 从 audit split 的不同全局策略中轮流选择 10 个 E1 candidates。
- 将每条 demonstration 保存为独立 Rerun Recording，打开归档后可在左侧切换。

`next.done` 只代表 episode 结束，不能单独解释成成功。当前数据的完成质量使用
`max_reward` 和原始 terminal/success 字段分别记录。

## 执行

若数据尚未准备：

```bash
bash scripts/experiments/pusht/prepare_dataset.sh
```

执行完整 E0：

```bash
bash scripts/experiments/pusht/inspect_demonstrations.sh \
  --num-episodes=206 \
  --num-candidates=10 \
  --output-dir=outputs/pusht_diffusion/data_inspection
```

已有旧结果时重新生成：

```bash
bash scripts/experiments/pusht/inspect_demonstrations.sh \
  --num-episodes=206 \
  --num-candidates=10 \
  --output-dir=outputs/pusht_diffusion/data_inspection \
  --overwrite
```

打开完整可视化：

```bash
conda run --no-capture-output -n robot-practice \
  rerun --persist-state outputs/pusht_diffusion/data_inspection/recording.rrd
```

Rerun 左侧 Recording selector 中应出现 `episode_000` 到 `episode_205`。每条
Recording 有自己的本地 `frame` 时间轴。

## 输出

```text
outputs/pusht_diffusion/data_inspection/
├── demonstrations/                 # 206 个独立 episode RRD
├── recording.rrd                   # 含 206 个可切换 Recording 的归档
├── trajectories.parquet            # 全部逐帧 agent/block/action/reward 数据
├── demonstration_features.parquet  # 每条完整轨迹的分类特征
├── episode_summaries.json
├── candidate_groups.json           # 全局策略分组，不代表条件多模态
├── candidates/
├── candidates.json                 # E1 使用的 10 个 audit candidates
├── inspection.md
├── metrics.json
└── metrics.md
```

## 判断标准

- `num_demonstrations = 206`，并覆盖所有 25,650 帧。
- Rerun 归档包含 206 个 Recording。
- `block_geometry_coverage` 被明确测量，提取失败不能静默忽略。
- 每条 demonstration 都有接近方向、rotation-first、block motion 和 recontact 属性。
- 每个 candidate 保存 previous/current state、两帧图像、未来 8 步 action 和未来图像。
- 报告明确区分全局轨迹多样性与严格条件多模态。
