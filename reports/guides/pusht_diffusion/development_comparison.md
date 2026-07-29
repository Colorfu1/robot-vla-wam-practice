# PushT Development Checkpoint 对比

## 实验目标

在同一张目标 GPU 上，对选中的 BCChunk 与 Diffusion checkpoint 执行完全配对的
development closed-loop rollout，找出仅 BC 成功、仅 Diffusion 成功、均失败和均成功
的环境初始状态。每个模型按 seed 保存 20 个独立 RRD，并在两个 Viewer 窗口中使用相同
的 `env_1000..env_1019` Recording ID 对照。

## 实验思路

- 从 checkpoint selection JSON 读取两个选中 checkpoint，并校验权重 SHA256。
- 固定 development environment seed `1000..1019` 和 `policy_seed=0`。
- Diffusion 固定为 DDPM 100 步；两个模型在同一张 GPU 上顺序执行。
- 每条 rollout 在仿真完成时直接写入自己的 `env_<seed>.rrd`。
- 所有任务 Recording 使用相同的 `/rollout/*` 实体路径和默认 Blueprint。
- 分类结果写入 `comparison.json`，不会覆盖 checkpoint selection 的原始聚合值。

## 模式

`Closed-loop / Formal development diagnostic`。该结果复用 checkpoint selection 所使用的
development reset，不是 final seed 上的独立泛化评测。

## 可直接执行命令

生成完整对比结果：

```bash
bash scripts/experiments/pusht/compare_development_checkpoints.sh \
  --selection=outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json \
  --device=cuda:0 \
  --output-dir=outputs/pusht_diffusion/pilot/development_comparison
```

分别拉起 BCChunk 和 Diffusion Viewer。启动器会检查并加载目录中的 20 个 RRD：

```bash
bash scripts/experiments/pusht/view_development_recordings.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/development_comparison/recordings/bc_chunk \
  --port=9876

bash scripts/experiments/pusht/view_development_recordings.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/development_comparison/recordings/diffusion \
  --port=9877
```

两个窗口都选择同一个 `env_<seed>`，即可对照相同确定性 reset。例如两边都选择
`env_1008`。如果本机曾经保存过旧版 seed 路径布局，第一次打开新结果后在 Rerun 中执行
一次 **Reset Blueprint**；之后切换 Recording 不需要再次重置。

默认 Blueprint 只保留两个窗格：较大的 `PushT rollout` 仿真画面和较窄的 `Reward`
曲线。metadata、success、rollout index 与 `/task/*` 仍保存在 RRD 中，需要时从数据树
或 Selection 面板查看，不再长期占用独立窗格。

## 结果落点

```text
outputs/pusht_diffusion/pilot/development_comparison/
├── comparison.json
├── rollouts.parquet
├── replans.parquet
├── recordings/
│   ├── bc_chunk/
│   │   ├── env_1000.rrd
│   │   └── ... env_1019.rrd
│   └── diffusion/
│       ├── env_1000.rrd
│       └── ... env_1019.rrd
├── videos/
├── metrics.json
├── metrics.md
└── run_manifest.json
```

每个 RRD 只包含一个 Recording，Recording ID 与文件名中的 seed 一致。不同 Recording
内部都使用 `/rollout/*`，数据隔离由 Recording ID 提供。`comparison.json` 保存四类
environment seed、逐 seed success/reward、初始 agent/block 状态、相对 RRD 路径和
`/rollout` 实体根。Parquet 保留完整轨迹、动作、reward 和 Diffusion replan 信息。

## 判断标准

- 两个 checkpoint SHA256 与 selection JSON 完全一致。
- `rollouts.parquet` 恰好有 40 行，两个模型各覆盖 seed `1000..1019`。
- `comparison.json` 的四个类别互斥且并集恰好为全部 20 个 seed。
- `videos/` 恰好有 40 个 MP4；两个模型目录各有 20 个非空 RRD。
- 每个 RRD 恰好有一个 `env_<seed>` Recording，timeline 从零开始。
- 每个 Recording 只使用统一的 `/rollout/*` 与 `/task/*` 应用实体。
- 每个 RRD 均通过 `rerun rrd verify` 和代码内语义校验。
- Reset Blueprint 后只显示 `PushT rollout` 与 `Reward` 两个数据窗格。
- 切换 `env_1000`、`env_1001` 和后续 seed 时，图像、轨迹、reward 与元数据同步更新，
  不出现黑屏。
- `run_manifest.json` 记录相对 Recording 目录、目标 GPU、checkpoint ID、seed 和
  DDPM100 配置。
