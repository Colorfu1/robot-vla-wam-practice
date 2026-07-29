# PushT E5 Prototype-ID 条件控制实施计划

**Date:** 2026-07-30

## Goal

实现已批准的 E5 第一阶段：从现有 PushT demonstrations 提取 K-medoids prototype，
通过 sidecar episode assignment 给本地 Mode-BCChunk/Mode-Diffusion 注入
prototype-ID/NULL embedding，并提供可复现训练、固定 observation 和闭环评测入口。

## Task 1: Prototype 与 sidecar

Files:

- `src/robot_practice/experiments/pusht_diffusion/e5_prototypes.py`
- `configs/experiments/pusht_diffusion/e5_mode_conditioned.json`
- `scripts/experiments/pusht/fit_e5_prototypes.sh`
- `tests/test_pusht_e5_prototypes.py`

Work:

1. 从 E0 `trajectories.parquet` 构造事件对齐 object-centric sketch。
2. 实现预注册 component distance、缩放、K-medoids、多 seed 和 subsample stability。
3. 规范化 medoid ID，输出 assignment/manifest/leakage audit。
4. 测试缺失事件、tie-break、canonical ID 和 ineligible-to-NULL 语义。

## Task 2: Mode policies

Files:

- `src/robot_practice/policies/mode_diffusion/`
- `src/robot_practice/policies/mode_bc_chunk/`
- `src/robot_practice/experiments/pusht_diffusion/model_configs/mode_*.py`
- `tests/test_pusht_e5_policies.py`

Work:

1. 使用本地 Diffusion model 子类扩展 `global_cond`。
2. 增加 `nn.Embedding(K+1, 32)` 和 20% train-only NULL dropout。
3. 保持原有 observation encoder、U-Net、queue、action horizon 和 loss 语义。
4. 验证 ID 持久性、dropout accounting、保存加载和成对初始化。

## Task 3: Dataset 与训练

Files:

- `src/robot_practice/experiments/pusht_diffusion/e5_dataset.py`
- `src/robot_practice/experiments/pusht_diffusion/e5_initialization.py`
- `src/robot_practice/experiments/pusht_diffusion/train_mode_*.py`
- `scripts/experiments/pusht/train_mode_*.sh`
- `scripts/experiments/pusht/train_mode_*_multi_gpu.sh`

Work:

1. 用 sidecar wrapper 按 `episode_index` 注入整数 ID，不修改原始 dataset。
2. 在本地 LeRobot runtime adapter 中安装 wrapper，保留官方 train loop。
3. 生成 Mode-BCChunk/Mode-Diffusion 共同 step-zero checkpoint。
4. 提供单卡和显式 GPU 数的多卡 launcher 与 `--print-command`。

## Task 4: 评测与可视化

Files:

- `src/robot_practice/experiments/pusht_diffusion/e5_fixed_observation.py`
- `src/robot_practice/experiments/pusht_diffusion/e5_rollout_eval.py`
- `scripts/experiments/pusht/eval_e5_*.sh`
- `scripts/experiments/pusht/view_e5_rollouts.sh`

Work:

1. 固定 E0 audit candidates、initial noise 和 reverse stream，只切换 ID。
2. 实现 20 environment × K × 10 policy seed controlled rollouts。
3. 实现 20 × 10 NULL rollouts、medoid rejection、adherence 和 controlled success。
4. 沿用现有完整 rollout Rerun 布局并增加 prototype/NULL 标识。

## Task 5: Verification 与 guide

Files:

- `tests/test_pusht_e5_*.py`
- `reports/guides/pusht_diffusion/e5_mode_conditioned.md`
- `reports/guides/pusht_diffusion/README.md`
- `reports/experiment_log.md`

Work:

1. 运行 focused pytest、Ruff、shell syntax 和 launcher print-command。
2. 运行最小 prototype fixture、policy checkpoint round-trip 和 1-frame closed-loop smoke。
3. 写入完整命令、输出结构、Formal/Smoke 标签和结论边界。
4. 不把 smoke 或未训练行为写成正式实验结论。
