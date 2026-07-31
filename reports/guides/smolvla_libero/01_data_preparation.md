# SmolVLA LIBERO：B0 与目标任务数据准备

## 实验目标

为两阶段 SmolVLA 实验冻结两套数据：

1. B0 使用 37 个非目标 LIBERO 任务完成 domain adaptation；
2. C1/C2/C3 共用三个目标任务的 demonstration split 和独立 closed-loop split。

结果类别：`Offline`。

## 实验思路

B0 从当前 combined LeRobot 数据中显式排除 bowl→stove、bowl→cabinet 和 bowl→plate，
再按每个背景任务 90%/10% 划分 Train/Validation。目标三任务下载完整原始
LIBERO-Goal，通过完整 action trajectory 将 LeRobot episode 映射回 raw demonstration，
再将每任务 10 条 mapped demonstrations 固定为 Offline Validation，其余全部 Train。
官方 `.pruned_init` 是独立 benchmark 状态，每任务划分为 20 个 closed-loop Validation
和 30 个 frozen Test，不与训练 demonstration 强行配对。

## 实验结构与数据

```text
lerobot/smolvla_base
        │
        ▼
B0：37 个非目标 LIBERO 任务
        │
        ├── Z0：目标三任务零样本迁移
        ├── C1：Standard Fine-tuning
        ├── C2：Canonical LoRA
        └── C3：LoRA + Paraphrase
```

### B0 数据

| 数据 | Tasks | Episodes | Frames |
| --- | ---: | ---: | ---: |
| Combined 全量 | 40 | 1,693 | 273,465 |
| 目标三任务 | 3 | 143 | 13,991 |
| B0 背景任务 | 37 | 1,550 | 259,474 |

真实元数据按 `seed=1000` 划分后的 B0 数量：

| Split | Episodes | Frames |
| --- | ---: | ---: |
| Train | 1,412 | 236,470 |
| Validation | 138 | 23,004 |

B0 排除的 combined task indices 为 `10,17,18`。B0 checkpoint 只能依据 background
Validation offline loss 选择，不能用目标三任务结果反向选择。

### Target 数据

| LIBERO-Goal task ID | Combined task index | Canonical instruction | Episodes |
| ---: | ---: | --- | ---: |
| 1 | 17 | `put the bowl on the stove` | 48 |
| 4 | 18 | `put the bowl on top of the cabinet` | 46 |
| 8 | 10 | `put the bowl on the plate` | 49 |
| 合计 | — | — | 143 |

Target demonstration split：

| Task ID | Train episodes | Train frames | Offline Validation episodes | Validation frames |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 38 | 3,809 | 10 | 1,045 |
| 4 | 36 | 3,594 | 10 | 980 |
| 8 | 39 | 3,619 | 10 | 944 |
| 合计 | 113 | 11,022 | 30 | 2,969 |

C1/C2/C3 共用这 113 条 Train demonstrations，并采用
`uniform_task_then_uniform_episode` 采样，避免 `38/36/39` 的数量差异改变任务权重。

Official closed-loop split：

| Split | 每任务 init states | 三任务合计 |
| --- | ---: | ---: |
| Validation | 20 | 60 |
| Test | 30 | 90 |

Z0、C1、C2、C3 的 smoke、调试和 checkpoint selection 只能使用 closed-loop
Validation。Test 只在 checkpoint 与配置冻结后运行。

## Provenance 契约

目标任务建立：

```text
LeRobot episode
  → raw HDF5 demonstration
```

映射使用 `raw_action_noop_v1`：按照 OpenVLA/LIBERO regeneration 的固定 no-op 规则过滤
raw actions，再比较 task、完整 float32 action 序列长度和 SHA-256。不得按 episode 顺序
猜测，不做额外降采样或 gripper 变换。映射必须全局单射：一个 raw demo 不能被多个
LeRobot episodes 复用。

真实验收已经证明 raw training demonstrations 与官方 `.pruned_init` benchmark states
不是同一集合。两者使用独立 split，不通过 state 最近邻或 demo 序号强行建立关联。

原始 HDF5 只承担 provenance，不替代当前 LeRobot 训练数据，也不重新转码视频。

## 实际构建审计结果

本次正式构建已完成以下检查：

- 143/143 个目标 LeRobot episodes 均唯一匹配到 raw demonstration；
- 全部匹配的 `action_max_abs_error` 均为 `0.0`；
- task 1/4/8 分别还有 2/4/1 条 raw demonstrations 未出现在 combined LeRobot 数据中，
  它们不进入当前 Train 或 Offline Validation；
- 同一输入、审核文件与 `seed=1000` 重复执行后，三个 tracked configs
  byte-for-byte 一致。

这些结果只证明 action provenance、划分互斥性和生成确定性，不代表模型训练或闭环评测
已经完成。完整逐 episode 映射保存在未提交的
`outputs/smolvla_libero/data_audit/target_provenance.json`。

## 第一步：下载并冻结完整 LIBERO-Goal

该命令会下载完整10个 HDF5，约5.94 GiB。第一次允许从 `main` 解析，但成功 manifest
记录 immutable resolved commit；后续 build 只认 manifest。

```bash
bash scripts/experiments/smolvla_libero/prepare_data.sh \
  --stage=fetch-sources \
  --raw-repo-id=yifengzhu-hf/LIBERO-datasets \
  --raw-revision=main \
  --raw-data-root=datasets/raw/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --source-manifest=outputs/smolvla_libero/data_audit/source_manifest.json \
  --output-dir=outputs/smolvla_libero/data_audit
```

该 stage 会复用已完整下载且 checksum 正确的文件。未完成分片能否按字节续传取决于
Hugging Face Hub 客户端和当前网络链路。下载完成后计算全部 HDF5 和三个官方 init-state
文件的 SHA-256，只有全部通过才写成功 manifest。

## 第二步：审核 Paraphrase candidates

当前候选文件为：

```text
configs/experiments/smolvla_libero/paraphrases.json
```

每个任务已经预填 8 条 train 和 5 条 held-out candidates，并于 `2026-07-30` 完成
人工审核，`semantic_review.status=approved`。任何文本修改都会
改变绑定的 SHA-256，必须重新审核并重新生成 `target_split.json`，不能沿用旧 attestation。

只有候选文件缺失、需要重建空模板时才执行：

```bash
bash scripts/experiments/smolvla_libero/prepare_data.sh \
  --stage=scaffold-paraphrases \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --output-paraphrases=configs/experiments/smolvla_libero/paraphrases.json
```

每个任务需要人工确认：

- 8 条 train paraphrases；
- 5 条 held-out paraphrases；
- `semantic_review.status=approved`；
- 四项语义检查均为 `true`；
- `reviewed_at` 使用 `YYYY-MM-DD`。

审计会对字符串执行 Unicode NFKC、大小写、连续空白和末尾标点归一化，拒绝
canonical/train/held-out 重复。审核字段只表示人工 attestation，程序不会伪称自动证明了
语言语义。

## 第三步：构建 B0 与 Target splits

Paraphrase 人工审核完成后执行：

```bash
bash scripts/experiments/smolvla_libero/prepare_data.sh \
  --stage=build-splits \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --raw-data-root=datasets/raw/libero \
  --source-manifest=outputs/smolvla_libero/data_audit/source_manifest.json \
  --init-states-source=installed_libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --offline-validation-per-task=10 \
  --closed-loop-validation-per-task=20 \
  --closed-loop-test-per-task=30 \
  --seed=1000 \
  --paraphrases=configs/experiments/smolvla_libero/paraphrases.json \
  --output-background-split=configs/experiments/smolvla_libero/background_split.json \
  --output-target-split=configs/experiments/smolvla_libero/target_split.json \
  --output-dir=outputs/smolvla_libero/data_audit
```

`build-splits` 不访问网络。输入 checksum、action provenance、官方 init-state 数量或
语言审核任一不通过时，命令直接失败，不发布成功 manifest。全部 split 固定使用
`pcg64_seedsequence_v1`，并将 seed、purpose namespace 和排序规则写入配置。

## 结果落点

供后续训练和评测读取、可以跟踪的配置：

```text
configs/experiments/smolvla_libero/
├── background_split.json
├── target_split.json
└── paraphrases.json
```

生成但不提交的审计产物：

```text
outputs/smolvla_libero/data_audit/
├── source_manifest.json
├── target_provenance.json
├── paraphrase_audit.json
├── manifest.json
└── metrics.json
```

`background_split.json` 保存37任务的 Train/Validation episode indices。
`target_split.json` 保存每个目标任务的 Train/Offline Validation episode/raw demo，
以及独立的 20/30 closed-loop Validation/Test init-state IDs。

## 判断标准

通过条件：

- 完整10个 LIBERO-Goal HDF5 使用 immutable revision 且 checksum 全部匹配；
- B0 只包含37个背景任务，目标 task indices `10,17,18` 数量为零；
- B0 Train/Validation 合计1,550 episodes、259,474 frames；
- 每个目标 LeRobot episode 唯一对应一个 raw demonstration；
- Target Train 为 `38/36/39`，每任务 Offline Validation 为 10；
- 每个目标任务得到20个 closed-loop Validation 和30个 frozen Test states；
- demonstration Train/Offline Validation 互斥，closed-loop Validation/Test 互斥；
- C1/C2/C3 读取同一个 `target_split.json`；
- Paraphrase 数量、去重和人工审核 attestation 全部通过；
- 相同输入和 seed 两次生成的 tracked configs byte-for-byte 一致。

如果 action provenance 不是全局单射、每任务 mapped demos 不足以保留10条 Offline
Validation，或官方 init states 不是50条，不得启动 B0/C1/C2/C3 正式训练。
