# SmolVLA LIBERO 域适配与数据准备设计

## 目标与结论边界

本工作建立一个两阶段 SmolVLA 实验：

1. 使用 37 个非目标 LIBERO 任务完成 LIBERO 域适配，得到统一的 B0 checkpoint；
2. 从同一个 B0 checkpoint 出发，在三个目标任务上公平比较普通微调、Canonical LoRA
   和 LoRA + Paraphrase。

准备阶段的 Smoke 只证明训练、checkpoint 加载和闭环评测链路可运行。Smoke 使用过的
episode、init state 和两步 checkpoint 都不参与模型选择，也不作为正式结果或 split
约束。

本设计不回答“无任何 LIBERO 训练的 SmolVLA 是否能够零样本完成 LIBERO”。原始
`lerobot/smolvla_base` 的 state/action contract 与当前 LIBERO 数据不同，因此它只作为
B0 的初始化权重。

## 实验结构

```text
lerobot/smolvla_base
        │
        ▼
B0：37 个非目标任务 LIBERO 域适配
        │
        ├── Z0：目标三任务零样本闭环基线
        ├── C1：目标三任务 Standard Fine-tuning
        ├── C2：目标三任务 Canonical LoRA
        └── C3：目标三任务 LoRA + Paraphrase
```

### B0：LIBERO 域适配

B0 使用当前 combined LeRobot 数据中的 37 个非目标任务。它负责让模型适应两路 LIBERO
图像、8 维 state、7 维 relative end-effector action 和 LIBERO 的语言/视觉/控制分布。

B0 使用 SmolVLA 默认非 PEFT 训练范围：

- `train_expert_only=true`
- `freeze_vision_encoder=true`
- 使用 LIBERO dataset features 和 normalization stats 覆盖 base checkpoint contract

B0 不是“解冻整个视觉语言模型”的全参数微调。全参数解冻会显著改变显存、优化器状态和
模型容量，不属于本轮 LoRA 对照；需要时另立 C1b 实验。

### Z0：目标任务迁移基线

Z0 不训练目标任务，直接用选定的 B0 checkpoint 在三个目标任务的 validation init
states 上运行闭环评测。它衡量 37 任务域适配带来的任务迁移能力。Z0 不查看正式 Test
结果，也不参与 B0 checkpoint selection。

### C1/C2/C3：目标任务平行对照

三者必须从同一个 B0 checkpoint 开始，并使用完全相同的 target Train、Validation 和
Test init states：

| 配置 | PEFT | Instruction | 目的 |
| --- | --- | --- | --- |
| C1 | 关闭 | Canonical | 普通 SmolVLA 微调对照 |
| C2 | LoRA，rank 32 | Canonical | 参数高效适配 |
| C3 | LoRA，rank 32 | Canonical + train paraphrases | 语言增强 |

C3 与 C2 的模型、优化器、训练步数、batch、seed 和 demonstrations 必须相同；唯一实验
变量是训练 instruction 的采样方式。

## 数据范围

### Combined LeRobot 数据

训练继续使用现有的 `datasets/lerobot/lerobot/libero`，不重新转码视频：

| 数据集合 | Tasks | Episodes | Frames |
| --- | ---: | ---: | ---: |
| 全部 combined 数据 | 40 | 1,693 | 273,465 |
| 三个目标任务 | 3 | 143 | 13,991 |
| B0 非目标任务 | 37 | 1,550 | 259,474 |

三个目标任务为：

| LIBERO-Goal task ID | Combined task index | Canonical instruction | Episodes |
| ---: | ---: | --- | ---: |
| 1 | 17 | `put the bowl on the stove` | 48 |
| 4 | 18 | `put the bowl on top of the cabinet` | 46 |
| 8 | 10 | `put the bowl on the plate` | 49 |

B0 必须通过 task index 显式排除 `10,17,18`。不得根据 episode 编号范围推断任务，因为
combined dataset 中不同任务的 episodes 已经交错。

### 原始 LIBERO-Goal 数据

从公开的 `yifengzhu-hf/LIBERO-datasets` 下载完整 `libero_goal/`，共 10 个 HDF5，
约 5.94 GiB。下载解析一次 `main` 后，必须在 `source_manifest.json` 中冻结 resolved
commit；后续运行只允许使用该 commit。

原始 HDF5 不直接参与 B0/C1/C2/C3 训练。它只用于：

- 将目标任务的 combined LeRobot episode 映射回原始 demonstration；
- 读取原始 simulator initial state；
- 将 demonstration 映射到 installed LIBERO 的官方 `.pruned_init`；
- 证明 target Train、Validation 和 Test 没有 init-state 泄漏。

完整下载的另外七个 LIBERO-Goal HDF5 只做版本与 checksum 冻结，本轮不生成正式 split。

## B0 数据划分

B0 的 37 个任务完全排除目标任务，因此不要求为所有背景任务下载原始 HDF5 或完成
init-state provenance。

每个背景任务按照 `seed=1000` 对 episode index 做确定性打乱：

- 90% episodes 进入 B0 Train；
- 10% episodes 进入 B0 Validation；
- 每个任务至少保留 1 条 Validation episode。

比例取整采用“先计算 `floor(0.1 * n)`，再限制为至少 1”的规则。B0 checkpoint 只允许
依据 background Validation offline loss 选择；不得依据三个目标任务的 Z0、Validation
或 Test 成功率反向选择 B0。

输出 `background_split.json`，记录每个 task 的 train/validation episode indices、帧数和
确定性 seed。

## Target provenance

目标任务需要建立以下双重映射：

```text
combined LeRobot episode
        ↓ trajectory fingerprint
raw HDF5 demonstration
        ↓ simulator initial state
official LIBERO init_state_id
```

### Combined episode 到 raw demonstration

映射使用固定的 `raw_action_noop_v1` contract，复现 LIBERO 256×256/no-noop
regeneration 的动作保留规则。对 raw HDF5 的 `/data/demo_N/actions` 按原始顺序执行：

```python
kept = []
for action in raw_actions:
    previous_kept = kept[-1] if kept else None
    near_zero_motion = np.linalg.norm(action[:-1]) < 1e-4
    same_gripper = previous_kept is not None and action[-1] == previous_kept[-1]
    is_noop = near_zero_motion if previous_kept is None else near_zero_motion and same_gripper
    if not is_noop:
        kept.append(action)
```

identity fingerprint 由以下字段组成：

- canonical task instruction；
- retained action 数量；
- 将 retained actions 转为 C-contiguous little-endian float32 后得到的 SHA-256。

当前 LeRobot episode 的 action 使用同一 dtype/layout 计算 fingerprint。fingerprint 命中后
仍逐元素比较完整 action 序列，最大绝对误差不得超过 `1e-6`。不做额外降采样、不尝试
offset、不改变 gripper 符号或范围，也不依靠 episode/demo 顺序。

每个 combined episode 必须只有一个满足 contract 的 raw demonstration；零匹配或多匹配
均记为 provenance failure。raw 中没有进入 combined dataset 的失败 replay 可以保持
unmatched，但必须记录；它不能被错误地配给现有 episode。

8 维 LeRobot state 只做 shape、finite 和 episode length sanity check，不参与 identity
fingerprint。原始 HDF5 的 observation 是 replay/regeneration 之前的记录，不能在不重新
运行 simulator 的情况下据此重建当前 8 维序列。图像同样不参与映射，因为存在重新渲染、
编码和方向处理。

### Raw demonstration 到 official init state

读取 raw HDF5 中每个 demonstration 的首个完整 simulator state，并与该任务的官方
`.pruned_init` 数组逐项比较：

- state shape 必须一致；
- 记录最近 init state、最大绝对误差和 L2 误差；
- 最近项必须唯一；
- 最大绝对误差不得超过 `1e-6`。

如果 raw 文件保存的 state 与 installed LIBERO 版本无法在阈值内匹配，整个 target split
构建失败，不允许退化为只比较 8 维 robot state。

## Target Train/Validation/Test 划分

每个目标任务独立划分，主键为 official `init_state_id`：

| Split | 每任务 unique init states | 三任务合计 |
| --- | ---: | ---: |
| Train | 30 | 90 |
| Validation | 10 | 30 |
| Test | 10 | 30 |

算法如下：

1. 收集能够唯一映射到 official init state 的 combined demonstrations；
2. 按 init state 分组，同一 state 的所有 demonstrations 保持在同一 split；
3. 要求至少存在 40 个具有 demonstration 的 unique states；
4. 使用由 `seed=1000` 和 task ID 派生的独立 RNG，确定性选择 30 个 Train states；
5. 从剩余 mapped states 中确定性选择 10 个 Validation states；
6. 该任务官方 50 个 init states 中未进入 Train/Validation 的 10 个 state 全部作为 Test；
7. Test 对应的 demonstration 即使存在，也不得进入任何训练或 offline validation 输入。

如果 installed LIBERO 的某个任务不是恰好 50 个 init states，或无法得到 30/10/10
互斥划分，构建过程失败。不得用 simulator seed 代替 init state，也不得移动 Test state
来掩盖 provenance failure。

C1/C2/C3 共用同一个 `target_split.json`。训练入口消费 Train episode indices；offline
validation 消费 Validation episode indices；闭环 Validation/Test 消费对应的
`init_state_id`。

## Paraphrase 冻结

每个目标任务固定：

- 1 条 canonical instruction；
- 8 条 train paraphrases；
- 5 条 held-out test paraphrases。

scaffold 阶段只生成结构和 canonical instruction，不自动生成可直接进入实验的语言内容。
`paraphrases.json` 使用以下 versioned contract；每个 task 的 scaffold 初始状态均为
`pending`：

```json
{
  "schema_version": 1,
  "suite": "libero_goal",
  "canonical_probability": 0.25,
  "tasks": {
    "1": {
      "canonical": "put the bowl on the stove",
      "train_paraphrases": [],
      "held_out_paraphrases": [],
      "semantic_review": {
        "status": "pending",
        "reviewed_at": null,
        "checks": {
          "object_preserved": false,
          "target_preserved": false,
          "completion_condition_preserved": false,
          "train_test_separated": false
        }
      }
    }
  }
}
```

人工填写后，审核者必须为每个 task 将四个 checks 设为 `true`，将 status 设为
`approved`，并填写 `YYYY-MM-DD` 格式的 `reviewed_at`。不记录审核者姓名或其他个人标识。
该字段是人工语义审核的机器可读 attestation；程序不声称能够自动证明自然语言语义。

审计器要求：

- 每个列表数量准确；
- canonical/train/test 之间标准化后无字符串重复；
- 每项均为非空字符串；
- 每个 task 的 semantic review status/checks/date 完整有效；
- 文件内容 hash 被写入 target manifest。

C3 每次取样以 `0.25` 概率使用 canonical，以 `0.75` 概率均匀选择 train paraphrase。
随机选择由 train seed、epoch/step 和 sample index 派生，不使用多 worker 共享的全局
`random.random()`。

Canonical、held-out paraphrase 和 Prompt Swap 评测均使用同一组冻结 Test init states。

## 实现边界与入口

新增：

```text
scripts/experiments/smolvla_libero/prepare_data.sh
src/robot_practice/experiments/smolvla_libero/data_preparation.py
configs/experiments/smolvla_libero/paraphrases.json
configs/experiments/smolvla_libero/background_split.json
configs/experiments/smolvla_libero/target_split.json
tests/test_smolvla_libero_data_preparation.py
```

Python 模块提供三个独立、可重入 stage：

### `fetch-sources`

- 解析并冻结 upstream commit；
- 下载完整 `libero_goal/`，支持 Hugging Face Hub 自带的断点续传；
- 计算并验证全部 HDF5 SHA-256；
- 计算三个 installed init-state 文件 SHA-256；
- 写入 `source_manifest.json`；
- 已存在文件只有 checksum 与 manifest 一致时才复用。

### `scaffold-paraphrases`

- 创建三个目标任务的 canonical 和空 paraphrase slots；
- 已存在非空文件时拒绝覆盖，除非显式 `--overwrite`；
- 不调用在线语言模型，不自动宣称语义审核通过。

### `build-splits`

- 验证 source manifest 与所有 checksum；
- 生成 B0 background split；
- 完成目标任务双重 provenance；
- 审计人工填写的 paraphrases；
- 原子写入两个 tracked split config；
- 将详细 provenance、错误和统计写入 ignored output directory。

三个 stage 必须能够单独重跑。`build-splits` 不发起网络下载，防止输入版本在构建中变化。

## 输出

可跟踪、供训练和评测读取的配置：

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

`metrics.json` 记录：

- B0 每个 task/split 的 episodes 和 frames；
- target 每个 task/split 的 unique states、episodes 和 frames；
- raw、combined 和 official init-state 的匹配数量；
- unmatched/ambiguous demonstrations；
- 所有 state/action 匹配最大误差；
- canonical/train/test instruction 数量及重复审计结果。

`manifest.json` 记录输入文件 hash、resolved revisions、CLI 参数、split seed 和输出 config
hash。不得记录缓存路径或其他机器特定信息。

三个 tracked config 均使用 `schema_version=1`：

- `background_split.json`：suite/task identity、split seed、每个 task 的 train/validation
  episode indices 和 frame counts；
- `target_split.json`：source manifest hash、mapping contract version、每个 task 的
  Train/Validation/Test init-state IDs，以及 Train/Validation 对应 episode/demo identity；
- `paraphrases.json`：上述 canonical、train、held-out 和人工审核 attestation。

所有列表按数值 ID 或 canonical name 排序，使相同输入产生 byte-for-byte 相同输出。

## 错误处理

以下情况必须 fail closed，且不得发布部分 `background_split.json` 或
`target_split.json`：

- source manifest revision 或 checksum 不一致；
- raw HDF5 schema 缺少 provenance 所需字段；
- combined task/episode/frame 统计与 metadata 不一致；
- trajectory fingerprint 无匹配或多匹配；
- raw initial state 无法唯一对应 official init state；
- target mapped unique states 少于 40；
- Train/Validation/Test init states 有交集；
- Test state 的 episode 出现在训练或 offline validation；
- paraphrase 数量、重复或人工语义审核 attestation 不合格。

下载中断只保留 Hugging Face Hub 的可恢复临时状态，不写成功 manifest。配置文件使用同
目录临时文件写完并验证后原子替换。

## 测试与验收

### 单元测试

- B0 永远排除 combined task indices `10,17,18`；
- 90/10 划分确定、每任务至少一个 Validation episode；
- `raw_action_noop_v1` 正确处理首个静止动作、重复 gripper 静止动作和 gripper 切换；
- trajectory fingerprint 对唯一、缺失和歧义匹配分别给出正确结果；
- init-state 最近邻在唯一、超阈值和并列情况下正确处理；
- target split 精确得到每任务 30/10/10 unique states；
- 同一 init state 的多条 demonstrations 不会跨 split；
- Test episode 不会被训练入口消费；
- paraphrase 归一化重复、错误数量和不完整人工审核 attestation 被拒绝；
- manifest/config 原子写入和 checksum 验证可重复。

### 本地真实验收

1. `fetch-sources` 下载 10 个 LIBERO-Goal HDF5，并冻结非 `main` 的 resolved commit；
2. 第二次运行不重复下载已校验文件；
3. 三个目标任务所有进入 Train/Validation 的 LeRobot episodes 都有唯一 provenance；
4. 每任务得到 30 Train、10 Validation、10 Test unique init states；
5. B0 split 只包含 37 个非目标任务；
6. 两次使用相同 seed 构建的 tracked configs byte-for-byte 一致；
7. 修改任一输入 checksum 或制造重复 paraphrase 时构建失败；
8. C1/C2/C3 能读取同一 target split，且 C3 只改变 instruction 采样。

验收通过后才能启动 B0 正式训练。B0 完成并冻结 checkpoint 后，才能启动 Z0 和
C1/C2/C3。
