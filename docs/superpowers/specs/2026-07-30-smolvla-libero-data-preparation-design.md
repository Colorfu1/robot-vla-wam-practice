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

三者必须从同一个 B0 checkpoint 开始，并使用完全相同的 target demonstration split、
closed-loop Validation 和 Test init states：

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
- 验证目标 demonstration 的来源和去重关系；
- 记录没有进入 combined dataset 的 replay failure。

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

所有 split 使用固定 RNG contract `pcg64_seedsequence_v1`：

- BitGenerator 固定为 `PCG64`；
- B0 task 使用 `SeedSequence([seed, 0, task_index])`；
- Target demonstration 使用 `SeedSequence([seed, 1, task_id])`；
- Target closed-loop 使用 `SeedSequence([seed, 2, task_id])`；
- 选择操作固定为 `Generator(PCG64(seed_sequence)).permutation(sorted_ids)`。

purpose namespace `0/1/2` 防止三个划分因 ID 数值碰巧相同而复用同一随机流。
`background_split.json` 和 `target_split.json` 都必须记录 RNG contract、seed 和 namespace。

B0 对排序后的 task episode IDs 生成 permutation，取
`permutation[:validation_count]` 为 Validation，其余为 Train；两个列表均在持久化前按
episode index 升序排序。

输出 `background_split.json`，记录每个 task 的 train/validation episode indices、帧数和
确定性 seed。

## Target provenance

目标任务只建立可由现有数据证明的映射：

```text
combined LeRobot episode
        ↓ trajectory fingerprint
raw HDF5 demonstration
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

映射必须是全局单射：每个 combined episode 必须只有一个满足 contract 的 raw
demonstration，而且同一个 raw demo identity 最多只能被一个 combined episode 使用。
episode 零匹配、多匹配或多个 episodes 复用同一 raw demo 均记为 provenance failure。
raw 中没有进入 combined dataset 的失败 replay 可以保持 unmatched，但必须记录；它不能
被错误地配给现有 episode。

8 维 LeRobot state 只做 shape、finite 和 episode length sanity check，不参与 identity
fingerprint。原始 HDF5 的 observation 是 replay/regeneration 之前的记录，不能在不重新
运行 simulator 的情况下据此重建当前 8 维序列。图像同样不参与映射，因为存在重新渲染、
编码和方向处理。

### 为什么不再建立 raw demonstration 到 official init state 的映射

真实数据验收否定了“训练 demonstration 来自官方 `.pruned_init`”这一前提：

- raw demonstration 的首个 simulator state 已经过采集环境的稳定步骤，不能与
  `.pruned_init` 逐项相等；
- 重放相同稳定步骤后仍不能形成 50 对 50 的唯一映射；
- 只比较场景物体平面位姿时，每任务 50 条 raw demonstrations 的最近邻也只覆盖约
  29～30 个官方 state，同编号对应几乎不存在。

这与 LIBERO 将 `.pruned_init` 用作固定 benchmark 初始状态的定位一致。训练
demonstrations 与官方 closed-loop states 是两个独立集合，不能通过放宽阈值、demo 序号
或最近邻强行关联。目标数据因此拆成两个互不冒充 provenance 的 split：

1. demonstration split，以已证明的 LeRobot episode ↔ raw demo 映射为主键；
2. closed-loop split，以 installed LIBERO 的官方 `init_state_id` 为主键。

## Target demonstration 划分

每个任务先通过完整 action trajectory 建立全部唯一映射，再从 mapped demonstrations 中
固定选取 10 条 Offline Validation，其余全部进入 Train：

| LIBERO-Goal task ID | Train demos | Offline Validation demos |
| ---: | ---: | ---: |
| 1 | 38 | 10 |
| 4 | 36 | 10 |
| 8 | 39 | 10 |
| 合计 | 113 | 30 |

算法如下：

1. 按 episode index 排序所有唯一映射；
2. 按 `pcg64_seedsequence_v1` 使用 purpose namespace `1` 派生 demonstration RNG；
3. 取 `permutation[:10]` 为 Offline Validation；
4. 取 `permutation[10:]` 为 Train；
5. Train 与 Offline Validation 的 episode 和 raw demo identity 必须分别无交集；
6. 两组记录均在持久化前按 episode index 升序排序。

本实验没有额外 offline Test demonstrations。正式结论来自独立的 closed-loop Test。
C1/C2/C3 使用同一组 113 条 Train 和 30 条 Offline Validation。训练采样采用
task-balanced sampling，先均匀选择 task，再在该 task 内选择 episode，避免
`38/36/39` 的轻微数量差异改变任务权重。

## Target closed-loop 划分

每个任务的 50 个官方 `.pruned_init` 独立划分：

| Split | 每任务 official init states | 三任务合计 |
| --- | ---: | ---: |
| Validation | 20 | 60 |
| Test | 30 | 90 |

算法如下：

1. 要求 installed LIBERO 的每个目标任务恰好提供 50 个官方 init states；
2. 按 `pcg64_seedsequence_v1` 使用 purpose namespace `2` 派生独立的 closed-loop RNG；
3. 取 `permutation[:20]` 为 Validation；
4. 取 `permutation[20:]` 的 30 个 states 全部冻结为 Test；
5. 两组 state IDs 必须互斥且并集覆盖 50 个官方 states；
6. 两组 state IDs 均在持久化前按数值升序排序。

Z0、C1、C2、C3 的 smoke、调试、checkpoint selection 只能消费 Validation states。
Test 在训练配置和 checkpoint 冻结后才运行，不能参与超参数调整、checkpoint selection
或失败案例驱动的修改。

C1/C2/C3 共用同一个 `target_split.json`。训练入口消费 Train episode indices；offline
validation 消费 Offline Validation episode indices；闭环 Validation/Test 消费对应的
官方 `init_state_id`。

所有 Z0/C1/C2/C3 Offline Validation、closed-loop Validation、checkpoint selection 和
主 Test 指标都固定使用 canonical instruction。C3 的 train paraphrases 只允许出现在
C3 Train 输入，不得进入任何 Validation/Test 路径。

Held-out paraphrase 和 Prompt Swap 是 checkpoint 与全部配置冻结后，在同一组 frozen
Test states 上运行的辅助语言协议。它们不参与 checkpoint selection，也不得用于修改
模型、超参数、paraphrase 集或数据 split；其结果与 canonical 主 Test 指标分开报告。

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
- 文件内容 hash 被写入 `target_split.json` 和 ignored audit `manifest.json`。

这里的 hash 落点明确为 `target_split.json.paraphrases_sha256` 和
`manifest.json.inputs.paraphrases_sha256`，`build-splits` 必须保证两处记录一致。训练与
评测入口加载 `target_split.json` 时，必须重新计算当前 `paraphrases.json` 的 SHA-256
并与 tracked target split 中的记录一致；不一致时 fail closed。

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
- 下载完整 `libero_goal/`，复用已完成且 checksum 正确的文件；
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
- 完成目标 demonstration action provenance；
- 独立生成 demonstration split 与 official closed-loop split；
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
- target demonstration Train/Offline Validation 的 episodes 和 frames；
- official closed-loop Validation/Test 的 init-state 数量；
- raw 与 combined demonstration 的匹配数量；
- unmatched/ambiguous demonstrations；
- 所有 action 匹配最大误差；
- canonical/train/test instruction 数量及重复审计结果。

`manifest.json` 记录输入文件 hash、resolved revisions、CLI 参数、split seed 和输出 config
hash。不得记录缓存路径或其他机器特定信息。

三个 tracked config 均使用 `schema_version=1`：

- `background_split.json`：suite/task identity、split seed、每个 task 的 train/validation
  episode indices 和 frame counts，以及 `pcg64_seedsequence_v1` RNG contract；
- `target_split.json`：source manifest hash、mapping contract version、每个 task 的
  Train/Offline Validation episode/demo identity，以及独立的 closed-loop
  Validation/Test init-state IDs、RNG contract 和 `paraphrases_sha256`；
- `paraphrases.json`：上述 canonical、train、held-out 和人工审核 attestation。

所有列表按数值 ID 或 canonical name 排序，使相同输入产生 byte-for-byte 相同输出。

## 错误处理

以下情况必须 fail closed，且不得发布部分 `background_split.json` 或
`target_split.json`：

- source manifest revision 或 checksum 不一致；
- raw HDF5 schema 缺少 provenance 所需字段；
- combined task/episode/frame 统计与 metadata 不一致；
- trajectory fingerprint 无匹配或多匹配；
- 多个 combined episodes 复用同一个 raw demo identity；
- 每任务 mapped demonstrations 少于 11 条；
- Train 与 Offline Validation 的 episode 或 raw demo identity 有交集；
- installed LIBERO 的官方 init states 不是每任务 50 条；
- closed-loop Validation/Test init states 有交集或没有覆盖全部 50 条；
- paraphrase 数量、重复或人工语义审核 attestation 不合格。

下载中断保留 Hugging Face Hub 临时状态，但未完成分片能否按字节续传取决于客户端和
网络代理；无论如何不得写成功 manifest。配置文件使用同目录临时文件写完并验证后原子
替换。

## 测试与验收

### 单元测试

- B0 永远排除 combined task indices `10,17,18`；
- 90/10 划分确定、每任务至少一个 Validation episode；
- `raw_action_noop_v1` 正确处理首个静止动作、重复 gripper 静止动作和 gripper 切换；
- trajectory fingerprint 对唯一、缺失和歧义匹配分别给出正确结果；
- 多个 combined episodes 复用同一 raw demo 时 fail closed；
- demonstration split 固定保留每任务 10 条 Offline Validation，其余全部 Train；
- `pcg64_seedsequence_v1` 的三个 purpose namespaces 产生确定且隔离的 permutation；
- closed-loop split 精确得到每任务 20 Validation / 30 Test；
- Train/Offline Validation demos 互斥，closed-loop Validation/Test states 互斥；
- paraphrase 归一化重复、错误数量和不完整人工审核 attestation 被拒绝；
- paraphrase hash 不一致、train paraphrase 进入 Validation/Test，或 held-out paraphrase
  进入 Train/checkpoint-selection 路径时被拒绝；
- manifest/config 原子写入和 checksum 验证可重复。

### 本地真实验收

1. `fetch-sources` 下载 10 个 LIBERO-Goal HDF5，并冻结非 `main` 的 resolved commit；
2. 第二次运行不重复下载已校验文件；
3. 三个目标任务所有 Train/Offline Validation LeRobot episodes 都有唯一 action
   provenance；
4. demonstration Train 数量为 `38/36/39`，每任务 Offline Validation 为 10；
5. 每任务得到 20 closed-loop Validation 和 30 frozen Test init states；
6. B0 split 只包含 37 个非目标任务；
7. 两次使用相同 seed 构建的 tracked configs byte-for-byte 一致；
8. 修改任一输入 checksum 或制造重复 paraphrase 时构建失败；
9. C1/C2/C3 能读取同一 target split，且 C3 只改变 instruction 采样。

验收通过后才能启动 B0 正式训练。B0 完成并冻结 checkpoint 后，才能启动 Z0 和
C1/C2/C3。
