# SmolVLA LIBERO：数据、Split 与 Paraphrase 准备

## 实验目标

为 C1/C2/C3 生成完全一致、可追溯且无初始状态泄漏的 train/validation/test split，并
冻结 C3 使用的 train paraphrases 和评测使用的 held-out paraphrases。

结果类别：`Offline`。

## 实验思路

从 LIBERO-Goal 选择 bowl/stove、bowl/cabinet、bowl/plate 三个同构任务。先把 combined
dataset episode 映射回原始 demonstration 与官方 init state，再按 unique init state
进行 30/10/10 分组。同一物理状态的所有 episodes 必须进入同一 split。Paraphrase
训练集和 test 集独立，并在正式训练前冻结。

## 任务与现有数据

| LIBERO-Goal task ID | Canonical instruction | Combined task index | Episodes | Frames | 时长（10 Hz） |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | `put the bowl on the stove` | 17 | 48 | 4,854 | 485.4 秒 |
| 4 | `put the bowl on top of the cabinet` | 18 | 46 | 4,574 | 457.4 秒 |
| 8 | `put the bowl on the plate` | 10 | 49 | 4,563 | 456.3 秒 |
| 合计 | — | — | 143 | 13,991 | 1,399.1 秒 |

三个 BDDL 任务共享 fixtures、objects 和初始化区域定义，适合比较语言指定的不同放置目标。
但三个任务各自保存的 50 个 init-state 数组并不逐项相同，相同 `init_state_id` 不表示
完全相同的物理状态。

本地 combined dataset 有 1,693 episodes、273,465 frames、40 tasks、10 Hz，但没有完整
的 demonstration/init-state provenance；原始 LIBERO HDF5 当前也未缓存。仅凭 combined
dataset 的全局 episode index 不能证明没有状态泄漏。

表中的 143 episodes 是三个任务的 demonstration 总数，不等于 143 个 unique init
states。Train/Validation 的实际可用规模必须等 provenance 映射完成后再冻结。

### Provenance 阻塞输入

生成 split 前必须具备：

- 三个任务的原始 HDF5 demonstrations
- 原始数据仓库的 resolved commit SHA
- 每个 HDF5 的 SHA-256
- installed LIBERO 官方 init-state 文件及其 SHA-256
- combined dataset episode 到原始 demonstration 的映射依据

原始 HDF5 中的初始 simulator state 用于匹配官方 79 维 init-state 向量；combined
dataset 的两路图像和 8 维 robot state 不能单独完成该匹配。

## 固定 Split

| Split | 每任务 unique init states | 三任务总量 | Episode 规则 |
| --- | ---: | ---: | --- |
| Train | 30 | 90 | 同一状态的全部 mapped demonstrations |
| Validation | 10 | 30 | 同一状态的全部 mapped demonstrations |
| Test | 10 | 30 | 只需要官方 init state，不要求有 demonstration |

这里把原计划的 30 个 test seeds 解释为三任务合计 30 个，即每任务 10 个。

Split 主键必须是 `init_state_id` 或经过容差验证的状态向量，而不是 simulator seed：

- simulator seed 控制 episode 内的随机过程
- init state 决定物理场景是否重复
- 映射到同一物理状态的所有 episodes 必须进入同一个 split
- provenance 失败只减少可用 train/validation episodes，不能移动 test state 补位
- 若非 test mapped states 少于 40，训练前必须重新冻结缩减后的 train/validation 数量
- 报告中分别记录 unique state 数和 episode 数

## Paraphrase 契约

每个任务准备：

- 1 条 canonical instruction
- 5～10 条 train paraphrases
- 5 条完全不参与训练的 test paraphrases

要求：

- 对象、目标位置和任务完成条件必须保持不变
- train/test paraphrase 不得字符串重复
- 人工检查目标语义，不依赖自动生成结果直接入库
- test paraphrase 与 test state、policy seed、统计协议一起冻结

C3 训练时使用：

```python
canonical_probability = 0.25
paraphrase_probability = 0.75
```

正式实现必须由可复现的 dataset wrapper 或 processor 完成。随机数应由 train seed、
epoch/step 和 sample index 派生；不得在多 worker `__getitem__` 中直接使用未受控的
全局 `random.random()`。

## 数据准备命令

以下入口尚未实现，属于 `Planned / CLI 未实现 / 当前不可运行`。

第一步获取并冻结原始数据 provenance。允许第一次从 `main` 解析，但
`source_manifest.json` 必须记录 resolved commit SHA，后续 build 只认该 manifest：

```bash
bash scripts/experiments/smolvla_libero/prepare_data.sh \
  --stage=fetch-sources \
  --raw-repo-id=yifengzhu-hf/LIBERO-datasets \
  --raw-revision=main \
  --raw-data-root=datasets/raw/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --source-manifest=outputs/smolvla_libero/data_audit/source_manifest.json
```

第二步生成 paraphrase 人工填写模板：

```bash
bash scripts/experiments/smolvla_libero/prepare_data.sh \
  --stage=scaffold-paraphrases \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --output-paraphrases=configs/experiments/smolvla_libero/paraphrases.json
```

人工填写并审查 `paraphrases.json` 后，它成为下一步的只读输入；build 不负责自动生成或
改写语言内容。

第三步映射 provenance、生成 split 并审计 paraphrases：

```bash
bash scripts/experiments/smolvla_libero/prepare_data.sh \
  --stage=build-split \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --raw-data-root=datasets/raw/libero \
  --source-manifest=outputs/smolvla_libero/data_audit/source_manifest.json \
  --init-states-source=installed_libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --train-per-task=30 \
  --validation-per-task=10 \
  --test-per-task=10 \
  --seed=1000 \
  --exclude-development-episodes=379,392,398 \
  --exclude-development-init-states=1:0 \
  --paraphrases=configs/experiments/smolvla_libero/paraphrases.json \
  --output-split=configs/experiments/smolvla_libero/split.json \
  --output-dir=outputs/smolvla_libero/data_audit
```

排除项来自准备阶段的训练和闭环 Smoke。Builder 必须先通过 provenance 将 episode
映射为 init state，再对所有 development-used states 去重并禁止进入 test；未来新增
Smoke 状态时必须扩展这两个参数。`manifest.json` 需记录最终排除列表及原因。

## 结果落点

```text
configs/experiments/smolvla_libero/
├── split.json
└── paraphrases.json

outputs/smolvla_libero/data_audit/
├── source_manifest.json
├── paraphrase_audit.json
├── manifest.json
└── metrics.json
```

`paraphrases.json` 由 scaffold 生成骨架、人工填写并冻结，是 build-split 的输入。

`source_manifest.json` 至少记录 upstream repo、requested/resolved revision、raw HDF5
相对路径、SHA-256、LIBERO 版本和官方 init-state 文件 SHA-256。

`split.json` 至少记录：

- suite/task ID
- combined dataset episode
- 原始 demonstration 名称
- init-state ID
- state 匹配误差
- split

`metrics.json` 至少记录：

- 每个 task/split 的 unique states、episodes 和 frames
- 无法映射或存在歧义的 demonstrations
- canonical/train/test instruction 数量
- train/test paraphrase 重复检查结果

## 判断标准

通过条件：

- Train、Validation、Test 的 init states 两两互斥
- Smoke 使用过的 episode/init state 均未进入 Test
- 同一状态的全部 episodes 位于同一 split
- Source manifest 使用 resolved commit，所有 HDF5/init-state checksum 验证通过
- C1/C2/C3 使用完全相同的 train demonstrations
- 三个任务都有足够的 train/validation mapped states
- train/test paraphrase 无重复且语义人工检查通过
- 所有 provenance 异常都记录在 manifest 中

如果 provenance 无法证明 split 无泄漏，不得开始 C1/C2/C3。

下一步：[C2 Canonical LoRA](02_c2_canonical_lora.md)。
