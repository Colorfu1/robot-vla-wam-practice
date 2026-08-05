# 2026-08-04 C1C2C3 共用 checkpoint 选模入口

## 状态

- 状态：`done`
- 范围：实现 C1/C2/C3 共用的一个 checkpoint selection 入口，包含三任务闭环成功率、
  离线开环动作误差、三级判据排名与 selected 实体发布，并去掉三份 guide 里
  `Planned / CLI 未实现` 的标注。三个模型的正式训练、目标协议评测（Canonical /
  Held-out Paraphrase / Prompt Swap）、B0 的 Z0 baseline 都不在本计划内。
- 最后更新：2026-08-04

## 目标

三份 guide 现在都调用
`scripts/experiments/smolvla_libero/select_checkpoint.sh`，但仓库里没有这个文件，
guide 自己标了 `Planned / CLI 未实现 / 当前不可运行`。训练跑完会卡在选模。
本计划补上这一个入口，供 C1/C2/C3 用同一份协议选模。

## 前置条件

- [ ] 确认入口确实缺失：`ls scripts/experiments/smolvla_libero/select_checkpoint.sh`
      应报 No such file
- [ ] 确认三份 guide 的调用只差 `--run-dir` 与两个输出路径，其余 flag 完全相同：

  ```bash
  grep -A 12 "select_checkpoint.sh" reports/guides/smolvla_libero/0{3,4,5}_*.md
  ```

- [ ] `configs/experiments/smolvla_libero/target_split.json` 可用，三个任务各有
      20 个 `validation_init_state_ids`、30 个 `test_init_state_ids`（互不重叠），
      以及各 10 条 `offline_validation_demonstrations`（合计 30 条 / 2,969 帧）
- [ ] `datasets/raw/libero_assets` 是完整离线资产树
- [ ] 至少一个训练完成的 run 目录可用于联调（可先用 C1；没有正式 run 时用 smoke run
      的两个 checkpoint 走通链路，但结果只能标 `Smoke`）

## 执行步骤

### 1. 复用 B0 选模器的骨架

新增 `src/robot_practice/experiments/smolvla_libero/target_checkpoint_selection.py`。
从 `b0_checkpoint_selection.py` 复用这些部件，不要重写：

| 复用对象 | 位置 | 用途 |
| --- | --- | --- |
| `discover_candidate_checkpoints` | 第 179 行 | 校验候选集合恰好覆盖冻结的 6 个 step |
| `build_evaluation_command` | 第 257 行 | 拼 `robot_practice.evaluation.libero_smolvla` 命令 |
| `run_evaluation_jobs` | 第 637 行 | 跨 GPU 有界并发、失败即终止、心跳与进度 |
| `_completed_job_metrics` | 第 288 行 | `--resume` 时复用协议完全匹配的已完成 job |
| `_copy_selected_checkpoint` / `publish_selected_checkpoint` | 第 564 / 585 行 | 实体复制发布，绝不软链 |
| `_write_metrics_markdown` | 第 525 行 | `metrics.md` |

与 B0 的差别只有三处：任务集来自 `target_split.json` 的三个目标任务、多一个开环判据、
排名键换成三级判据。

### 2. 闭环部分

读 `target_split.json` 每个任务的 `closed_loop_split.validation_init_state_ids`（各 20 个），
对 6 个 checkpoint × 3 任务建 job，共 `6 × 3 × 20 = 360` 条 300-step rollout。
`--policy-seed=1500`、`--suite=libero_goal`、`--task-ids=1,4,8` 按 guide 冻结。

C2/C3 的 checkpoint 是 adapter 形态（`pretrained_model/` 下是
`adapter_config.json` + `adapter_model.safetensors` + `config.json`）。评测端已经支持：
`src/robot_practice/evaluation/libero_smolvla.py` 第 191–197 行检测 adapter 成对存在，
第 294 行用 `PeftModel` 挂载。所以选模器只传路径即可，**不要**为 C2/C3 加分支。

主指标是三任务 macro canonical 闭环成功率（先按任务算成功率，再对三个任务取平均）。

### 3. 开环动作误差（本计划唯一的新实现）

这是 B0 选模器没有的部分。数据源是 `target_split.json` 每个任务的
`offline_validation_demonstrations`（各 10 条，合计 30 条 / 2,969 帧），
这 30 条不参与训练。

要求：

- 同一批 frame：帧的选取只依赖 split 与固定 seed，对全部 6 个 checkpoint 完全相同；
- 固定 flow-noise：SmolVLA 是 flow matching，采样噪声必须由固定 seed 决定，
  同一 frame 在不同 checkpoint 上用同一份噪声；
- 同一 action horizon：与训练一致（`chunk_size=50`，执行 `n_action_steps=10`），
  误差在同一段上计算；
- 先按任务求均值，再对三个任务取 macro 平均；
- 归一化空间与去归一化空间的误差都记录，排名只用其中一个并写死是哪一个。

`src/robot_practice/evaluation/offline_act.py` 可作为结构参照（数据集构造、逐帧预测、
parquet 落盘），但它是 ACT 的，不要直接复用其策略调用路径。

### 4. 三级判据排名

`b0_checkpoint_selection.py` 第 515 行的 `checkpoint_rank_key` 是参照，但判据不同。
新的排序键按 guide 冻结为：

1. 三任务 macro canonical 闭环成功率，高优先；
2. 平局时比 macro 开环动作误差，低优先；
3. 仍平局时取更早的 checkpoint step。

不要保留 B0 的 `mean_max_reward` 判据——LIBERO 奖励是稀疏 0/1，它恒等于成功率，
永远打不破平局（这一点在 B0 结论文档里已经记录）。

同时写一份 `ranking_audit`，记录实际生效的是第几级判据，供报告引用。

### 5. 发布 selected

用复用来的 `publish_selected_checkpoint`：`--selected-dir` 必须是自包含实体副本，
不能软链；目标已存在时需要 `--overwrite`；发布前校验 `source_checkpoint` 目录名与
选中 step 一致、权重 sha256 与源一致。

### 6. Shell 入口

新增 `scripts/experiments/smolvla_libero/select_checkpoint.sh`，参数与三份 guide 里
已经写好的调用完全一致，不要改名：

```bash
bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/c1_default_non_peft \
  --dataset-root=datasets/lerobot/lerobot/libero \
  --suite=libero_goal \
  --task-ids=1,4,8 \
  --split-file=configs/experiments/smolvla_libero/target_split.json \
  --checkpoint-steps=5000,10000,15000,20000,25000,30000 \
  --policy-seed=1500 \
  --output-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoint_selection \
  --selected-dir=outputs/smolvla_libero/pilot/c1_default_non_peft/checkpoints/selected
```

另需支持 `--num-gpus=N`、`--libero-assets-dir=DIR`、`--resume`、`--overwrite`、
`--print-command`，语义与 `select_b0_checkpoint.sh` 一致。C2/C3 只改 `--run-dir` 与两个
输出路径。

### 7. 测试

新增 `tests/test_smolvla_libero_target_checkpoint_selection.py`：

- 候选集合不是恰好 6 个冻结 step（缺少、重复、跨 run、绝对路径）时在启动 GPU 前失败；
- 闭环 job 数为 `6 × 3 × 20 = 360`，每个 job 的 init state 集合等于 split 的
  `validation_init_state_ids`；
- 三级判据：构造闭环成功率相同的两个 checkpoint，验证由开环误差分出胜负；再构造两者
  都相同的情况，验证取更早 step；
- 排名键**不含** `mean_max_reward`；
- 使用 `test_init_state_ids` 或 held-out paraphrases 时拒绝运行；
- adapter 形态的 checkpoint 路径能被接受（不需要真跑模型，校验命令构造即可）；
- 开环误差：同一 checkpoint 重复评测两次结果完全相同（固定 frame + 固定 flow-noise）；
- 发布：sha256 不一致时拒绝，目标已存在且无 `--overwrite` 时拒绝；
- `--print-command` 打印的工作量为 6 个 step、3 个任务、360 条 rollout。

```bash
conda run --no-capture-output -n robot-practice python -m pytest \
  tests/test_smolvla_libero_target_checkpoint_selection.py \
  tests/test_smolvla_libero_b0_checkpoint_selection.py -q
conda run --no-capture-output -n robot-practice ruff check \
  src/robot_practice/experiments/smolvla_libero tests/test_smolvla_libero_target_checkpoint_selection.py
conda run --no-capture-output -n robot-practice ruff format --check \
  src/robot_practice/experiments/smolvla_libero tests/test_smolvla_libero_target_checkpoint_selection.py
```

### 8. 干跑与冒烟

```bash
bash scripts/experiments/smolvla_libero/select_checkpoint.sh \
  --run-dir=outputs/smolvla_libero/pilot/c1_default_non_peft \
  --print-command
```

再用一个只有两个 checkpoint 的 smoke run 走通完整链路（闭环 + 开环 + 排名 + 发布），
标注 `Smoke`，不作为结果。

### 9. 更新 guide

三份 guide 的 `## Checkpoint Selection` 段去掉
`Planned / CLI 未实现 / 当前不可运行`，补上：判据顺序、开环误差的固定条件（同一批 frame、
固定 flow-noise seed、同一 action horizon、先任务均值再 macro）、360 条闭环规模、
`ranking_audit` 的位置，以及"不得使用 test split 或 held-out paraphrases 选模"。
C1/C2/C3 三份写同一套，只有路径不同。

## 判断标准

**执行前写死。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| 候选集合校验 | 缺少/重复/跨 run/绝对路径都在启动 GPU 前失败 | 修实现 |
| 闭环工作量 | 恰好 360 条（6 × 3 × 20），init states 等于 split 的 validation 集 | 修实现 |
| test split 隔离 | 传入 test init states 或 held-out paraphrases 被拒 | 停止，这是选模公平性的底线 |
| 排名键 | 恰好三级，且不含 `mean_max_reward` | 修实现 |
| 判据分级 | 成功率相同时由开环误差分出；两者都相同时取更早 step | 修实现 |
| 开环误差可复现 | 同一 checkpoint 两次评测结果逐值相同 | 停止，固定 frame 或固定噪声没生效 |
| adapter checkpoint | C2/C3 形态的路径被接受，且选模器里没有 C2/C3 专用分支 | 修实现 |
| 发布 | 实体副本、非软链；sha256 不符或目标已存在无 `--overwrite` 时拒绝 | 修实现 |
| `--resume` | 只复用协议完全匹配的已完成 job | 修实现 |
| 单测与 ruff | 全绿 | 修完再往下 |
| 步骤 8 冒烟 | 产出 `checkpoint_results.json`、`selected_checkpoint.json`、`ranking_audit`、`metrics.md`，`checkpoints/selected/` 为实体副本 | 排查后再跑正式选模 |

正式选模只有在 6 个 checkpoint、360 条闭环、30 条离线 validation demonstration 全部
完成时才可标记为 `Formal / Closed-loop / Validation`。

## 产物路径

- 代码：`src/robot_practice/experiments/smolvla_libero/target_checkpoint_selection.py`（新增）
- 脚本：`scripts/experiments/smolvla_libero/select_checkpoint.sh`（新增）
- 测试：`tests/test_smolvla_libero_target_checkpoint_selection.py`（新增）
- 文档：`reports/guides/smolvla_libero/03_c1_non_peft.md`、`04_c2_canonical_lora.md`、
  `05_c3_paraphrase_lora.md`
- 选模产物：`outputs/smolvla_libero/pilot/<c1_default_non_peft|c2_canonical_lora|c3_paraphrase_lora>/checkpoint_selection/`
  （`checkpoint_results.json`、`selected_checkpoint.json`、`selection_manifest.json`、
  `ranking_audit` 相关字段、`metrics.md`、`jobs/`、`logs/`）
- 发布副本：同一 run 下 `checkpoints/selected/`

## 不在本计划范围

- C1/C2/C3 的正式训练与结果登记。
- 目标协议评测（Canonical / Held-out Paraphrase / Prompt Swap）。那套 21-job /
  210-rollout 的 evaluator 已经存在，是选模之后的步骤。
- B0 的 Z0 target baseline。
- `--overwrite` 的正式目录保护（跨 B0/C1/C2/C3 的安全改动，另起计划）。
- 把 B0 选模器和本入口合并成一个通用实现。两者任务集与判据不同，先各自跑通再谈合并。

## 风险与停止条件

- **不得用 test split 选模。** 三个目标任务各有 30 个 test init states 与 30 条
  held-out paraphrases，任何一项进入选模都会让后续 test 结论作废。测试里要有专门用例
  拦这件事。
- **开环误差必须可复现。** 帧选取或 flow-noise 任一没固定住，6 个 checkpoint 的误差就
  不可比，第二级判据失效且不会报错。上线前必须验证同一 checkpoint 两次结果逐值相同。
- **不要给 C2/C3 开专用分支。** 评测端已经支持 adapter，加分支只会引入 C1 与 C2/C3
  路径不一致的风险。
- 360 条 300-step rollout 加 30 条离线 demonstration 的评测量不小，务必用 `--num-gpus`
  并支持 `--resume`，中断后不要从头重跑。
- 三个模型的选模必须用同一份协议与同一个 `--policy-seed=1500`。任何一个模型单独放宽
  validation 规模，C1/C2/C3 的对照就不成立。
- 本计划只交付入口。看到某个模型的选模结果后，不得回头修改判据或增减 validation states。

## 收尾

- [ ] 把本文件 `状态` 改为 `done`
- [ ] 更新 `plans/README.md` 索引
- [ ] guide 与结果登记按 `reports/AGENTS.md` 处理，不在本目录续写
