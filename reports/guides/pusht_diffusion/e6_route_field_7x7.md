# E6 7×7 Route Field 条件控制

## 先说结论

RF7 的正式 held-out audit 已完成，结论是**负面的**：模型读取了 route field，但没有把它
当作可执行的顺序指令。

- 45/45 条闭环 rollout 完成（audit 21 条 episode × oracle 21 / reversed 19 / permuted 5），
  21/21 首帧重建通过，`empty_achieved_route_count` 全为 0。
- 离线闸门通过：固定 observation + 固定 initial noise + 固定 reverse noise stream，只把
  路线换成逆序，action chunk 移动 113.6 px（逐 episode 最小 56.6 px），末端动作移动
  54.1 px。因此不能把负结论归因于"路线没接上"。
- 两条冻结主判据均判负：`obeyed_rate` 为 0/19（逆序）与 0/5（置换）；
  `commanded_first_segment_positive_rate` 为 4/19（0.2105）。
- 执行轨迹与 GT 路线的吻合度**高于**与被请求路线的吻合度：
  `commanded_minus_oracle_adherence` 均值 −0.1368（逆序）与 −0.1300（置换）。
- oracle 条件任务成功率仅 1/21（4.8%），与选模阶段 8 个 checkpoint 的 0.05–0.20 一致
  （Fisher p = 0.18），**不是评测回退**。
- 负结论的**强度**弱于 0/19 的字面读数：19 个逆序配对中 13 个 adherence 差值恰为 0，
  结构上不可判定；可判定的只有 6 个，Wilson 95% 上界 0.390。

判读纪律不变：探针的 raw success rate 是反号的服从度读数，不是性能指标；逆序 2 次成功
高于 oracle 的 1 次纯属噪声。`permuted` 首段进度 4/5 为正也不算部分服从——中段反转保留
首格，路线盲策略天然为正。

正式结论、完整矩阵与根因分析见：

```text
reports/2026-08-03-RF7 路线条件遵循正式评测结论.md
```

## 统一名称

后续统一把这个模型输入称为 **RF7 condition**（RouteField-7×7 路线条件），简称
**RF7**。代码中的 batch key 是 `route_field`。`route_manifest.json`、
`episode_assignments.parquet` 和 `route_fields.npz` 是 RF7 sidecar；真正传给模型的是
每个训练样本携带的 `[4,7,7]` 浮点 tensor。

## 实验定位

RF7 使用固定 BEV 坐标系中的空间路线场替代 E5 的离散 prototype ID。它要验证的是：
固定 observation 和 Diffusion noise 时，把一条带有明确局部方向的粗粒度路线换成
**占据支撑完全相同、只有遍历顺序不同**的另一条，能否稳定改变动作和完整 rollout 的
T-block 中心路径。这样的错配路线把"顺序与方向通道是否被读取"从"覆盖的格子不同"里
隔离出来，是 E6 与 E5 在实验设计上的主要差别。

7×7 相比 5×5 能保留更多轨迹结构。使用 3 帧稳定判定时，165 条 train episode 的
原始路线统计为：

- 5×5：路线长度中位数 3，7 条 episode 至多确认一个稳定格；
- 7×7：路线长度中位数 4，3 条 episode 至多确认一个稳定格；
- 5×5 有 52 条原始路线含重复格，7×7 有 44 条；
- 7×7 的局部有向转移中，训练支持数不低于 3 的转移覆盖约 89.8%。

首版固定 7×7，不加 NULL、condition dropout、classifier-free guidance、waypoint cursor
或外部阶段机。

## 数据和 condition 契约

每帧从已有 `trajectories.parquet` 读取 `block_position`，按 512×512 PushT 世界坐标映射到
7×7 row-major 网格。进入新格后连续保持 3 帧才确认切换，并压缩连续重复。

RF7 不再按路线长度、重复格或精确路线支持度过滤训练 episode：

- 全部 165 条 train episode 参与训练；
- LeRobot 原有的全部合法随机 index/action chunk 都可采样；
- 只有一个稳定格的 episode 也合法，编码为单格 occupancy/progress，方向为 0；
- 精确路线支持度只用于数据统计和静态图示，不决定训练资格，也不参与闭环条件的构造。

固定空间场不能直接表达同一格被多次访问的先后顺序，因此 sidecar 同时保存：

- `raw_route_cells`：去抖后的原始稳定路线；
- `route_cells`：模型实际使用的无重复 macro route；
- `backtrack_removed_cells`：迭代压缩 `A→B→A` 删除的格数；
- `loop_removed_cells`：对残余重复格做按时间顺序 loop erasure 删除的格数。

这项简化只改变 condition 表达，不删除 episode 或训练 chunk。后续分析可按审计字段单独
检查受路线简化影响的样本。

每个 macro route 编码为 `[4,7,7]`：

1. `occupancy`：路线格为 1；
2. `progress`：按路线顺序从 0 线性增加到 1，单格路线取 1；
3. `direction_x`：当前路线格指向下一格的归一化 x 分量；
4. `direction_y`：当前路线格指向下一格的归一化 y 分量。

路线外和终点方向都为 0。同一 episode 中所有 action chunk 使用同一份 route field；
模型从当前 observation 自行判断路线进度。

### 条件错配探针（occupancy-matched probes）

早期版本用两条写死的绝对路线 `16 → 17 → 24` / `16 → 23 → 24` 做 A/B 诊断。sidecar
统计显示这个设计不成立：audit 21 条 episode 中只有 2 条的 T 初始格落在 `16`，其余
19 条从一开始就不在路线起点，adherence 由几何决定而不是由可控性决定；而且两条路线
共享首尾格、只差第 17 与 23 一个格，可分辨信号最大只有 1/3。这两条路线现在只保留在
sidecar 和静态图示里，不再进入闭环评测协议。

正式协议改为**逐 episode 生成占据支撑完全相同的错配路线**：

| 条件 | 角色 | 定义 | 前置条件 | audit 可用 |
| --- | --- | --- | --- | --- |
| `oracle` | task_reference | episode 自己的 GT macro route | 无 | 21/21 |
| `reversed` | obedience_probe | 同一批格子逆序遍历 | 路线 ≥ 2 格 | 19/21 |
| `permuted` | obedience_probe | 只逆转中段，保留首尾格 | 路线 ≥ 4 格 | 5/21 |

逆序路线的 `occupancy` 通道与前向逐格完全一致，`progress` 镜像为 `1-p`，方向通道按
相反遍历顺序重算（方向为 0 的终点格从末格移到首格）。因此前向与逆序之间的任何行为
差异都不能归因于"覆盖的格子不同"或"其中一条更 OOD"，这把 E6 真正要问的问题隔离了
出来：模型到底读没读 progress/direction 通道，还是只用了 occupancy。

单格路线逆序等于自身，会贡献保证为 0 的差异，因此被排除在探针之外。中段置换额外保留
首尾格，所以起点仍紧邻 T 的真实初始位置、终点仍在目标区，是唯一不与 observation
矛盾的匹配对，代价是要求 ≥ 4 格、样本量小。

同一 episode 的三个条件共用 environment seed 与 policy seed，因此 Diffusion 的 initial
noise 和 reverse noise stream 逐条一致，commanded route 是唯一变量。

### 判读纪律

逆序条件在物理上要求模型**离开**目标区，因此它的 success 一定会掉，而这不构成任何
证据——"服从方向"和"OOD 输入让模型崩了"都预测同一个结果。协议因此规定：

- `condition_role = obedience_probe` 的 success rate 是反号的服从度读数，**不是**性能
  指标，`metrics.md` 的注释里也这样标注；
- 决定性读数是 `commanded_minus_oracle_adherence`（同一条执行轨迹分别对 commanded
  route 和 GT route 算 adherence 的差值）与 `commanded_first_segment_progress_cells`
  （T 中心净位移在 commanded 首段方向上的带符号投影，单位为格）；
- 三种结局据此区分：服从 = 差值 > 0 且投影 > 0；忽略条件 = 差值 ≈ 0 且与前向结果
  几乎相同；OOD 崩塌 = 两个 adherence 都低、编辑距离都大。

`paired_condition_metrics.parquet` 按 `(model, oracle vs probe)` 汇总 `obeyed_rate`
（差值 > 0 的配对占比，带 Wilson 区间）和 `commanded_first_segment_positive_rate`。

现有 demonstration 没有同初始状态的配对监督，所以"condition 被模型读取"仍不等于
"路线可任意切换"。逆序探针能证伪前者；证实后者还需要同起点、同终点、都能完成任务的
前向路线对，中段置换是这个方向上现有数据能支持的唯一形式。

## 生成 sidecar 与图示

先确认 E0 已生成完整轨迹表，然后执行：

```bash
bash scripts/experiments/pusht/build_route_fields.sh
```

默认读取：

```text
outputs/pusht_diffusion/data_inspection/trajectories.parquet
```

默认输出：

```text
outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/
├── route_manifest.json
├── episode_assignments.parquet
├── route_fields.npz
├── metrics.json
└── metrics.md
```

重新生成时显式加 `--overwrite`。训练入口会校验 split hash、artifact hash 和
网格/通道契约，并在创建 run 前拒绝损坏或不匹配的输入。

生成实际 A/B condition 图：

```bash
conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.pusht_diffusion.route_field_visualization
```

图像默认保存到
`outputs/pusht_diffusion/pilot/route_field_7x7/visualizations/rf7_route_a_b.png`。

## Smoke 训练

```bash
bash scripts/experiments/pusht/train_route_field_bc_chunk.sh \
  --mode=smoke \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=1000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda
```

```bash
bash scripts/experiments/pusht/train_route_field_diffusion.sh \
  --mode=smoke \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=1000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda
```

## 正式多卡训练

推荐 8 卡、每卡 batch 8，使 global batch size 为 64：

```bash
bash scripts/experiments/pusht/train_route_field_bc_chunk_multi_gpu.sh \
  --num-gpus=8 \
  --mode=pilot \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=200000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda \
  --output-dir=outputs/pusht_diffusion/pilot/route_field_7x7/training/route_field_bc_chunk
```

```bash
bash scripts/experiments/pusht/train_route_field_diffusion_multi_gpu.sh \
  --num-gpus=8 \
  --mode=pilot \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --steps=200000 \
  --batch-size=8 \
  --num-workers=4 \
  --device=cuda \
  --output-dir=outputs/pusht_diffusion/pilot/route_field_7x7/training/route_field_diffusion
```

`--batch-size` 是每卡 batch size。运行前可追加 `--print-command` 检查最终 Accelerate
命令、resolved episodes 和模型配置，此时不会创建输出目录。

## 选模与验证入口

正式训练结束后按顺序执行下面四步。每一步都可以先加 `--print-command` 只打印最终命令，
不创建输出目录；重跑同一个默认路径需要 `--overwrite`。

### 0. 零场消融与条件敏感性诊断（全 checkpoint 离线前置闸门）

在花掉任何 rollout 预算之前，用 `ablate_route_field_condition.sh` 对 **run 目录里全部
可发现 checkpoint**（不只选中的那一个）做离线前向诊断：固定每条 episode 的 dataset
中间帧 observation，固定同一份 DDPM initial noise 与 reverse noise stream，只替换
commanded route。这是 development checkpoint selection 之前的**前置闸门**；步骤 2 的
`--gate-only` 仍然保留，用于选模之后对选中 checkpoint 按正式协议再复核一次。

```bash
bash scripts/experiments/pusht/ablate_route_field_condition.sh \
  --diffusion-run=outputs/pusht_diffusion/pilot/route_field_7x7/training/route_field_diffusion \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --split=development \
  --device=cuda:0
```

`--bc-run` / `--diffusion-run` 至少提供一个（BC 未训练时只给 Diffusion）；
`--split=audit` 可换到 audit；`--num-inference-steps` 默认 100，与闭环协议一致；
`--num-gpus=N` 默认 1，>1 时按 checkpoint 分发给各 GPU 并行（此时不可设置
`--device`；单卡用 `--device=cuda:N` 选卡）；先加 `--print-command` 只打印命令，
重跑默认路径需要 `--overwrite`。

每个 episode 生成五个条件：

| 条件 | 定义 |
| --- | --- |
| `oracle` | episode 自己的 GT macro route |
| `zero` | 全零 `[4,7,7]` 场 |
| `reversed` | 同格逆序（occupancy 逐格相同），单格路线跳过 |
| `shuffled` | 另一条 episode 的真实路线 |
| `observation_swap` | 保持 oracle route，换成另一条 episode 的 observation |

`shuffled` / `observation_swap` 的配对是确定性的：序号 `i` 取
`(i + n // 2) mod n`。差异用去归一化后的世界坐标度量，主指标是每步位移差均值
`mean_step_l2 = mean_t ||a_t^cond − a_t^oracle||`（PushT 像素）。

默认输出到 `outputs/pusht_diffusion/pilot/route_field_condition_ablation/`：

- `condition_ablation.parquet`：每 (checkpoint, episode, condition) 一行，含
  `model`、`step`、`checkpoint_id`、`episode_index`、`route_length`、`condition`、
  `mean_step_l2`、`chunk_l2`、`endpoint_l2`、`oracle_chunk_norm`、`policy_seed`、
  `initial_noise_seed`、`reverse_seed`、`scheduler`、`num_inference_steps`；
- `checkpoint_metrics.parquet`：每 checkpoint 每条件聚合，含
  `zero_observation_swap_ratio` 与 `reversed_observation_swap_ratio` 两个比值判据；
- `metrics.json` / `metrics.md`；
- `run_manifest.json`：`label` 为 **`Offline`**（不执行动作，永远不能报告 success），
  并固化 `interpretation_asymmetry` 三条约束。

判断标准（阈值在看结果前写死，执行后不得修改）：

- `zero` 在**全部** 8 个 checkpoint 的 `mean_step_l2` 都 < 1.0 px → route 通道未被
  使用；
- `reversed` 全部 < 1.0 px → 顺序/方向通道未被使用；
- `reversed / observation_swap` 比值（取比值最大的 checkpoint，即对"route 被读取"最
  有利的那个）< 0.05 → route 影响相对 observation 可忽略。

三者都指向"未使用"时闸门判 `route_not_read`，RF7 闭环计划应改为 `blocked`；任一指明
"已使用"且与比值判据一致时判 `route_read`，继续选模与闭环；两者结论不一致时记为
`inconclusive`，不得只挑支持某一方向的那个报。解读必须遵守的不对称性：

- `zero` 距离 ≈ 0 是强结论，route 通道确定没被读取；
- `zero` 距离很大**不能**下结论：全零场是 OOD 输入，模型从未见过；
- `reversed` 距离大只说明顺序/方向通道**被读取**，不说明被正确使用（方向正确与否
  只能由闭环的带符号读数回答）。

### 1. Development checkpoint selection

```bash
bash scripts/experiments/pusht/select_route_field_checkpoints.sh \
  --diffusion-run=outputs/pusht_diffusion/pilot/route_field_7x7/training/route_field_diffusion \
  --bc-run=outputs/pusht_diffusion/pilot/route_field_7x7/training/route_field_bc_chunk \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --num-gpus=8
```

`--bc-run` / `--diffusion-run` 至少提供一个，可只选 Diffusion。选模条件是 **每个
development episode 自己的精确 macro route**（oracle），不依赖 prototype manifest，也不
使用错配探针。单卡时可用 `--device=cuda:N` 指定卡（`--num-gpus>1` 时禁止该参数）。

默认输出到 `outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection/`，包含
`selected_checkpoints.json`、`selection.parquet`、`rollouts.parquet`、
`reconstruction_metrics.parquet`、`metrics.json` 和 `recording.rrd`。

### 2. 条件闸门（离线，不执行任何 rollout）

```bash
bash scripts/experiments/pusht/eval_route_field.sh \
  --gate-only \
  --selection-dir=outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --split=audit \
  --fixed-observation-episodes=5
```

固定 observation（dataset 中间帧）、固定 initial noise 和 reverse noise stream，只把
commanded route 从 GT 换成逆序，测 action-chunk 距离。逆序路线的 occupancy 逐格相同，
因此 `chunk_l2_distance` ≈ 0 就说明顺序/方向通道没被读取，闭环 rollout 不可能证明路线
可控，应当停在这一步。

默认输出到 `outputs/pusht_diffusion/pilot/route_field_condition_gate/`，只有
`condition_response.parquet`、`reconstruction_metrics.parquet`、`metrics.json` 和
`run_manifest.json`。该 run 的 `label` 是 **`Offline`**：它不执行动作，因此永远不能报告
success。`--gate-only` 强制单卡，且要求 `--fixed-observation-episodes>=1`。

### 3. Held-out 闭环验证（GT 路线 + 错配探针）

```bash
bash scripts/experiments/pusht/eval_route_field.sh \
  --selection-dir=outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --split=audit \
  --num-gpus=8
```

小范围冒烟（先确认 Rerun/PNG/parquet 都能落地再开正式跑）：

```bash
bash scripts/experiments/pusht/eval_route_field.sh \
  --selection-dir=outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection \
  --route-manifest=outputs/pusht_diffusion/pilot/route_field_7x7/sidecar/route_manifest.json \
  --split=audit \
  --max-episodes=6 \
  --output-dir=outputs/pusht_diffusion/pilot/route_field_held_out_eval_smoke
```

冒烟 episode 必须覆盖全部绘图边界，不能只跑前两条：audit 前 6 条包含单格路线
（59、81）、≥4 格路线（127，触发 `permuted` 配对图）和 6 格路线（40，触发路线末格
透明度上限）。只用 `--max-episodes=2` 会漏掉 6 格路线，`route_field_visualization`
里按路线序号算 alpha 的路径就没有被验证。

该入口对每个 audit episode 从 dataset-first-frame 状态执行该 episode 支持的全部闭环
条件（`oracle` / `reversed` / `permuted`，定义见上文"条件错配探针"），并在所有 rollout
**之前**再跑一次同样的闸门（默认前 5 条合格 episode，`--fixed-observation-episodes=0`
可关闭）。

条件是逐 episode 的，因此每个模型的 rollout 数不是 `3 × episodes`。当前 sidecar 下：

```text
audit:       oracle 21 / reversed 19 / permuted  5  → 45 rollouts per model
development: oracle 20 / reversed 19 / permuted 10  → 49 rollouts per model
```

每条 rollout 记录 requested/achieved T-cell 序列、有序 waypoint adherence（LCS）、
Levenshtein 编辑距离、对 GT route 的第二组 adherence、
`commanded_minus_oracle_adherence`、`commanded_first_segment_progress_cells`、
`initial_block_cell`、final rotation error、success/reward/latency，并保存 Rerun
Recording 和 **PNG 轨迹图**：

- `figures/<model>/condition_<oracle|reversed|permuted>/episode_<idx>.png`：单条
  rollout 的 requested vs achieved 路线 + 实际 T 中心轨迹 + 指标标题；
- `figures/<model>/pair_oracle_vs_<probe>/episode_<idx>.png`：同一初始状态、同一噪声
  下 GT 路线与错配探针的四格对比，直接看条件切换是否改变实际路径；
- `figures/<model>/condition_<...>/contact_sheet.png`：全部 episode 的
  requested（蓝）/achieved（橙）叠加总览。

结构化结果落在 `rollouts.parquet`、`condition_metrics.parquet`、
`paired_condition_metrics.parquet`、`condition_response.parquet`、`metrics.json` 和
`eval_info.json`。默认输出到
`outputs/pusht_diffusion/pilot/route_field_held_out_eval/`。`--split=development` 可把
同一协议跑到 development 上；单卡时用 `--device=cuda:N`；不需要图片时加 `--no-figures`。

### 4. 打开保存的 Recording

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun \
  outputs/pusht_diffusion/pilot/route_field_held_out_eval/recordings/route_field_diffusion/condition_reversed/episode_000826.rrd
```

## 训练后必须做的评测

按下面的顺序执行，把最便宜的判据放在最前面当闸门（命令见上文"选模与验证入口"）：

1. **全 checkpoint 离线消融**（步骤 0）：固定 observation + 固定 noise，对 run 目录
   里全部 checkpoint 比较 `oracle` / `zero` / `reversed` / `shuffled` /
   `observation_swap` 的 action-chunk 距离。`zero` 与 `reversed` 在全部 checkpoint 都
   ≈ 0，或 `reversed / observation_swap` 比值 < 0.05 时，判定 route 通道未被读取，
   停止后续 rollout。
2. **选中 checkpoint 的固定 observation + 固定 noise 前向 vs 逆序复核**：`--gate-only`
   （步骤 2）。DP 同时固定 initial noise 和 reverse noise stream。差异 ≈ 0 即可判定
   方向通道未被读取，停止后续 rollout。
3. **oracle 闭环**：定任务能力天花板。注意它单独不能证明控制——oracle route 取自同一
  条 episode 的执行轨迹，初始状态也是同一条 episode 的首帧，忽略 condition 的策略也
  可能拿到高 adherence。
4. **逆序闭环 + 有向读数**：同初始状态、同噪声，记录实际 T-cell 序列、
   `commanded_minus_oracle_adherence` 和 `commanded_first_segment_progress_cells`。
5. **中段置换闭环**（仅 ≥ 4 格的 episode）：唯一不与 observation 矛盾的匹配对。

第 3–5 步由步骤 3 的同一条命令一次跑完；`condition_metrics.parquet` 按条件分行，
`paired_condition_metrics.parquet` 按 `(model, oracle vs probe)` 分行。

第一阶段成功标准不是"condition 让动作数值变化"，而是：换成错配路线后实际 T 中心路径
按被请求的顺序改变，即 `obeyed_rate` 与 `commanded_first_segment_positive_rate` 明显
高于 0.5，且其 Wilson 下界高于 0.5。audit 只有 21 条 episode（逆序探针 19 条），
Wilson 区间宽度约 ±20pp，因此这个阈值必须在看结果之前写死，否则结论只会像 E5 一样
"不确定"。

正式结论与完整矩阵见：

```text
reports/2026-08-03-RF7 路线条件遵循正式评测结论.md
```

## 已完成的正式结果（Closed-loop / Formal）

`policy_seed=0`，每个 (episode, condition) 各 1 条 rollout，DDPM full sampling 100 步，
episode 长度 300（38 次 replan），初始状态为 dataset 第一帧重建。

### Development checkpoint selection（步骤 1）

条件为每条 development episode 自己的精确宏观路线，不使用错配探针。

| step | Development success | Mean max reward | Offline normalized MSE |
|---:|---:|---:|---:|
| 25000 | 0.05 | 0.5783 | 0.04894 |
| 50000 | 0.15 | 0.6604 | 0.05148 |
| 75000 | 0.15 | 0.6820 | 0.05087 |
| **100000** | **0.20** | **0.6577** | **0.04886** |
| 125000 | 0.10 | 0.6191 | 0.04778 |
| 150000 | 0.15 | 0.5808 | 0.04502 |
| 175000 | 0.10 | 0.6118 | 0.04616 |
| 200000 | 0.05 | 0.5879 | 0.04561 |

选中 step 100000。0.20 是 8 个 n=20 噪声估计的最大值，带选择性偏差；整族水平是 5%–20%。

### 条件闸门（步骤 2，Offline）

| 读数 | 值 |
|---|---:|
| `condition_response_chunk_l2_mean` | 113.630 px |
| `condition_response_chunk_l2_min` | 56.640 px |
| `condition_response_endpoint_mean` | 54.089 px |

逐 episode 均非零，闸门通过，后续 45 条 rollout 的支出合理。上表取自独立闸门产物
`route_field_condition_gate/`；audit run 内嵌复测同一读数为 113.625 px / 54.078 px
（两次独立执行的数值差，`chunk_l2_min` 仅闸门产物记录）。

### Held-out 条件级指标（步骤 3）

| 条件 | 角色 | n | Success | Wilson 95% | Mean max reward | 对被请求路线 adherence | 对 GT 路线 adherence | 首段进度（格） | 旋转误差 |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| oracle | task_reference | 21 | 1（4.8%） | [0.85%, 22.7%] | 0.5561 | 0.5516 | 0.5516 | +0.9339 | 89.503° |
| reversed | obedience_probe | 19 | 2（10.5%） | [2.9%, 31.4%] | 0.5333 | 0.2930 | 0.4298 | −0.9918 | 80.975° |
| permuted | obedience_probe | 5 | 0（0%） | [0%, 43.4%] | 0.2782 | 0.3333 | 0.4633 | +1.4294 | 122.599° |

旋转误差取自修正后产物（参考角为目标块角 π/4）。

### 配对服从度与可判定性

| 配对 | 总数 | 不可判定 | 可判定 | 其中 2 格探针 | `obeyed_rate` | 可判定上的 Wilson 上界 | 首段进度为正 |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle vs reversed | 19 | 13 | 6 | 9 | 0/19 → 0/6 | 0.168 → 0.390 | 4/19（0.2105） |
| oracle vs permuted | 5 | 3 | 2 | 0 | 0/5 → 0/2 | 0.434 → 0.658 | 4/5（0.8000） |

对 2 格路线 `[a,b] → [b,a]`，LCS 对被请求路线与 GT 路线都只能匹配 1 格，差值被结构性
钉死在 0；而真正服从所需的 `a→b→a` 又会被 `simplify_macro_route` 当作回头路删除（逆序
条件共删除 28 个回头格 + 9 个环格）。这类配对无法承载服从度证据，原口径把它们计入分母
并当作"不服从"。`undecidable_pair_count` / `decidable_pair_count` /
`obeyed_rate_decidable` / `two_cell_probe_count` 为后补的诊断列，原 `obeyed_rate` 定义
未变。

### 指标修复（Re-derived，未新增 rollout）

`achieved_route_metrics` 原先以 0 rad 为旋转误差参考角，而 gym_pusht 的目标位姿是
`[256, 256, π/4]`。判决性证据：唯一成功的 ep 62（覆盖率 > 0.95）原报 45.83°，修正后
0.83°。条件级均值 oracle 102.192° → 89.503°、reversed 76.665° → 80.975°、
permuted 107.393° → 122.599°。该列此前无阈值判据，修正不改变任何通过/否决结论，但修正
前后的值不可比。

```bash
bash scripts/experiments/pusht/rederive_route_field_metrics.sh \
  --source-dir=outputs/pusht_diffusion/pilot/route_field_held_out_eval/route_field_held_out_eval \
  --output-dir=outputs/pusht_diffusion/pilot/route_field_held_out_eval_corrected
```

重导出以只读方式读取源目录，`success` / `max_reward` / 两个 adherence / 首段进度等 13 列
逐值不变，manifest 记录 `schema_version=2`、`label="Re-derived"`、`derived_from`、
`derived_from_rollouts_sha256`、`goal_block_angle_rad`。**引用旋转误差必须使用
`_corrected` 产物**，其余指标两者等价。

### 结果落点

```text
outputs/pusht_diffusion/pilot/route_field_oracle_checkpoint_selection/
outputs/pusht_diffusion/pilot/route_field_condition_gate/
outputs/pusht_diffusion/pilot/route_field_held_out_eval/route_field_held_out_eval/
outputs/pusht_diffusion/pilot/route_field_held_out_eval_corrected/
```

## 已知限制

- 精确 RF7 路线族更稀疏；全量训练依赖共享的空间卷积编码和局部转移复用，而非每个
  exact route 都有大量样本；
- loop erasure 会丢失回退和闭环的时间顺序，必须结合 sidecar 审计字段报告其影响；
- 现有数据不是同一状态下的配对监督数据，observation 可能压过 condition；
- 逆序探针与 observation 互相矛盾（T 物理上位于前向 progress-0 格），训练中不存在这种
  样本，因此它只能证伪"方向通道被读取"，不能单独证明路线可任意切换；
- 中段置换保真度更高但只覆盖 audit 的 5/21，样本量不足以单独下结论；
- 7×7 hard assignment 更靠近边界，3 帧 debounce 只能缓解抖动；
- route field 只约束 T 中心平移，不直接表达最终旋转。

以下限制由本次正式评测**实测确认**，不是设计期预判：

- 2 格逆序探针在结构上不可判定，audit 的 19 个逆序配对中有 13 个 adherence 差值恰为 0
  （其中 9 个为 2 格路线）。可判定子集仅 6 条，Wilson 95% 上界 0.390；
- audit 路线长度中位数为 2（7×7 每格 73 px），路线指标与任务指标近乎解耦：ep 62 在
  achieved macro route 仅 1 格、adherence 为 0 的情况下就完成了任务；
- `commanded_first_segment_progress_cells` 用的是整段 300 步净位移投影，不是首段窗口读数；
  窗口化需要持久化 `block_path` 并重跑 rollout；
- 每个 (episode, condition) 仅 1 条 rollout、单一 `policy_seed=0`，oracle 成功率的
  Wilson 区间为 [0.85%, 22.7%]；正式结论应有 ≥ 3 个 policy seed；
- 165 条训练 episode 对应 113 个 canonical route family（每族平均 1.46 条，pilot route
  A/B 支撑仅 3/2 条），路线近似 episode ID、与 observation 高度共线。这是当前负结论的
  **结构性根因候选**，尚未通过同架构去条件消融证明；
- audit run 的 runtime repository `commit=null`，正式结果无法绑定到唯一源码 commit。
