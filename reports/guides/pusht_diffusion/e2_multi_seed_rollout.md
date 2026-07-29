# PushT E2：双协议闭环 Sampling Rollout

## 实验目标

E2 的所有样本都在 PushT 环境中闭环执行，不把离线 action chunk 当作 rollout：

- E2a `Initial noise sensitivity rollout`（低优先级）：验证不同 \(x_T\) 引起的动作差异
  是否被闭环交互放大。
- E2b `Full DDPM sampling rollout`（主实验）：测量完整 DDPM 随机采样下的 success
  variance、trajectory diversity 和 successful mode count。

## 实验思路

- 每条样本都重新创建环境、`env.reset(seed=...)`、`policy.reset()`，再循环执行
  observation → policy → `env.step()`，不续接任何旧状态。
- E2a 使用 2 个 environment seed × 5 个 policy seed，只改变初始 noise stream。
- E2b 使用 20 个 environment seed × 10 个 policy seed，初始 noise 与反向 DDPM stream
  都随 policy seed 改变。
- E2b 另执行同 20 个环境上的 BCChunk deterministic baseline。
- 轨迹多样性和成功模式只在同一 environment seed 内比较，再对环境等权汇总。
- 每条 Diffusion rollout 保存为独立、含完整帧时间线的 Rerun Recording。

## 模式

`Closed-loop / Formal`。E2a 和 E2b 都是完整闭环实验；E2a 只是降低结论优先级。

## 已测结果

两套协议已按注册矩阵从空目录完整执行并通过 schema/count/RRD 校验：

| 指标 | E2a | E2b |
|---|---:|---:|
| Diffusion rollout | 10 | 200 |
| BCChunk rollout | 0 | 20 |
| Diffusion 全采样成功率 | 50% | 52.5% |
| `policy_seed=0` 成功率 | 50% | 55% |
| BCChunk 成功率 | N/A | 25% |
| mean within-env success variance | 0.0000 | 0.1425 |
| between-env success-rate variance | 0.2500 | 0.1069 |
| first-replan chunk diversity | 0.0000417 | 0.03080 |
| all-trajectory diversity | 0.05692 | 0.14434 |
| successful-trajectory diversity | 0.000483（1 个 eligible env） | 0.11576 |
| failed-trajectory diversity | 0.11335（1 个 eligible env） | 0.18217 |
| 至少 2 个成功 mode 的环境 | 0/2 | 16/20 |
| mean / max successful mode count | 0.5 / 1 | 3.9 / 8 |

### 当前结论

结论先行：

1. **E2a 不支持 Diffusion Policy 具有多模能力。** 它只表明改变 \(x_T\) 产生的微小
   action 差异，可能在部分场景中被闭环交互放大为明显不同的轨迹；目前没有 success
   翻转或多个成功 mode 的证据。
2. **E2b 支持 Diffusion Policy 具有多模闭环行为能力。** 完整 DDPM sampling 能从相同
   初始场景产生多个不同的成功闭环轨迹 mode，并且该结果对当前三个轨迹聚类阈值稳定。
3. **目前不能声称条件动作分布 \(p(a\mid o)\) 是多峰分布或 DP 已克服 MSE mode
   averaging。** E2b 提供下游闭环行为证据；复用现有 E1 20-sample 数据完成的固定
   observation action-mode 分析为 0/10 candidate 通过预注册多模判据。

#### E2a：数据与解释

E2a 中，env 2000 为 0/5 成功，first-replan chunk diversity 为 `0.0000201`，但完整
闭环轨迹 diversity 达到 `0.11335`；env 2001 为 5/5 成功，完整轨迹 diversity 仅
`0.000483`。两个环境内部都没有 success 翻转，mean within-env success variance 为
0。这说明总成功率 50% 完全来自场景差异，不能归因于 initial-noise variation。

E2a 在每次 replan 都使用随 policy seed 改变的 \(x_T\)，而同一环境和 replan 下的
reverse-noise stream 在五条 rollout 间保持一致。因此它测量的是持续 initial-noise
stream 的闭环敏感性，并不是“只扰动第一次 \(x_T\)，之后冻结全部随机性”。E2a 的
0/2 多成功 mode 结果不能作为 Diffusion Policy 具有多模能力的证据。

#### E2b：数据与解释

E2b 中有 16/20 个环境同时出现成功和失败 seed，说明完整 DDPM sampling 会实质性改变
闭环成败；同样有 16/20 个环境在主阈值 0.05 下出现至少两个成功轨迹 cluster。把轨迹
距离阈值改为 0.025 或 0.075 时仍然都是 16/20，说明这个操作性结论对当前阈值选择稳定。

该结论是 Diffusion Policy 具有多模闭环行为能力的直接实验依据，但不能严格证明固定
observation 下学习到的条件动作分布 \(p(a\mid o)\) 本身是多峰分布。E2b 的
observation 会随闭环交互变化，轨迹分叉同时包含环境动力学放大和后续 observation
反馈。现有 E1 固定-observation action-mode 探索分析中，E1a negative control 为
0/10，E1b 也为 0/10，BC averaging-consistent signature 为 0/10。因此当前证据组合
支持“stochastic sampling 具有多个成功闭环行为 mode”，不支持“已经证明 DP 克服了
MSE regression 的 mode averaging”。20 samples 的阴性结果同样不能证明真实分布必然
单峰，只能说明当前实验没有检测到多峰 action distribution。

在与 E2a 完全匹配的 2 env × 5 seed 子集上，E2b 的 first-replan chunk diversity 为
`0.02214`，而 E2a 为 `0.0000417`；两者使用相同的 initial-noise seed 集合，主要差异
是 E2b 的 reverse-noise stream 也随 policy seed 改变。这表明完整反向 DDPM 随机性是
当前 action diversity 的主要来源。

E2b 的 `success_any_at_10=90%` 只表示 oracle coverage，不能替代单次部署成功率
`policy_seed=0` 的 55%。DP 单样本相对 BC 的成功率为 55% vs 25%，但 20 个配对环境的
证据量仍不足以声称统计显著优于 BC。失败轨迹 diversity `0.18217` 也高于成功轨迹
`0.11576`，所以不能把更大的 diversity 本身解释为更好的策略。

## 可直接执行的完整命令

先解析 Development 阶段冻结的正式 checkpoint：

```bash
BC_STEP="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["bc_chunk"]["step"])')"
DIFFUSION_STEP="$(python -c 'import json; print(json.load(open("outputs/pusht_diffusion/pilot/checkpoint_selection/selected_checkpoints.json"))["diffusion"]["step"])')"
BC_POLICY="$(printf 'outputs/pusht_diffusion/pilot/bc_chunk/checkpoints/%06d/pretrained_model' "${BC_STEP}")"
DIFFUSION_POLICY="$(printf 'outputs/pusht_diffusion/pilot/diffusion/checkpoints/%06d/pretrained_model' "${DIFFUSION_STEP}")"
```

E2a（2 × 5，共 10 条 Diffusion 闭环 rollout）：

```bash
bash scripts/experiments/pusht/eval_multi_seed_rollouts.sh \
  --protocol=initial-noise-sensitivity \
  --diffusion-policy="${DIFFUSION_POLICY}" \
  --num-gpus=1 \
  --output-dir=outputs/pusht_diffusion/pilot/e2a_initial_noise_sensitivity \
  --overwrite
```

E2b（20 × 10，共 200 条 Diffusion + 20 条 BCChunk 闭环 rollout）：

```bash
bash scripts/experiments/pusht/eval_multi_seed_rollouts.sh \
  --protocol=full-ddpm-sampling \
  --bc-policy="${BC_POLICY}" \
  --diffusion-policy="${DIFFUSION_POLICY}" \
  --num-gpus=1 \
  --output-dir=outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling \
  --overwrite
```

`--overwrite` 会整体替换对应输出目录；原子临时目录不会被恢复或续跑。正式运行不传
`--num-environment-seeds` / `--num-policy-seeds`，由注册配置强制使用完整矩阵。缩小计数
只允许配合 `--label=Smoke`。

## 显式多卡执行

`--num-gpus=N` 必须是正整数，默认值和上面的单卡正式命令都显式使用 1。多卡时，每张
逻辑可见 GPU 启动一个独立进程，使用 `cuda:0 ... cuda:N-1`；如需选择物理卡，先设置
`CUDA_VISIBLE_DEVICES`。例如用 4 张可见卡重跑 E2b：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/experiments/pusht/eval_multi_seed_rollouts.sh \
  --protocol=full-ddpm-sampling \
  --bc-policy="${BC_POLICY}" \
  --diffusion-policy="${DIFFUSION_POLICY}" \
  --num-gpus=4 \
  --output-dir=outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling \
  --overwrite
```

分片单位是 environment seed，不是单条 rollout。20 个场景在 4 卡上按 round-robin
分为 `2000,2004,...`、`2001,2005,...`、`2002,2006,...`、`2003,2007,...`；一个场景
的全部 10 个 Diffusion policy seed 以及该场景的 BC baseline 始终属于同一 worker。
因此 worker 间不会写同一个 RRD/video 文件，最终仍由父进程一次性汇总 Parquet、指标、
mode analysis 和 manifest。任一 worker 失败都会终止其余 worker，带 `.incomplete`
标记的临时目录不会成为正式结果。

卡数不能超过当前协议的 environment seed 数，也不能超过可见 CUDA device 数：E2a
最多 2 卡，E2b 最多 20 卡。多卡只改变执行并行度，不改变 seed、闭环语义、Recording
目录或查看顺序；实际分片记录在 `run_manifest.json` 的 `execution` 字段。

## 打开 E2a / E2b 完整可视化

E2a（先加载 env 2000 的 5 个 policy seed，再加载 env 2001 的 5 个 policy seed）：

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9878 \
  outputs/pusht_diffusion/pilot/e2a_initial_noise_sensitivity/recordings/policy_seed_{00..04}/env_2000.rrd \
  outputs/pusht_diffusion/pilot/e2a_initial_noise_sensitivity/recordings/policy_seed_{00..04}/env_2001.rrd
```

E2b：

```bash
bash scripts/experiments/pusht/view_e2b_rollouts.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling/recordings \
  --port=9878
```

查看器校验 10 个 policy seed 目录、200 个 Diffusion RRD 和相邻目录中的 20 个 BC RRD，
然后按 `environment_seed → BC / policy_seed` 顺序加载。Rerun 的 Recording selector
中，同一个场景的 1 条 BC baseline 和 10 条 Diffusion 采样连续相邻。显示名明确包含
协议、模型、场景、policy seed 和结果，例如
`E2a | DP | env 2000 | seed 00 | success`、
`E2b | DP | env 2000 | seed 00 | success`、`E2b | BC | env 2000 | success`；
内部 Recording ID 同样分别使用 `e2a_dp_...`、`e2b_dp_...`、`e2b_bc_...`，切换时不会
混淆。每条都沿用完整 rollout Blueprint，可切换查看 simulator、agent path、block path、
action targets、reward 和 success。`complete_action_targets` 是实际执行的全部 action
target 序列，作为独立可选实体从视频第 0 帧持续显示到最后一帧；逐帧的
`current_action_target` 仍保留。

当前 E2a/E2b Formal RRD 已原地补入激活的完整 rollout Blueprint，不需要重跑 rollout；
补丁只增加 Blueprint store，没有改动 observation、action、轨迹、reward 或 success
数据。后续新生成的 RRD 会在写入时将同一 Blueprint 同时设为 active 和 default，因此
可直接显示橙色 `complete_action_targets`。

## 结果落点

```text
outputs/pusht_diffusion/pilot/e2b_full_ddpm_sampling/
├── recordings/
│   ├── policy_seed_00/
│   │   ├── env_2000.rrd
│   │   └── ... env_2019.rrd
│   └── ... policy_seed_09/
├── bc_recordings/
│   └── env_2000.rrd ... env_2019.rrd
├── videos/
├── rollouts.parquet
├── replans.parquet
├── per_environment_metrics.parquet
├── metrics.json
├── metrics.md
├── mode_analysis.json
└── run_manifest.json
```

E2a 使用同样结构，但包含 5 个 policy seed 目录、每个目录 2 个场景，不包含 BC baseline。
每条 rollout 和每次 replan 都记录 sampling protocol、environment/policy seed、
initial-noise seed/hash、reverse seed/state hash、scheduler、success、reward、episode
length、termination reason 和完整数值轨迹。

## 指标与判断标准

- success variance：报告每个环境的 \(p(1-p)\) 均值，以及环境成功率的 population
  variance。
- trajectory diversity：agent xy、block xy 除以 512，block angle 使用 wrapped
  difference 除以 π；轨迹重采样到 100 点后计算 RMS distance。
- successful mode count：仅对同一环境中的成功轨迹做 complete-linkage clustering；
  主阈值 0.05，并报告 0.025 / 0.075 敏感性。
- single-sample success 使用 `policy_seed=0`；`success_any_at_10` 只作为 oracle coverage，
  不能替代部署成功率。
- E2b Formal 必须恰好有 200 条 Diffusion 与 20 条 BCChunk rollout；任一缺失、重复、
  非闭环或旧 schema 输出都会被最终报告拒绝。
- 只有同一环境中出现至少两个成功 cluster，才能声称观测到多个成功闭环 mode。

## 统一实验字段

- Task suite：PushT
- Dataset size：206 episodes；训练 split 165 episodes
- Observation keys：96×96 RGB、二维 agent position；历史长度 2
- Action space：二维绝对 target position
- Action horizon：内部 16，每次闭环执行 `[1:9]` 共 8 步后 replan
- Control frequency：10 Hz
- Train loss / open-loop action error：引用冻结 checkpoint 的 Development 选择结果
- Closed-loop success rate：E2b `single_sample_success`、all-sample success、BC baseline
- Latency：`replans.parquet` 的 `chunk_latency_ms`；正式速度结论由 E3 给出
- Failure cases：按 environment seed、policy seed、termination reason、完整轨迹记录
