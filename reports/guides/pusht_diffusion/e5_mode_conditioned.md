# PushT E5：Prototype-ID 条件控制

## 先说结论

E5 第一阶段已经实现并通过本地 Smoke，正式训练与正式闭环评测尚未执行。

- 现有 165 条 train demonstrations 可稳定提取 3 个全局 trajectory prototypes；
  K=3 是满足预注册门槛的最小 K。
- Mode-BCChunk 和 Mode-Diffusion 均只接收离散 Prototype ID，不接收未来 trajectory
  sketch。输入使用 `nn.Embedding(K+1, 32)`，其中 ID `K` 是 trainable NULL。
- 两个模型均从 `train_seed` 控制的随机权重直接开始训练，不需要生成或同步 E5
  step-zero checkpoint。
- 训练时每个条件样本有 20% 概率替换为 NULL；一个 rollout 内 ID 固定，环境不会代替
  用户切换 condition。
- 固定 observation 实验会固定 observation、initial noise 和完整 DDPM reverse stream，
  只切换 P0/P1/P2/NULL。
- 所有 E5 rollout 都是闭环；正式主网格是 20 个环境、每个 prototype 10 个
  Mode-Diffusion seeds，另含完整 NULL 网格和确定性 Mode-BCChunk reference。

目前不能宣称 E5 已证明“Diffusion Policy 的无条件分布完整覆盖多个成功模式”。只有
正式训练后同时满足：

```text
指定 Prototype ID → 对应 prototype adherence + task success
NULL + 不同 seed → 同一环境至少两个可分类的成功 prototype
```

才能分别支持“显式可控性”和“该 E5 checkpoint 的 NULL 分支覆盖多个成功模式”。
现有 demonstrations 也不是严格的相同初始状态配对数据，因此即使 E5 成功，仍不能把
结果提升为严格的训练数据同状态多解证明，更不能反向替代原 E2 checkpoint 的结论。

## 实验目标

验证一个全局、离散、用户指定的 trajectory prototype 是否能让同一条件策略选择并闭环
执行对应行为；同时检查同一 E5 Mode-Diffusion checkpoint 的 NULL 分支是否保留多个
可分类、成功的行为 prototype。

## 实验思路

1. 只用 train demonstrations 的物体中心化完整轨迹做 K-medoids，medoid 仅用于标签和
   评价。
2. 模型只输入 Prototype ID/NULL embedding，不输入未来 sketch。
3. 固定 observation 和完整 RNG，只切换 ID，确认动作是否响应 condition。
4. 对每个 ID 和 NULL 执行完整闭环，以 task success、adherence 和 controlled success
   判断能力。

## 已完成的离线原型结果（Offline / Formal）

输入为 E0 导出的 206 条完整轨迹，其中只用 165 条 train demonstrations 拟合。
prototype sketch 使用物体中心化的 8 个事件 token；K-medoids 的 medoid 是真实轨迹，
不是左右模式的平均。

| 项目 | 结果 |
|---|---:|
| 选中 K | 3 |
| cluster sizes | 57 / 70 / 38 |
| medoid episodes | 16 / 108 / 140 |
| silhouette | 0.278858 |
| initialization ARI median | 1.000000 |
| 80% subsample ARI median | 0.829350 |
| eligible train episodes | 165 / 165 |

K=2 因 subsample ARI median 只有 0.642008 未通过；K=4 也通过稳定性门槛，但按“选择
最小通过 K”的冻结规则选择 K=3。K=3 的初始两帧 leakage audit 中，logistic regression
balanced accuracy 为 0.395238，random forest 为 0.529825：初始状态与 prototype
存在相关性，但不是近乎完美可预测。该 audit 不能消除数据未严格配对的限制。

结果位于：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/
├── prototype_manifest.json
├── prototype_candidates.json
├── prototype_sketches.json
├── episode_assignments.parquet
├── leakage_audit.json
└── visualizations/
```

## K=3 是如何聚类出来的

### 聚类输入

K-medoids 没有直接使用原始 action，也没有把 E0 的 `approach_side` 或
`strategy_label` 当作监督标签。每条完整 demonstration 先被压缩成 8 个按事件和运动
进度对齐的 object-centric tokens：

| Token | 对应轨迹事件 |
|---:|---|
| 0 | 第一个 T pose 有效的 frame |
| 1 | 第一次稳定接触前的最后一个有效 frame |
| 2 | 第一次稳定接触 |
| 3 | T 第一次发生显著运动 |
| 4 | T 累计运动弧长达到 25% |
| 5 | T 累计运动弧长达到 50% |
| 6 | T 累计运动弧长达到 75% |
| 7 | 最后一个 T pose 有效的 frame |

稳定接触定义为 agent 到 T 表面的距离不超过 20 workspace pixels；二值接触序列先做
长度为 5 的 closing，再丢弃少于 3 frames 的短接触段。显著运动定义为 T 相对初始
位置平移至少 12 pixels，或相对旋转至少 15°。25%/50%/75% 使用如下平移和旋转联合
弧长计算，64 pixels 是把旋转换算为等效位移的半径：

```text
step_arc = sqrt(||Δposition||² + (64 × Δangle_rad)²)
```

每个 token 保存：

- agent 相对当前 T 的位置，并旋转到当前 T 的局部坐标系；
- T 相对初始 T pose 的位移，并旋转到初始 T 的局部坐标系；
- T 相对初始姿态的连续角度；
- 当前 token 是否处于稳定接触；
- 当前 token 在完整 episode 中的归一化时间。

因此，平移或旋转后的同类物体交互可以保持可比；聚类关注的是相对接近方式、接触过程和
T 的完整运动形状，而不是绝对世界坐标。

### 轨迹距离

两条 8-token sketches 的距离由 6 个分量组成：

| 距离分量 | 比较内容 | 权重 |
|---|---|---:|
| first contact | token 2 的 agent-T 局部接触几何 | 2.00 |
| approach | tokens 0–2 的 agent-T 局部接近路径 | 1.50 |
| block path | tokens 3–6 的 T 局部位移路径 | 1.00 |
| rotation | tokens 3–6 的 T 相对旋转过程 | 1.00 |
| contact sequence | 8 个 contact flags 及接触段数量 | 1.50 |
| edge | tokens 0、7 的 agent 和 T 起终点 | 0.25 |

每个分量先除以 165 条 eligible train trajectories 两两距离的非零中位数，再做加权
平均：

```text
D(i,j) = Σc wc × dc(i,j) / median_nonzero(dc) ÷ Σc wc
```

然后对预计算的 `D(i,j)` 运行 K-medoids/PAM。medoid 必须是一条真实 train
demonstration，不会产生把两条路线平均后的虚构轨迹。K 扫描 2、3、4；K=2 未通过
80% subsample stability，K=3 和 K=4 通过，因此按冻结规则选择最小的 K=3。

### 三个 medoids 代表什么

P0/P1/P2 按 medoid 的 `episode_index` 从小到大固定编号。下表中的文字标签来自 E0，
只用于解释 medoid 本身，不参与 K-medoids 拟合：

| ID | Cluster size | Medoid | Medoid 的代表轨迹 |
|---|---:|---:|---|
| P0 | 57 | episode 16 | left approach；frame 12/126 首次接触；rotation-first、down、recontact；T 净平移 70.84 pixels，净旋转 +81.91°，总旋转 268.54° |
| P1 | 70 | episode 108 | right approach；frame 22/112 首次接触，是三个 medoids 中接触最晚且整体运动最紧凑的一条；rotation-first、down、recontact；T 净平移 57.71 pixels，净旋转 +77.22°，总旋转 170.94° |
| P2 | 38 | episode 140 | top approach；frame 12/112 首次接触，并在 frame 31 达到 50% 累计运动进度；rotation-first、down、recontact；T 净平移 73.80 pixels，净旋转 +87.67°，总旋转 279.87° |

三个 medoids 都包含“先旋转、主要向下运动、发生重接触”的共同任务结构；主要区别是
agent 从哪一侧接近和接触、接触发生时间，以及 T 的中间位移/旋转路径。原型图位于：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/visualizations/
k_03_prototypes.png
```

不能把上述 medoid 说明直接提升为 `P0=left`、`P1=right`、`P2=top` 的硬语义。
整个 cluster 是按完整加权轨迹距离分配的，成员的 E0 approach-side 统计如下：

| ID | bottom | left | right | top |
|---|---:|---:|---:|---:|
| P0 | 36.8% | 29.8% | 22.8% | 10.5% |
| P1 | 20.0% | 11.4% | 52.9% | 15.7% |
| P2 | 34.2% | 18.4% | 7.9% | 39.5% |

所以更准确的解释是：

- P1 是 right-approach 占主导的完整轨迹 prototype；
- P2 的 medoid 和最大单一成员组都是 top approach，但仍包含其他接近方式；
- P0 是没有单一 approach-side 过半的混合 prototype，其真实定义仍是“距离 episode 16
  的完整轨迹 sketch 最近”，而不是 “left prototype”。

E5 后续评测也使用相同 8-token 表示和完整距离做 adherence 分类，不会用
`approach_side` 单字段替代 prototype。

## Mode-BCChunk 与 Mode-Diffusion 如何改造

### 共同的外插式 condition adapter

原始 `lerobot/pusht` 数据保持只读。`episode_assignments.parquet` 按
`episode_index` 为训练 sample 增加标量 `prototype_id`：

```text
0 / 1 / 2  → P0 / P1 / P2
3          → NULL
```

训练仍然按原逻辑随机抽取 frame/chunk；sidecar dataset wrapper 只根据该 sample 的
`episode_index` 附加一个整数 ID，不复制或改写原始 observations/actions。第一阶段
模型不会读入 8-token sketch、medoid 轨迹或未来状态；sketch 只用于离线生成 ID 和
rollout adherence 评价。

K=3 时新增 `nn.Embedding(4, 32)`。observation history 仍由原 RGB encoder 和 state
encoder 编码；对应 ID 的 32 维 embedding 只在 observation global condition 的末尾
拼接一次，并把现有 conditional U-Net 的 `global_cond_dim` 增加 32：

```text
image history ─→ RGB encoder ─┐
agent-state history ──────────┼─→ observation global condition ─┐
prototype_id ─→ Embedding(4,32) ────────────────────────────────┼─→ concat
                                                                └─→ existing U-Net FiLM
```

这个 adapter 对两种模型完全相同：

- `prototype_id` 是范围 `[0,3]` 的 categorical integer，不按连续状态做 normalization；
- 训练时每个非 NULL sample 有 20% 概率把 ID 替换为 trainable NULL ID 3，显式 NULL
  不再重复 dropout；
- rollout 开始时 latch 一次 ID；在 `policy.reset()` 前禁止切换，之后每次闭环 replan
  都重用同一 ID；
- 模型不是 recurrent policy，300-step rollout 的 condition persistence 来自执行器每次
  replan 都重新传入同一 ID，而不是环境状态或隐藏状态。

### 两种模型分别保留和改变的部分

| 项目 | Mode-BCChunk | Mode-Diffusion |
|---|---|---|
| 基类 | 本地 `BCChunkPolicy` | LeRobot `DiffusionPolicy` |
| 条件改造 | observation global condition 末尾拼接同一个 32-D Prototype/NULL embedding | 与 Mode-BCChunk 完全相同 |
| 训练输入 | zero action-trajectory query，固定 diffusion timestep 0 | 对真实 action horizon 随机采样 diffusion timestep 并加噪 |
| 训练目标 | 直接回归完整 clean action horizon 的 MSE | 预测加入的 epsilon noise |
| 推理 | 单次 U-Net 前向；不接受 sampling noise，结果确定 | DDPM 100-step reverse sampling；由 policy seed 控制随机性 |
| Action horizon | 预测 16 步，闭环执行 `[1:9]` 的 8 步 | 生成 16 步，闭环执行 `[1:9]` 的 8 步 |
| 条件内随机性 | 同一 observation、同一 ID 只有一个 deterministic chunk | 同一 observation、同一 ID 可由不同 diffusion seeds 采样不同 chunks |

Mode-BCChunk 的核心仍是 deterministic clean-action chunk regression：它把长度 16 的
全零 query 和固定 timestep 0 送入同一容量的 conditional U-Net，然后直接对 16 步
ground-truth action 做 MSE。改造只让 U-Net 同时看到 Prototype ID embedding。

Mode-Diffusion 的噪声训练和采样机制没有改写：训练仍随机选 timestep、给 16 步 action
加噪并学习 epsilon；推理仍进行完整 DDPM reverse process。Prototype ID embedding 在
一次 replan 的所有 reverse steps 中保持不变，用于从该 ID 对应的条件分布中采样。

两者的 observation encoder、U-Net 结构、horizon、normalization、optimizer 等共同配置
保持匹配。E5 训练将 `policy.pretrained_path` 保持为 `None`，由 LeRobot 在设置
`train_seed` 后直接构造随机初始化模型；多卡训练再由 Accelerate/DDP 同步各 rank 的
参数。两个模型使用相同 `train_seed` 以保证各自训练可复现，但不再依赖共享 checkpoint
强制逐 tensor 配对。实现位于本地新增的 `mode_conditioning`、`mode_bc_chunk` 和
`mode_diffusion` policy classes；没有修改安装的 LeRobot，也没有改变 E0–E4 原始策略和
checkpoint 的行为。

## 从头执行

### 0. 准备数据和 E0 完整轨迹

```bash
bash scripts/experiments/pusht/prepare_dataset.sh

bash scripts/experiments/pusht/inspect_demonstrations.sh \
  --num-episodes=206 \
  --num-candidates=10 \
  --output-dir=outputs/pusht_diffusion/data_inspection_e0_full
```

### 1. 生成离线 prototypes

```bash
bash scripts/experiments/pusht/fit_e5_prototypes.sh \
  --trajectories=outputs/pusht_diffusion/data_inspection_e0_full/trajectories.parquet \
  --output-dir=outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes \
  --overwrite
```

### 2. 多卡随机初始化训练

`--num-gpus` 必须显式给出。`--batch-size` 是每张卡的 batch size；下面 8 卡命令的
global batch size 为 64。`--train-seed=1000` 显式控制随机初始化和训练随机性。

```bash
bash scripts/experiments/pusht/train_mode_bc_chunk_multi_gpu.sh \
  --mode=pilot \
  --prototype-manifest=outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json \
  --train-seed=1000 \
  --num-gpus=8 \
  --batch-size=8 \
  --steps=200000 \
  --output-dir=outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_bc_chunk

bash scripts/experiments/pusht/train_mode_diffusion_multi_gpu.sh \
  --mode=pilot \
  --prototype-manifest=outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json \
  --train-seed=1000 \
  --num-gpus=8 \
  --batch-size=8 \
  --steps=200000 \
  --output-dir=outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_diffusion
```

训练入口不会读取 E5 step-zero 或 Smoke checkpoint。两个模型必须使用同一 prototype
manifest、train seed、steps 和 global batch size；训练配置中的
`initialization=random_from_train_seed` 和 `initialization_seed=1000` 用于确认协议。

## 正式评测

以下命令假设使用 200000-step checkpoint。如果最终 checkpoint selection 选择了不同
step，只替换两个 `--*-policy` 路径，并在实验记录中写明。

### 固定 observation 条件切换

```bash
bash scripts/experiments/pusht/eval_e5_fixed_observations.sh \
  --bc-policy=outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_bc_chunk/checkpoints/200000/pretrained_model \
  --diffusion-policy=outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_diffusion/checkpoints/200000/pretrained_model \
  --prototype-manifest=outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json \
  --candidates=outputs/pusht_diffusion/data_inspection_e0_full/candidates.json \
  --num-policy-seeds=20 \
  --num-inference-steps=100 \
  --label=Offline \
  --overwrite
```

输出：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/fixed_observation/
├── actions.parquet
├── condition_metrics.parquet
├── metrics.json
├── manifest.json
└── recordings/
    ├── condition_P0.rrd
    ├── condition_P1.rrd
    ├── condition_P2.rrd
    └── condition_NULL.rrd
```

启动可视化：

```bash
bash scripts/experiments/pusht/view_e5_fixed_observations.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/e5_mode_conditioned/fixed_observation/recordings \
  --port=9878
```

固定 observation 的主要量是相同 sampling seed 下的跨 prototype action-chunk RMS，以及
各 prototype 内不同 seeds 的 pairwise RMS。它只能证明 condition 是否改变动作，不能
单独报告 prototype adherence。

### 完整闭环 Prototype/NULL rollout

```bash
bash scripts/experiments/pusht/eval_e5_rollouts.sh \
  --bc-policy=outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_bc_chunk/checkpoints/200000/pretrained_model \
  --diffusion-policy=outputs/pusht_diffusion/pilot/e5_mode_conditioned/training/mode_diffusion/checkpoints/200000/pretrained_model \
  --prototype-manifest=outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/prototype_manifest.json \
  --num-gpus=8 \
  --num-environment-seeds=20 \
  --num-policy-seeds=10 \
  --episode-length=300 \
  --num-inference-steps=100 \
  --label=Closed-loop \
  --overwrite
```

K=3 时正式输出应恰好包含：

```text
Mode-Diffusion conditional: 20 × 3 × 10 = 600
Mode-Diffusion NULL:        20 × 1 × 10 = 200
Mode-BCChunk conditional:   20 × 3      = 60
Mode-BCChunk NULL:          20 × 1      = 20
total: 880 closed-loop rollouts
```

输出：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/closed_loop/
├── rollouts.parquet
├── condition_metrics.parquet
├── null_mode_support.parquet
├── metrics.json
├── manifest.json
└── recordings/
    ├── mode_bc_chunk/condition_<P0|P1|P2|NULL>/env_*.rrd
    └── mode_diffusion/condition_<P0|P1|P2|NULL>/policy_seed_*/env_*.rrd
```

可视化按 environment → condition → model/policy seed 排序，相同场景相邻：

```bash
bash scripts/experiments/pusht/view_e5_rollouts.sh \
  --recordings-dir=outputs/pusht_diffusion/pilot/e5_mode_conditioned/closed_loop/recordings \
  --port=9878
```

每条 Rerun recording 沿用完整 rollout 视图，包含 simulator、measured agent/block path、
reward、当前 action target，以及从视频开始到结尾可见的完整实际执行 action-target
序列。

## 判定标准

闭环 rollout 使用与 train clustering 相同的图像几何、8-token sketch、component
scales 和 medoid distance。每个 cluster 的拒识半径是 train member 到 medoid 距离的
q95 higher quantile：

```text
classifiable =
  rollout sketch eligible
  AND nearest-medoid distance <= nearest prototype radius

prototype_adherent =
  classifiable
  AND nearest prototype == commanded prototype

controlled_success =
  task success
  AND prototype_adherent
```

unclassifiable 的 strict adherence 记为 false，并单独报告 classifiable rate。

NULL 网格中，同一环境的某个 prototype 至少要有 2 个成功、可分类的 seeds 才算一个
supported NULL mode；同一环境至少有 2 个 supported modes 才记为
`multi_mode_success=true`。单个离群 seed 不算模式覆盖。

## 已完成的验证（Smoke）

Smoke 只证明实现链路可执行，不是模型能力结果：

- Mode-BCChunk 和 Mode-Diffusion 分别完成 1 update；最后记录 loss 约为 0.221 和
  1.189。
- 两个模型均为约 264M 参数；checkpoint 保存/加载、条件 dropout、固定 ID 与显式
  DDPM RNG 测试通过。
- 10 candidates × P0/P1/P2/NULL × 单 seed 的固定 observation Smoke 完成，输出
  640 action rows 和 4 个 Rerun recordings。
- 1 environment × P0/P1/P2/NULL × 两种模型的一帧闭环 Smoke 完成 8/8 rollouts，
  输出 8 个独立 Rerun recordings。
- 一帧 rollout 不足以形成稳定接触或完整 sketch，因此其 0 success/adherence 不得作为
  E5 负面结论；正式 open-loop error、closed-loop success 和 DDPM100 latency 仍未测量。

## 必填实验字段

- Task suite：PushT
- Dataset size：206 episodes；prototype fit/train 只用 165 train episodes
- Observation keys：96×96 RGB、二维 agent position；历史长度 2
- Action space：二维绝对 action target
- Action horizon：内部 16；闭环每次执行 `[1:9]` 共 8 actions
- Control frequency：10 Hz
- Train loss：从正式训练日志记录；当前只有 1-step Smoke
- Open-loop action error：正式 fixed-observation/checkpoint evaluation 后记录
- Closed-loop success rate：正式 880-rollout 汇总后记录
- Latency：正式 DDPM100 replan latency 后记录
- Failure cases：按 environment、condition、policy seed、classifiability 和终止原因记录
