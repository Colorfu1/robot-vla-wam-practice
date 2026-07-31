# 2026-07-30 E5 Prototype条件控制正式评测结论

## 重要进展

- development checkpoint selection 已正式完成：Mode-BCChunk 选中 step 175000，
  Mode-Diffusion 选中 step 125000。
- held-out audit 已完成 21 条 audit episodes × P0/P1/P2/NULL × 两种模型，共
  168/168 条闭环 rollout；21/21 首帧重建通过。
- Mode-Diffusion 保留了任务完成能力：oracle 子集成功率 57.1%，NULL 和 P1 都为
  52.4%；但 P0/P1/P2 adherence 仅 14.3%/28.6%/19.0%。
- condition ID 确实进入模型且会改变动作；固定噪声下不同 ID 产生不同 action chunk。
  失败发生在“条件变化没有稳定映射为指定 prototype”，不是评测链路回退到 NULL。
- GT dense-trajectory 可视化和数据结构共同表明 P0/P1/P2 不是已验证的任意可切换命令：
  165 个训练初始状态每个只有一条 demonstration、一个 prototype，没有同状态配对监督。
- 当前 E5 的正式结论是负面的：它没有证明 prototype-conditioned controllability。
  各 condition 的 raw success rate 只能说明任务能力，不能解释为模态控制成功。

## 核心结论

### 1. 模型会响应 condition，但没有学会按 condition 选择语义策略

P0/P1/P2/NULL 的 embedding 和实际传入 ID 均有效。对同一个 reconstructed initial
state，固定 initial noise 和完整 DDPM reverse noise stream 后切换 ID，会改变最早的
action chunk；因此不能把结果归因于“condition 没传入”。

但是，指定 condition 与最终 achieved prototype 的对应关系很弱：

- Mode-BCChunk 的 P0/P1/P2 adherence 为 19.0%/23.8%/19.0%；
- Mode-Diffusion 的 P0/P1/P2 adherence 为 14.3%/28.6%/19.0%；
- controlled success 最高也只有 19.0%；
- 当显式 condition 与 NULL 都可分类时，BC 有 35/36、DP 有 34/38 得到相同的
  achieved prototype。

因此当前 embedding 更像会扰动动作的附加特征，而不是主导高层行为的离散控制开关。

### 2. Mode-Diffusion 的任务成功率较高，但不能等价为可控性

Mode-Diffusion 在 oracle 子集上成功 12/21，优于 Mode-BCChunk 的 7/21；P0 和 P1
全任务 success rate 也较高。但同一模型的 NULL 为 11/21，而显式 P1 也是 11/21，
并且 requested→achieved 矩阵没有形成明显对角线。

这说明 Mode-Diffusion 仍有不错的 PushT 任务能力，但成功并不依赖它遵守指定
prototype。当前结果不能支持“告诉模型 P0/P1/P2 后，它会选择相应策略”。

### 3. 根因优先指向数据可辨识性与 prototype 定义，而不是输入实现

P0/P1/P2 是根据完整 demonstration 的 8 个 object-centric event tokens 聚类得到的
事后轨迹类别。训练集中：

- P0/P1/P2 分别有 57/70/38 条；
- medoid episodes 分别为 16/108/140；
- 165 个训练 episode 对应 165 个不同初始状态；
- 每个初始状态只有一条示范和一个 prototype 标签；
- 没有同一初始状态下 P0/P1/P2 的 paired/counterfactual supervision。

初始状态 leakage audit 的 balanced accuracy 为 logistic regression 39.5%、
random forest 53.0%，说明初始状态与 prototype 有相关性，但不是完全决定关系。
模型仍然可以主要依赖 observation 预测动作，而不需要把 condition 学成因果控制变量。

三个 full dense-trajectory 图也显示 cluster 内部离散且跨 prototype 轨迹明显重叠。
medoid 是 E5 八事件 token 加权距离下的中心，不表示整个 cluster 都具有简单的
left/right/top 硬语义。

## Development checkpoint selection

selection 使用 20 条 development episodes，每条只输入其 GT demonstration 分配出的
oracle prototype；排序键为闭环成功率、mean max reward、更早 step。

| 模型 | 选中 step | Development success | Mean max reward | Open-loop normalized MSE |
|---|---:|---:|---:|---:|
| Mode-BCChunk | 175000 | 10/20（50.0%） | 0.8506 | 0.03322 |
| Mode-Diffusion | 125000 | 14/20（70.0%） | 0.9563 | 0.03834 |

selection 只负责从现有 checkpoint 中选择任务表现最好的参数；它不能证明被选模型具有
condition controllability。

## Held-out audit 正式结果

### 全任务 condition 指标

每个 condition 都在相同的 21 条 audit episodes 上执行。Mode-Diffusion 使用
DDPM-100、`policy_seed=0`；同 episode、同 replan index 的 P0/P1/P2/NULL 使用相同
initial noise 和 reverse noise stream。

| 模型 | Requested | Success | Wilson 95% | Classifiable | Adherence | Controlled success |
|---|---|---:|---:|---:|---:|---:|
| BC | P0 | 5/21（23.8%） | [10.6%, 45.1%] | 76.2% | 19.0% | 4.8% |
| BC | P1 | 8/21（38.1%） | [20.8%, 59.1%] | 76.2% | 23.8% | 19.0% |
| BC | P2 | 8/21（38.1%） | [20.8%, 59.1%] | 66.7% | 19.0% | 14.3% |
| BC | NULL | 11/21（52.4%） | [32.4%, 71.7%] | 71.4% | N/A | N/A |
| DP | P0 | 10/21（47.6%） | [28.3%, 67.6%] | 76.2% | 14.3% | 9.5% |
| DP | P1 | 11/21（52.4%） | [32.4%, 71.7%] | 76.2% | 28.6% | 19.0% |
| DP | P2 | 7/21（33.3%） | [17.2%, 54.6%] | 76.2% | 19.0% | 14.3% |
| DP | NULL | 11/21（52.4%） | [32.4%, 71.7%] | 71.4% | N/A | N/A |

P0/P1/P2/NULL 的 BC–DP 配对差异在 21 条样本上均不显著；exact McNemar p 分别为
0.180、0.453、1.000、1.000。这里不能根据点估计把某个 condition 宣称为稳定优势。

### Oracle 子集与 open-loop

oracle 子集是 168 条 sweep 中 `requested condition == GT mode` 的 42 条，不是额外
rollout。

| 模型 | Oracle task success | Oracle adherence | Oracle controlled success | Normalized MSE | Denormalized MSE | Closed-loop replan p95 |
|---|---:|---:|---:|---:|---:|---:|
| Mode-BCChunk | 7/21（33.3%） | 38.1% | 23.8% | 0.01333 | 804.90 | 10.22 ms |
| Mode-Diffusion | 12/21（57.1%） | 33.3% | 28.6% | 0.01315 | 793.59 | 608.93 ms |

两种模型的 held-out open-loop action error 很接近。Mode-Diffusion 的额外价值主要体现
为当前样本上的闭环任务成功率，而不是更清晰的 prototype adherence。

## Requested → Achieved 混淆矩阵

### 成功且可分类的 rollout

单元格只计 `task success AND classifiable AND achieved prototype=列`；
`Other/Failed` 包含失败和不可分类轨迹。

Mode-BCChunk：

| Requested \ Achieved | P0 | P1 | P2 | Other/Failed |
|---|---:|---:|---:|---:|
| P0 | 1 | 1 | 3 | 16 |
| P1 | 1 | 4 | 3 | 13 |
| P2 | 2 | 2 | 3 | 14 |

Mode-Diffusion：

| Requested \ Achieved | P0 | P1 | P2 | Other/Failed |
|---|---:|---:|---:|---:|
| P0 | 2 | 4 | 3 | 12 |
| P1 | 3 | 4 | 3 | 11 |
| P2 | 3 | 1 | 3 | 14 |

两个矩阵都没有形成清晰对角线。

### 不考虑 task success 的几何分类

Mode-BCChunk：

| Requested \ Achieved | P0 | P1 | P2 | Unclassifiable |
|---|---:|---:|---:|---:|
| P0 | 4 | 5 | 7 | 5 |
| P1 | 6 | 5 | 5 | 5 |
| P2 | 4 | 6 | 4 | 7 |

Mode-Diffusion：

| Requested \ Achieved | P0 | P1 | P2 | Unclassifiable |
|---|---:|---:|---:|---:|
| P0 | 3 | 8 | 5 | 5 |
| P1 | 4 | 6 | 6 | 5 |
| P2 | 4 | 8 | 4 | 5 |

BC 没有完全坍缩到单一 prototype；DP 对 achieved P1 有偏置，但也不是完全坍缩。
更准确的描述是：requested condition 对 achieved geometry 的控制弱，DP 存在默认
P1 倾向。

## 不可分类轨迹与失败形态

在 P0/P1/P2 的 63 条显式条件 rollout 中：

| 模型 | Classifiable | Unclassifiable | Sketch 不可提取 | Sketch 可提取但超出所有 radius |
|---|---:|---:|---:|---:|
| Mode-BCChunk | 46/63 | 17/63 | 2 | 15 |
| Mode-Diffusion | 48/63 | 15/63 | 3 | 12 |

大多数不可分类轨迹不是姿态提取器失效，而是完整几何轨迹落在所有 train prototype
拒识半径之外。这进一步说明模型产生了 out-of-prototype behavior，而不是只在三个
离散 prototype 之间切换。

## 固定噪声与条件传递诊断

代码和最小复现确认：

- condition ID 会经过 preprocessor 写入 `prototype_id`，在 rollout 内 latch，并在每次
  replan 重新传给 U-Net；
- P0/P1/P2/NULL 不参与 initial-noise seed 或 reverse-noise seed 计算；
- 同 episode、同 replan ordinal 的四个 condition 使用相同噪声；
- 四行 embedding 权重不同，切换 ID 会改变生成的 action chunk。

episode 13 的探索性单-replan probe 中，固定噪声切换 condition 的 DP action-chunk
RMSE 中位数为 0.027；固定 P0、只切换 DDPM noise 的中位数为 0.040。该单 episode
probe 不是正式总体指标，只说明 condition 影响存在但没有压过普通采样随机性。

当前 artifact 仍有一个审计缺口：每个 replan 的 seed/hash 在运行时记录中存在，但
`closed_loop_rollouts.parquet` 没有持久化这些字段。噪声固定结论已由代码路径和局部
复现验证，但未来正式因果对比应把 per-replan seed/hash 写入结果表并做跨 condition
断言。

## Prototype 可选择性与 GT 可视化

完整训练 demonstration 图位于：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/visualizations/
├── p0_train_gt_trajectories.png
├── p1_train_gt_trajectories.png
└── p2_train_gt_trajectories.png
```

| Prototype | Train members | Radius 内 | Radius 外 | Medoid |
|---|---:|---:|---:|---:|
| P0 | 57 | 55 | 2 | episode 16 |
| P1 | 70 | 67 | 3 | episode 108 |
| P2 | 38 | 37 | 1 | episode 140 |

图中左侧是 agent 相对当前 T 的 dense path，右侧是标准 PushT BEV 下的 agent 与 T
中心路径。三组都存在明显内部离散和跨组重叠。因此 prototype 是描述完整轨迹的聚类
标签，不是已经验证可从任意初始状态执行的行为命令。

要证明 prototype 真正“可选择”，至少需要在相同初始状态下获得多个成功且可分类的
P0/P1/P2 demonstration，定义 state–mode feasibility mask，再只在物理可行的组合上
评价 adherence。当前数据不具备这一证据。

## 必填实验字段

- Task suite：PushT
- Dataset size：206 episodes；165 train / 20 development / 21 audit
- Observation keys：96×96 RGB、二维 agent position；历史长度 2
- Action space：二维绝对 target position
- Action horizon：内部 16；闭环执行 `[1:9]` 共 8 步
- Control frequency：10 Hz
- Train loss：选中 checkpoint 附近日志中，BC 为 `0.000`（日志精度舍入），DP 约
  `0.007–0.009`
- Open-loop action error：BC normalized/denormalized MSE 为 0.01333/804.90；
  DP 为 0.01315/793.59
- Closed-loop success rate：oracle 子集 BC 33.3%，DP 57.1%；全条件结果见上表
- Latency：BC closed-loop replan p95 10.22 ms；DP DDPM-100 为 608.93 ms
- Failure cases：task timeout、指定 prototype 不一致、轨迹超出所有 rejection radius、
  少数 sketch 不可提取

## Pretrain / Provenance

- 两种 E5 模型均为 `train_seed=1000` 控制的随机初始化，没有外部 pretrained
  checkpoint；`verified=true`。
- dataset revision、prototype manifest 和选中权重标识由 selection/audit artifact
  交叉校验；`verified=true`。
- selection 与 audit 的 runtime repository `commit=null`，因此无法把正式结果绑定到
  唯一源码 commit；code provenance `verified=false`。
- simulator 初始状态来自 dataset 第一帧重建，不是完整 Pymunk snapshot replay。

## 结果位置

```text
outputs/pusht_diffusion/pilot/e5_oracle_checkpoint_selection/
outputs/pusht_diffusion/pilot/e5_oracle_audit/
outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/
```

## 结论边界

本次实验支持：

- 离散 condition 已正确进入 BC/DP 模型，并能改变动作；
- Mode-Diffusion 在当前 held-out audit 上保留了较好的任务成功能力；
- 当前 prototype/data/training 组合没有产生可靠的显式模态控制。

本次实验不支持：

- P0/P1/P2 是任意初始状态下都物理可行的命令；
- raw success 差异来自模型遵守 condition；
- NULL 已证明覆盖多个成功且稳定的 prototype；
- Mode-Diffusion 相对普通 Diffusion Policy 有严格同状态性能优势。
