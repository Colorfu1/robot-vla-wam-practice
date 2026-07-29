# PushT E5 Prototype-ID 条件控制实验设计

## 状态

`Approved for implementation planning / Not implemented / Not executed`

本文冻结 E5 第一阶段的实现边界。它描述计划与判断标准，不记录尚未运行的结果。

## 目标

E5 验证离散的全局轨迹模式条件能否让同一个 PushT 策略，在固定 observation 下选择并
闭环执行对应的行为分支。

实验使用现有 `lerobot/pusht` demonstrations。完整 demonstration 只用于离线发现和解释
prototype；模型第一阶段只接收离散 `prototype_id`，不接收未来轨迹 sketch、逐帧 action
或人工阶段状态。

E5 分开回答三个问题：

1. 现有 demonstrations 是否包含稳定、可解释的全局轨迹 prototype？
2. 给定 prototype ID，BCChunk 和 Diffusion 是否会遵循指定模式？
3. 同一个带 condition-dropout 的 Diffusion checkpoint 输入 `NULL` 后，多个 sampling
   seeds 是否覆盖多个成功 prototype？

第三项必须由 NULL 闭环采样直接验证。条件控制成功本身不能证明无条件分布完整，也不能
反向证明现有 E2 checkpoint 已经学习了完整分布。

## 非目标

- 不向策略输入完整 prototype trajectory sketch。
- 不实现 trajectory-token cross-attention 或自动阶段对齐。
- 不修改安装的 LeRobot 源码。
- 不修改原始 PushT 数据集。
- 不把聚类得到的不同全局轨迹直接写成“相同 observation 下多模”的证据。
- 不在本阶段补采配对 demonstrations；数据覆盖不足时明确记录为结论限制。

## 数据与派生制品

原始数据保持只读。E5 将 prototype 分析结果写到 ignored output：

```text
outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/
├── prototype_manifest.json
├── episode_assignments.parquet
├── prototype_candidates.json
├── leakage_audit.json
└── visualizations/
```

`episode_assignments.parquet` 至少包含：

```text
episode_index
split
prototype_id
distance_to_medoid
eligible
exclusion_reason
```

训练 dataset wrapper 使用 `episode_index` 读取 sidecar assignment，并向 sample 增加整数
`prototype_id`。它不复制、不重写原始数据帧。manifest 必须记录数据 revision、split
hash、特征版本、距离版本、K、medoid episode、cluster size 和制品 hash。

prototype 只在 train split 上拟合。development/audit demonstrations 只能分配到最近的
train medoid，不能参与 K 或 medoid 选择。

ineligible train demonstration 不参与 prototype fit，也不伪造最近 medoid。它的所有
frames 在训练 wrapper 中固定输出 NULL ID，因此仍参与无条件分支训练，但不向任何
conditional prototype 提供监督。manifest 和训练摘要必须分别报告 eligible conditional
frames、condition-dropout NULL frames 和 ineligible-always-NULL frames。development/
audit 中的 ineligible demonstration 保持 unclassified。

## Object-Centric 轨迹表示

离线聚类使用完整 demonstration 的低频物体中心化轨迹表示。它只用于 prototype 发现、
解释、可视化和 rollout 分类，不作为第一阶段模型输入。

每条 demonstration 先使用现有 E0 图像几何恢复得到：

- agent workspace position；
- moving T block position and orientation；
- stable contact mask；
- block motion and task progress。

轨迹转换到初始 T 的规范坐标系。表示必须降低绝对初始平移、旋转、最终公共目标和执行
速度对聚类的影响，并强调：

- agent approach path；
- 第一次稳定接触位置；
- block 中间平移路径；
- block rotation evolution；
- contact/recontact sequence。

实现使用八个事件或进度 token：

1. initial valid frame；
2. frame immediately before first stable contact；
3. first stable contact；
4. first significant block motion；
5. 25% block-motion progress；
6. 50% block-motion progress；
7. 75% block-motion progress；
8. final valid frame。

每个 token 保存：

```text
agent position relative to the current block in the block-local frame
block displacement relative to its initial pose in the initial block frame
sin/cos of block rotation relative to its initial angle
stable-contact flag
normalized episode progress
```

stable contact 沿用 E0 的 codec-tolerant block mask、20 workspace-pixel surface-distance
阈值、binary closing 和最少连续三帧规则。first significant block motion 定义为相对
初始 pose 达到 12 workspace pixels 平移或 15 degrees 旋转的首帧。block-motion
progress 使用相邻 block pose 的累计运动弧长：

\[
\Delta s_t =
\sqrt{
  \lVert p_t-p_{t-1}\rVert_2^2 +
  \left(64\operatorname{wrap}(\theta_t-\theta_{t-1})\right)^2
}
\]

其中长度单位为 workspace pixel，64 是固定的旋转折算半径。25%/50%/75% token 取累计
弧长第一次达到对应比例的帧。不存在 stable contact、first significant motion、完整
四个 progress token 或任一 token block pose 无效的 demonstration 标记为 ineligible，
不以静默填充值参与拟合。若多个帧同时满足同一事件，使用最早帧。

## Prototype 拟合

候选 `K` 固定扫描：

```text
K = 2, 3, 4
```

使用可接收预计算距离矩阵的 K-medoids/PAM。禁止用平均轨迹作为模型 prototype。每个
prototype 必须是一条真实 train demonstration 的 medoid。

pairwise distance 是预注册的加权和：

```text
first-contact geometry     2.0
agent approach path        1.5
block intermediate path    1.0
rotation evolution         1.0
contact sequence           1.5
initial/final-only terms    0.25
```

对 episode \(i,j\)，令 \(a^i_t\) 是 token 的 agent-to-current-block local position，
\(b^i_t\) 是 block-to-initial-block local displacement，\(r^i_t\) 是相对初始角度，
\(c^i_t\) 是 stable-contact bit。六个 raw distance component 固定为：

\[
\begin{aligned}
d_{\mathrm{first}} &=
  \lVert a^i_2-a^j_2\rVert_2,\\
d_{\mathrm{approach}} &=
  \frac{1}{3}\sum_{t=0}^{2}\lVert a^i_t-a^j_t\rVert_2,\\
d_{\mathrm{block}} &=
  \frac{1}{4}\sum_{t=3}^{6}\lVert b^i_t-b^j_t\rVert_2,\\
d_{\mathrm{rotation}} &=
  \frac{1}{4}\sum_{t=3}^{6}
  \left|\operatorname{wrap}(r^i_t-r^j_t)\right|,\\
d_{\mathrm{contact}} &=
  \frac{1}{8}\sum_{t=0}^{7}\mathbf{1}[c^i_t\ne c^j_t]
  \frac{|n^i_{\mathrm{contact-runs}}-n^j_{\mathrm{contact-runs}}|}
       {\max(1,n^i_{\mathrm{contact-runs}},n^j_{\mathrm{contact-runs}})},\\
d_{\mathrm{edge}} &=
  \frac{1}{2}\sum_{t\in\{0,7\}}
  \left(
    \lVert a^i_t-a^j_t\rVert_2+
    \lVert b^i_t-b^j_t\rVert_2
  \right).
\end{aligned}
\]

公式中的 token index 是零起始。每个 component 分别除以所有 eligible train episode
pair 的非零中位数；若某 component 不存在非零 pair，则该 component 从总距离中移除并
记录到 manifest。缩放后按上表权重求和，再除以实际参与 component 的权重和。由于所有
eligible episode 必须有完整 token，不做 pairwise 缺失值插补。

每个 K 使用固定多个初始化 seeds 和 bootstrap 子样本评估：

- silhouette；
- assignment stability；
- medoid stability；
- minimum cluster size；
- prototype 可解释性；
- initial-observation leakage。

聚类算法使用 k-medoids++ 初始化和交替 assignment/medoid update，直到 medoid 不再变化
或达到 300 iterations。完整 train set 固定运行十个 initialization seeds：

```text
20260730, 20260731, ..., 20260739
```

选择 objective 最小的 run；objective 在绝对误差 `1e-12` 内相同则选择排序后 medoid
episode tuple 字典序更小的 run。assignment stability 是十个 full-data runs 的所有
45 对 assignment adjusted Rand index 的中位数。

最终 prototype ID 不继承聚类实现返回的任意 label。将选定 run 的 medoid episode
indices 升序排列，并依次规范化为 ID `0..K-1`；所有 assignment、checkpoint config、
可视化和评测都使用这个 canonical mapping。

bootstrap stability 使用 seed `20260740` 生成 100 个不放回 80% train subsamples。每个
subsample 使用上面十个 initialization seeds 的最优 run。其 assignment stability 是
subsample assignment 与 full-data 最优 assignment 在该 subsample 上的 adjusted Rand
index；报告 100 次的中位数和 5%/95% quantiles。使用 Hungarian matching 将 subsample
medoids 对齐 full-data medoids，medoid stability 报告匹配 medoid 距离的中位数和
5%/95% quantiles。

自动推荐只允许从满足以下硬条件的候选中选择：

- 每个 cluster 至少 8 条 train demonstrations；
- 多个初始化下 assignment adjusted Rand index 的中位数至少 0.8；
- 100 个 subsamples 的 assignment adjusted Rand index 中位数至少 0.7；
- medoid 与 cluster 可视化成功生成。

若多个 K 通过，优先选择较小 K，再由人检查 medoid 行为是否存在清晰差异。工具必须
支持显式 `--num-prototypes=2|3|4` 覆盖推荐值。没有候选通过时停止，不能生成可训练的
complete manifest。任何人工 override 必须在 manifest 保存选择者提供的非空 rationale；
工具本身不记录用户或机器身份。

## 初始 Observation 泄漏审计

对每个候选 K，仅使用最初两帧 observation/state 训练轻量 prototype 分类器，并使用
train 内交叉验证报告：

- majority accuracy；
- logistic-regression accuracy；
- random-forest accuracy；
- balanced accuracy；
- confusion matrix。

该分类器只诊断 prototype 是否主要由初始状态决定。它不能创造同状态多解，也不能代替
配对数据。

如果分类性能接近完美，E5 仍可作为条件可学习性 pilot 执行，但不能将结果解释成严格的
同状态多模态证据。正式结论必须同时报告固定 simulator snapshot 的 counterfactual
condition swap 和 NULL rollout 覆盖。

## Prototype-ID 模型

新增两个仅位于本仓库的策略类型：

```text
mode_bc_chunk
mode_diffusion
```

它们分别继承当前 BCChunk 和 Diffusion 的 observation encoder、action horizon、loss、
noise scheduler 和 action queue 语义。新增参数固定为：

```text
num_prototypes = selected K
mode_embedding_dim = 32
condition_dropout_prob = 0.20
null_prototype_id = K
```

prototype ID 是 categorical integer，不能作为连续 state 做 min-max normalization。
模型使用：

```python
nn.Embedding(num_prototypes + 1, 32)
```

将 embedding 与原有 observation `global_cond` 拼接，再送入原有 U-Net 的 FiLM
residual blocks。condition 在每次 model query 时显式输入；rollout runner 将用户选择的
ID 锁存并在整个最多 300 步 episode 中重复传入，不需要 recurrent memory 或环境阶段
标签。

训练时每个 sample 独立以 0.20 概率把真实 prototype ID 替换为 trainable NULL ID。
dropout 只在 training mode 生效，并使用 PyTorch RNG，以保持分布式训练的标准 seed
语义。第一阶段不使用 classifier-free guidance。

模式模块是本地子类和本地 dataset/runtime adapter。现有 `bc_chunk`、`diffusion`、
E0-E4 checkpoint 和安装的 LeRobot 代码不得改变行为。

## 训练采样

首版保持现有随机 action-chunk 训练语义。所有 frame 使用所属 episode 的 prototype ID。
不向模型输入离线 first-contact phase、APPROACH/PUSH 状态或 prototype token index。

为了避免条件信号只出现在少量早期决策帧，可选的 decision-window weighting 必须作为
后续明确 ablation；它不进入首版主实验，不能在 BCChunk 和 Diffusion 间不对称使用。

模式条件策略必须从共同的 mode-aware step-zero 初始化开始。Mode-BCChunk 与
Mode-Diffusion 的 observation encoder、U-Net 和 embedding tensor 初始值需逐 tensor
一致；仅训练目标和 diffusion sampling 机制按各自模型定义不同。

## 评测

### Offline prototype audit

报告 K=2/3/4 的 cluster sizes、stability、leakage、medoid 可视化和最终选择理由。该结果
只能称为 `Offline`。

### Fixed-observation condition swap

formal observation set 固定复用 E0 在 audit split 预注册的十个 E1 candidates，不允许
根据 E5 结果重新筛选。若 E0 artifact 不存在，评测入口必须失败并提示先执行 E0，不得
自动改选 observation。

固定：

- two-frame observation；
- checkpoint；
- Diffusion initial noise and complete reverse-noise stream；
- inference steps。

只切换 prototype ID。报告实际执行 `[1:9]` action chunk 的跨 prototype 距离、模式内
距离。Diffusion 对每个 observation × prototype 使用 policy seeds `0..19`，不同
prototype 复用相同 seed 和完整 reverse-noise stream；NULL 也使用相同 20 个 seeds。
Mode-BCChunk 对每个 observation × prototype/NULL 运行一次。

fixed-observation action chunk 不直接与完整 demonstration medoid 比较，也不报告
prototype adherence。它只回答 condition 是否在固定 observation/noise 下改变动作。

### Closed-loop controlled rollout

对相同 environment seed 分别输入每个 prototype ID。每个 ID 在整个 episode 中固定，
不由环境自动改变。Diffusion 的不同 prototype 使用相同 policy seed 集合。

Formal grid 固定为：

```text
environment seeds: 2000..2019
Mode-Diffusion policy seeds per prototype: 0..9
Mode-BCChunk rollouts per prototype/environment: 1
maximum episode length: 300
```

因此选定 K 后，Mode-Diffusion controlled 部分恰好有 `20 × K × 10` 条 rollout，
Mode-BCChunk controlled 部分有 `20 × K` 条。减少数量只能标记为 Smoke。正式输出缺失、
重复或 seed 不匹配时必须拒绝汇总。

至少报告：

- task success；
- prototype adherence；
- controlled success；
- agent/block trajectory；
- first stable contact；
- environment seed × prototype 的可行成功数；
- latency and failure cases。

每条 rollout 使用与 train clustering 完全相同的 geometry、token extraction、component
scales 和 medoid distance 分类。对 prototype \(k\)，拒识半径 \(R_k\) 固定为 eligible
train cluster member 到 medoid 距离的 95% empirical quantile，quantile 使用 NumPy
`method="higher"`。定义：

```text
classifiable =
    rollout sketch eligible
    AND nearest-medoid distance <= that medoid's R_k

prototype_adherent =
    classifiable
    AND nearest prototype ID == commanded prototype ID

controlled_success =
    environment task success
    AND prototype_adherent
```

unclassifiable rollout 的 strict prototype adherence 记为 false，同时另报
`classifiable_rate` 和 classifiable 子集上的 conditional adherence，避免用拒识隐藏失败。
cluster medoid 自身距离为零；若某 cluster 的 \(R_k=0\)，保持零阈值并在报告中标记
degenerate radius。

### Closed-loop NULL rollout

同一 mode-conditioned Diffusion checkpoint 输入 NULL ID，用多个 policy seeds 执行相同
environment seed。只有 NULL rollouts 中出现至少两个可分类、成功且与 conditional
prototype 对齐的模式，才支持“该 checkpoint 的 NULL 分支覆盖多个成功模式”。

Formal NULL grid 固定为 environment seeds `2000..2019` × policy seeds `0..9`，恰好
200 条 rollout。对单个 environment，某 prototype 必须至少有 2 个成功且 classifiable
NULL seeds 才算一个 supported NULL mode；该 environment 至少有两个 supported NULL
modes 才记为 `multi_mode_success=true`。主结果报告 20 个 environments 中满足该条件的
数量，不把单个离群 seed 写成模式覆盖。Mode-BCChunk NULL 每个 environment 运行一次，
只作为 deterministic reference。

该结果不能反向替代 E1 固定-observation action-distribution 结论，也不能自动推广到
原始 E2 checkpoint。

## 对照与解释

主矩阵：

| Policy | Condition |
|---|---|
| BCChunk | existing unconditional baseline |
| Diffusion | existing unconditional baseline |
| Mode-BCChunk | prototype ID or NULL |
| Mode-Diffusion | prototype ID or NULL |

Mode-BCChunk 用于判断显式 ID 是否把动作歧义拆成可学习映射。Mode-Diffusion 的 NULL
分支用于测试同一 checkpoint 是否保留可采样模式。可选 shuffled-ID negative control
只在主实现验证完成后加入。

若现有 demonstrations 缺少相同或足够相近 initial state 下的多个 prototype，E5 的
条件控制结果只能作为迁移/组合泛化证据。严格的训练数据多解结论仍需要配对
demonstrations。

## 实现边界与验证

实现应提供：

- prototype 提取、K 扫描、medoid 选择和 leakage audit CLI；
- sidecar assignment loader 和原始数据只读 wrapper；
- Mode-BCChunk/Mode-Diffusion config、policy、保存与加载；
- 单卡和显式 GPU 数的多卡训练 launcher；
- fixed-observation、closed-loop ID 和 NULL 评测入口；
- prototype/rollout Rerun 可视化；
- E5 guide 的完整可复制命令和结果落点；
- 单元测试、checkpoint round-trip、condition dropout、固定 ID 持久性和小型闭环 smoke。

首版实现完成不代表实验结论成立。只有实际训练和正式评测结果可以写入 experiment log
的 measured-result 部分。
