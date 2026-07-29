# PushT E4：同中心纯旋转场景

## 当前状态

`Implemented / Smoke validated / Closed-loop OOD diagnostics completed`

E4 已定义并验证自定义 PushT 场景，并使用原始 PushT 数据训练的 DP-175000 执行了
同中心 5 条和接触点偏移 15 条闭环 OOD 诊断。E4 仍然没有 controller、
demonstration、训练数据或适配后的模型，因此不能据此声称 CW/CCW 策略可行或模型
具有双模态能力。

## 实验目标

构造一个几何上只需旋转即可到达目标的 PushT 场景：

- 初始 T 与目标 T 的几何中心均为 `(256, 256)`；
- 目标相对初始姿态旋转 180°；
- T 的物理质心与面积中心一致；
- T 仍是自由刚体，不增加中心转轴；
- agent 从 T 横线中点外侧、相距 1 px 的位置开始。

## 实验思路

PushT 的 body origin 不在 T 的几何中心，因此不能直接令初始和目标 pose 使用相同
`(x, y)`。E4 先计算 T 的面积中心 `local_y=285/7`，再分别反算初始和目标 body origin。
环境继承原始 PushT 的 observation、二维绝对位置 action、物理仿真、reward 和 success
定义。场景 Smoke 只检查几何、reset、渲染和重合度判定；额外的 OOD 诊断直接复用现有
Diffusion 闭环推理，不把它当作 E4 正式评测。

## Smoke 命令

以下命令可以直接复制执行：

```bash
conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.pusht_diffusion.e4_rotation_only_env \
  --output-dir=outputs/pusht_diffusion/pilot/e4_rotation_only_scene \
  --overwrite
```

对应的定向测试为：

```bash
conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_pusht_e4_rotation_only_env.py
```

## 结果落点

```text
outputs/pusht_diffusion/pilot/e4_rotation_only_scene/
├── scene_preview.png
└── scene_metrics.json
```

- `scene_preview.png`：灰色初始 T、绿色目标 T 和蓝色 agent 的初始画面。
- `scene_metrics.json`：场景常量、中心坐标、初始重合度和精确目标状态的成功判定。

这两个文件是生成物，不进入 Git。

## 判断标准

该 Smoke 只有同时满足以下条件才通过：

- Pymunk `block.center_of_gravity=(0, 285/7)`；
- 初始 T、目标 T 和两个姿态下的世界质心均为 `(256, 256)`；
- 不同 reset seed 得到相同初始状态；
- 初始 agent 与 block 没有接触，表面间距为 1 px；
- 初始 `coverage≈0.3877551020`、`reward≈0.4081632653`，不成功；
- 精确目标姿态 `coverage=1.0`、`reward=1.0`，并返回 `is_success=true` 和
  `terminated=true`；
- 场景没有 Pymunk pivot constraint；
- 预览图尺寸正确且非空。

PushT 沿用以下成功定义：

\[
coverage =
\frac{Area(T_{block}\cap T_{goal})}{Area(T_{goal})}
\]

```text
reward = clip(coverage / 0.95, 0, 1)
success = coverage > 0.95
```

## DP-175000 闭环 OOD 诊断

该诊断使用原始 PushT development selection 选出的 DP-175000。模型没有见过 E4
场景，因此以下结果只能解释为“未适配模型面对新场景时的行为”。

完整执行命令：

```bash
conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.pusht_diffusion.e4_diffusion_ood_rollout \
  --policy-path=outputs/pusht_diffusion/pilot/diffusion/checkpoints/175000/pretrained_model \
  --policy-seeds=0,1,2,3,4 \
  --device=cuda \
  --episode-length=300 \
  --output-dir=outputs/pusht_diffusion/pilot/e4_rotation_only_dp_ood \
  --overwrite
```

实测结果标记为 `Closed-loop / OOD diagnostic`：

| Policy seed | Success | Max reward | Final reward | Steps |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.4597 | 0.4215 | 300 |
| 1 | 0 | 0.5515 | 0.5515 | 300 |
| 2 | 0 | 0.4612 | 0.4222 | 300 |
| 3 | 0 | 0.5858 | 0.5858 | 300 |
| 4 | 0 | 0.5836 | 0.5836 | 300 |

汇总：

```text
success = 0/5
mean max reward = 0.5284
best max reward = 0.5858
all termination reasons = timeout
```

结果落点：

```text
outputs/pusht_diffusion/pilot/e4_rotation_only_dp_ood/
├── metrics.json
├── rollouts.json
├── run_manifest.json
├── recordings/policy_seed_00.rrd ... policy_seed_04.rrd
└── videos/policy_seed_00.mp4 ... policy_seed_04.mp4
```

可视化必须让 shell 展开五个 RRD；当前 `robot-rerun` 不能直接读取目录本身：

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9877 \
  outputs/pusht_diffusion/pilot/e4_rotation_only_dp_ood/recordings/*.rrd
```

## 左下到中心的接触点偏移诊断

该控制场景把初始 T 的几何中心放在 `(128,384)`、角度设为 `-135°`，目标中心保持
`(256,256)`、角度为 `+45°`。两者姿态严格相差 180°，初始和目标 T 不重叠。

只沿初始 T 横杆移动 agent 接触点，block、goal、policy checkpoint、环境 seed 和
policy seeds 均保持不变：

```text
-20 px：左上
  0 px：横杆中心
+20 px：右下
```

完整执行命令：

```bash
for spec in 'left_up -20' 'center 0' 'right_down 20'; do
  name="${spec%% *}"
  offset="${spec##* }"
  conda run --no-capture-output -n robot-practice \
    python -m robot_practice.experiments.pusht_diffusion.e4_diffusion_ood_rollout \
    --policy-path=outputs/pusht_diffusion/pilot/diffusion/checkpoints/175000/pretrained_model \
    --scene=lower-left \
    --contact-offset-px="${offset}" \
    --policy-seeds=0,1,2,3,4 \
    --device=cuda \
    --episode-length=300 \
    --output-dir="outputs/pusht_diffusion/pilot/e4_contact_offset_probe/${name}" \
    --overwrite
done
```

结果标记为 `Closed-loop / OOD diagnostic`：

| 接触点 | Offset | Success | Mean max reward | Best max reward | Mean final reward |
|---|---:|---:|---:|---:|---:|
| 左上 | -20 px | 0/5 | 0.1595 | 0.1648 | 0.1595 |
| 中心 | 0 px | 0/5 | 0.2266 | 0.4110 | 0.1905 |
| 右下 | +20 px | 0/5 | 0.5664 | 0.8441 | 0.5064 |

右下偏移显著提高了重合度，其中 seeds 0、1、3 的 max reward 分别达到
`0.8357/0.8441/0.6671`。但三个条件合计 15 条 rollout 都没有成功；录像中也没有
出现从相反侧接近并反向推动的语义策略。该结果支持“现有模型对接触点和力矩高度
敏感”，不支持“固定 observation 下存在左右双模”。

结果落点：

```text
outputs/pusht_diffusion/pilot/e4_contact_offset_probe/
├── metrics.json
├── reference_demonstrations.json
├── left_up/
├── center/
└── right_down/
```

同时打开 15 条 rollout：

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9879 \
  outputs/pusht_diffusion/pilot/e4_contact_offset_probe/*/recordings/*.rrd
```

## 已有 demonstration 参考筛选

该步骤标记为 `Offline / diagnostic`。E0 的 206 条 demonstration 中存在全局左右接近
行为，但不存在当前 E4 初始状态的严格左右配对数据。使用以下条件筛选动作参考：

- `rotation_first=true`；
- `approach_side` 为 `left` 或 `right`；
- block 几何覆盖率不低于 0.99；
- `max_reward>=0.85`；
- 按初始位置、初始角度和 max reward 与当前场景的组合距离排序。

选出的前三条参考为：

| Approach | Episode | Max reward | 初始位置误差 | 初始角度误差 |
|---|---:|---:|---:|---:|
| Left | 110 | 0.9042 | 42.2 px | 6.1° |
| Left | 132 | 0.8654 | 80.9 px | 33.7° |
| Left | 151 | 0.9238 | 170.2 px | 13.3° |
| Right | 171 | 0.8937 | 92.4 px | 11.2° |
| Right | 134 | 0.9224 | 101.9 px | 16.7° |
| Right | 90 | 0.8845 | 117.2 px | 8.1° |

打开六条参考 Recording：

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state --detach-process --port=9880 \
  outputs/pusht_diffusion/data_inspection/demonstrations/episode_110.rrd \
  outputs/pusht_diffusion/data_inspection/demonstrations/episode_132.rrd \
  outputs/pusht_diffusion/data_inspection/demonstrations/episode_151.rrd \
  outputs/pusht_diffusion/data_inspection/demonstrations/episode_171.rrd \
  outputs/pusht_diffusion/data_inspection/demonstrations/episode_134.rrd \
  outputs/pusht_diffusion/data_inspection/demonstrations/episode_090.rrd
```

其中 episode 110 是最接近当前初始姿态的 left reference；最近的 right reference
episode 171 仍有 92.4 px 位置误差。这些轨迹只适合参考接触和旋转过程，不能直接
作为当前场景的 paired demonstrations。

## 训练数据局部偏置假说（待因果验证）

以下内容是基于已测统计提出的解释假说，不是已经完成的因果结论。

### 已测事实

模型实际使用 165 条 train demonstration。训练集中 `left_direct=12`、
`right_direct=6`，但全部右侧接近轨迹有 53 条、左侧接近轨迹有 29 条，因此不能用
全局的 `left_direct > right_direct` 单独解释模型偏好。

更相关的是初始 block pose 的局部条件分布。对于每条 train demonstration，取第一帧
有效 block pose \((p_i,\theta_i)\)，并与 E4 初始 pose
\((p_{E4},\theta_{E4})=((128,384),-3\pi/4)\) 比较：

\[
d_i =
\frac{\lVert p_i-p_{E4}\rVert_2}{512}
+
\frac{
\left|\operatorname{wrap}(\theta_i-\theta_{E4})\right|
}{\pi}
\]

按 \(d_i\) 排序后，训练数据得到：

```text
最近 10 条：10/10 为同一 net rotation 符号
最近 20 条：20/20 为同一 net rotation 符号
最近 40 条：40/40 为同一 net rotation 符号

最近 40 条：
positive = 40
negative = 0
zero = 0
net rotation 范围 = +21.43° 至 +174.20°
```

固定邻域而非 Top-K 的结果同样是：

```text
位置误差 ≤ 64 px，角度误差 ≤ 30°：3/3 同一符号
位置误差 ≤ 96 px，角度误差 ≤ 30°：7/7 同一符号
位置误差 ≤ 128 px，角度误差 ≤ 45°：18/18 同一符号
```

当前接触点 probe 的 15 条 E4 rollout 也全部采用同一个 `net_block_rotation_deg`
符号，没有出现相反旋转方向。

### 当前解释假说

与 E4 初始姿态接近的训练 demonstration 几乎只提供一个旋转方向，而 E4 的 15 条
随机 rollout 也全部选择该方向。这强烈支持以下解释：

> 原始 PushT 训练数据的局部条件偏置很可能引导了当前模型的旋转方向偏好。

该证据仍是相关性而不是因果证明。E4 对现有模型属于 OOD 场景，Agent 初始接触几何、
仿真动力学以及有限 sampling 数量也可能造成方向偏置。正负角度符号对应屏幕中的
CW/CCW 还需要单独标定，不能只凭数值符号命名。

真正的因果验证仍需在相同初始状态下构造数量平衡、方向相反的 paired
demonstrations，重新训练相同模型，并检查旋转偏好是否随数据平衡而消失。

## 统一实验字段

- Task suite：自定义 PushT E4 rotation-only 与 lower-left-to-center 180° scene
- Dataset size：E4 数据为 0；OOD policy 由原始 PushT 165 个 train episodes 训练
- Observation keys：96×96 RGB、二维 agent position
- Action space：二维绝对 target position，范围 `[0, 512]`
- Action horizon：内部 16；闭环执行 `[1:9]` 共 8 actions
- Control frequency：10 Hz（继承 PushT）
- Train loss：继承原始 DP-175000 训练；E4 不训练
- Open-loop action error：E4 未测量
- Closed-loop success rate：同中心 OOD 为 0/5；接触点 probe 为 0/15；均非正式成功率
- Latency：本次未汇总
- Failure cases：20 条 OOD rollout 均运行 300 步后 timeout；右下接触点能显著提高
  overlap，但未越过 0.95 success threshold

## 限制与后续

场景 Smoke 只证明目标几何和环境判定正确；OOD rollout 只说明现有 DP 没有直接完成新
场景。已有数据只能提供全局左右接近参考，不能替代严格配对数据。后续若继续 E4，
需要在当前场景生成 paired demonstrations，再训练无条件 BCChunk 和 Diffusion。
原 Mode-Conditioned 正对照顺延为
[E5](e5_mode_conditioned.md)。
