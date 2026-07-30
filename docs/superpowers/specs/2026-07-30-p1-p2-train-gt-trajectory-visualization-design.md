# 2026-07-30 P1/P2 训练集 GT 轨迹可视化设计

## 目标

分别生成 P1、P2 两张训练 demonstration 轨迹图，用于与已经生成的 P0 图
进行同构比较。这里的 GT 指数据集 demonstration、不是 policy prediction；
T 位姿来自 E0 图像几何提取。

本设计复用
`docs/superpowers/specs/2026-07-30-p0-train-gt-trajectory-visualization-design.md`
中已经审查通过的数据校验、坐标定义、双面板布局、视觉编码、原子输出和验证
规则。除 prototype 参数外不得改变图形语义。

## Prototype 参数

| Prototype | 训练成员 | Radius 内 | Radius 外 episode | Medoid episode |
|---|---:|---:|---|---:|
| P1 | 70 | 67 | 116、125、200 | 108 |
| P2 | 38 | 37 | 205 | 140 |

每个 prototype 都必须由 manifest 的 `canonical_prototypes` 记录和
`episode_assignments.parquet` 重新验证，不能只依赖表中的预期值。

## 输出

分别输出：

- `outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/visualizations/p1_train_gt_trajectories.png`
- `outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/visualizations/p2_train_gt_trajectories.png`

每张图片包含：

1. 左侧：agent 相对当前 T 的完整逐帧路径，使用 P0 规格中的 `_rotate_local`
   等价坐标变换。
2. 右侧：固定 512×512、Y 轴向下的标准 PushT workspace-coordinate BEV，
   同时显示完整 agent 与 T 中心路径。

颜色只编码对象类型，线型和透明度编码 rejection-radius 状态。每张图最后绘制
并突出各自 medoid，只为 medoid 添加起点、终点和方向标记。标题、图例和图注
中的 prototype、成员数、radius 统计和 medoid 必须与当前图片对应。

## 实现方式

使用一个一次性参数化绘图流程依次生成 P1、P2，不新增长期维护脚本。两个
prototype 必须复用同一绘图函数和样式常量，避免视觉尺度或编码漂移。每个输出
在全部输入校验通过后写入同目录临时 PNG，重读验证成功后再原子覆盖目标文件。

P1 与 P2 相互独立：如果任一 prototype 校验或生成失败，不得把该 prototype
的临时文件替换为正式输出；已成功生成并验证的另一张图可以保留。

## 验证

- 校验 manifest、assignment、source trajectories 的字段与 SHA256。
- 对每个 prototype 校验成员数量、inside/outside 数量、异常 episode 和 medoid。
- 对每个 episode 校验 split、逐帧连续性、帧数、位姿有效性和有限坐标。
- 每张图分别校验左侧 N 条 agent-local 路径，右侧 N 条 agent 和 N 条 T 路径。
- 校验 medoid artist、radius 外样式、固定 BEV 坐标范围及 Y 轴方向。
- 保存后重读两张 PNG，检查格式、尺寸和非空内容，再进行人工视觉检查。
