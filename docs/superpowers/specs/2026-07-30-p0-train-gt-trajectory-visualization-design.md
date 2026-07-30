# 2026-07-30 P0 训练集 GT 轨迹可视化设计

## 目标

生成一张本地静态图片，展示训练集中全部 57 条由完整 demonstration 轨迹
聚类得到的 P0 episode，并突出 P0 medoid episode 16。这里的 GT 指数据集
demonstration、不是 policy prediction；T 位姿来自 E0 图像几何提取，不是
simulator 特权状态。该图片用于理解 P0 的实际轨迹形态，不改变训练、评测或
prototype 产物。

## 数据来源

- 从 E5 prototype manifest 读取源轨迹文件位置及 P0 medoid。manifest 必须
  满足 `status=complete`、`schema_version=1`、`selected_k=3`，且 P0 canonical
  记录必须为 `cluster_size=57`、`medoid_episode=16`。
- 从 `episode_assignments.parquet` 选择同时满足 `split=train` 和
  `prototype_id=0`、`training_condition_id=0`、`eligible=true` 的 episode。
  P0 的 57 个训练成员中有 55 个位于 rejection radius 内，另有 2 个仍是
  P0 训练 cluster 成员但会被 held-out classifier 拒绝；两者都纳入图中，
  并对这 2 条轨迹使用不同线型。
- 从 E0 `trajectories.parquet` 读取这些 episode 的完整逐帧
  `agent_position`、`block_position` 和 `block_angle_rad`。
- manifest 中的源轨迹路径相对仓库根目录解析，assignment 路径相对 manifest
  所在目录解析。使用 manifest 中登记的 SHA256 校验两个文件。
- 使用 `episode_index` 精确连接 assignment 和 E0 数据，并再次确认每行
  `split=train`。每集按 `frame_index` 排序；帧号必须唯一、从 0 连续递增，
  且行数等于 assignment 的 `num_frames`。
- 每帧要求 `block_pose_valid=true`、agent/T 坐标均为有限 `(2,)` 数组且角度
  有限。绘图前断言 P0 train episode 数为 57，且 medoid episode 16 位于集合中。

## 图形设计

输出一张横向双面板 PNG：

1. 左侧为 agent 相对当前 T 的完整路径。对每一帧 \(t\)，令
   \(d_t = agent_t - block_t\)，并计算
   \(d_{local,t}=R(-\theta_t)d_t\)，其中
   \(R(-\theta)=\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}\)。
   该公式与 E5 `_rotate_local` 完全一致。左图在 `(0,0)` 绘制固定 T 中心参考点。
2. 右侧为 PushT workspace-coordinate BEV，同时绘制完整 agent 路径和 T
   中心路径。E0 workspace 是 512×512 图像坐标：原点左上、X 向右、Y 向下。
   右图固定 `xlim=(0,512)`、`ylim=(512,0)`，绘制工作区边界，但不伪造源数据
   未记录的环境背景或目标 T 轮廓。

普通 episode 使用低透明度细线；超出 rejection radius 的两个 P0 cluster
成员使用淡紫色虚线。medoid episode 16 最后绘制，使用更粗、更醒目的颜色和
更高层级。左图只为 medoid 添加起点、终点和方向标记，避免 57 组标记遮挡；
右图分别使用蓝色实线表示 agent、绿色虚线表示 T，并同样只标记 medoid 的
起终点和方向。图例只包含语义样式，不为每个 episode 生成条目。两图使用等
比例坐标并保持图像坐标的 Y 向下显示。标题注明
`P0 · 57 train demonstrations · medoid episode 16`，图注说明 medoid 是 E5
八事件 token 加权距离下的中心，不保证在逐帧稠密轨迹图中视觉最居中，也不把
P0 命名成固定的 approach-side 策略。

## 输出与隔离

图片写入：

`outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/visualizations/p0_train_gt_trajectories.png`

该文件位于生成物目录中，不提交到版本控制。绘图使用一次性本地命令完成，不为
单张诊断图新增长期维护脚本。

## 失败处理

以下情况应立即失败且不输出误导性图片：

- manifest、assignment 或源轨迹文件缺失；
- manifest 字段或输入文件 SHA256 不匹配；
- P0 train episode 数不是 57；
- medoid episode 16 不属于 P0；
- 任一选中 episode 的帧号、帧数、split 或位姿不满足上述约束；
- 输出图片写入后无法重新读取。

所有输入断言在写文件前完成。图片先写入目标目录中的临时 PNG，成功重读并
验证尺寸后，再用原子替换覆盖已有目标文件；失败时保留已有图片不变并清理临时
文件。

## 验证

- 验证输入筛选得到 57 个唯一 episode，其中 55 条 classifiable、2 条超出
  rejection radius。
- 验证左侧包含 57 条 agent-local 路径；右侧包含 57 条 agent 和 57 条 T
  中心路径。
- 验证 medoid episode 16 的 artist 样式、绘制层级、起终点和方向标记。
- 验证右图坐标范围固定为 512×512、Y 轴向下，所有 artist 数据有限且位于
  预期坐标范围内。
- 保存后检查 PNG 存在、尺寸非零且可由图像库读取。
- 人工查看图片，确认图例、坐标方向和轨迹没有明显裁切。
