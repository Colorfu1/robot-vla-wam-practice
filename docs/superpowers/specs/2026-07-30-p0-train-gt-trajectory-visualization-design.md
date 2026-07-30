# 2026-07-30 P0 训练集 GT 轨迹可视化设计

## 目标

生成一张本地静态图片，展示训练集中全部 57 条由完整 GT 轨迹聚类得到的
P0 episode，并突出 P0 medoid episode 16。该图片用于理解 P0 的实际轨迹
形态，不改变训练、评测或 prototype 产物。

## 数据来源

- 从 E5 prototype manifest 读取源轨迹文件位置及 P0 medoid。
- 从 `episode_assignments.parquet` 选择同时满足 `split=train` 和
  `prototype_id=0` 的 episode。
- 从 E0 `trajectories.parquet` 读取这些 episode 的完整逐帧
  `agent_position`、`block_position` 和 `block_angle_rad`。
- 绘图前断言 P0 train episode 数为 57，且 medoid episode 16 位于集合中。

## 图形设计

输出一张横向双面板 PNG：

1. 左侧为 agent 相对当前 T 的完整路径。每帧先将 agent 与 T 中心的差向量
   旋转到 T 的局部坐标系，使不同初始位置和朝向的 episode 可以直接比较。
2. 右侧为标准 PushT BEV 工作区坐标，同时绘制完整 agent 路径和 T 中心路径。

普通 episode 使用低透明度细线。medoid episode 16 使用更粗、更醒目的颜色。
agent 与 T 使用不同颜色和线型；起点、终点及总体运动方向使用不同标记。两侧
均采用等比例坐标并保持标准 BEV 的 Y 轴显示方向。标题注明 P0、57 条训练
episode 和 medoid episode 16。

## 输出与隔离

图片写入：

`outputs/pusht_diffusion/pilot/e5_mode_conditioned/prototypes/visualizations/p0_train_gt_trajectories.png`

该文件位于生成物目录中，不提交到版本控制。绘图使用一次性本地命令完成，不为
单张诊断图新增长期维护脚本。

## 失败处理

以下情况应立即失败且不输出误导性图片：

- manifest、assignment 或源轨迹文件缺失；
- P0 train episode 数不是 57；
- medoid episode 16 不属于 P0；
- 任一选中 episode 缺少有效 agent 或 T 位姿；
- 输出图片写入后无法重新读取。

## 验证

- 验证输入筛选得到 57 个唯一 episode。
- 验证左右面板均包含 57 条 episode 轨迹。
- 验证 medoid episode 16 被单独突出。
- 保存后检查 PNG 存在、尺寸非零且可由图像库读取。
- 人工查看图片，确认图例、坐标方向和轨迹没有明显裁切。
