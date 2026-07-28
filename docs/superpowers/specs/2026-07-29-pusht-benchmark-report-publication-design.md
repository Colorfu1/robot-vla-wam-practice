# PushT Benchmark 阶段性结果发布设计

## 目标

把截至 2026-07-29 已产生的 PushT Pilot benchmark 结果整理成一条可追溯的
实验记录，同时服务项目内复现和 Obsidian 阅读，不改写历史计划为已完成事实。

## 项目仓库

新增 `reports/daily/2026-07-29-PushT-Benchmark阶段性结论.md`，并同步：

- `reports/daily/README.md`
- `reports/README.md`
- `reports/experiment_log.md`

日报采用以下叙事顺序：

1. H1 后立即设置 `## 重要进展`，用 4–6 条要点区分已完成、实测结果、
   证据限制和下一步；
2. 固定实验协议和数据划分；
3. 正式训练与 checkpoint selection；
4. development closed-loop 配对比较；
5. E0 与当前 E1 能支持的结论；
6. 明确统计限制、尚未完成的 E1/E2/E4 和下一步判断标准。

每组结果必须使用 Offline、Closed-loop、Smoke 或 Formal 标签；“阶段性诊断”
只能作为补充状态，不能替代模式标签。日报和 `experiment_log.md` 必须覆盖任务
套件、数据量、观测、动作空间、动作时域、控制频率、训练损失、开环误差、闭环
成功率、延迟和失败案例，尚未测量的字段明确写“未测量”。训练损失、离线 MSE
和闭环成功率不做跨语义直接比较。优先链接仓库内 `metrics.md`，机器可读结果
保留为补充来源。

## Obsidian

新建
`BestPractice/2026-07-29-PushT-Benchmark阶段性结论.md`。核心数值和结论与
项目日报保持一致，但增加标准 Obsidian frontmatter、2–5 个相关双链和前后笔记
导航；同时更新 `index/robotics_papers.md`，并在
`BestPractice/2026-07-27-PushT-BCChunk与Diffusion模型规划和训练.md` 中加入
反向链接。验证所有新增双链仅指向一个现有或本次新增文件。若主题索引已有用户
改动，只按 hunk 暂存本次新增行，不把既有改动混入提交。

## 发布边界

- 不记录用户名、绝对路径、机器名、session ID、缓存位置或完整唯一 ID。
- 不把 development selection-set 表现描述为独立泛化结果。
- 不把旧版 E1 的固定 reverse-noise 结果描述为完整 DDPM 确定性结论。
- 只提交本次文档和索引改动，保留两个仓库中已有的其他未提交内容。
- 项目仓库和 Obsidian 仓库分别提交、分别推送。
- 用户已在本次请求中明确授权 push。推送前仍需分别声明并核查两个仓库的目标
  remote、branch、待推送提交范围和敏感内容。

## 验证

- 检查日报必填实验字段、内部链接和日期索引。
- 核对项目版与 Obsidian 版的核心数值、限制和结论一致。
- 提交前检查 staged diff、敏感路径和个人标识。
- 项目仓库和 Obsidian 仓库均以各自配置的 `origin/main` 为目标；若检查结果
  显示实际目标不同，则暂停推送并说明。
- 推送后确认两个远端分支均包含对应提交。
