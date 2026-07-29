# 7 月 29 日进展总结与双仓库发布设计

## 目标

把 2026-07-29 已完成的 PushT 工作整理为一篇完整、可追溯的 Daily，并生成一份适合
Obsidian 阅读的独立笔记。两个版本共享事实与结论边界，但分别服从实验仓库和知识库的
写作规则。

## 事实来源

只汇总已经落盘并可由机器指标或现有报告支持的内容：

- BCChunk 与 Diffusion Formal 训练和 checkpoint selection；
- E0 demonstration 审计；
- E1 固定 observation 双协议采样；
- E2a/E2b 闭环多 seed rollout；
- E3 采样步数、成功率与延迟结论；
- E4 场景构造和未适配 Diffusion Policy 的 OOD 诊断。

E5 是 7 月 30 日实现，不写入 7 月 29 日已完成进展。未完成的训练、因果解释和正式
held-out 泛化评测只能放入限制或后续工作。

事实源按以下优先级逐项对账：

1. 机器可读 `metrics.json`、Parquet 汇总和 manifest；
2. `reports/experiment_log.md` 中与机器指标一致的最新摘要；
3. guide 和旧 Daily 中的解释性文字。

旧 Daily、guide 或 experiment log 与机器指标冲突时，不能把冲突版本继续当作结果
证据；新总结采用机器指标，并明确旧文档属于早期版本。写作前必须完成训练、E0、E1、
E2、E3、E4 关键数字的对账。

## 仓库 Daily

新增 `reports/daily/2026-07-29-PushT实验进展总结.md`，保留现有 Benchmark 和 E2
专题报告作为详细证据，不覆盖或重命名它们。新 Daily 使用以下结构：

1. H1 后立即放置 `## 重要进展`，包含 4–6 条；
2. 实验链路与统一协议；
3. 训练、选模、E0、E1、E2、E3、E4 的实测结果；
4. 可以支持的结论；
5. 不能支持的结论与风险；
6. 下一步；
7. 相关文档和相对路径。

同步更新 `reports/daily/README.md` 与 `reports/README.md`，使新 Daily 可从两个索引
进入。结果使用规定类别，并把诊断性质写成修饰：`Offline / Formal`、
`Closed-loop / Formal`、`Closed-loop / Smoke` 或 `Closed-loop / OOD diagnostic`。

可点击链接只指向本次发布后真实存在且已跟踪的文档。未进入 Git 的 `outputs/` 证据
位置只以代码格式记录，不能伪装成发布后可点击链接。E2/E3/E4 的旧专题或 guide 如果
不在远端基线上，本次不链接；聚合事实直接完整写入新 Daily。

## Obsidian 适配

新增 `BestPractice/2026-07-29-PushT实验进展总结.md`，不覆盖当前有未提交修改的旧
Benchmark 笔记。适配版：

- 添加知识库要求的 frontmatter；
- 将仓库内 Markdown 相对链接改为真实存在的 Obsidian wikilink；
- 删除无法在知识库中解析的 `outputs/` artifact 链接；
- 保留核心指标、实验逻辑、证据边界和下一步；
- 至少连接 PushT 训练规划、Benchmark 专题笔记和一个知识库导航入口。

冻结 Obsidian 修改集为：

- `BestPractice/2026-07-29-PushT实验进展总结.md`；
- `reading_status.md`，为新笔记提供导航入链；
- `BestPractice/2026-07-27-PushT-BCChunk与Diffusion模型规划和训练.md`，加入强关系
  backlink。

旧 Benchmark 和两个已有用户修改的主题索引不在本次修改集内。

## 校验

- 检查两份笔记的关键数字和结论边界一致；
- 验证 Daily 的相对 Markdown 链接；
- 验证 Obsidian 新增 wikilink 唯一且目标存在；
- 对两个发布 worktree 分别运行并审查 `git diff --cached --check`、
  `git diff --cached --stat` 和每个精确目标文件的 staged diff；
- 扫描 staged patch 中的绝对路径、用户名、凭据和生成制品引用；
- 推送前检查 `git log --oneline origin/main..HEAD` 和完整 outgoing diff。

## 提交与推送

实验仓库和 Obsidian 仓库分别从最新 `origin/main` 创建独立临时 worktree 和
publication branch，在临时 worktree 中只应用本次批准的文件与 hunk。不得 stash、
rebase 或改写原 dirty worktree。

实验仓库 publication branch 只包含已审查的设计 spec、新 Daily 和两个索引 hunk；
Obsidian publication branch 只包含冻结的三个文件。验收时
`origin/main..publication-branch` 只能出现本任务提交，且 outgoing diff 不能包含
未授权的 E2 专题、guide、代码或用户笔记修改。满足后才分别推送 `HEAD:main`；若
远端在最终检查后再次前进，停止并重新基于最新远端构造，不做强推。
