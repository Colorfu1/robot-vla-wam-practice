# 7 月 29 日进展总结与双仓库发布设计

## 目标

把 2026-07-29 已完成的 PushT 工作整理为一篇完整、可追溯的 Daily，并生成一份适合
Obsidian 阅读的独立笔记。两个版本共享事实与结论边界，但分别服从实验仓库和知识库的
写作规则。

## 事实来源

只汇总已经落盘并可由现有报告、指南或指标文件支持的内容：

- BCChunk 与 Diffusion Formal 训练和 checkpoint selection；
- E0 demonstration 审计；
- E1 固定 observation 双协议采样；
- E2a/E2b 闭环多 seed rollout；
- E3 采样步数、成功率与延迟结论；
- E4 场景构造和未适配 Diffusion Policy 的 OOD 诊断。

E5 是 7 月 30 日实现，不写入 7 月 29 日已完成进展。未完成的训练、因果解释和正式
held-out 泛化评测只能放入限制或后续工作。

## 仓库 Daily

新增 `reports/daily/2026-07-29-PushT实验进展总结.md`，保留现有 Benchmark 和 E2
专题报告作为详细证据，不覆盖或重命名它们。新 Daily 使用以下结构：

1. 重要进展；
2. 实验链路与统一协议；
3. 训练、选模、E0、E1、E2、E3、E4 的实测结果；
4. 可以支持的结论；
5. 不能支持的结论与风险；
6. 下一步；
7. 相关文档和相对路径。

同步更新 `reports/daily/README.md` 与 `reports/README.md`，使新 Daily 可从两个索引
进入。结果明确区分 `Offline`、`Closed-loop`、`Formal`、`Smoke` 和 OOD diagnostic。

## Obsidian 适配

新增一篇单独的 BestPractice 笔记，不覆盖当前有未提交修改的旧 Benchmark 笔记。
适配版：

- 添加知识库要求的 frontmatter；
- 将仓库内 Markdown 相对链接改为真实存在的 Obsidian wikilink；
- 删除无法在知识库中解析的 `outputs/` artifact 链接；
- 保留核心指标、实验逻辑、证据边界和下一步；
- 至少连接 PushT 训练规划、Benchmark 专题笔记和一个知识库导航入口。

新增笔记通过 `reading_status.md` 或合适主题索引获得入链；优先选择当前无用户修改的
文件，避免把无关工作纳入提交。

## 校验

- 检查两份笔记的关键数字和结论边界一致；
- 验证 Daily 的相对 Markdown 链接；
- 验证 Obsidian 新增 wikilink 唯一且目标存在；
- 对两个仓库分别运行 `git diff --check`；
- 发布前扫描本次提交内容中的绝对路径、用户名、凭据和生成制品引用。

## 提交与推送

实验仓库和 Obsidian 仓库分别形成独立提交并推送到各自的 `origin/main`。只提交本次
新增总结、必要索引和导航修改；保留两个工作区中已有的其他 staged、modified 和
untracked 内容。推送前 fetch 并检查 outgoing commit；若远端已前进，则先安全保存
未提交工作，再 rebase 本次提交，恢复原工作区状态后推送。
