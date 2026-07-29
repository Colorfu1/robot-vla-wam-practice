# PPO 技术笔记重组设计

## 目标

将 Obsidian Vault 中现有的 `RL/PPO.md` 重组为一篇从问题背景到算法实现逐层展开的
中文技术笔记。新版保留完整数学推导、关键直觉、最小例子和 PPO 训练流程，同时合并
主体综述与后续问答中重复出现的 Monte Carlo、log-derivative trick 和 policy gradient
解释。

## 文件范围

- 原文：个人笔记库中的 `RL/PPO.md`
- 新文：个人笔记库中的 `RL/PPO_逻辑重构版.md`
- 索引：个人笔记库中的 `index/ai_fundamentals.md`
- 原文必须保持字节级不变。
- 不重命名或删除现有笔记。
- 新文作为草稿供用户审阅，不替换索引中的正式 `[[PPO]]` 入口；索引只新增一条明确标注
  为“重构草稿”的入链。

## 内容原则

- 保留原文中的技术推导、公式、代码片段和有效直觉。
- 同一个概念只完整解释一次；后续需要复用时使用章节引用。
- 删除对话导出的 Prompt、Response、时间戳、用户信息和重复总结。
- 中文为默认语言，PPO、Actor、Critic、Advantage、GAE 等术语保留英文。
- 修正明显的 Markdown/LaTeX 排版错误，但不在没有依据时扩展新结论。
- 明确区分轨迹级回报、单步 reward、return、value 与 advantage。
- 对 PPO 的 “near on-policy”、clip 作用和 Monte Carlo 估计使用准确而不过度简化的措辞。

## 章节结构

1. PPO 的定位与完整流程
2. 为什么 reward 不能直接反向传播
3. 从期望回报到 Policy Gradient
   - 为什么目标是期望回报
   - Monte Carlo 如何估计期望
   - 为什么不能直接采样估计 `∫ ∇p R`
   - log-derivative trick
   - 从轨迹概率化简到 `log π(a|s)`
4. Actor、动作分布与探索
5. Advantage、Critic 与 baseline
6. Return、TD error 与 GAE
7. PPO clipped surrogate objective
8. Actor loss、Critic loss与 Entropy bonus
9. Rollout、bootstrap 和多轮更新
10. Near on-policy 以及与 DQN、SAC 的区别
11. 常见误解与最小数值例子
12. 完整伪代码和总结

## Obsidian 元数据与链接

- frontmatter 固定包含 `title`、`type`、`topic`、`status`、`importance`、`updated`
  和 `tags`。其中 `type: concept_note`、`topic: reinforcement_learning`、
  `status: draft`、`importance: high`、`updated: 2026-07-27`。
- 主标题只出现一次。
- 文末保留并验证与 `SAC_PPO_compare`、OPD、pi*0.6 / RECAP 的相关笔记双链。
- 新文至少从 `index/ai_fundamentals.md` 获得一个入链，但必须标注为“重构草稿”，不替换
  现有正式 PPO 条目。

## 验证

- 记录原文校验和，并在完成后确认未变化。
- 检查新文只有一个一级标题，标题层级连续。
- 搜索并确认不存在 `Prompt:`、`Response:`、导出时间戳或匿名用户信息。
- 验证所有新增 wikilink 均能解析到唯一文件。
- 运行 `git diff --check` 并检查 Vault 的最终 `git status -sb`。

## 无损补全

用户审阅初稿后要求将原版中被压缩的知识点和数值例子全部补回，同时保持重构版现有的
十二章主结构与连贯语气：

- 第 2 章加入监督学习数据分布不依赖参数、RL 轨迹分布依赖策略参数的公式对照。
- 第 3 章加入 `∫g = E_p[g/p]` 的一般改写、两轨迹 `0.8/0.2` 例子，以及
  `1e-20` 概率的 log 数值稳定性例子。
- 第 4 章加入连续高斯 Actor 的 `mean`、`std` 和采样动作数值例子。
- 第 5 章加入 `V^π(s) = E_{a~π}[Q^π(s,a)]`、离散动作加权平均例子，以及连续动作
  价值曲面为何更难拟合的例子。
- 第 8 章加入离散 entropy 数值例子、可选的 value loss clipping 公式，以及原版提及
  的 PPO 稳定化手段汇总。
- 第 10 章加入完整 DQN Bellman loss 和 `9/10/13` 的 Q 高估数值例子。
- 第 11 章保留现有最小动作例子，不重复复制已经在前文完整讲解的结论。

补回内容必须融入对应概念首次出现的位置，不恢复对话导出结构，也不将相同结论再次写成
独立问答。
