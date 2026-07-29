# ALOHA 与 ACT 完整笔记融合设计

## 目标

将 Obsidian Vault 中两篇互补的 ACT 笔记融合成一篇连续、完整、去重的技术笔记：

- `Robot/VLA/ALOHA硬件与ACT算法.md`
- `Robot/VLA/ACT-补充理解-从Action-Chunking到闭环稳定性.md`

新笔记从 ALOHA 硬件与示范数据采集开始，依次解释 ACT 的动作表示、Transformer/CVAE
结构、训练目标、闭环执行方式、论文消融、本地 LIBERO 实验、失败机制修正和相关方法
边界。读者不需要来回切换两篇源文即可建立完整理解。

## 文件范围

- 新建：`Robot/VLA/ALOHA与ACT完整笔记.md`
- 修改：`index/robotics_papers.md`，只新增一条明确标注为融合草稿的入口。
- 两篇源文保持字节级不变，不重命名、不删除。
- 不替换现有 `[[ALOHA硬件与ACT算法]]` 正式索引入口。
- 不修改当前未提交的 PPO 草稿及 `index/ai_fundamentals.md`。

## 元数据

新文件使用：

- `title: ALOHA 与 ACT：从低成本双臂数据采集到闭环稳定性`
- `type: concept_note`
- `topic: robot_imitation_learning`
- `status: draft`
- `importance: high`
- `updated: 2026-07-27`
- 标签覆盖 `aloha`、`act`、`bimanual-manipulation`、`action-chunking`、`cvae`、
  `temporal-ensemble`、`imitation-learning`、`closed-loop-evaluation`。

## 叙事结构

1. 阅读范围与证据层级
2. ALOHA 解决的数据采集问题
3. Observation、Action 与 joint-space 遥操作
4. ACT：从单步动作到连续 Action Chunk
5. 当前观测、Action Query 与时间语义
6. `chunk_size`、`n_action_steps` 与闭环频率
7. ACT Transformer 完整结构
8. CVAE、Diagonal Gaussian、KL 与 `z=0`
9. 训练目标：L1、Padding、ObsEqual、Delta-action loss
10. 推理、重规划与 Temporal Ensemble
11. Release horizon 停滞与闭环固定点
12. 论文结果与本地 LIBERO 实验结果
13. Offline 指标与 Closed-loop success
14. 被实验修正的五个猜想
15. 与 Diffusion Policy、Flow Matching、VLA、World Model、RL 的关系
16. 完整训练/推理流程、优势、局限与结论

## 融合原则

- 保留两篇源文中的公式、关键数值、代码片段、论文结果、本地实验表和结论边界。
- 同一个概念只完整解释一次。后续复用时引用前文，不恢复重复问答或重复总结。
- ALOHA、ACT 论文陈述标记为“论文报告”；本地 A-Sanity、A1/A2/ObsEqual/VAE 等结果
  标记为“本地 LIBERO 单任务实验”。
- 不把单任务、单训练 seed 的本地结果推广为 ACT 的一般性质。
- 对后续实验修正过的猜想，先说明原始观察，再给出更新后的证据与更稳妥结论。
- 保留论文模型结构图与已有来源说明。
- 清理源文中的匿名用户、创建/导出时间、重复一级标题和对话导出痕迹。
- 将补充文档末尾的纯文件名实验记录改为可解析的 Obsidian wikilink。

## 知识覆盖要求

必须覆盖以下知识簇：

- ALOHA leader-follower 硬件、WidowX/ViperX、joint-space mapping、50 Hz 数据接口、
  四路相机、14 维连续关节 action。
- Action Chunking 的 effective horizon、compounding error、连续 action embedding、
  当前观测 reactive policy、Action Query 的相对时间语义。
- `chunk_size` 与 `n_action_steps` 的差异，以及闭环反馈与计划持续性的权衡。
- Temporal Ensemble 的时间对齐、指数权重、与普通 smoothing 的差异、作用和非必要性。
- CVAE posterior、diagonal Gaussian、reparameterization、KL、`[CLS]`、训练期 latent、
  推理期 `z=0` 及可能机制。
- ACT 图像/关节/latent token、Transformer encoder-decoder、训练和推理数据流。
- 固定 horizon padding 权重、ObsEqual、动作差分损失及其准确作用。
- 论文真实任务结果、chunk/CVAE/Temporal Ensemble 消融。
- 本地 A-Sanity 与 20-demonstration 实验表、证据限制。
- Offline teacher-forced 与 closed-loop observation distribution 的差异。
- Release horizon 固定点解释及五个被修正的猜想。
- 与 Diffusion Policy、Flow Matching、VLA、World Model、RL 的职责边界。

## 链接

- 文首说明两篇源文及新文之间的关系，并以 wikilink 链接源文。
- 文末加入 2 至 5 个最高相关笔记链接，并保留必要的实验记录链接。
- `index/robotics_papers.md` 为新文提供入链；正式旧条目保持不变。

## 验证

- 修改前后分别记录两篇源文的 SHA-256，确认未变化。
- 新文只有一个一级标题，标题层级连续。
- 公式与代码围栏成对。
- 不包含 `Prompt:`、`Response:`、匿名用户或导出时间字段。
- 所有新增 wikilink 均能唯一解析到真实 Markdown 文件。
- 逐项核对两篇源文标题映射与知识覆盖清单。
- 运行 `git diff --check`，最终状态只包含新文、robotics 索引和进入本轮前已有的 PPO
  本地改动。
