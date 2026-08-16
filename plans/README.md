# 计划目录（plans/）

本目录是**待执行工作**的唯一入口。每份文件描述一件要做的事：目标、前置条件、可直接
复制执行的命令、判断标准、停止条件。

## 职责边界

本目录只管**要执行什么**和**怎么判断做完了**。

- 不写实验结果、指标解读或进展叙述 —— 那属于 `reports/`，规范见 `reports/AGENTS.md`
- 不写设计论证与接口契约 —— 那属于 `docs/superpowers/specs/`
- 不写可复现的实验说明书 —— 那属于 `reports/guides/<实验族>/`

判断放哪里：**还没跑 → `plans/`；怎么跑 → `reports/guides/`；跑出什么 → `reports/`。**

计划文件不承载结论。执行完成后只做两件事：把 `状态` 改为 `done`，并在本文件索引里标注。
结果去哪里登记由 `reports/AGENTS.md` 决定，本目录不重复那套规则。

## 命名与状态

文件名 `YYYY-MM-DD-<清晰中文主题>.md`，日期是**创建日期**，不随执行进度改。H1 必须包含
同样的日期与主题。

`状态` 取以下之一：

- `todo`：尚未开始
- `in-progress`：正在执行
- `blocked`：被明确的前置条件阻塞，必须写出阻塞项
- `done`：已完成
- `dropped`：不再执行，必须写出原因

新增或改名计划时，同步维护本文件的索引表。

## 必填段落

复制 [`TEMPLATE.md`](TEMPLATE.md) 开始写。每份计划必须包含：

1. `## 状态` —— 状态值、负责范围、最后更新日期
2. `## 目标` —— 要回答什么问题或交付什么产物，不超过 5 行
3. `## 前置条件` —— 逐条可检查，写清如何确认已满足
4. `## 执行步骤` —— 按依赖顺序编号，每步给**拷贝即跑**的完整命令
5. `## 判断标准` —— 成功/失败条件与阈值，必须在执行前写死
6. `## 产物路径` —— 命令会写到哪些仓库相对路径
7. `## 风险与停止条件` —— 什么情况下应当中止，不要硬着头皮跑完

只用仓库相对路径。不写用户名、绝对路径、机器名、session ID、token。预期与猜测不得
写成结论。

## 索引

| 计划 | 状态 | 主题 |
| --- | --- | --- |
| [2026-08-03 RF7 零场消融与条件敏感性诊断](2026-08-03-RF7零场消融与条件敏感性诊断.md) | `done` | 离线判定 route field 是否被读取（下一份计划的前置闸门） |
| [2026-08-03 RF7 正式训练与条件遵循评测](2026-08-03-RF7正式训练与条件遵循评测.md) | `in-progress` | PushT E6 7×7 Route Field 选模与条件遵循评测 |
| [2026-08-03 RF7 评测指标修复与重导出](2026-08-03-RF7评测指标修复与重导出.md) | `done` | 修正旋转误差参考角与服从度可判定性口径，离线重导出修正指标 |
| [2026-08-04 C1 多卡可配置与选中 B0 加载修复](2026-08-04-C1多卡可配置与选中B0加载修复.md) | `done` | C1 全局 batch 不变量、GPU 预检、selected B0 发布入口与回归测试 |
| [2026-08-04 C2 与 C3 多卡可配置改造](2026-08-04-C2与C3多卡可配置改造.md) | `done` | C2/C3 复制 C1 的全局 batch 不变量与 GPU 预检，C3 instruction 快照随布局约束 |
| [2026-08-04 C1C2C3 共用 checkpoint 选模入口](2026-08-04-C1C2C3共用checkpoint选模入口.md) | `done` | 共用选模：360 条闭环 + 固定帧开环误差 + 三级排名与实体发布 |
| [2026-08-04 选模开环评测六路并行](2026-08-04-选模开环评测六路并行.md) | `done` | 开环误差按 checkpoint 分派多卡并行，数值与串行逐值一致 |
| [2026-08-04 C1 背景任务回退评测](2026-08-04-C1背景任务回退评测.md) | `in-progress` | B0/C1 背景 test 重跑 + episodes.parquet + 配对 McNemar 回退分析 |
| [2026-08-04 Z0 基线与 C1 目标协议正式评测](2026-08-04-Z0基线与C1目标协议正式评测.md) | `done` | Z0 与 C1 各 210 条目标协议 test，协议一致，C1 canonical 96.7% vs Z0 0% |
| [2026-08-05 C1 遗忘权衡曲线诊断](2026-08-05-C1遗忘权衡曲线诊断.md) | `done` | 完成 A/B 学习率诊断，确认降 LR 可部分缓解背景遗忘但不能恢复到 B0 |
| [2026-08-05 背景数据回放缓解遗忘](2026-08-05-背景数据回放缓解遗忘.md) | `in-progress` | 已完成 k=4/8 探针；按预注册规则待补 k=2/8、k=6/8 并完成权衡汇总 |
| [2026-08-05 Prompt Swap 目标判定修复](2026-08-05-PromptSwap目标判定修复.md) | `done` | 修复参考场景原生目标 reset 判定并完成三模型 Prompt Swap 补测与文档更正 |
| [2026-08-06 Mini-WAM与残差RL三阶段实验计划](2026-08-06-Mini-WAM与残差RL三阶段实验计划.md) | `dropped` | 已由 2026-08-11 完整母计划取代，不再执行 |
| [2026-08-06 M0接通ManiSkill端到端探针](2026-08-06-M0接通ManiSkill端到端探针.md) | `done` | PickCube-v1 端到端探针已完成；结果与后续路由见 `reports/` |
| [2026-08-10 对齐论文的PickCube四维基线](2026-08-10-对齐论文的PickCube四维基线.md) | `done` | motionplanning + 4D + CPU + horizon 100 的论文参照已完成；结果见 `reports/` |
| [2026-08-11 Mini-WAM与Residual-RL-完整实验总计划](2026-08-11-Mini-WAM与Residual-RL-完整实验总计划.md) | `in-progress` | 当前先收口 M1-W，再按 M2-W→W0→W1 推进 W 线；Residual 线独立，不 gate M2-W |
| [2026-08-11 M1三任务数据与控制协议预检](2026-08-11-M1三任务数据与控制协议预检.md) | `dropped` | 三任务同栈路线终止；StackCube 由新 M1-W 接替，Peg/Plug 不进入 MS3 正式数据线 |
| [2026-08-11 PegInsertionSide M1 继续诊断](2026-08-11-PegInsertionSide-M1-继续诊断计划.md) | `done` | P1 exact replay、P2 controller contract、P3 精密接触分叉诊断完成，作为分线证据保留 |
| [2026-08-12 M1 WAM 数据与控制协议预检](2026-08-12-M1-WAM数据与控制协议预检.md) | `done` | StackCube baseline/repair/final live RGB、D_core、LeRobot、state banks 与 frozen protocol 全部通过 |
| [2026-08-13 M1-R PolicyDecorator MS2/v2 协议预检](2026-08-13-M1-R-PolicyDecorator-MS2v2协议预检.md) | `blocked` | 用户暂缓；待切换 `maniskill2-pidec` 环境后继续 full replay、D_core 与 state banks |
| [2026-08-13 M2-W StackCube Action-only Flow 基线与 Headroom](2026-08-13-M2-W-StackCube-Action-only-Flow基线与Headroom.md) | `in-progress` | M1-W frozen protocol 已通过；实现、训练、development 选模并冻结 Action-only Flow；其后先 W0、再 W1 |
