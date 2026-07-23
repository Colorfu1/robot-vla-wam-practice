# 2026-07-24 ACT 进展总结设计

## 目标

新增一份可公开分享的中文进展总结，记录 A-Sanity 收尾和后续三组 ACT 实验指南。离线评测与
可视化只作为支撑背景简要提及，不重复已有专项日志。

正式产物：

```text
reports/2026-07-24-ACT-A-Sanity收尾与后续实验指南.md
```

H1 使用相同日期和主题。

## 总结范围

重点记录：

1. A-Sanity 已完成的 teacher-forced offline 诊断指标与正式 closed-loop 结果；
2. A-Sanity 当前证据限制；
3. 后续 development 数据从 5 条扩展到同任务 20 条、共 3233 帧；
4. A0/A1 Action Chunking、A2 动作差分损失、A3 Temporal Ensemble 的设计；
5. 三份实施指南已经写好，但其中描述的脚本、policy 插件和 evaluator 参数尚未实现；
6. 已有离线评测与 Rerun 可视化为后续实验提供的观察基础。

不重复记录：

- 离线相机标定、坐标转换和 Rerun 布局的完整实现过程；
- 原始预测、动作、视频、RRD、checkpoint 或终端日志；
- 机器、本地缓存、个人笔记库或会话信息。

## 文档结构

H1 后立即写 `## 重要进展`，包含 5 条：

1. A-Sanity 完成及 20k checkpoint 的主要离线诊断指标和正式闭环结果；
2. step-0 B0 和完整训练 loss 未保留这一关键限制；
3. 下一阶段固定 20 demonstrations；
4. 三组实验设计和独立审查完成；
5. 三份实施指南已完成，但实验代码尚未实现。

正文按以下章节组织：

```text
## 总结范围
## A-Sanity 实测结果
## 后续实验设计
## 已完成的实验指南
## 离线评测与可视化基础
## 当前限制
## 下一步
```

## 可记录的实测结果

A-Sanity 使用公开 LIBERO Object task 8、5 条 demonstrations、859 帧，ACT 配置为
`chunk_size=20`、`n_action_steps=5`、`use_vae=false`。

允许记录以下 aggregate result：

| 指标 | 结果 |
|---|---:|
| 20k first-action MAE | 0.02076 |
| 20k delta-action MAE | 0.00972 |
| 20k teacher-forced action jerk | 0.01146 |
| Closed-loop success rate | 18/20（90%） |
| Closed-loop action TV | 0.02195 |
| Closed-loop action jerk | 0.02210 |
| Model-query latency p50/p95 | 18.28/24.47 ms |

100-step checkpoint 只能作为近似参考，不能标为 B0。训练 loss 曲线和零更新 B0 均未保留，
两次 closed-loop 失败也尚未分类。

## 后续实验描述

三组实验使用相同 LIBERO task。Development 阶段固定使用该任务 20 条 demonstrations、
3233 帧，训练默认使用 batch size 8、20,000 updates、seed 1000 和 `use_vae=false`。

- A0/A1：`chunk_size=1, n_action_steps=1` 对比
  `chunk_size=20, n_action_steps=5`；
- A2：在 A1 上增加 `delta_loss_weight=0.1`；
- A3：不重新训练，使用 source-matched Standard、Replan control 和
  Temporal Ensemble `0.01` 做闭环比较。

正式报告使用可点击的仓库相对链接：

- [ACT 单步预测与 Action Chunking 指南](../../../reports/act_action_chunking.md)；
- [ACT 动作差分损失指南](../../../reports/act_delta_action_loss.md)；
- [ACT Temporal Ensemble 指南](../../../reports/act_temporal_ensemble.md)。

必须明确这些是实施指南，不得把尚未运行的 20-demo 训练或评测写成实验结果。

## 离线评测与可视化背景

只用一个简短段落说明现有工具能够：

- 在 demonstration 上做 teacher-forced chunk prediction；
- 展示两路图像、Panda/EEF 3D 几何和 GT/Pred action；
- 展示逐维曲线与 metrics；
- 切换多个 recording。

同时说明 teacher-forced offline prediction 不等于模拟器执行，任务成功必须使用 LIBERO
closed-loop rollout。

## 发布安全

- 只使用仓库相对链接；
- 不写绝对路径、用户信息、session ID、缓存位置或远端凭证；
- 不包含生成物的内容或未忽略路径；
- 已完成、实测、限制和下一步必须分开；
- 不同步外部笔记库，不 push；
- 正式文档写完后检查敏感标记、Markdown whitespace 和事实一致性。
