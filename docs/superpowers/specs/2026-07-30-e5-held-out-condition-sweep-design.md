# 2026-07-30 E5 Held-out 全条件闭环评测设计

## 目标

升级现有 E5 held-out audit，使同一组 21 个 audit episodes 分别在固定输入
P0、P1、P2、NULL 下执行。两个已选模型都覆盖全部四种条件，因此一次正式评测
生成 168 条闭环 rollout。新评测覆盖原有 GT-oracle 评测，并继续使用原入口和
原输出目录。

## 不变接口

- 启动脚本保持为 `scripts/experiments/pusht/eval_e5_oracle_audit.sh`。
- Python 入口保持为
  `robot_practice.experiments.pusht_diffusion.e5_oracle_audit`。
- 默认输出保持为 `outputs/pusht_diffusion/pilot/e5_oracle_audit/`。
- 重新执行正式评测时显式传入 `--overwrite`，原子替换旧结果。
- checkpoint 选择、held-out split、初始状态重建、open-loop 评测和多 GPU
  启动参数保持不变。

## 闭环执行

每个模型 worker 仍按 episode 分片。worker 加载模型后，对分配到的每个 episode
顺序执行 condition ID `0..selected_k`；当前配置对应 P0、P1、P2、NULL。每个
condition 使用相同的 dataset 首帧重建状态、episode length 和 policy seed。
NULL condition 必须取自 sidecar 的 `null_prototype_id`，不得假定它恒为 3。

闭环执行函数必须显式接收 commanded condition，不能继续隐式使用 episode 的
GT prototype。P0/P1/P2/NULL 均通过现有 fixed-prototype processor 传给模型。
每条结果同时保存：

- `oracle_prototype_id/name`：由完整 GT 轨迹得到的分析标签；
- `commanded_condition_id/name`：本次实际输入模型的条件；
- 成功、reward、延迟和 rollout prototype 分类结果；
- `is_oracle_condition`：commanded condition 是否等于 GT prototype。

录制文件按 `model/condition/episode` 分目录，避免四种条件互相覆盖。
NULL rollout 仍计算 nearest prototype、sketch eligibility 和 classifiability 作为
行为诊断，但其 commanded adherence 与 controlled success 在明细和聚合中均为
null/undefined，不能写成 false。

## 可视化身份

每个 Rerun recording 必须在三个位置标明身份：

- 文件目录和 recording name 同时包含模型短名 BC/DP 与 commanded condition；
- recording 内的 metadata 文本明确显示 model、commanded condition、GT mode、
  episode、success；
- 默认 blueprint 直接展示 metadata 面板，打开 recording 后无需从文件名或实体树
  推断当前查看的是哪个模型和条件。

BC/DP 是面向实验阅读者的显示名，原始 `mode_bc_chunk`/`mode_diffusion` 仍保存在
结构化结果中。commanded condition 与 GT mode 必须使用不同字段和标签，避免把
P0/P1/P2/NULL 输入误读为轨迹真值。

## 完整性检查

正式结果必须精确包含：

- 两个模型；
- 21 个 audit episodes；
- P0、P1、P2、NULL 四个 condition；
- 每个 `(model, episode, condition)` 恰好一条记录；
- 每条闭环记录都有非空 Rerun recording；
- recording identity、metadata 字段和默认 blueprint 满足可视化身份契约；
- 每个模型仍只有一条 full open-loop 汇总和 21 条 open-loop episode 记录。

任一 worker、coverage 或 recording 检查失败时，原子输出不得标记为 complete。
多 GPU 进度总数从 42 调整为 168；worker 必须在每个
`(model, episode, condition)` 完成后发出一次 completion event，不能仅按 episode
报告 42 次完成。

## 指标

`metrics.json` 保留 open-loop 和 reconstruction 指标，并提供三种闭环视图：

1. **Condition sweep**：按模型和 commanded condition 汇总全部 21 个任务，
   计算成功数、总数、成功率、Wilson 95% 区间、平均最大 reward 和延迟。
   P0/P1/P2 额外统计 prototype adherence 与 controlled success；NULL 不定义
   commanded-prototype adherence。
2. **Oracle GT**：仅筛选 `is_oracle_condition=true` 的 42 条记录，计算与旧
   audit 相同的全部 legacy oracle metric keys 和分母，使新评测完整包含旧评测。
   GT mode unclassifiable 的 episode 也必须保留在原有总体分母中。
3. **GT × commanded matrix**：按模型、GT prototype 和 commanded condition
   汇总成功数、总数和成功率，用于分析条件控制效果。

`closed_loop_rollouts.parquet` 保存全部 168 条明细。另写结构化的
`condition_metrics.parquet` 和 `gt_commanded_metrics.parquet`，避免仅靠扁平
JSON key 读取分组结果。

## Manifest

`run_manifest.json` schema version 递增，并记录：

- evaluation protocol 为 `held_out_all_condition_sweep`；
- condition IDs/names；
- 每个 episode 的 condition 数；
- 预期与实际闭环 rollout 数；
- oracle 指标由 sweep 子集派生；
- 原有 checkpoint、数据、随机性和运行环境 provenance。

## 验证

针对性测试覆盖：

- 单 episode 生成 P0/P1/P2/NULL 四条结果；
- condition 被显式传给 rollout 与 classifier；
- 168 条精确 coverage、重复/缺失检测和 recording 路径；
- condition、oracle 和 GT×commanded 三层聚合；
- NULL 指标不错误宣称 prototype adherence；
- recording name、metadata 和默认 blueprint 可直接区分 BC/DP、
  P0/P1/P2/NULL 与 GT mode；
- 单卡与多卡 worker 进度和返回条数；
- 启动脚本继续正确转发原参数。

实现阶段只运行 mocked/unit/static checks 和命令预检，不在本机启动正式 held-out
rollout；168 次正式评测由用户在远端执行。
