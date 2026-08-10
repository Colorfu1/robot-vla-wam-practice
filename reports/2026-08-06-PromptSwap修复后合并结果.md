# 2026-08-06 Prompt Swap 修复后合并结果

## 重要进展

- Z0/C1/C2 的 30 条 Prompt Swap fixed rollout 已分别合并回原始 210 条主题结果目录。
- 主指标统一使用每一步 predicate 的 `first_goal_reached`，不再使用 reset 后的终态作为主判定。
- canonical 与 held-out paraphrase 原始 180 条记录未重跑、未替换。
- 每个模型的 30 条 Prompt Swap episode 均通过 300 步完整性校验。

## 汇总

| Model | First-goal compliance | Terminal-goal compliance | First/terminal disagreement | Full horizon |
| --- | ---: | ---: | ---: | ---: |
| z0_b0_zero_shot | 0/30 (0.0%) | 0/30 (0.0%) | 0 | 30/30 |
| c1_default_non_peft | 29/30 (96.7%) | 27/30 (90.0%) | 2 | 30/30 |
| c2_canonical_lora | 28/30 (93.3%) | 28/30 (93.3%) | 0 | 30/30 |

## 结论

- 旧 Prompt Swap 汇总值因参考场景原生目标在 step 返回前触发 reset 而偏低；合并后的主指标应引用首次达成目标。
- `terminal_goal` 低于 `first_goal_reached` 时，说明策略达成后没有保持目标状态；该差异是保持性诊断，不应重新解释为未达成。
- 这次合并只改变 Prompt Swap 判定结果，不改变 canonical、held-out paraphrase、训练或 checkpoint selection 结论。

## 结果位置

- `z0_b0_zero_shot`：`outputs/smolvla_libero/eval/z0_b0_zero_shot`；合并说明：`outputs/smolvla_libero/eval/z0_b0_zero_shot/prompt_swap_merged.md`
- `c1_default_non_peft`：`outputs/smolvla_libero/eval/c1_default_non_peft`；合并说明：`outputs/smolvla_libero/eval/c1_default_non_peft/prompt_swap_merged.md`
- `c2_canonical_lora`：`outputs/smolvla_libero/eval/c2_canonical_lora`；合并说明：`outputs/smolvla_libero/eval/c2_canonical_lora/prompt_swap_merged.md`

结果类别：Formal / Closed-loop / Test。
