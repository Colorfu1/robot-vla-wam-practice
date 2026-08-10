# 2026-08-09 ManiSkill M0 development checkpoint selection 结论

## 重要进展

- 在固定的 PickCube-v1 development reset batch（200 条状态）上完成了 `010000` 与 `020000` 两个正式 checkpoint 的闭环选模。
- 评测使用 `physx_cuda`、`pd_ee_delta_pose`、50-step horizon、8 个 GPU worker；每张卡完成 25 条 episode，合计 200 条。
- `020000` 以终态成功率 120/200（60.0%）胜出，`010000` 为 6/200（3.0%），差距为 57 个百分点。
- `020000` 达到 M0 预注册的 development ≥50% 门槛；它已被冻结为后续 test 使用的 checkpoint。
- 本次结果仍是 development 选模结果，不是 test 400 条上的最终泛化结论。
- 检查发现初版 ManiSkill selector 只记录了训练目录路径，没有像既有 LIBERO selector 一样发布
  `checkpoint_selection/selected/` 实体副本；已补上复制逻辑，远端需用 `--select-only` 完成发布。

## 评测口径

- Task suite：ManiSkill `PickCube-v1`。
- Dataset：RL 生成的 `pd_ee_delta_pose` 原生 7D 数据；本次每个候选使用同一批 development 200 条状态。
- Observation：128×128 RGB、21D student state（9D qpos + 9D qvel + 3D goal_pos）。
- Action：7D `pd_ee_delta_pose`；policy `n_obs_steps=2`、`horizon=16`、`n_action_steps=8`。
- Control frequency：20 Hz；最大 episode 长度 50 steps。
- Train loss：本条记录不重新测量训练 loss；选模只使用闭环 success 指标。
- Open-loop action error：本条记录未测量。

## Checkpoint 对比

| checkpoint | success_once | success_at_end | first success | stage signals | termination |
| ---: | ---: | ---: | ---: | --- | --- |
| `010000` | 6/200（3.0%） | 6/200（3.0%） | 6，median 15 steps | grasp 7.0%，placed 3.0% | success 6，truncated 194 |
| `020000` | 120/200（60.0%） | 120/200（60.0%） | 120，median 15 steps | grasp 73.5%，placed 63.0% | success 120，truncated 80 |

两者的 `success_once` 与 `success_at_end` 相同，说明本批中没有“中途成功后终态丢失”的额外差异。`020000` 的成功轨迹首次成功 p90 为 17 steps；`010000` 的 p90 为 33.5 steps。

## 结论与限制

`020000` 是无歧义的 development winner，并通过 `success_at_end_rate ≥ 50%` 的 M0 选模门槛。这个门槛只说明视觉基座在固定 development batch 上已具备可用的闭环能力，不能替代独立 test 泛化评测。

本次远端 rollout 与 worker 合并均完成；首次远端收尾时 selector 的相对仓库根目录解析有 bug，导致 selection JSON 未在远端落盘。修复后使用已有 checkpoint 和 metrics 做了纯离线重排，没有新增 rollout；当前排名 JSON 与评测 metrics 已生成。随后又发现初版 ManiSkill selector 没有发布实体 winner copy，已在代码中修复，远端 `--select-only` 后才算完成冻结产物发布。

Latency 记录中，`020000` policy forward p50 为约 2179.6 ms、p95 为约 2367.7 ms；environment step p50 为约 190.5 ms。该延迟只作当前 8-worker 实现的运行记录，不改变 checkpoint 选择。

## 产物

- `outputs/maniskill/m0_probe/checkpoint_selection/selected_checkpoint.json`
- `outputs/maniskill/m0_probe/checkpoint_selection/checkpoint_results.json`
- `outputs/maniskill/m0_probe/checkpoint_selection/selected/pretrained_model`（修复后由 `--select-only` 发布）
- `outputs/maniskill/m0_probe/checkpoint_selection/evaluations/checkpoint_010000/development/metrics.json`
- `outputs/maniskill/m0_probe/checkpoint_selection/evaluations/checkpoint_020000/development/metrics.json`
- `reports/guides/maniskill/00_m0_pickcube.md`

## 下一步

远端同步修复后执行 `--select-only`，使用 `outputs/maniskill/m0_probe/checkpoint_selection/selected/pretrained_model`
作为冻结策略，在不重新选模的前提下执行 random、privileged PPO 与 visual 三个 system 的 test 400 条评测。
