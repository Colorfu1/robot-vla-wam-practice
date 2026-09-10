# 2026-08-28 Policy Decorator 在 ManiSkill2 的 Residual RL 复现

## 状态

- 状态：`done`
- 范围：在 ManiSkill2 `PegInsertionSide-v2` 上按 Policy Decorator 官方接口准备并审计 frozen
  Diffusion Policy，做 residual 接线 smoke，并在官方 pinned checkout 上产出 frozen base 与
  `+ residual SAC` 的对照。边界之外的事不在本计划内。
- 最后更新：2026-09-01
- 后续路径：Residual RL 的下游由
  [2026-09-01 Residual RL 下游改接 robomimic + DPPO pixel](2026-09-01-Residual-RL下游改接robomimic-DPPO-pixel.md)
  承担。本计划早期版本还包含另外两个阶段（自建 Light-WAM 的 residual 对照、RLinf 的 GRPO/PPO
  复现），二者已取消并从本文件移除，不再作为待执行项。

## 目标

在 ManiSkill2 `PegInsertionSide-v2` 上，用同一个 frozen 官方 Diffusion Policy checkpoint 回答：
bounded residual SAC 相对 zero-residual base 有多少增益，且该增益在同一条上游 `evaluate()`
路径内可相减。

## 前置条件

- [x] 仓库根目录执行，`PYTHONNOUSERSITE=1` 保持开启；生成数据、模型、视频和日志只写入
  `datasets/`、`outputs/` 或 ignored `third_party/`，不纳入版本控制。
- [x] Policy Decorator checkout 为 `92ba9ba442587ae5989c286355ace2200a0537fb`，ManiSkill2 为
  `0.5.3`；source/environment audit 与 50/50 native/strict replay smoke 全部通过。
- [x] 官方 Diffusion Policy checkpoint 目录同时包含 `best.pt` 与训练时 `args.json`；下载后先记录
  文件 hash，不以本地重新训练的 base 冒充官方 checkpoint。
- [x] 每个官方 worker 使用单卡 16 env；十个 seed 由外层 launcher 跨多卡并行。

`PegInsertionSide-v2` 不是 `mani-skill2==0.5.3` wheel 自带的注册项，而是 pinned checkout 里
`envs.maniskill_fixed` 注册的 repo-local corrected runtime。正式 run 必须把该 checkout 作为
`--reference-root`/`PYTHONPATH` 并显式导入该模块；不能把原生 `v0` 冒充 `v2`。

## 统一实验合同

- `configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml`：revision、task、seed、horizon、
  action contract、预算和阈值。**文件名沿用历史命名**，因为
  `src/robot_practice/experiments/rl/validation_runner.py` 与 `tests/experiments/rl/` 直接引用它；
  改名需同步改代码与测试，本次不做。
- `scripts/experiments/rl/run_three_stage_validation.sh`：`preflight` / `evaluate` / `summarize` 入口。
- `scripts/experiments/rl/run_policy_decorator_official_base_eval_multigpu.sh`：base 评测的单卡
  worker 并行 launcher。
- `scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh`：base 评测完成后再启动
  官方 SAC 的串行 wrapper。
- `src/robot_practice/experiments/rl/`：episode-level manifest、paired bootstrap、checkpoint hash、
  progress heartbeat 和失败分类。官方结果不依赖这些本地 evaluator。
- 任何 long-running 入口每 60 秒内至少向 stdout 与自身 `run.log` 写一次
  `STATUS completed=<n> total=<N>`，结束写 `DONE completed=<N> total=<N>`。

```bash
PYTHONNOUSERSITE=1 conda run -n robot-practice \
  python -m pytest -q tests/experiments/rl

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=preflight \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root=outputs/rl_three_stage_20260828/preflight
```

`preflight` 必须保存 resolved config、git/Hugging Face revisions、环境包版本、GPU 信息、文件 hash、
seed 列表和 train/development/test 无交集审计；不把机器名或绝对路径写进可分享汇总。

## 执行步骤

可复制执行的命令以配套 guide
[Policy Decorator 在 ManiSkill2 的 Residual RL 复现指南](../reports/guides/rl/policy_decorator_maniskill2_residual_rl.md)
为准；本计划只规定顺序、冻结合同和验收标准。

### 1. artifact 与 runtime audit

按 guide §1 下载官方 checkpoint 并记录 hash，然后跑 audit。audit 只检查本地 checkpoint、
`args.json`、checkout 与运行时合同，不下载数据、不跑 episode。

### 2. frozen base 十 seed 闭环

```bash
bash scripts/experiments/rl/run_policy_decorator_official_base_eval_multigpu.sh \
  --gpus=0,1,2,3 \
  --base-policy-ckpt=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt \
  --output-root=outputs/rl_three_stage_20260828/stage1/official_base_eval_paper10
```

该入口通过 `importlib` 导入 pinned 的 `online.pi_dec_diffusion_maniskill2` 并逐字复用其
`make_env` / `BasePolicy` / `evaluate`，base 行即同一 `evaluate()` 在 `residual_scale=0.0` 下的情形。

### 3. residual 接线 smoke

极短预算跑通「采样 → buffer → SAC 更新 → 存 ckpt」全链路，并断言 `base_parameters_frozen=true`、
`zero_residual_action_equal=true`、base hash 前后一致。smoke 不是结果。

### 4. 官方 residual SAC 正式训练与内置周期评测

```bash
bash scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh \
  --gpus=0,1,2,3 \
  --output-root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac
```

官方脚本自身的周期评测（`eval_freq`、`num_eval_episodes`）是 residual 的唯一结果来源；本地
wrapper/evaluator 输出只能标记为诊断。

### 5. 登记实验记录

完成后立即追加 `reports/experiment_log.md`，必须填写：task suite、dataset size、observation keys、
action space、action horizon、control frequency、train loss、open-loop action error、closed-loop
success rate、latency、failure cases。base 报告 checkpoint hash 与协议来源，不伪造代理指标。

## 判断标准

**执行前写死，看到数字之后不得修改。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| protocol | commit/version/hash/schema 与 50/50 replay smoke 全匹配 | 停止，先修协议 |
| headroom | pooled base success 在 `[10%, 90%]` | 停止，检查 artifact/protocol |
| residual 接线 | `zero_residual_action_equal=true`；base hash 前后不变 | 修接线，不进入正式训练 |
| residual 增益 | 相对 base `>=+10pp`；十个 seed 增益同向（以 4M final checkpoint 计） | 记录负复现，不追加预算、不 sweep |
| observability | 每 60 秒 `STATUS`，结束 `DONE` | 停止任务，先补进度 |

立即停止对应 seed 的情况：NaN/Inf、base hash 改变、OOM 重启三次、simulator crash rate `>1%`。

## 产物路径

```text
configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml
scripts/experiments/rl/run_three_stage_validation.sh
scripts/experiments/rl/run_policy_decorator_official_base_eval_multigpu.sh
scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh
src/robot_practice/experiments/rl/
tests/experiments/rl/
outputs/rl_three_stage_20260828/{preflight,artifacts,stage1}/
```

`outputs/` 下的数据、checkpoint、视频与日志均 gitignored。

## 风险与停止条件

- 官方 checkpoint 或 `args.json` 缺失、hash 不匹配时停止，不用本地重训的 base 顶替。
- residual 不得读取 object pose、success predicate 内部状态等特权信息；训练时使用而部署不可得的
  信号一律不允许进入 residual 观测。
- 禁止根据 formal 结果选任务、调超参、挑 checkpoint 或重新划 seed；任何协议变化都要新建计划。
- 官方 `evaluate()` 用 `eval_envs.reset()` 且不 seed，评测状态是环境 RNG 流上的新鲜抽样，不是冻结
  test state bank。结果不得读成对新物体摆放的泛化。

## 收尾

- [x] 把本文件 `状态` 改为 `done`
- [x] 更新 `plans/README.md` 索引
- [x] required experiment fields 已写入 `reports/experiment_log.md`（结论见 README §6.1）
