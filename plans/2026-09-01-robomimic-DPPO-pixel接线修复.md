# 2026-09-01 robomimic + DPPO pixel 接线修复

## 状态

- 状态：`done`
- 范围：修复 `residual_robomimic_dppo_square.yaml`、`robomimic_dppo_residual.py`、
  `run_residual_robomimic_dppo.sh` 中五处与上游不一致或会导致结论不可信的接线问题，并补上锁住这些
  行为的测试。本计划**不执行任何实验**，不下载数据，不训练。
- 最后更新：2026-09-01
- 关系：这是
  [2026-09-01 Residual RL 下游改接 robomimic + DPPO pixel](2026-09-01-Residual-RL下游改接robomimic-DPPO-pixel.md)
  的前置修复。该实验计划在本计划完成前不得进入其第 2 步。

## 目标

让本地 adapter 在动作管线、奖励形态、优化器结构上与 DPPO 上游逐项一致，并让评测的开局状态可控、
可配对、可审计。

## 前置条件

- [x] 仓库根目录执行，`PYTHONNOUSERSITE=1` 保持开启。
- [x] `third_party/dppo/dppo` 已 clone 到 commit `cc7234ad7ff39a8f32de3af903606723a16f0648`
  （上游对照的唯一依据）。
- [ ] 本计划只改这四个文件，不动 `third_party/`，不动 Dockerfile：
  - `configs/experiments/rl/residual_robomimic_dppo_square.yaml`
  - `src/robot_practice/experiments/rl/robomimic_dppo_residual.py`
  - `scripts/experiments/rl/run_residual_robomimic_dppo.sh`
  - `tests/experiments/rl/test_robomimic_dppo_residual.py`

## 五处问题与依据

| # | 问题 | 现状 | 上游依据 |
| --- | --- | --- | --- |
| F1 | 评测开局不可控 | `robomimic_dppo_residual.py:482` 是无参数 `env.reset()`；config 的 `random_seed_base: 1001` 从未被读取 | 无（是我们自己的缺陷）；修法沿用 `policy_decorator_eval.py` 的 `contiguous_disjoint_v1` |
| F2 | `residual_bound` 量纲错位 | config 写 `[0.05,0.05,0.05,0.5,0.5,0.5,1.0]`，是从 `square-img.json` 的 controller `output_max` 抄来的物理单位（5 cm / 0.5 rad） | policy 输出在 `[-1,1]`，由 `robomimic_image.py:91 unnormalize_action` 映射；DPPO **无 residual 概念**，无可对齐的 bound |
| F3 | 奖励形态与上游不一致 | config `reward_shaping: true`，且 `robomimic_dppo_residual.py:421` 把它套用到所有环境构造（含 anchor 评测） | 上游 8 个 `env_meta/*.json` **全部** `reward_shaping=False`；12 个 finetune 配置**全部** `best_reward_threshold_for_success: 1`；无任务级差异 |
| F4 | `critic_lr` 从未生效 | 第 279/477 行把 actor 与 critic 塞进一个 `Adam`，统一用 `actor_lr` | 上游 `train_ppo_diffusion_img_agent.py` 有独立 `actor_optimizer` / `critic_optimizer` 与各自 scheduler，分别 `.step()`（第 364/370、400/403 行） |
| F5 | 数据路径未按用途分离 | `paths.train_dataset` 指向上游 100 条那份；anchor 与 pretrain 共用一套 normalization 路径 | 上游 `state_4000.pt` 训于 300 条并配同期 `normalization.npz`；两份 normalization 混用会静默产生错误反归一化 |

关于 F3 的一处口径纠正：上游的 sparse **不是「只在终局给一次」**。`nut_assembly.py` 在
`reward_shaping=False` 时返回 `np.sum(objects_on_pegs)`，**每一步都算**，螺母在柱上的每一步都 +1。
证据是上游处理日志 `Train - Mean/Std episode reward: 5.0367, 0.188`——若只给一次，300 条均值应约为 1。

## 执行步骤

### 1. F1 评测开局可控、可配对

`env.reset()` 改为按显式种子表 reset。种子公式与 `policy_decorator_eval.py` 一致：

```text
reset_seed = RESET_SEED_BASE + 1 + seed_index * episodes_per_seed + episode_index
```

三件事必须同时成立，缺一条就仍可能重演之前那类误判：

1. 种子含 `seed_index`——否则多个 seed 会复用同一批开局状态，池化时把重复样本当独立样本。
2. **重复 `reset_seed` 直接抛错**——否则将来改动会静默把上一条退回去。
3. **base 与 residual 消费同一份 manifest**——先生成一次只读的 `episode_manifest.jsonl`（含
   `episode_key`、`seed_index`、`episode_index`、`reset_seed`），两条臂逐行读同一份；汇总器拒绝
   key 或条数不一致的输入。第 3 条防的是「两条臂在不同题目上比分」，与第 1 条是不同的成因。

汇总时按**状态**而非按 episode 计算区间；`contiguous_disjoint` 下每个状态恰好出现一次，池化二项区间
才成立。

### 2. F2 移除量纲错误的 bound

从 config 删除 `train.residual_bound` 与 `train.residual_scale`，动作合成严格照上游管线：policy 输出
`[-1,1]` → `unnormalize_action` → env。`compose_bounded_residual` 保留但改为**在 residual RL 真正启用
时**才从 config 读 bound；本轮 residual 相关参数一律不预设。

`residual_bound` 与渐进放开调度（`prog_explore`）**推迟到 residual RL 那一轮再定**，不在本计划内。

### 3. F3 奖励与上游一致，并把 anchor 与训练分开

- config `env.reward_shaping` 改为 `false`。
- 环境构造拆成两条路径，各自独立传 `reward_shaping`：anchor 评测**强制** `False`（否则与已发表数字
  不可比，实验计划里「差 >5pp 就停」的判据会失真）；本地训练/评测读 config。
- 在 config 与 metrics 中记录本次实际使用的 `reward_shaping`，使产物自带口径。

### 4. F4 拆成两个独立 optimizer

按上游结构改：`actor_optimizer` 用 `train.actor_lr`、`critic_optimizer` 用 `train.critic_lr`，各自
`zero_grad()` / `step()`。上游还带 `actor_lr_scheduler` / `critic_lr_scheduler`（`CosineAnnealingWarmupRestarts`）
与可选 `eta_optimizer`；本轮只做两个 optimizer 与两个学习率，scheduler 与 eta 记为待办，不静默省略。

### 5. F5 config 按用途分离数据

`paths` 拆成两组，禁止交叉：

```yaml
paths:
  anchor:                     # 只用于复现上游 pixel 基线
    normalization: ${data_dir}/robomimic/square-img/normalization.npz
    base_checkpoint: ${log_dir}/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt
  pretrain:                   # 只用于训练我们自己的 base
    train_dataset: ${data_dir}/robomimic/square-img-mh300/train.npz
    normalization: ${data_dir}/robomimic/square-img-mh300/normalization.npz
```

删除指向上游 100 条 `train.npz` 的路径——它对本实验无用途（RL 在线采样不需要数据集，pretrain 用
重生成的 300 条）。代码在读取时断言两组不互相引用，并把实际使用的路径写进 metrics。

### 6. 补测试

`tests/experiments/rl/test_robomimic_dppo_residual.py` 至少覆盖：

- F1：3 seeds × 50 episodes 产生 **150 个互不重复**的 `reset_seed`；构造重复种子时抛错；base 与
  residual 读同一 manifest 时 `episode_key` 逐行相同，条数不一致时汇总器拒绝。
- F2：config 中不存在 `residual_bound` / `residual_scale` 时仍能构图；动作经 `unnormalize_action` 后
  落在 `[action_min, action_max]` 内。
- F3：anchor 路径构造的环境 `reward_shaping is False`，且不受 config 取值影响；训练路径读 config。
- F4：`actor_optimizer` 与 `critic_optimizer` 是两个不同对象，且 `param_groups[0]["lr"]` 分别等于
  `actor_lr` 与 `critic_lr`。
- F5：anchor 与 pretrain 的路径组交叉引用时抛错。

**每个新测试都必须先注入对应 bug、确认它会 FAIL，再修回。** 只加不验的测试不算完成。

### 7. 验证

```bash
cd "$(git rev-parse --show-toplevel)" && PYTHONPATH=src \
  conda run --no-capture-output -n robot-practice \
  python -m pytest -q tests/experiments/rl
```

```bash
docker run --rm -v "$PWD":/repo -w /repo --entrypoint bash \
  practice:latest -lc \
  'conda run --no-capture-output -n robot-practice python -m ruff check --output-format=concise \
   src/robot_practice/experiments/rl/robomimic_dppo_residual.py \
   tests/experiments/rl/test_robomimic_dppo_residual.py'
```

ruff 必须**不新增**报错项；先记录修改前的基线报错数再比对。

## 判断标准

**执行前写死，看到结果之后不得修改。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| F1 种子唯一性 | 3 seeds × 50 得到 150 个互不重复 `reset_seed`；重复时抛错 | 未达则不允许进入实验计划第 2 步 |
| F1 配对 | base 与 residual 的 `episode_manifest.jsonl` 逐行 `episode_key` 相同；不一致时汇总器抛错 | 同上 |
| F2 | config 无 `residual_bound` / `residual_scale`；动作反归一化后落在 `[action_min, action_max]` | 修实现 |
| F3 | anchor 环境 `reward_shaping is False` 且不受 config 影响；config 默认 `false` | 修实现 |
| F4 | 两个 optimizer 为不同对象，学习率分别匹配 config | 修实现 |
| F5 | anchor / pretrain 路径组交叉引用时抛错 | 修实现 |
| 回归 | `tests/experiments/rl` 全绿；ruff 无新增报错 | 先修再提交 |

## 产物路径

本计划只改源码与测试，不产生 `outputs/` 产物：

```text
configs/experiments/rl/residual_robomimic_dppo_square.yaml
src/robot_practice/experiments/rl/robomimic_dppo_residual.py
scripts/experiments/rl/run_residual_robomimic_dppo.sh
tests/experiments/rl/test_robomimic_dppo_residual.py
```

## 风险与停止条件

- **F1 是最容易做假的一条**。只加种子公式而不加重复检测与共享 manifest，仍会重演之前那类误判
  （两条臂在不同题目上比分、或多 seed 复用同一批题）。三条缺一即视为未完成。
- **不要顺手改 residual 的 bound 或加渐进放开**。那属于 residual RL 那一轮，现在预设会污染后续的
  单变量判断。
- **F3 改成 sparse 后，residual SAC 能否在该信号下学会是未验证的**。本轮只对齐上游；等 residual RL
  启动时必须重新讨论奖励形态，不得把「对齐上游」当成「已验证可训」。
- **不动 `third_party/dppo`**。所有差异都在本地 adapter 侧修，保持上游可比。
- 若发现第六处不一致，先记录再决定是否纳入本计划，不在同一次改动里扩大范围。

## 收尾

- [x] 把本文件 `状态` 改为 `done`
- [x] 更新 `plans/README.md` 索引
- [x] 在实验计划的前置条件里勾掉「接线修复已完成」
