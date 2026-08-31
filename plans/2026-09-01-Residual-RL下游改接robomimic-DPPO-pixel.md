# 2026-09-01 Residual RL 下游改接 robomimic + DPPO pixel

## 状态

- 状态：`in-progress`
- 范围：接续 [2026-08-28 Policy Decorator 复现计划](2026-08-28-PolicyDecorator-ManiSkill2-Residual-RL复现.md)
  的 ManiSkill2 结果，把下游 residual RL 对照改到 robomimic `square`（`NutAssemblySquare`）的
  **pixel** 工作点。本计划只负责到「frozen base vs frozen base + bounded residual，含 Zero/Random
  对照」为止；DPPO 全参数微调只作为外部锚点复现，不做算法排名。
- 最后更新：2026-09-01

## 目标

1. 在一个**无特权观测**的 pixel 工作点上重做 ManiSkill2 那轮的 residual 对照：观测只有相机图像 +
   机器人自身 proprioception，不含物体真值位姿与目标位置。
2. 交付同一 frozen base 下的四行对照：Frozen Base、+ Bounded Residual、Zero Residual、
   Random Bounded Residual。
3. 复现 DPPO 已发表的 pixel 微调数字作为外部锚点，用于判断本地 harness 是否可信。

## 任务与工作点

`NutAssemblySquare`（robosuite `1.4.1`，单臂 Panda、`OSC_POSE`、20 Hz）：桌面有方形与圆形两根立柱和
一个方形螺母，螺母的初始位置与绕 z 轴朝向都随机；要抓起方螺母套到方柱上并滑到底。成功判定要求
三条同时成立——`|nut_x − peg_x| < 0.03`、`|nut_y − peg_y| < 0.03`、`nut_z < table_z + 0.05`（必须到柱底
而非悬空），且 `r_reach < 0.6` 即夹爪须撤离螺母柄 4.2 cm 以上（不能握着宣称成功）。难点是**方孔套
方柱要求 xy 位置与 yaw 朝向同时对准**，而初始 yaw 随机。

DPPO 侧的固定合同（全部读自 commit `cc7234ad7ff39a8f32de3af903606723a16f0648` 的配置）：

| 项 | 值 |
| --- | --- |
| pixel 配置 | `cfg/robomimic/{pretrain,finetune,eval}/square/*_diffusion_mlp_img.yaml` |
| 观测 | 1 路 `agentview_image` `[3,96,96]` + `state [9]` |
| `low_dim_keys` | `robot0_eef_pos` / `robot0_eef_quat` / `robot0_gripper_qpos` |
| 动作 | `[7]`，`horizon_steps = act_steps = 4` |
| base checkpoint | `square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint/state_4000.pt` |
| 去噪 | `denoising_steps=100`、`use_ddim=True`、`ft_denoising_steps=5` |
| 预算 | `n_train_itr=301 × n_envs=50 × n_steps=400 × act_steps=4` = **24.08M env steps** |
| episode | `max_episode_steps=400` |

## 为什么是 Square 而不是 Transport

- **dense reward 只有 Square 有**。`nut_assembly.py` 的 `staged_rewards()` 实现了 reach/grasp/lift/hover
  四阶段；`two_arm_transport.py` 的 `reward()` 里 `reward_shaping=True` 会被**强制改回 False** 并打印
  `Warning! No dense reward current implemented for this task`，只剩「payload 进目标箱 **且** 垃圾进
  垃圾桶」的终局二元奖励，800 步 episode 内连子任务里程碑都没有。
- **单臂 7D 可直接沿用 ManiSkill2 那轮的 residual 配方**；Transport 是双臂 14D，两臂处于不同任务阶段，
  bound 需重新设计。
- 预算约 27.5 h 对 48.6 h（按上游 Table A4 的 `328.5 s/iter` × 301 与 `871.3 s/iter` × 201 外推）。
- 代价：Square 比 Transport 简单，headroom 可能更小。这由第 2 步的实测基线决定，不预设。

## 前置条件

### 已完成的预检（结论落盘在 `outputs/rl_residual_robomimic_20260901/preflight/gates.json`）

- [x] **G1 数据变体**：`square/` 下确有 `*_ph` 变体，但都属于 Cal-QL / IBRL / `ta1` 那几条 baseline
  对照（`pre_diffusion_mlp_ta1_ph.yaml`、`calql_mlp_offline_ph.yaml`、`ibrl_mlp_ph.yaml`）。
  **本计划用的 pixel 主线不带 ph 后缀**，数据路径是 `${DPPO_DATA_DIR}/robomimic/square-img/train.npz`，
  是 DPPO 自己预处理的 npz。`cfg/pretraining.md:8` 只说用了 "more than enough expert demonstrations"。
  → **Diffusion Policy 论文的 Square-PH/MH 数字不能直接当同口径参照**；唯一开放项是下载后数 npz 的
  episode 数。
- [x] **G2 pixel 配置名**：`pre_diffusion_mlp_img.yaml` / `ft_ppo_diffusion_mlp_img.yaml` /
  `eval_diffusion_mlp_img.yaml`，另有 unet 与 gaussian 的 img 变体。
- [x] **G3 观测无泄漏**：pixel config 的 `low_dim_keys` 只有三个自身状态键，`image_keys` 只有
  `agentview_image`；pretrain config 第 15 行上游自注 `obs_dim: 9  # proprioception only`；
  `env/gym_utils/wrapper/robomimic_image.py:64` 是 `obs_keys = low_dim_keys + image_keys`，循环只读
  白名单，`object` 进不来。
  **注意**：`cfg/robomimic/env_meta/square-img.json` 里 `use_object_obs: true`，底层 robosuite 仍在算
  物体观测，泄漏是靠 wrapper 的 key 白名单挡住的。任何人改 `low_dim_keys` 泄漏就回来。
- [x] **G4 reward 形态**：`square-img.json` 里 `reward_shaping: False`，DPPO 默认跑 sparse
  （`best_reward_threshold_for_success: 1`）。但 Square 的 dense **已实现**，翻开关即生效。
- [x] **checkpoint 可自动下载**：`script/download_url.py:417` 列的正是上表那条 `state_4000.pt`；
  `script/run.py:14-18` 用 `gdown.download_folder`。**源是 Google Drive，有限流；离线机器需自建镜像。**

### 环境与执行前提

- [ ] 仓库根目录执行，`PYTHONNOUSERSITE=1` 保持开启。生成物只写 `datasets/`、`outputs/` 或
  ignored `third_party/`，不纳入版本控制。
- [ ] DPPO 环境按上游装：`conda create -n dppo python=3.8`、`pip install -e .[robomimic]`，并按
  `installation/install_mujoco.md` 装 MuJoCo。**不复用** `robot-practice` / `maniskill2-pidec` 环境。
- [ ] **不要执行 `script/set_path.sh`**——它是交互式的（`read -p`）且会往 `~/.bashrc` 追加 export。
  改为显式 `export DPPO_DATA_DIR=... DPPO_LOG_DIR=...`；无 WandB 时命令行传 `wandb=null`。
- [ ] CPU 线程数够跑 `env.n_envs=50`。不够时按上游说明降 `env.n_envs` 并同比升 `train.n_steps`，
  使 `n_envs × n_steps × act_steps` 不变，并把实际值写进 resolved config。
- [ ] `reward_shaping=True` 已在 resolved config 中显式开启（见「reward 设计」一节）。

## reward 设计

**只用 staged dense**（`reward_shaping=True`）。

```python
reward = objects_on_pegs + max(r_reach, r_grasp, r_lift, r_hover)

r_reach = (1 - tanh(10.0 · d_夹爪→螺母柄)) × 0.1          # [0, 0.1]
r_grasp = int(检测到抓握) × 0.35                            # {0, 0.35}
r_lift  = 0.35 + (1 - tanh(15.0 · z_距目标高度)) × 0.15     # [0.35, 0.5]，未抓握为 0
r_hover = r_lift + (1 - tanh(10.0 · d_螺母→peg_xy)) × 0.2   # [0.35, 0.7]
```

四项取值区间首尾相接，`max` 使其成为分段单调爬升：远离 `0→0.1` → 抓住 `0.35` → 抬升 `0.35→0.5`
→ 移向 peg `0.5→0.7` → 套上并撤离 `≈1.0`，每段内是 `tanh(距离)` 的连续插值。这与 ManiSkill2 那轮
residual SAC 所用的 reward 形态同类，因此 `res_scale` / `prog_explore` 配方可以复用。

DPPO 的 `square-img.json` 默认 `reward_shaping: False`，本计划必须显式开启；**第 2 步复现上游锚点时
例外**——那一步必须用上游默认（sparse），否则与已发表数字不可比。

reward 允许读仿真真值（螺母/peg 位姿、抓握判定）——它是训练信号不是观测。**条件 3 约束的是观测**，
residual 的观测里不得出现这些量。

## 统一实验合同

沿用 ManiSkill2 那轮已生效的三条约束，不重新设计：

- **base 冻结的三重保证**：base 参数不进任何 optimizer；base 前向在 `torch.no_grad()` 下并返回
  numpy（与计算图完全隔离）；训练与评测各记 `base_parameter_hash` before/after，不等即
  `RuntimeError`。
- **reset seed 必须依赖 policy seed**：新增评测入口采用
  `policy_decorator_eval.py` 的 `contiguous_disjoint_v1` 形式，即
  `reset_seed = BASE + 1 + seed_index * episodes_per_seed + episode_index`，并对重复 `reset_seed`
  抛错。汇总区间按**状态**而非按 episode 计算。
- **进度可观测**：任何 long-running 入口每 60 秒内至少向 stdout 与自身 `run.log` 写一次
  `STATUS completed=<n> total=<N>`，结束写 `DONE completed=<N> total=<N>`。

需要新增并冻结：

- `configs/experiments/rl/residual_robomimic_dppo_square.yaml`：DPPO commit、pixel config 名、seed
  列表、`env.n_envs`/`train.n_steps`、`reward_shaping=True`、residual bound、预算、以及本计划
  「判断标准」里的全部阈值。
- `scripts/experiments/rl/run_residual_robomimic_dppo.sh`：`preflight` / `base-eval` / `train` /
  `eval` / `summarize` 五个阶段。
- `src/robot_practice/experiments/rl/robomimic_dppo_residual.py`：把 DPPO 的 frozen diffusion policy
  包成 base，接上 bounded residual actor/critic。
- 对应测试，放 `tests/experiments/rl/`。

## 执行步骤

### 1. 装环境、下载 checkpoint 与数据

```bash
cd third_party/dppo/dppo && git rev-parse HEAD
```

按上游装 `dppo` 环境与 MuJoCo，显式导出 `DPPO_DATA_DIR` / `DPPO_LOG_DIR`。首次跑 eval 或 finetune
时 `script/run.py` 会用 `gdown` 自动下载 checkpoint、数据与 normalization。下载后**数一次
`square-img/train.npz` 的 episode 数**，补上 G1 的开放项。

### 2. 复现上游 pixel 基线（外部锚点，用上游默认 sparse）

```bash
cd third_party/dppo/dppo && python script/run.py \
  --config-name=eval_diffusion_mlp_img \
  --config-dir=cfg/robomimic/eval/square wandb=null
```

不带 `ft_denoising_steps` 时评的是预训练策略，即 frozen base 的上游成绩。记录 episode 数、seed 与
成功率。这一步不产生本地正式结果，只判断 harness 可信，且**成本远低于训练**，必须先做。

### 3. 本地 frozen base 评测（Zero Residual 对照）

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=base-eval \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/base
```

`residual_scale=0` 的输出必须与纯 base 逐步一致（这同时就是 Zero Residual 控制组）。

### 4. residual 接线 smoke

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=train --smoke \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/smoke
```

极短预算跑通「采样 → buffer → 更新 → 存 ckpt」全链路，并断言 `base_parameter_hash` 前后一致、
`zero_residual_action_equal=true`。smoke 不是结果。

### 5. 单 iteration 计时，并实测 base 推理占比

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=train --max-iterations=1 --time-stages \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/timing
```

base 是 diffusion policy，每个动作要跑 5 步 DDIM 穿过 ViT + MLP，比 ManiSkill2 那轮的单次前向贵得多。
必须把 `s/iter` 拆成「env step / base 推理 / residual 更新」三段写进 `timing/metrics.json`，再外推总时长。

### 6. 正式 residual 训练与评测

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=train \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/residual
```

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=eval \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --residual-root=outputs/rl_residual_robomimic_20260901/residual \
  --output-root=outputs/rl_residual_robomimic_20260901/residual_eval
```

### 7. Random Bounded Residual 对照

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=eval --residual-mode=random-bounded \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/random_residual
```

### 8. 汇总与登记

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=summarize \
  --output-root=outputs/rl_residual_robomimic_20260901
```

按 `reports/AGENTS.md` 登记到 `reports/experiment_log.md`，并在 README §6 增加一行；不把 smoke、
timing、第 2 步的上游锚点写成本地正式结果。

## 判断标准

**执行前写死，看到数字之后不得修改。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| G3 观测无泄漏 | pixel config 的观测 key 中不出现 `object` 或任何目标真值 | **立即停止本计划**，换任务的前提不成立 |
| 上游锚点可信 | 第 2 步复现的 pixel base 成功率与 DPPO 已发表值差 ≤ 5pp | 差更多则先查 harness，不进入第 4 步 |
| headroom | 本地 frozen base pooled success 在 `[10%, 90%]` | 记录任务不适合作 residual probe，停止 |
| Zero Residual 一致性 | `residual_scale=0` 的动作与纯 base 逐步一致 | 不一致说明接线错，修完再跑 |
| base 冻结 | 训练与评测前后 `base_parameter_hash` 相同 | 不同即 `RuntimeError`，视为实现缺陷 |
| Residual 增益 | 相对第 3 步 frozen base ≥ **+10pp**，且 ≥3 个 seed 方向一致 | 记为负结果，不调阈值 |
| Random 对照 | Random Bounded Residual 的增益 < Residual 增益的一半 | 若随机也能拿到同等增益，则增益不来自学习，作废该结论 |
| Residual 不长期顶界 | 训练后期动作触界比例 < 20% | bound 设置不当，重设后重跑 |

## 产物路径

- `outputs/rl_residual_robomimic_20260901/preflight/gates.json`
- `outputs/rl_residual_robomimic_20260901/base/metrics.json`
- `outputs/rl_residual_robomimic_20260901/smoke/metrics.json`
- `outputs/rl_residual_robomimic_20260901/timing/metrics.json`
- `outputs/rl_residual_robomimic_20260901/residual/`（checkpoint、`run.log`、`progress.jsonl`）
- `outputs/rl_residual_robomimic_20260901/residual_eval/metrics.json`
- `outputs/rl_residual_robomimic_20260901/random_residual/metrics.json`
- `outputs/rl_residual_robomimic_20260901/summary.json`

均 gitignored。

## 风险与停止条件

- **G1 的开放项**：`square-img/train.npz` 的 episode 数未知。若 demo 数很大，base BC 可能偏高、
  headroom 缩小；这由第 3 步的 headroom 判据裁决，不预设也不因此调阈值。
- **base 推理可能成为瓶颈**：5 步 DDIM + ViT。第 5 步若显示 base 推理占单 iteration 的多数时间，
  先考虑降 `ft_denoising_steps` 或减少 residual 的 base 调用频率，**不要**直接硬跑满预算。
- **时间上限**：外推约 27.5 h 单 seed。第 5 步实测显著高于此时，先减 `n_train_itr` 做短预算版本并
  在结论里标注预算不足，不静默延长。
- **CPU 不足**：`env.n_envs` 降低后必须同比提高 `train.n_steps` 保持每 iteration 采样量不变，
  否则与上游数字不可比；改了要在 resolved config 里记下来。
- **Google Drive 依赖**：`gdown` 有限流，远端无网机器需先在有网机器上建镜像。
- **不得混列**：ManiSkill2 那轮的 `69.40%` / `0.9960` 是 state 工作点，与本计划的 pixel 工作点不同
  harness、不同观测、不同 base，两者的成功率**不可相减、不可并入同一张表**。
- **不做算法排名**：DPPO 全参数微调与 bounded residual 的可训练参数量与预算不同，只能各自相对
  自己的 base 报增益。

## 收尾

- [ ] 把本文件 `状态` 改为 `done`
- [ ] 更新 `plans/README.md` 索引
