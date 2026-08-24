# 2026-08-13 M2-W StackCube Action-only Flow 基线与 Headroom

## 状态

- 状态：`dropped`
- 原因：本文件冻结的 7D `pd_ee_delta_pose` 正式路线不再执行；M2-W 里程碑已由独立的
  M1-W native-4D 数据 + 4D Action-only Flow 100k 路线完成。结果只保留在 ManiSkill M2 guide，
  后续由 2026-08-17 WAM 接入计划继续。
- 最后更新：2026-08-17

## 目标

在 M1-W 冻结的 ManiSkill3 StackCube GPU/7D/RGB 协议上建立唯一的 Action-only Flow
基线，证明它已掌握任务主体且保留方法 headroom。冻结模型、训练预算、Flow sampler、
checkpoint、`n_action_steps` 和 development 选模规则，作为 W0–W3 的公平对照。

## 顺序决议

W 线的依赖固定为：

```text
M1-W done → M2-W → W0 → W1 → W2 → W3
```

不能从 M1-W 直接跳到 W1。M2-W 先提供 Action-only 对照；W0 再验证 future branch 确实依赖
action tokens。W0 的机制锚点固定为 M2-W 选中的 `Δ=n_action_steps`；只有该锚点通过后，W1
才比较 `Δ∈{1,n_action_steps,H_chunk}`。M1-R/M2-R 属于独立 Residual 线，不再作为 M2-W
的前置 gate。

## 前置条件

- [ ] M1-W 已标记为 `done`，且下面命令返回 `gate_pass: true`：

  ```bash
  bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
    --phase=validate-frozen \
    --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
    --protocol=outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json
  ```

- [ ] frozen protocol 只含 `StackCube-v1`，固定 `physx_cuda`、20 Hz、
  `pd_ee_delta_pose`、7D action、formal horizon 500。
- [ ] LeRobot 数据集为 `datasets/lerobot/maniskill/stackcube_dcore_7d`，manifest 为 987 episodes、
  106528 frames、两路 `128×128` RGB、18D `qpos/qvel` 和 7D action。
- [ ] offline split 固定为 train/development/test = 787/100/100；训练只能读取 train，开环选模
  只能读取 development，test 在 W3 前保持未读。
- [ ] exact state banks 固定为 train/development/test = 2000/200/400，fingerprint 跨 split
  无重叠；M2-W 闭环只读取 200 条 development states。
- [ ] `./scripts/doctor.sh` 通过，且 global batch 256 能被可见 GPU 数量整除。

## 冻结实验合同

### 数据与环境

- 数据：只用 M1-W `D_core` 的 787 条 offline train episodes；不加 policy rollout、失败数据、
  RL demonstrations 或 privileged state。
- 训练读取：从冻结 D_core H5 按需读取窗口，不把 H5 全量载入内存；正式训练每个 rank 使用
  4 个 persistent DataLoader worker，并由 worker 复用只读 H5 句柄。LeRobot 产物继续作为
  经过 hash/feature gate 的转换与审计产物，但本基线暂不把 AV1 视频解码路径作为训练 I/O。
- 输入：最近 2 步的 `base_camera` RGB、`hand_camera` RGB 和 18D `qpos/qvel`；单任务只使用
  一个 learned StackCube task token，不加入 goal pose、object pose、success 或 stage label。
- 输出：长度 16 的原生 7D `pd_ee_delta_pose` action chunk。
- 环境：ManiSkill3 `StackCube-v1`、`physx_cuda`、20 Hz、500 步；成功后不停，主指标为
  `success_at_end`，同时报告 `success_once`、first-success、grasp/on-cube stages、动作与 latency。
- normalization：RGB 转 `[0,1]` 后使用 train-only channel mean/std；18D state 和 7D action
  使用 train-only mean/std，标准差下限 `1e-6`。动作送入环境前反归一化并裁剪到 `[-1,1]`，
  必须报告裁剪比例；development/test 不参与统计量估计。

### Action-only Flow 模型

- 视觉编码器：两路相机、两个 observation steps 共用同一个无预训练 ResNet-18；BatchNorm
  替换为 GroupNorm，使用 32-keypoint SpatialSoftmax，与本仓库 LeRobot Diffusion 的视觉
  归纳偏置保持一致。
- token trunk：`d_model=256`、6 层 pre-norm Transformer、8 heads、MLP ratio 4、dropout 0.1；
  加 camera/time/type embeddings。context 与 16 个 noisy action tokens 做非因果 self-attention。
- action head：每个 action token 线性投影到 7D velocity。M2-W 不创建 future target、future
  decoder 或 auxiliary loss；W0 后续只能在同一 action architecture 上增加 world/future 分支。
- Flow path：`x0 ~ N(0,I)`、`t ~ Uniform(0,1)`、`xt=(1-t)x0+t*x1`、
  `v*=x1-x0`，优化 `MSE(v_theta(xt,t,context), v*)`。
- inference：固定 Euler ODE、10 integration steps、每个 state fingerprint/chunk index 派生
  可复现 Gaussian source-noise seed；不做 classifier-free guidance、candidate reranking 或 Oracle。

### 训练与选模

- training seed：1000；global batch 256；AdamW，learning rate `1e-4`、betas
  `(0.9,0.999)`、weight decay `1e-4`、gradient clip 1.0。
- 训练 50000 optimizer steps；linear warmup 1000 steps 后 cosine decay；AMP bf16；每 10000
  steps 保存可恢复 checkpoint，并将 10000/20000/30000/40000/50000 全部纳入选模。
- 不把训练 loss 当选模主指标。每个候选先在固定 offline-development 窗口与固定 Flow noise
  上算 normalized action MSE，再在相同 200 条 development state bank 上分别评测
  `n_action_steps=4` 和 `8`。
- checkpoint/执行步联合排序：`success_at_end` 降序 → `success_once` 降序 →
  `is_cubeA_on_cubeB` 降序 → normalized action MSE 升序 → amortized policy latency 升序 →
  更早 checkpoint → 更小 `n_action_steps`。任何并列都按该顺序机械消解。
- test split 在 M2-W 不加载、不评测。最终 test 的 checkpoint set 与 paired evaluation
  只由新的 literature-aligned W3 在 W2 完成后冻结。

## 执行步骤

### 1. 收口并验证 M1-W

```bash
bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=state-banks --task=StackCube-v1 \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=finalize \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=validate-frozen \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --protocol=outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json
```

M1-W 前置 gate 已完成；不重复生成 state banks 或 D_core。若复核命令失败，停止 M2-W
实现并回到 M1-W protocol 诊断。

### 2. 实现 M2-W 专用入口并锁定合同

实现以下仓库文件：

```text
configs/experiments/maniskill/m2_w_action_only_flow.json
src/robot_practice/policies/wam_tiny/__init__.py
src/robot_practice/policies/wam_tiny/action_flow.py
src/robot_practice/experiments/maniskill/m2_w_action_flow.py
src/robot_practice/experiments/maniskill/eval_stackcube_action_flow.py
scripts/experiments/maniskill/run_m2_w_action_flow.sh
tests/test_maniskill_m2_w_action_flow.py
tests/test_maniskill_m2_w_selection.py
```

runner 必须拒绝非 M1-W frozen hash、test split、privileged key、非 7D action、非 500 horizon、
global batch 不为 256、Flow solver/steps 漂移和已有不完整输出目录。完成实现后运行：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  pytest -q tests/test_maniskill_m2_w_action_flow.py \
    tests/test_maniskill_m2_w_selection.py \
    tests/test_maniskill_stackcube_gpu_route.py

bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=validate-config \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json
```

### 3. 单卡端到端 smoke

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=smoke \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=1 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/smoke
```

smoke 固定跑 20 个 optimizer steps、保存/加载一次 checkpoint、计算 16×7 Flow loss，并在
development state bank 前 10 条上以 4 steps 和 8 steps 各跑一次短闭环。它只验证 shape、
device、noise determinism、action queue 和 exact-state reset，不记为实验结果。

### 4. 训练唯一正式 Action-only Flow run

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=train \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --num-gpus=8 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000
```

launcher 必须在日志和 manifest 中写明 `per_process_batch=32 × 8 = global_batch=256`。如果目标
机器不是 8 张可见 GPU，先修改子计划与 config 中的卡数/每卡 batch，但 global batch、训练步数
和其余合同不能改变；不得靠 gradient accumulation 静默改 optimizer-step 语义。

### 5. development 参考策略与候选联合评测

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=evaluate-references \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --seed-set=development --num-gpus=8 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/development

bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=select-development \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --seed-set=development --num-gpus=8 \
  --train-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000 \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/development
```

reference 固定为 zero-action 与 iid Gaussian-random action 两条；候选是 5 个 checkpoint ×
2 个 `n_action_steps`。每条都使用同一 200-state 顺序、相同 fingerprint-derived Flow noise、
500 步 horizon，不允许成功即停或 partial reset。

### 6. 冻结并复核最终 baseline

```bash
bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=freeze \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --train-root=outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000 \
  --selection=outputs/maniskill/m2_task_base/StackCube-v1/development/selected_checkpoint.json \
  --output-root=outputs/maniskill/m2_task_base/StackCube-v1/frozen_base

bash scripts/experiments/maniskill/run_m2_w_action_flow.sh \
  --phase=validate-frozen \
  --config=configs/experiments/maniskill/m2_w_action_only_flow.json \
  --protocol=outputs/maniskill/m2_task_base/StackCube-v1/frozen_base/frozen_base.json
```

冻结物必须实体引用唯一 checkpoint，记录 checkpoint SHA-256、config hash、M1-W frozen protocol
hash、dataset/split hashes、normalization stats hash、training seed、Flow solver/steps、选中的
`n_action_steps` 和完整 development ranking。禁止用 symlink 指向可变的 `latest`。

## 判断标准

**以下阈值必须在 development 数字出现前锁定。**

| 判据 | 通过阈值 | 不满足时 |
| --- | --- | --- |
| 数据/协议 identity | 所有 M1-W、dataset、split、state-bank hash 一致；test 读取次数为 0 | 停止，修 protocol |
| Flow 学习 | selected checkpoint 的 offline-development normalized MSE 比 train-action-mean predictor 至少低 10% | 停止，先修模型/normalization |
| 主体掌握 | development `success_at_end ≥ 0.30`、`success_once ≥ 0.40`、ever-grasped `≥0.70`、ever-on-cube `≥0.40` | baseline floor；不进入 W0 |
| 高于无策略参考 | 相对 zero/random 中较强者，paired `success_at_end` 至少 `+0.20`，fingerprint-paired bootstrap 95% CI 下界 `>0` | baseline 不成立；不进入 W0 |
| 保留 headroom | development `success_at_end ≤ 0.90` | 视为 ceiling；另建候选任务计划，不自动切 Plug/Peg |
| action 合法 | finite rate 100%，送入环境前越界裁剪元素比例 `≤1%` | 修 normalization/sampler；不放宽 action bound |
| 实时性 | policy chunk-forward p95 / selected `n_action_steps ≤50 ms` | 先做 profile；架构变化后重跑整个 M2-W |
| 冻结完整性 | `frozen_base.json` 验证通过，checkpoint/config/data/split hash 均可重建 | 不进入 W0 |

M2-W 通过只允许进入 W0，不直接允许进入 W1，也不构成 Mini-WAM 方法结论。

## 产物路径

- `configs/experiments/maniskill/m2_w_action_only_flow.json`
- `outputs/maniskill/m2_task_base/StackCube-v1/smoke/`
- `outputs/maniskill/m2_task_base/StackCube-v1/formal_seed1000/`
- `outputs/maniskill/m2_task_base/StackCube-v1/development/reference_metrics.json`
- `outputs/maniskill/m2_task_base/StackCube-v1/development/candidate_metrics.json`
- `outputs/maniskill/m2_task_base/StackCube-v1/development/selected_checkpoint.json`
- `outputs/maniskill/m2_task_base/StackCube-v1/frozen_base/frozen_base.json`
- `outputs/maniskill/m2_task_base/StackCube-v1/frozen_base/checkpoint/`

## 风险与停止条件

- M1-W 未 frozen、hash 漂移、state-bank leakage 或 exact reset 失败时停止，不用 seed 代替
  fingerprint identity。
- smoke 不通过时不得启动 full training；训练出现 NaN、global batch 漂移或 checkpoint 无法
  deterministic reload 时停止。
- development floor/ceiling 任一触发时，M2-W 记录为负结果或 blocked，不通过修改 gate、查看
  test、切 RL demonstrations 或把 historical Plug/Peg 数据混入来“救” baseline。
- 若模型架构、训练 budget、Flow sampler、ODE steps 或 observation schema 需要实质修改，创建
  新 config id 并从训练重跑；不能只重跑选模。
- 本计划不声称 future/world representation 有效；该命题必须由 W0 的 action intervention 先验证。

## 收尾

- [ ] 将本文件状态改为 `done`。
- [ ] 更新 `plans/README.md` 索引和母计划的当前进度。
- [ ] 按 `reports/` 规范登记训练、开环、闭环、latency、failure cases 与限制。
- [ ] 另建 W0 可执行子计划；W0 通过前不创建 W1 正式 run。
