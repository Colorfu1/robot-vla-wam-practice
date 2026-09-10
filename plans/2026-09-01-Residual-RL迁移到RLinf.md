# 2026-09-01 Residual RL 迁移到 RLinf：冻结 VLA base 加轻量操纵

## 状态

- 状态：`blocked`
- 阻塞项：等
  [Residual RL 下游改接 robomimic + DPPO pixel](2026-09-01-Residual-RL下游改接robomimic-DPPO-pixel.md)
  这一轮小实验出结论。本计划不与它并行执行。
- 范围：把「冻结 base + 轻量可训练模块」这条线从小规模 diffusion policy 抬到 VLA 规模，用 RLinf 已
  发布的成对 SFT / RL checkpoint 作为外部锚点。只做到「同一 frozen base 下轻量操纵 vs 全参数 RL 各自
  相对自己 base 的增益」为止；不跨 base、不跨 base 规模做算法排名。
- 最后更新：2026-09-01

## 目标

1. 复现 RLinf 已发表的 SFT base 与全参数 RL 数字，确认本地 harness 可信。
2. 在**同一** frozen base 上跑 DSRL（噪声空间轻量 SAC），拿到它相对该 base 的增益。
3. 回答一个我们自己的问题：约 500K 可训练参数的冻结 base 操纵，能拿回全参数 RL 增益的多少。

## 为什么切到 RLinf

三条都是已核实的事实，不是推测：

- **headroom 极大且有公开锚点**。RoboTwin 七个任务 OpenVLA-OFT 的 SFT 平均只有 **28.79%**
  （`lift_pot` 3.13%、`move_can_pot` 9.37%），GRPO 后 **86.16%**，七个任务**全部**有成对的
  SFT + RL checkpoint 公开在 HF。
- **观测天然无特权信息**：`images [B,224,224,3]` uint8 + 可选 `wrist_images` + `states [B,14]`
  float32 + 语言描述。无物体真值、无目标位置。
- **现成的冻结 base + 轻量模块实现**：DSRL 让约 **500K** 参数的 SAC 生成 32 维噪声注入 π₀ 的
  `sample_actions()` 作为初始噪声，π₀ 的 VLM 与 diffusion expert 全程冻结。这与我们在 ManiSkill2 上做的
  bounded residual 属同类，但机制不同（噪声空间操纵 vs 动作空间有界偏置）。

## 固定合同

| 项 | 值 | 依据 |
| --- | --- | --- |
| 仓库 | `https://github.com/RLinf/RLinf`（Apache-2.0） | OSDI 2026 / RSS 2026 |
| 镜像 | `rlinf/rlinf:agentic-rlinf0.4-robotwin`（国内镜像 `docker.1ms.run/rlinf/...`） | RoboTwin 文档 |
| 环境切换 | 容器内 `source switch_env openvla-oft` / `openpi` / `lingbotvla` | 同上 |
| RoboTwin | `git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support` + `bash script/_download_assets.sh` | 同上 |
| 环境变量 | `PYTHONPATH=<RoboTwin>`、`ROBOT_PLATFORM=ALOHA`；`env.{train,eval}.assets_path` 指向 `assets/` 的**父目录** | 同上 |
| 硬件 | 1–2 节点 / 8–16 GPU；基准配置 8×H100 + 192 核 + 1.8 TB 内存 | RoboTwin 页 Overview 与 Requirements 页 |
| 关键指标 | `env/success_once` | 同上 |
| 评测种子 | `rlinf/envs/robotwin/seeds/` 下的 train/eval seed 文件 | 同上 |

**DSRL 只能配 diffusion / flow 类 base**（它操纵的是 denoiser 的初始噪声），所以 base 只能是
π₀ / π₀.₅，**不能是 OpenVLA-OFT**（非 diffusion 架构）。这条约束决定了下面的任务选择。

## 任务与 base 的选择

RLinf 在 RoboTwin 上给 π₀ 系列公开的成对锚点只有 `adjust_bottle`（`demo_clean` 设置）：

| base | SFT | RLinf-PPO | 可用增益空间 |
| --- | ---: | ---: | ---: |
| π₀ | 76.56% | 98.44% | +21.88 |
| π₀.₅ | 85.94% | 96.09% | +10.15 |

而 DSRL 上游只在 **LIBERO-Spatial + π₀** 上给了现成配方
（`examples/embodiment/config/libero_spatial_dsrl_openpi.yaml`）。

因此分两步走，先低风险跑通、再迁到有锚点的任务：

- **主线**：π₀ + `adjust_bottle`（RoboTwin，`demo_clean`）。有 SFT 76.56% 与 PPO 98.44% 两个锚点，
  可以问「500K 参数的冻结 base 操纵能拿回这 +21.88 中的多少」。
- **前置跑通**：π₀ + LIBERO-Spatial，用上游现成的 DSRL 配方，只为验证冻结 base + SAC 的链路能跑。

## 前置条件

- [ ] robomimic + DPPO 那一轮已出结论并登记，本计划才解除 `blocked`。
- [ ] 1–2 节点 / 8–16 GPU 可用；显存需容纳 7B 级 π₀ 推理加 SAC 训练与 rollout。
- [ ] 生成物只写 `datasets/`、`outputs/` 或 ignored `third_party/`，不纳入版本控制。
- [ ] RLinf 与 RoboTwin(`RLinf_support` 分支) 各固定到一个 commit 并记录；RoboTwin assets 已下载。
- [ ] 下面六道闸门全部有明确答案（见「必须先查清的六件事」），未答完不进入第 3 步。

## 必须先查清的六件事

沿用上一轮的做法：**逐条在代码/配置里核实，不靠文档措辞推断**，结论落盘
`outputs/rl_rlinf_<date>/preflight/gates.json`。

- **G1 观测无泄漏**：在 RLinf 的 RoboTwin env wrapper 里逐 key 确认送进 policy 的观测只有相机图像、
  `states[14]` 与语言，**不含物体位姿 / 目标位置 / 抓握标志**。上一轮的教训是：底层仿真器可能仍在算
  特权量，是否泄漏取决于 wrapper 的 key 白名单，改动白名单就会把泄漏放回来。
- **G2 reward 形态**：确认 `adjust_bottle` 与 LIBERO-Spatial 各自实际用的 reward。文档明写
  **"Dense reward functions are still being expanded across tasks"**，不能假设 dense 可用；DSRL 的
  LIBERO 配方用的是 `use_step_penalty: True`（每步 −1 + 成功 termination bonus）。逐任务记录。
- **G3 base 真的冻结**：核实 DSRL 实现中 π₀ 的 VLM 与 diffusion expert 是否满足我们那三重保证——
  参数不进 optimizer、前向在 `no_grad` 下、训练前后参数指纹不变。上游只说「frozen」，要看代码。
- **G4 评测种子方案**：读 `rlinf/envs/robotwin/seeds/` 的 train/eval seed 文件，确认**跨 seed 的评测
  开局互不重复**，且 base 与 treatment 消费同一批开局。这正是 ManiSkill 那轮误判的成因，必须先验。
- **G5 `unnorm_key` 匹配**：文档明确警告 config 里的 `unnorm_key`（如 `"place_empty_cup"`）必须与 SFT
  checkpoint 训练时所用一致，否则**动作反归一化错误**。必须逐字核对，并把实际值写进 metrics。
- **G6 模型路径一致**：`rollout.model.model_path` 与 `actor.model.model_path` 必须指向**同一**路径。

## 执行步骤

### 1. 环境与 artifact

```bash
git clone https://github.com/RLinf/RLinf.git && cd RLinf && git rev-parse HEAD
```

```bash
docker run -it --rm --gpus all --shm-size 32g --network host --name rlinf \
  -v .:/workspace/RLinf rlinf/rlinf:agentic-rlinf0.4-robotwin
```

容器内切到 π₀ 环境，克隆 RoboTwin 并下资源：

```bash
source switch_env openpi
git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
cd RoboTwin && git rev-parse HEAD && bash script/_download_assets.sh
export PYTHONPATH=$PWD:$PYTHONPATH
export ROBOT_PLATFORM=ALOHA
```

下 π₀ 的成对 checkpoint（HF 在本机 Clash 规则下判 `DIRECT`，不占代理带宽）：

```bash
hf download RLinf/RLinf-Pi0-RoboTwin-SFT-adjust_bottle  --local-dir <models>/pi0-sft-adjust_bottle
hf download RLinf/RLinf-Pi0-RoboTwin-PPO-adjust_bottle  --local-dir <models>/pi0-ppo-adjust_bottle
hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT --local-dir <models>/pi0-libero-sft
```

### 2. 回答 G1–G6，落盘 gates.json

只读代码与配置，不训练。G1 与 G4 不通过则**停止本计划**——那是「观测干净」和「结论可信」两个前提。

### 3. 锚点复现（外部对照）

用上游 eval 配方分别评 SFT 与 PPO checkpoint，对照已发表值：

```bash
# 评测配方见 RoboTwin evaluation guide，形如 robotwin_adjust_bottle_openpi_eval
bash examples/embodiment/run_embodiment.sh <eval config>
```

判据：SFT 落在 `76.56% ± 5pp`、PPO 落在 `98.44% ± 5pp`。超出则先查 harness（优先怀疑 G5 的
`unnorm_key` 与 G6 的路径一致性），不进入第 5 步。

### 4. DSRL 链路跑通（π₀ + LIBERO-Spatial，上游现成配方）

```bash
bash examples/embodiment/run_embodiment.sh libero_spatial_dsrl_openpi
```

只跑到能看见 `env/success_once` 上升、且 G3 的三重冻结断言全部成立为止。这一步**不产生本计划的正式
结果**，只证明「冻结 base + 500K SAC」的链路在本地可运行。

### 5. 主线：π₀ + `adjust_bottle` 上的 DSRL

把 DSRL 配方迁到 RoboTwin `adjust_bottle`。上游没有这个组合的现成 config，需要新建，且必须显式记录
从 LIBERO 配方改动的每一项（观测维度 8→14、`max_episode_steps`、`total_num_envs`、reward 形态）。

DSRL 侧的关键参数（上游 LIBERO 配方的取值，作为起点，改动必须记录）：

```yaml
actor.model.openpi:
  use_dsrl: True
  dsrl_state_dim: 8              # RoboTwin 是 14，必须改
  dsrl_action_noise_dim: 32
  dsrl_num_q_heads: 10
  dsrl_image_latent_dim: 64
  dsrl_state_latent_dim: 64
  dsrl_hidden_dims: [128, 128, 128]
algorithm:
  adv_type: embodied_sac
  loss_type: embodied_sac
  gamma: 0.999
  tau: 0.005
  update_epoch: 200
  train_actor_steps: 10
  entropy_tuning: {alpha_type: softplus, initial_alpha: 1.0, target_entropy: -16, optim: {lr: 3.0e-4}}
```

### 6. 对照与汇总

三行对照，**每行只与自己的 base 比**：

| 行 | 可训练参数 | 相对谁报增益 |
| --- | --- | --- |
| π₀ SFT（frozen base） | 0 | — |
| π₀ + DSRL（噪声空间操纵） | 约 500K | 相对 SFT |
| π₀ + PPO（上游全参数，外部数字） | 全部 | 相对 SFT |

按 `reports/AGENTS.md` 登记，并在 README 增加一行。上游 PPO 那一行是**外部已发表数字**，必须标注，
不得写成本地测量。

## 判断标准

**执行前写死，看到数字之后不得修改。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| G1 观测无泄漏 | policy 观测中不出现物体位姿 / 目标位置 / 抓握标志 | **停止本计划** |
| G4 评测开局 | 跨 seed 的评测开局互不重复；base 与 treatment 消费同一批 | **停止本计划**，这是 ManiSkill 那轮误判的成因 |
| G3 base 冻结 | 参数不进 optimizer、前向在 `no_grad` 下、训练前后参数指纹不变 | 修实现，不进入第 5 步 |
| 锚点可信 | SFT 在 `76.56% ± 5pp`、PPO 在 `98.44% ± 5pp` | 先查 G5/G6，不进入第 5 步 |
| DSRL 有效 | 相对 SFT base ≥ **+5pp**，且 ≥3 个 seed 方向一致 | 记为负结果，不调阈值、不 sweep |
| 不做跨规模排名 | 只报告各自相对自己 base 的增益 | 违反即视为结论无效 |

DSRL 的 +5pp 阈值低于 ManiSkill2 那轮的 +10pp，理由写在前面：π₀ 在 `adjust_bottle` 上的 SFT 已达
76.56%，可用空间只有 +21.88，而 DSRL 只有约 500K 可训练参数。**这个阈值现在定，不因结果调整。**

## 产物路径

```text
outputs/rl_rlinf_<date>/preflight/gates.json
outputs/rl_rlinf_<date>/anchor/{sft,ppo}/metrics.json
outputs/rl_rlinf_<date>/dsrl_libero_smoke/
outputs/rl_rlinf_<date>/dsrl_adjust_bottle/seed_*/
outputs/rl_rlinf_<date>/summary.json
```

均 gitignored。

## 风险与停止条件

- **DSRL 无法配 OpenVLA-OFT**。它操纵 denoiser 初始噪声，只适用于 diffusion / flow 类 base。若要用
  RoboTwin 上 headroom 最大的那批任务（`lift_pot` SFT 3.13%），得先有对应的 π₀ SFT checkpoint——
  上游只发布了 `adjust_bottle`。**不要为了追 headroom 而把 DSRL 硬套到 OpenVLA-OFT。**
- **π₀.₅ 的 headroom 更小**（SFT 85.94%），若主线在 π₀ 上失败，不要顺手换 π₀.₅ 当救火，那只会让
  可用空间更小。
- **reward 形态未定就不要开跑**。文档明写 dense reward 尚在逐任务补齐；`adjust_bottle` 是否有 dense
  必须由 G2 给出答案。若只有 sparse，DSRL 的 `use_step_penalty` 是上游的绕法，但它在 RoboTwin 上未
  经验证。
- **四个任务 RLinf 不支持**：`place_fan`、`open_laptop`、`place_object_scale`、`put_object_cabinet`。
  不要选它们。
- **不与前一轮结果混列**。robomimic Square pixel 那一轮的 base 是 13.6 MiB diffusion policy、单相机
  96×96、7D 单臂；这里是 7B 级 π₀、224×224、ALOHA 双臂 14D。不同 harness、不同观测、不同 base 规模，
  成功率**不可相减、不可并入同一张表**。
- **`unnorm_key` 错配是静默失败**。它只会让动作反归一化错误，不报错，表现为成功率异常低。锚点复现
  对不上时先查它。
- 若第 3 步锚点在两次不同配置下都对不上，停止并重新评估是否值得继续，不要靠调参把数字凑上去。

## 收尾

- [ ] 把本文件 `状态` 改为 `done`
- [ ] 更新 `plans/README.md` 索引
- [ ] required experiment fields 已写入 `reports/experiment_log.md`
