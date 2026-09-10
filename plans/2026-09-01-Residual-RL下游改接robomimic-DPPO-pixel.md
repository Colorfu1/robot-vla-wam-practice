# 2026-09-01 Residual RL 下游改接 robomimic + DPPO pixel

## 状态

- 状态：`in-progress`
- 范围：接续 [2026-08-28 Policy Decorator 复现计划](2026-08-28-PolicyDecorator-ManiSkill2-Residual-RL复现.md)
  的 ManiSkill2 结果，把下游 residual RL 对照改到 robomimic `square`（`NutAssemblySquare`）的
  **pixel** 工作点。本计划先完成官方 DPPO 全参数微调基准，再回到同一个 frozen base 做 bounded
  residual（含 Zero/Random 对照）。官方 DPPO 与 residual 的参数量和预算不同，不做同一算法的直接排名，
  但必须将官方基准作为独立主基准记录。
- 最后更新：2026-09-01

## 目标

1. 在一个**无特权观测**的 pixel 工作点上重做 ManiSkill2 那轮的 residual 对照：观测只有相机图像 +
   机器人自身 proprioception，不含物体真值位姿与目标位置。
2. 先用上游 `ft_ppo_diffusion_mlp_img.yaml` 复现官方 DPPO pixel 全参数微调，并完成 3 seed × 50
   episode 的闭环评测。
3. 再交付同一 frozen base 下的四行 residual 对照：Frozen Base、+ Bounded Residual、Zero Residual、
   Random Bounded Residual。
4. 用 frozen base 的上游评测作为 harness 锚点；不把它误写成官方 full fine-tuning 结果。

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
- 代价：Square 比 Transport 简单，headroom 可能更小。这由 frozen base 锚点评测决定，不预设。

## 前置条件

### 已完成的预检（结论落盘在 `outputs/rl_residual_robomimic_20260901/preflight/gates.json`）

- [x] **G1 数据来源与 demo 数（已实测解决）**：源数据是 **robomimic Square-MH**。证据是 Drive folder
  里的处理日志 `_2024_07_28_22_48_55.log` 写明 `total trajectories: 300`、`traj length min/max:
  123, 1051`；DPPO 自己的 `script/dataset/README.md` 处理命令也是 `--dataset_types mh`。
  **但发布的 `train.npz` 只有 100 条**（27,214 步），原因是那条命令结尾带 `--n 100`——
  `dataset_states_to_obs.py` 只 replay 了前 100 条。文件本身完整（`zipfile.testzip()` 无损坏、
  `images.npy` 未压缩尺寸 752,412,800 = `27214×3×96×96+128` 逐字吻合、本地字节数与远端
  Content-Length 相同），也确实来自 `download_url.py` 声明的那个 folder，不是下载残缺。
  文件日期揭示了不一致：日志与 `normalization.npz` 是 8月27，`state_4000.pt` 是 8月29，
  而 `train.npz` 是 **9月9**——**发布的 checkpoint 训在 300 条上，现在能下到的数据是 100 条子集**。
  → 这只说明官方 checkpoint 的训练来源；当前路线不重新生成 demonstrations。若未来重新 pretrain，
  再按下面延期的数据路径生成 300 条，届时 `Diffusion Policy` 论文的 Square-MH 数字才作为同源参照。
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

- [x] **接线修复已完成**：见 [robomimic + DPPO pixel 接线修复](2026-09-01-robomimic-DPPO-pixel接线修复.md)。其中 reset seed 与 anchor 奖励口径已锁定。

- [ ] 仓库根目录执行，`PYTHONNOUSERSITE=1` 保持开启。生成物只写 `datasets/`、`outputs/` 或
  ignored `third_party/`，不纳入版本控制。
- [ ] DPPO 环境按上游装：`conda create -n dppo python=3.8`、`pip install -e .[robomimic]`，并按
  `installation/install_mujoco.md` 装 MuJoCo。**不复用** `robot-practice` / `maniskill2-pidec` 环境。
- [ ] **不要执行 `script/set_path.sh`**——它是交互式的（`read -p`）且会往 `~/.bashrc` 追加 export。
  改为显式 `export DPPO_DATA_DIR=... DPPO_LOG_DIR=...`；无 WandB 时命令行传 `wandb=null`。
- [ ] CPU 线程数够跑 `env.n_envs=50`。不够时按上游说明降 `env.n_envs` 并同比升 `train.n_steps`，
  使 `n_envs × n_steps × act_steps` 不变，并把实际值写进 resolved config。
- [x] reward 形态已按接线修复锁定为上游 sparse（`reward_shaping=False`）；dense 另立计划。

## reward 设计

本轮接线修复将 reward 形态锁为上游默认 sparse（`reward_shaping=False`），用于 anchor、训练与本地
评测的可比性。Square 的 sparse reward 是每个环境 step 的 `sum(objects_on_pegs)`，成功阈值为 1；并非
只在终局给一次。若后续 residual RL 需要 staged dense，须另立计划，不得在本轮混入。

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
  列表、`env.n_envs`/`train.n_steps`、`reward_shaping=False`、预算、以及本计划
  「判断标准」里的全部阈值。
- `scripts/experiments/rl/run_residual_robomimic_dppo.sh`：`preflight` / `base-eval` / `train` /
  `eval` / `summarize` 五个阶段。
- `scripts/experiments/rl/run_official_dppo_baseline.sh`：一次编排官方 full fine-tuning、上游 eval
  和 3-seed 汇总，不改上游入口；提供每个 seed 的 iteration/rollout/ETA 进度和
  `--resume`（从最新 `state_N.pt` 继续）。
- `scripts/experiments/rl/resume_official_dppo.py`：仅在 `--resume` 时使用的官方 agent 恢复 shim，
  复用上游 Hydra 配置、agent 和 `run()`，不修改上游 DPPO checkout。
- `src/robot_practice/experiments/rl/robomimic_dppo_residual.py`：把 DPPO 的 frozen diffusion policy
  包成 base，接上 bounded residual actor/critic。
- 对应测试，放 `tests/experiments/rl/`。

## 执行步骤

### 1. 装环境并准备上游 artifacts

```bash
cd third_party/dppo/dppo && git rev-parse HEAD
```

按上游装 `dppo` 环境与 MuJoCo，**显式** export 路径（不要跑 `script/set_path.sh`，它是交互式且会写
`~/.bashrc`）：

```bash
repo_root="$(git rev-parse --show-toplevel)"
export DPPO_DATA_DIR="${repo_root}/datasets/raw/dppo"
export DPPO_LOG_DIR="${repo_root}/outputs/rl_residual_robomimic_20260901/dppo_logs"
```

#### 1a. 上游 artifact（当前路线只需 normalization + checkpoint）

`gdown 6.1.0` 的 `download_folder` 在 Clash TUN 下会 `SSLEOFError`（curl 与 requests 同一 URL 均 200，
是 gdown 自身问题），**按 file id 逐个下**：

```bash
CKPT_DIR="$DPPO_LOG_DIR/robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint"
mkdir -p "$DPPO_DATA_DIR/robomimic/square-img" "$CKPT_DIR"
python3 -m venv /tmp/gdown-venv && /tmp/gdown-venv/bin/pip -q install gdown
```

```bash
/tmp/gdown-venv/bin/python - <<'PY'
import gdown, os
D = os.path.join(os.environ["DPPO_DATA_DIR"], "robomimic/square-img")
K = os.path.join(os.environ["DPPO_LOG_DIR"],
    "robomimic-pretrain/square/square_pre_diffusion_mlp_img_ta4_td100/2024-07-30_22-27-34/checkpoint")
for fid, out in [
    ("1jq5atfHdu-ZMQ8YaFjwcctbJESBYDgP8", os.path.join(D, "normalization.npz")),  # 1,274 B
    ("1miud5SX41xjPoW8yqClRhaFx0Q7EWl3V", os.path.join(K, "state_4000.pt")),      # 13.6 MiB
]:
    gdown.download(id=fid, output=out, quiet=False)
PY
```

checkpoint 路径的时间戳 `2024-07-30_22-27-34` 必须逐字一致（`ft_ppo_diffusion_mlp_img.yaml:10` 与
`download_url.py:417` 两处都是它），否则 DPPO 判定 checkpoint 不存在并重新下载。
`square-img/` 这一份**只用于第 2 步的上游 frozen-base anchor、官方 DPPO fine-tuning 和其评测**：
评官方 checkpoint 必须配上游 `normalization.npz`。当前路线不下载 `train.npz`；如果未来要重新
pretrain，再单独下载 100-episode 子集或按 1b/1c 重新生成数据。

#### 1b. （延期）自行重生成 300 条 MH（仅未来重新 pretrain 时使用）

当前路线不执行本小节：我们直接使用已下载的官方 `state_4000.pt`，官方 DPPO fine-tuning 和
Residual RL 都是在线环境训练，不读取 `train.npz`。以下内容只保留给未来需要从 demonstrations
重新 pretrain 的独立实验，不能作为本计划进入 residual 前的门槛。

**没有可直接下载的现成 300 条 96×96 数据**，三条路都查过：

| 来源 | 下载量 | 是否走代理 | 分辨率 | 结论 |
| --- | ---: | :-: | --- | --- |
| HF raw MH + 本地渲染 | 70.5 MiB | 否（HF 被 Clash 判 `DIRECT`） | 96×96（自己指定） | **采用** |
| Stanford `rt_benchmark/square/mh/image.hdf5` | 6.48 GiB | 是 | **84×84 写死** | 不可用，分辨率不符 |
| DPPO 现成 npz | 81.6 MiB | 是 | 96×96 | 仅 100 条 |

HF repo `robomimic/robomimic_datasets` 全部 31 个文件里**没有任何 image 数据集**（注册表中
`hdf5_type="image"` 的 `link=None`，`download_datasets.py` 会打印 "no url for dataset exists.
Create this dataset locally"）。raw hdf5 只存 MuJoCo 状态、图像靠重放渲染，所以才这么小。

需要的两个 raw 文件：

| HF 路径 | 体积 |
| --- | ---: |
| `v1.5/square/mh/demo_v15.hdf5` | 70.5 MiB |
| `v1.5/square/ph/demo_v15.hdf5` | 65.8 MiB（PH 对照用） |

```bash
python robomimic/scripts/download_datasets.py --tasks square --dataset_types mh ph --hdf5_types raw
```

渲染在**容器或远端**执行（`robomimic` + `robosuite` + MuJoCo 离屏渲染）。渲染是 CPU MuJoCo，不需要
GPU，但必须先确认离屏渲染后端可用——这是容器里最容易失败的一步：

```bash
python -c "
import os; print('MUJOCO_GL =', os.environ.get('MUJOCO_GL'))
import robosuite, robomimic, mujoco
print('robosuite', robosuite.__version__, '| robomimic', robomimic.__version__, '| mujoco', mujoco.__version__)
import numpy as np
env = robosuite.make('NutAssemblySquare', robots='Panda', has_renderer=False,
                     has_offscreen_renderer=True, use_camera_obs=True,
                     camera_names='agentview', camera_heights=96, camera_widths=96,
                     control_freq=20, reward_shaping=False)
o = env.reset(); img = o['agentview_image']
print('render OK', img.shape, img.dtype, 'nonzero =', bool(np.any(img)))
"
```

`MUJOCO_GL` 未设时 mujoco 会自行探测，容器里常需显式 `export MUJOCO_GL=egl`（有 GPU）或
`osmesa`（纯 CPU）。`nonzero = False` 说明渲染出全黑图，属于后端故障，**不要继续渲染 300 条**。

replay 渲染。**不要加 `--n`**——DPPO 原命令结尾的 `--n 100` 正是 demo 数变成 100 的原因。相机用
`agentview`（`ft_ppo_diffusion_mlp_img.yaml` 的 `image_keys` 是 `['agentview_image']`，不是 DPPO
README 示例里的 `robot0_eye_in_hand`）：

```bash
python robomimic/scripts/dataset_states_to_obs.py --done_mode 2 \
  --dataset datasets/square/mh/demo_v15.hdf5 --output_name image_v15.hdf5 \
  --camera_names agentview --camera_height 96 --camera_width 96 --exclude-next-obs
```

```bash
python script/dataset/process_robomimic_dataset.py \
  --load_path=<robomimic>/datasets/square/mh/image_v15.hdf5 \
  --save_dir=$DPPO_DATA_DIR/robomimic/square-img-mh300 \
  --normalize --cameras agentview
```

输出放 `square-img-mh300/`，**不覆盖** `square-img/`。自行 pretrain 时用这一份及其自带的
`normalization.npz`；评上游 checkpoint 时用 `square-img/` 那一份。两者不可交叉使用。

预估产物：`80,731 × 3 × 96 × 96` 未压缩约 2.08 GiB，按上游那份的 8.83× 压缩比约 240 MiB。

#### 1c. （延期数据路径）四项必须核实、不得假设

- **离屏渲染可用**：上面那段探针通过（能出非全黑的 `(96,96,3)`）后才渲染 300 条。
- **版本兼容，并有 fallback**：注册表现在是 `v1.5/.../demo_v15.hdf5`，而 DPPO 的
  `env_meta/square-img.json` 写 `env_version: 1.4.1`、README 示例是 `demo_v141.hdf5`。先确认 v1.5 的
  demo 能在 robosuite 1.4.1 契约下正确重放（对比重放后的 `traj_lengths` 与 raw 的 demo 长度）。
  **不通过时改用 v1.4.1 的 raw**——Stanford 旧服务器仍可用（实测 HTTP 200、60.0 MiB），与 DPPO 当年
  所用完全同版本：

  ```bash
  curl -fL -o demo_v141.hdf5 \
    http://downloads.cs.stanford.edu/downloads/rt_benchmark/square/mh/demo_v141.hdf5
  ```

  该域名不是 HF，会走代理节点，但只有 60 MiB。两条都不通时停止，不要静默用 v1.5 数据配 1.4.1 环境。
- **渲染耗时**：先只 replay 前 5 条计时（临时加 `--n 5`），再外推 300 条总时长，写进
  `preflight/render_timing.json`。正式渲染时务必去掉 `--n`。
- **重生成结果与上游日志对齐**：新 npz 的 `episodes` 应为 300、`traj_lengths` 的
  `mean/std ≈ 269.1/123.2`、`min/max = 123/1051`（上游日志 `_2024_07_28_22_48_55.log` 的值）。
  偏差明显时先查处理参数与数据版本。

```bash
/tmp/gdown-venv/bin/pip -q install numpy && /tmp/gdown-venv/bin/python -c "
import numpy as np, os
for name in ('square-img', 'square-img-mh300'):
    p = os.path.join(os.environ['DPPO_DATA_DIR'], 'robomimic', name, 'train.npz')
    if not os.path.exists(p): continue
    d = np.load(p); L = d['traj_lengths'].astype(int)
    print(f'{name}: episodes={len(L)} steps={L.sum()} len mean/std={L.mean():.1f}/{L.std():.1f} min/max={L.min()}/{L.max()}')
    for k in ('states','actions','images'): print('   ', k, d[k].shape, d[k].dtype)
"
```

### 2. 复现上游 frozen-base anchor（用上游默认 sparse）

```bash
cd third_party/dppo/dppo && python script/run.py \
  --config-name=eval_diffusion_mlp_img \
  --config-dir=cfg/robomimic/eval/square wandb=null
```

不带 `ft_denoising_steps` 时评的是预训练策略，即 frozen base 的上游成绩。记录 episode 数、seed 与
成功率。这一步不产生官方 full fine-tuning 结果，只判断 harness 可信，且**成本远低于训练**，必须先做。

### 3. 本地 frozen base 评测（Residual 的共同 anchor）

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=base-eval \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/base
```

`residual_scale=0` 的输出必须与纯 base 逐步一致（这同时就是 Zero Residual 控制组）。

### 4. 官方 DPPO 全参数微调基准（Residual 之前必须完成）

按 [guide 的 §5.2「官方 DPPO 全参数微调 + eval」](../reports/guides/rl/robomimic_dppo_pixel_residual.md)
执行 `scripts/experiments/rl/run_official_dppo_baseline.sh`。该入口直接调用官方
`ft_ppo_diffusion_mlp_img.yaml` 与 `eval_diffusion_mlp_img.yaml`，一次完成 3 个独立 seed 的微调、
每个 50 个 episode 的评测，以及 `official_dppo_baseline/metrics.json` 汇总；完成后才进入下面
residual 阶段。

正式 run 前先用同一入口的 `--smoke --gpus=0 --seeds=1` 完成 1 iteration、1 episode 的链路检查；
smoke 只验证入口和产物，不计入正式结果。训练期间 wrapper 必须在 top-level 和每个 seed 的 log
中周期性报告 `itr/total`、rollout、elapsed 和 ETA；中断后用 `--resume` 从最新非空 checkpoint
继续，未完成 seed 才重新启动。

这里的“全参数”沿用官方 pixel recipe：`actor_ft` 的全部参数参与 PPO 更新，但只对应最后 5 个 DDIM
step；这不是 residual，也不是重新 pretrain。

Residual 仍从同一个 frozen `state_4000.pt` 出发，官方 `state_300.pt` 作为独立主基准记录；如果要在
官方 full-finetuned policy 上继续叠 residual，需要另立实验合同。

### 5. residual 接线 smoke

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=train --smoke \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/smoke
```

极短预算跑通「采样 → buffer → 更新 → 存 ckpt」全链路，并断言 `base_parameter_hash` 前后一致、
`zero_residual_action_equal=true`。smoke 不是结果；官方 DPPO 基准必须先于本步骤完成。

### 6. 单 iteration 计时，并实测 base 推理占比

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=train --max-iterations=1 --time-stages \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/timing
```

base 是 diffusion policy，每个动作要跑 5 步 DDIM 穿过 ViT + MLP，比 ManiSkill2 那轮的单次前向贵得多。
必须把 `s/iter` 拆成「env step / base 推理 / residual 更新」三段写进 `timing/metrics.json`，再外推总时长。

### 7. 正式 residual 训练与评测

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

### 8. Random Bounded Residual 对照

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=eval --residual-mode=random-bounded \
  --config=configs/experiments/rl/residual_robomimic_dppo_square.yaml \
  --output-root=outputs/rl_residual_robomimic_20260901/random_residual
```

### 9. 汇总与登记

```bash
bash scripts/experiments/rl/run_residual_robomimic_dppo.sh \
  --phase=summarize \
  --output-root=outputs/rl_residual_robomimic_20260901
```

按 `reports/AGENTS.md` 登记到 `reports/experiment_log.md`，并在 README §6 增加一行；不把 smoke、
timing、第 2 步的上游 frozen-base anchor 写成官方 full fine-tuning 或 residual 正式结果。

## 判断标准

**执行前写死，看到数字之后不得修改。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| G3 观测无泄漏 | pixel config 的观测 key 中不出现 `object` 或任何目标真值 | **立即停止本计划**，换任务的前提不成立 |
| 重生成数据对齐 | 仅在未来重新 pretrain 时要求 `square-img-mh300` 的 `episodes=300`、`min/max=123/1051`、`mean/std≈269.1/123.2` | 当前路线不执行；未来偏差明显则先查处理参数与数据版本，不进入 pretrain |
| 上游 frozen-base anchor | 第 2 步复现的 pixel base 成功率与 DPPO 已发表曲线的工作点差异不超过预设容差 | 差异过大则先查 harness，不进入第 4 步 |
| 官方 DPPO 基准 | 3 个 full fine-tuning seed 均生成 `state_300.pt`，且每个 eval 完成 50 episodes | 不进入 residual |
| headroom | 本地 frozen base pooled success 在 `[10%, 90%]` | 记录任务不适合作 residual probe，停止 |
| Zero Residual 一致性 | `residual_scale=0` 的动作与纯 base 逐步一致 | 不一致说明接线错，修完再跑 |
| residual base 冻结 | Residual 训练与评测前后 `base_parameter_hash` 相同 | 不同即 `RuntimeError`，视为实现缺陷 |
| Residual 增益 | 相对第 3 步 frozen base ≥ **+10pp**，且 ≥3 个 seed 方向一致 | 记为负结果，不调阈值 |
| Random 对照 | Random Bounded Residual 的增益 < Residual 增益的一半 | 若随机也能拿到同等增益，则增益不来自学习，作废该结论 |
| Residual 不长期顶界 | 训练后期动作触界比例 < 20% | bound 设置不当，重设后重跑 |

## 产物路径

- `outputs/rl_residual_robomimic_20260901/preflight/gates.json`
- `outputs/rl_residual_robomimic_20260901/preflight/render_timing.json`
- `datasets/raw/dppo/robomimic/square-img/`（当前路线：上游 normalization；checkpoint 位于 `dppo_logs/`）
- `datasets/raw/dppo/robomimic/square-img-mh300/`（延期：未来自行重生成的 300 条 + 自带 normalization）
- `outputs/rl_residual_robomimic_20260901/base/metrics.json`
- `outputs/rl_residual_robomimic_20260901/official_dppo_baseline/finetune/seed_*/checkpoint/state_300.pt`
- `outputs/rl_residual_robomimic_20260901/official_dppo_baseline/eval/seed_*/result.npz`
- `outputs/rl_residual_robomimic_20260901/official_dppo_baseline/metrics.json`
- `outputs/rl_residual_robomimic_20260901/smoke/metrics.json`
- `outputs/rl_residual_robomimic_20260901/timing/metrics.json`
- `outputs/rl_residual_robomimic_20260901/residual/`（checkpoint、`run.log`、`progress.jsonl`）
- `outputs/rl_residual_robomimic_20260901/residual_eval/metrics.json`
- `outputs/rl_residual_robomimic_20260901/random_residual/metrics.json`
- `outputs/rl_residual_robomimic_20260901/summary.json`

均 gitignored。

## 风险与停止条件

- **数据不可交叉使用**：当前官方 DPPO fine-tuning 与 residual 不读取 demonstrations，只使用上游
  `state_4000.pt` 和配套 `square-img/normalization.npz`。若未来自行 pretrain，才使用
  `square-img-mh300/` 自带的另一份 normalization，不能与上游 checkpoint 混用。
- **v1.5 raw 配 1.4.1 环境契约**：若 1c 的重放核对不通过，改用 Stanford 旧服务器的
  `square/mh/demo_v141.hdf5`（实测 HTTP 200、60.0 MiB，与 DPPO 当年同版本；该域名走代理节点）。
  两条都不通时停止，不要静默用 v1.5 数据配 1.4.1 环境。
- **容器内离屏渲染**：`MUJOCO_GL` 未正确设置时 MuJoCo 可能渲出全黑图而不报错。必须先跑 1b 的探针
  确认能出非全黑的 `(96,96,3)`，再渲染 300 条；渲染出全黑数据会一路静默传到 pretrain。
- **预渲染数据不可用**：Stanford 的 `square/mh/image.hdf5` 虽然仍可下（6.48 GiB），但分辨率是
  84×84 写死，与 DPPO 要求的 96×96 不符，不得用它替代重新渲染。
- **自行 pretrain 复现不出上游 checkpoint**：上游那份来自 v0.6 配置（checkpoint 名 `state_4000.pt`
  而当前 `pre_diffusion_mlp_img.yaml` 是 `n_epochs: 2000`；`cfg/pretraining.md:3` 说明 2024-11-20
  修了 EMA 更新过疏并在 v0.7 下调 epoch，要复现论文原始结果需用 v0.6）。本地 pretrain 只能与
  本地 base 相比，不能声称复现了上游数字。
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

## 相关工作与备用路径（2026-09-01 调研，不改变本轮路线）

[RLinf](https://github.com/RLinf/RLinf)（OSDI 2026 / RSS 2026，Apache-2.0）有两样与本计划直接相关的
东西，但**硬件门槛使其无法作为本轮路线**。

### 为什么不切换

RoboTwin 页面 Overview 明写 `Hardware: 1–2 nodes · 8–16 GPUs`；全局 Requirements 的基准配置是每节点
**8×H100 + 192 核 CPU + 1.8 TB 内存 + NVLink/RoCE 3.2 Tbps**。base 模型是 OpenVLA-OFT / π₀ / π₀.₅ 等
7B 级 VLA，即使冻结也要装进显存推理。对比本轮路线的 base 是 ViT + MLP diffusion、**13.6 MiB**，单卡
加 50 个 CPU env 即可。切换等于把「能不能跑起来」这个已经消除的风险重新引入。

### 值得单独取用的两项

**1. `RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ`（HF 数据集，28.7k 行）**

若以后回到 PegInsertionSide，这份现成数据省掉自己渲染：

| 字段 | 规格 |
| --- | --- |
| Environment | `PegInsertionSideWideClearance-v1`（间隙放宽变体，比原版易） |
| Control mode | `pd_joint_delta_pos` |
| `image` | 第三人称 RGB `384×384×3` |
| `wrist_image` | 腕部 RGB |
| `state` | 9 维 proprioception |
| `actions` | 8 维 joint-control chunk |
| `prompt` | `insert the peg in the hole` |

观测只有两路 RGB + 自身状态，**无物体真值**，满足本计划的观测契约要求。

**2. DSRL 作为方法参考**（`examples/embodied/dsrl.html`，论文 Wagenmaker et al., CoRL 2025）

「冻结 diffusion policy + 轻量 SAC」，与本计划的 bounded residual 属同类但机制不同：DSRL 让 SAC
（约 **500K** 可训练参数：CNN 64×64→64 维 + 3 层 128 维 MLP + 10 个 Q head 集成）生成 32 维噪声，
**注入 π₀ 的 `sample_actions()` 作为初始噪声**替代随机采样，π₀ 的 VLM 与 diffusion expert 全程冻结。
即「在噪声空间操纵」而非「在动作空间加偏置」。等本计划进入 residual 设计时，这是一个值得对比的
替代设计。其 LIBERO-Spatial 配方用 `use_step_penalty: True`（每步 −1 + 成功 termination bonus）绕开
dense reward 的缺失，也是一个可借鉴的做法。

### RoboTwin 上的 headroom 参考（外部数字，非本地测量）

OpenVLA-OFT，`demo_randomized` 设置，七个任务均有成对的 SFT + RL checkpoint 公开：

| 任务 | SFT | RLinf-GRPO |
| --- | ---: | ---: |
| `lift_pot` | 3.13% | 70.31% |
| `move_can_pot` | 9.37% | 83.59% |
| `beat_block_hammer` | 10.15% | 96.09% |
| `pick_dual_bottles` | 20.31% | 92.96% |
| `handover_block` | 28.13% | 70.31% |
| `place_container_plate` | 54.69% | 95.31% |
| `place_empty_cup` | 75.78% | 94.53%（PPO 92.97%） |
| 平均 | **28.79%** | **86.16%** |

SFT 平均 28.79% 说明 RoboTwin 这批任务 headroom 很大——这与 LIBERO/ManiSkill 上 RL 后 97–98% 的
饱和数字不矛盾，后者是 RL 之后的值。观测为 `images [B,224,224,3]` + 可选 `wrist_images` +
`states [B,14]` + 语言，无物体真值。

**限制**：文档明写 "Dense reward functions are still being expanded across tasks"，dense reward 尚未
覆盖全部任务；另有四个 RoboTwin 任务（`place_fan`、`open_laptop`、`place_object_scale`、
`put_object_cabinet`）在 RLinf 中尚不支持。

以上均为**外部已发表数字与文档记载**，不是本地测量，不得写入本仓库的结果表。

## 收尾

- [ ] 把本文件 `状态` 改为 `done`
- [ ] 更新 `plans/README.md` 索引
