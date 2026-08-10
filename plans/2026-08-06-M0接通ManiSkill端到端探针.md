# 2026-08-06 M0接通ManiSkill端到端探针

## 状态

- 状态：`in-progress`
- 范围：只把 PickCube-v1 这一条数据管路从官方演示打通到闭环成功率，产出一个可用的冻结基线策略。
  不做 WAM、不做 RL、不做多任务。上游是
  [2026-08-06 Mini-WAM与残差RL三阶段实验计划](2026-08-06-Mini-WAM与残差RL三阶段实验计划.md) 的
  阶段零第一步。
- 最后更新：2026-08-07

## 目标

用最小代价确认或否掉整条 ManiSkill 数据管路：官方演示 → replay 生成 RGB 观测 → LeRobotDataset
→ 训练 diffusion policy → 闭环评测。同时产出后续 M1/M2/W/R 阶段要用的冻结基线策略与状态划分。

这一步的价值是**早失败**。管路不通就立刻走退路，不要在后面才发现。

## 前置条件

- [ ] `mani_skill` 可导入且版本为 3.0.1：
      `conda run -n robot-practice python -c "import mani_skill; print(mani_skill.__version__)"`
- [ ] **演示已下载并已审计过一轮**（2026-08-07 完成，见下面第 1 步的实测清单）。
      落在 `datasets/raw/maniskill_demos/PickCube-v1/`，共 90 MB。
- [ ] 已实测确认的环境能力（本轮已验，无需重复）：
      PickCube-v1 在 256 并行 + RGB 128×128 下 5716 步/秒；`get_state_dict`/`set_state_dict`
      可精确回滚；`info` 含 `is_grasped`/`is_obj_placed`/`is_robot_static`；
      `max_episode_steps = 50`（`sim_freq=100`、`control_freq=20`，即 2.5 秒）；
      `pd_ee_delta_pose` 为 7 维、`pd_ee_delta_pos` 为 4 维、`pd_joint_pos` 为 8 维。
- [ ] 仓库对 ManiSkill 仍是零基础：`envs/maniskill/` 是空占位，`src/`、`configs/`、`scripts/`
      下无引用，LeRobot 也没有 ManiSkill 适配（`lerobot/envs/` 只有 `libero.py` 与
      `metaworld.py`）。因此第 9 步的闭环评测需要新写代码。
- [ ] **数据集转换有官方工具**：`mani_skill/trajectory/convert_to_lerobot.py` 输出 LeRobot v3.0，
      与本仓库的 `CODEBASE_VERSION = v3.0` 一致。第 6 步先验它，别先自己写。
- [ ] **录制的初始状态分布与本机版本一致**（已实测，这是数据可用的正面证据）：

      | | cube x 范围 | goal z 范围 | qpos[0] 标准差 |
      | --- | --- | --- | ---: |
      | rl（`il-baselines` commit） | −0.1000..+0.0994 | 0.0201..0.3200 | 0.01940 |
      | motionplanning（`main` commit） | −0.0998..+0.0996 | 0.0201..0.3184 | 0.01977 |
      | 本机 3.0.1 | −0.0993..+0.0997 | 0.0206..0.3174 | 0.02080 |

      三者同分布，所以 3.3 的状态比对是在比同类东西。注意 `env_states` 里 panda 是 31 维 =
      根位姿(7) + 根速度(6) + qpos(9) + qvel(9)，**qpos 在偏移 13**，取错偏移会得到恒定值。
- [ ] **`env_states` 的状态恢复是完整的**：含 `actors/table-workspace`、`actors/cube`、
      **`actors/goal_site`**、`articulations/panda`。目标球也在里面，所以
      `--use-first-env-state` 能完整恢复首帧，不依赖 seed。
- [ ] **`mplib` / `toppra` 在本机是坏的**：NumPy 2.2.6 下报
      `numpy.core.multiarray failed to import`。所以本地无法重新生成运动规划演示。M0 用不到
      （数据是下载的），但 M2 若想给其他任务自制演示会撞上，届时那些任务官方也有现成演示可下。

## 执行步骤

### 1. 演示数据清单（已下载，已审计）

下载命令（已执行过，留档）：

```bash
conda run --no-capture-output -n robot-practice \
  python -m mani_skill.utils.download_demo PickCube-v1 \
  --output_dir datasets/raw/maniskill_demos
```

**实测清单。这不是预期值，是 2026-08-07 逐文件读出来的：**

| 来源 | 条数 | 动作维 | 总帧 | 均长 | 首次成功（中位） | 50 步内完成 | 后端 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `motionplanning/` | 1000 | 8（`pd_joint_pos`） | 77,976 | 78.0 | 73 | **1%** | `physx_cpu` |
| `rl/pd_ee_delta_pos` | 1022 | 4 | 51,001 | 49.9 | 10 | 100% | `physx_cuda` |
| **`rl/pd_ee_delta_pose`** | **1013** | **7** | **50,360** | **49.7** | **13** | **100%** | `physx_cuda` |
| `rl/pd_joint_delta_pos` | 997 | 8 | 49,539 | 49.7 | 18 | 100% | `physx_cuda` |
| `teleop/` | 10 | 8 | 1,058 | 105.8 | 90 | 10% | `physx_cpu` |

**录制形态（实测）**：rl 那三份的 `episode_id` 范围都是 0..1023 但不连续，seed 全不重复，所以是
**一次 1024 并行、每个环境只跑一局、把成功的留下**（1013/1024、1022/1024、997/1024），
`episode_id` 就是环境下标。整个过程没有自动重置。

**三份演示生成于三个不同的 commit，都不是本机装的 3.0.1 release**：

| 来源 | commit | branch |
| --- | --- | --- |
| `rl/` | `baab60ede2e8` | `il-baselines` |
| `motionplanning/` | `652ad9353c02` | `main` |
| `teleop/` | `ecc579b7567c` | `new-object-building-api` |

初始状态分布已验为同分布（见前置条件），但**动力学是否兼容未知**——replay 是"恢复首帧状态再重放
录制动作"，动力学有差异同一串动作就会走到别处。这是第 4 步 10 条试通要回答的问题，也是存活率
闸门的一条独立归因。

`rl/` 目录还附了三个 PPO checkpoint（`ppo_pd_ee_delta_pose_ckpt.pt` 等），第 9 步会用到。
五份数据的观测模式都是 `none`，**演示里没有任何图像**，只有 `actions`、`env_states`、
`terminated`、`truncated`、`success`（`motionplanning` 多一个 `rewards`）。所以图像只能靠
replay 重新仿真渲染，这是第 4、5 步存在的唯一理由。

**主线数据定为 `rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5`。** 原计划用
`motionplanning` 是错的，四条理由：

1. **motionplanning 的演示在标准评测 horizon 内解不出来。** 首次成功中位 **73 步**，而
   `max_episode_steps = 50`，只有 **1%** 落在 50 步内。原因在
   `mani_skill/examples/motionplanning/panda/run.py` 第 46 行：`gym.make` 没有覆盖 horizon，
   TimeLimit 就是 50，但求解器在 `truncated=True` 之后继续步进（最长 103 步），
   `RecordEpisode` 全录了下来。拿它训出来的策略照演示的节奏走，几乎必然在第 50 步被截断，
   而这会被误诊成管路故障。
2. **rl 那份原生就是 7 维 `pd_ee_delta_pose`**，不需要控制模式转换——原计划最大的风险点直接消失。
3. **均长 49.7、最长 50，与评测 horizon 匹配。**
4. 与 LIBERO 的 7 维 EEF 动作口径一致，后续 R2 移植不用换口径。

`motionplanning` 不删除，降级为辅助数据，三个用途写在第 11 步。

**数据量对照**（其余三行已核对本地产物）：

| | 训练 episode | 帧数 | 平均每条 | 观测 |
| --- | ---: | ---: | ---: | --- |
| PushT | 165（全库 206） | 全库 25,650 | 124.5 | 1 路 96×96，现成 |
| LIBERO 目标（C 家族） | 113 | 11,022 | 97.5 | 2 路 256×256，本仓库 replay 生成 |
| LIBERO 背景（B0） | 1,412 | 236,470 | 167.5 | 同上 |
| ManiSkill PickCube（主线） | 1,013 | 50,360 | 49.7 | **无，第 5 步生成** |

条数是 PushT 的 6 倍、LIBERO 目标任务的 9 倍，但每条短得多，换算成帧大约是 PushT 的两倍。

### 2. 原始数据审计：成功后的静止帧有多少

**这一步必须在任何 replay 与训练之前做完。** 起因是一个已经实测确认的问题：PPO 中位 13 步就成功，
但 episode 保存到接近 50 步，那中间那 37 步是什么？

**已实测的答案（这就是本步存在的理由，不是猜测）：**

```
首次成功步 中位 13    成功后剩余帧 中位 37
成功后帧占比：均值 72.8%  中位 74.0%  p10 68.0%  p90 79.6%
全库 50,360 帧，其中成功后 36,772 帧（73.0%）
成功一旦达成就保持到结尾：984/1013（97.1%）
动作绝对值均值：成功前 0.8009  成功后 0.3663  比值 0.46
```

**将近四分之三的训练信号在教"任务已经完成了，小幅乱动"。** 而且成功后动作不是零而是 0.37 的幅度，
行为克隆会把这个也学下来——放下之后又去碰，这是最容易破坏稳定性的部分。

本步要把上面这些数扩成完整报告并落盘，逐条记录：episode 长度、首次成功步、末次成功步、成功帧数、
成功是否一旦达成就持续、每个动作维度的分位数与标准差、成功前后的动作范数。

按 `τ = t − t_首次成功` 对齐所有轨迹，统计随 τ 变化的动作范数（平移与旋转分开）、夹爪动作、
TCP 位移、方块位移、成功概率。这张曲线要能直接看出"已成功 → 几乎不动 → 重复保存"的形状。

审计脚本用 CPU 后端（只读 h5，不需要 GPU），**必须与其他用 GPU 的脚本分成独立进程**，见"风险"一节。

产出 `outputs/maniskill/m0_probe/raw_data_audit.json` 与
`outputs/maniskill/m0_probe/success_tail_stats.json`。

### 3. 固定状态划分与初始状态泄漏审计

#### 3.1 seed 到状态的映射：三个必须知道的实测事实

这一节的三条都是实测的，而且第一条与第三条与直觉相反，实现评测入口时必须照着写。

**第一，一批环境的初始状态只由列表里的第一个 seed 和行号决定，其余 seed 完全不起作用。**

初始化代码 `pick_cube.py` 的 `_initialize_episode` 用的是**全局 `torch.rand`**，不是每个环境各自的
RNG：

```python
b = len(env_idx)
xyz[:, :2] = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
```

而播种在 `sapien_env.py` 第 947 行，**只用第一个环境的 seed**：

```python
torch.manual_seed(self._episode_seed[0])
self._initialize_episode(env_idx, options)
```

实测验证（改列表里第二个 seed 对所有环境都无影响，改第一个则全部改变）：

```
seeds=[2000, 2001] -> env0:[ 0.008543  0.050812] | env1:[ 0.093432 -0.059281]
seeds=[2000, 9999] -> env0:[ 0.008543  0.050812] | env1:[ 0.093432 -0.059281]   改 seed[1]，两个都没变
seeds=[8888, 2001] -> env0:[-0.019629 -0.027493] | env1:[-0.042937 -0.054013]   改 seed[0]，两个都变了
```

行号对应的状态**跨批大小稳定**。实测首个 seed 固定为 2000、其余 seed 每批都换，在 b = 1/2/4/8/16/64/200
下逐行比对，方块 xy、方块四元数、目标 xyz、机器人 qpos 全部一致（行 7 与行 15 在 b=8..200 下差异
在 1e-5 量级，是 GPU 浮点噪声不是不同状态）：

```
b=  8 row 7: cube_xy=[-0.017719  0.055131] goal=[ 0.05334  -0.09027   0.19693 ]
b= 16 row 7: cube_xy=[-0.017719  0.055131] goal=[ 0.05334  -0.09027   0.19693 ]
b=200 row 7: cube_xy=[-0.017719  0.05513 ] goal=[ 0.05334  -0.090271  0.19693 ]
```

**所以"一组评测种子"的正确说法是"一个首 seed 加一个条数"**，见 3.2。这也意味着要拿到第 199 行的
状态，必须跑一个至少 200 个环境的批——把它拆成两个 100 的批只会把前 100 行跑两遍。

**第二，同一个 seed 在 CPU 与 GPU 上给出不同状态。** 实测 seed=2000：

```
CPU  cube=[-0.089515 -0.094863 0.02]  goal=[-0.056865 -0.072339 0.236289]
GPU  cube=[ 0.008543  0.050812 0.02]  goal=[-0.031363  0.030398 0.255339]
```

所以**划分文件里必须把后端一起写死**。主线数据是 `physx_cuda` 录的，评测也用 `physx_cuda`，
划分按 GPU 定义。

**第三，rl 演示的 `episode_seed` 复现不出录制时的状态，而机制没定位。** rl 那份 1013 条**都有**
`episode_seed`（值是 40194941 这类大随机数，空的是 `reset_kwargs`），但复现不出来。

**这个事实已从四个方向验过，都是 0 命中**：

| 试法 | 结果 |
| --- | --- |
| CPU 后端、标量 seed、`num_envs=1` | 0/5 |
| `physx_cuda` 后端、seed 列表、`num_envs=5` 与 `1` | 0/5、0/1 |
| `physx_cuda`、首 seed 取 episode 0 的 seed、`num_envs=1024`、行号对 `episode_id` | **0/1013** |
| `reconfiguration_freq` 取 0 与取 1 | 两者结果相同，都不匹配 |

```
num_envs=5    ep0 seed=40194941  录制=[-0.02845 -0.028389]  reset=[0.09898 0.012233]
num_envs=1    ep0 seed=40194941  录制=[-0.02845 -0.028389]  reset=[0.09898 0.012233]
num_envs=1024 ep0 首seed=40194941 录制=[-0.02845 -0.028389]  reset=[0.09898 0.012233]
```

第三行那次是照第一条事实设计的：录制是一次 1024 并行、每环境一局（见第 1 步），`episode_id` 就是
环境下标，所以 `_episode_seed[0]` 应当是 episode 0 的 seed。若"首 seed 加行号"能复现，这一次就该
全中。结果 1013 条全不中。

**已排除的解释**（逐条都实测或读源码核过，不要再往这些方向猜）：

| 曾经的解释 | 排除依据 |
| --- | --- |
| GPU 随机化依赖 `num_envs` | 行号对应的状态跨 b=1..1024 稳定，见第一条 |
| `reconfiguration_freq` 的差异 | 取 0 与取 1 结果相同 |
| 批内首 seed 可复现整批 | `num_envs=1024` 那次 0/1013 |
| 版本间随机化数学变了 | 取 `baab60e` 的 `pick_cube.py` 与本机 3.0.1 逐行比：老代码写死 `*0.2-0.1` 与 `*0.3`，而本机运行时 `cube_spawn_half_size=0.1`、`cube_spawn_center=(0,0)`、`max_goal_height=0.3`，**两边等价**；`table_scene.initialize` 里 Panda 分支也未变 |
| 初始状态分布变了 | 录制分布与本机一致，见"前置条件"里的分布对照 |

**机制未定位，而且不影响本计划任何决定。** 剩下能让 RNG 流错位的候选太多（别处抽样次数、
`BatchedRNG` 后端、torch 版本、CUDA RNG 的 offset 记账、生成脚本自己的 seed 处理），即使把
`baab60e` 装起来复现成功也定位不到具体哪一条，失败更说明不了什么。**不要为此再花时间**，三条理由：

1. seed 能否复现只影响 seed-based isolation，本计划已换成 initial-state-based（3.3）
2. 录制分布与本机一致，所以 3.3 的状态比对是在比同类东西
3. 真正要紧的是"同一串动作在本机版本上会不会走到同一个地方"，那是动力学兼容问题，源码读不出来，
   由第 4 步的 10 条 replay 试通判定

**结论：比较 seed 整数无法判断训练与评测状态是否重叠**，必须比对实际状态，见 3.3。

#### 3.2 状态划分

写进 `configs/experiments/maniskill/split.json`，**一次写死不再改**。按 3.1 第一条，划分不是"一串
seed"而是**首 seed 加条数**，其余 seed 是装饰：

| 集合 | 首 seed | 条数 | 用途 |
| --- | ---: | ---: | --- |
| `train` | 0 | 2000 | M1 分支采样与后续 RL 在线交互 |
| `development` | 2000 | **200** | 选模、Full/Trim 对照、smoke |
| `test` | 3000 | **400** | 只在正式评测跑一次 |

划分文件必须同时记下四条约束：

1. `sim_backend: physx_cuda`（第二条：换后端就是另一批状态）
2. seed 以**列表**形式传入，列表首元素是该集合的首 seed
3. **一次 reset 的环境数等于该集合的条数**，不许拆批。拆批拿不到高行号的状态
4. 状态比对容差取 `1e-4`，因为 GPU 浮点噪声在 1e-5 量级

若显存装不下 400 个并行环境带 RGB（256 并行 128×128 已实测可行，400 未测），**不要改成拆批跑行号**，
而是给第二块换一个新的首 seed（例如 3400）并把两个首 seed 都记进划分文件，把并集当作 test 集。
这样每一块内部行号仍然唯一确定状态。

**样本量为什么是 200 和 400，而不是 50 和 100。** 判断标准里的阈值是 50%、10 个百分点这样的硬线，
样本量必须撑得住。在 p=0.5 处：

| 评测条数 | 标准误 | 95% 区间宽度 |
| ---: | ---: | ---: |
| 50 | 7.1 个百分点 | ±13.9 |
| 100 | 5.0 | ±9.8 |
| 200 | 3.5 | ±6.9 |
| 400 | 2.5 | ±4.9 |

50 条时 ±13.9 个百分点，"成功率 ≥50%"这条判据整个落在噪声带里。而代价几乎是零：按实测 5716 步/秒，
400 条 × 50 步 = 20,000 步，**纯仿真 3.5 秒**。策略前向才是主要开销，由第 7 步 smoke 实测。
PushT 那边 E2b 用 20 环境种子 × 10 策略种子 = 200 条 rollout 就是为了同一个理由。

#### 3.3 初始状态指纹与重复审计

对每条训练演示的首帧（直接读源 h5 的 `env_states`，不需要 replay）提取指纹：

```
f = [ 机器人 qpos, 方块 xyz, 方块四元数, 目标 xyz ]
```

评测状态那一侧不能靠 seed 推算，**必须真的 reset 一遍把状态取出来**（3.1 第三条的同一个理由：
seed 不是可算的状态标识）。取的时候按 3.2 的四条约束来，否则取到的不是评测时真正会用的状态。

对每个评测状态算它到训练集最近状态的距离，并**按分量分解报告**：机器人、方块位置、方块朝向、
目标位置各自的距离。两种结果都要记：

- **完全重复**：是否存在数值上相同的初始状态，**判定容差 `1e-4`**（GPU 浮点噪声在 1e-5 量级，
  用严格相等会漏判）
- **近邻分布**：每个评测状态离最近训练状态多远，给出分位数

**不要求训练与评测状态离得很远。** PickCube 的状态空间就那么大（方块 xy 在 ±0.1 米内，目标 xyz
在 ±0.1 米加 z 0.02–0.32 米），1000 条演示自然会铺满训练分布。要证明的只有一件事：**没有把某个
训练 episode 的初始状态原样拿来当评测状态。**

`motionplanning` 那份的 `reset_kwargs.seed` 是 0..1003 且在 CPU 单环境下录的，是唯一一份 seed
能复现状态的数据，可以用来交叉验证指纹提取实现本身是否正确。这是它的第三个用途。

产出 `outputs/maniskill/m0_probe/initial_state_fingerprints.parquet` 与
`outputs/maniskill/m0_probe/split_leakage_report.json`。

### 4. Replay 试通（10 条，独立一步）

**这一步单独列出来，因为它是最可能一上手就挂的地方，而全量 replay 是本探针的第一个闸门。**
不要直接跑 1013 条。

已知的两个隐患。第一，GPU 并行 replay 路径 `replay_parallelized_sim` 用
`episode["episode_seed"]` 做 reset：

```python
seeds = torch.tensor([episode["episode_seed"] for episode in episode_batch],
                     device=env.base_env.device)
```

而 3.1 已证明这个 seed 复现不出原状态。`--use-first-env-state` 应该会在 reset 之后覆盖状态，
但这个先后关系没有实测过。第二，CPU 后端不支持 `num_envs > 1`（实测报
`Cannot set the sim backend to 'cpu' and have multiple environments`），所以并行只能走 GPU。

```bash
traj="datasets/raw/maniskill_demos/PickCube-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"

conda run --no-capture-output -n robot-practice \
  python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${traj}" \
  --use-first-env-state \
  -c pd_ee_delta_pose -o rgb -b physx_cuda \
  --save-traj --allow-failure --max-retry 3 \
  --num-envs 10 --count 10
```

几个已核对的点：

- `--count N` 限制条数（第 459 行 `episodes = json_data["episodes"][: args.count]`）
- `-c pd_ee_delta_pose` 与源相同，**不会触发控制模式转换**（第 209 行
  `args.target_control_mode is None or ori_control_mode == args.target_control_mode`）。
  写出来是为了让命令自解释
- 用 `--use-first-env-state`（只对齐首帧状态）而**不是 `--use-env-states`**。后者与控制模式转换
  互斥，代码里有断言。这里虽然不转换，仍用前者以保持与 3b 诊断命令一致
- 后端名是 `physx_cpu` / `physx_cuda`
- 源是 `pd_ee_delta_pose`，转换代码只支持从 `pd_joint_pos` 或 `pd_joint_delta_pos` 出发
  （第 353 行），所以这份数据**转不成别的控制模式**——不需要，但要知道

试通要报三件事：10 条里成功 replay 几条、耗时多少、生成的 RGB 形状是否为 (128, 128, 3)。
**10 条不全过就停下来查，不要往下走。**

### 5. Replay 全量，算存活率与旋转通道统计

试通过了再跑全量：去掉 `--count 10`，`--num-envs` 按试通实测的显存占用上调。

**存活率必须算出来并落盘，不许假设是 100%。** 两个数分开记：

- `replay_ok`：replay 没报错的条数 / 源条数
- `replay_success`：replay 后仍达成 `success` 的条数 / 源成功条数（源是 1013/1013 全成功）

写进 `outputs/maniskill/m0_probe/replay_survival.json`。同时按 episode 保存诊断字段：
`success`、`is_grasped`、`is_obj_placed`、`is_robot_static`、时间步。

#### 5.1 旋转通道是否是死的

**这一项是必测产出，不是可选分析。**

PickCube 的目标信号是 `goal_pos`，形状 **(3,)——只有位置，没有朝向**。任务从不要求转手腕。所以
7 维动作里的三个旋转通道很可能接近常量，**残差在旋转上没有东西可纠正**。这也正是官方基准给
PickCube 和 StackCube 用 4 维、给 PegInsertionSide 和 PlugCharger 用 7 维的原因。已核对的对应关系：

| 任务 | 目标信号 | 朝向是否要紧 | 官方控制模式 | horizon |
| --- | --- | :-: | ---: | ---: |
| PickCube-v1 | `goal_pos` (3,) | 否 | 4 维 | 50 |
| StackCube-v1 | `cubeA_to_cubeB_pos` (3,) | 否 | 4 维 | 50 |
| PegInsertionSide-v1 | `box_hole_pose` (7,) | **是** | 7 维 | 100 |
| PlugCharger-v1 | `goal_pose` (7,) | **是** | 7 维 | 200 |

从源 h5 与 replay 产物两边都统计**每个动作维度的绝对值分位数与标准差**，尤其是第 4、5、6 维
（旋转）与平移三维的对比，写进 `outputs/maniskill/m0_probe/action_channel_stats.json`。

结论若是"旋转通道是死的"，那要换的是**任务**而不是控制模式——W 与 R 阶段改用
PegInsertionSide-v1 或 PlugCharger-v1。这个问题在 M0 就有答案，不用等到 R1 才发现旋转残差学不出
东西。

#### 5.2 关于 7 维口径的代价

官方基准对 PickCube 用的是 4 维 `pd_ee_delta_pos`，本计划用 7 维。所以 M0 的闭环成功率
**不能直接与官方基准的公开数字对比**。要对比就得另跑一份 4 维的（`rl/pd_ee_delta_pos` 那份现成，
1022 条）。M0 不做这件事。

选 7 维的理由：与 LIBERO 的 7 维口径一致，后续 R2 移植不用换；旋转通道死不死本身就是要测的。

### 6. 转成 LeRobotDataset（Full 与 Trim 两份）

#### 6.1 先试官方工具

**不要一上手自己写。** `mani_skill/trajectory/convert_to_lerobot.py`（589 行）目标格式就是
**LeRobot v3.0**，与本仓库装的 LeRobot 期望的 `CODEBASE_VERSION = v3.0` 一致，现有的 `pusht`
与 `libero` 两个数据集也都是 v3.0。

```bash
conda run --no-capture-output -n robot-practice \
  python -m mani_skill.trajectory.convert_to_lerobot \
  --traj-path <第 5 步的 replay 产物 .h5> \
  --output-dir datasets/lerobot/maniskill/pickcube_rl_7d_full \
  --image-size 128x128 \
  --task-name PickCube-v1 \
  --robot-type panda
```

**验收标准是"`lerobot-train` 能直接吃下去"**，不是"文件生成了"。跑第 7 步的 smoke 就是验收。

官方工具已知的三处不足，验收后再决定怎么补：

| | 官方转换器给的 | 本计划想要的 |
| --- | --- | --- |
| `observation.state` | **只有 `obs.agent.qpos`**（9 维） | qpos(9) + qvel(9) + `goal_pos`(3) = 21 |
| 视频编码 | OpenCV `cv2.VideoWriter`，`mp4v` | 仓库 LIBERO 路径用 `libsvtav1` |
| 默认图像尺寸 | `640x480` | 128×128（靠 `--image-size` 改） |

**处理顺序**：先用官方产物把管路走通（state 只有 qpos 也能训出策略，只是弱一些）。管路通了之后，
若第 9 步的成功率不达标，再判断是不是 state 太薄导致的，那时候才补一层后处理或自己写转换器。
**不要在管路还没验证时就先花时间做完整转换器**——这跟 M0"早失败"的定位相违。

自己写的话，可抄的模板是 `src/robot_practice/evaluation/libero_dataset_cache.py`：
`LeRobotDataset.create(repo_id=..., root=..., robot_type="panda", fps=..., features=...,
use_videos=True, image_writer_threads=4, vcodec="libsvtav1")`，然后逐帧 `add_frame` 再
`save_episode(parallel_encoding=False)`。

#### 6.2 观测与动作字段

| key | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `observation.images.base_camera` | video | (128, 128, 3) | 实测 `sensor_data.base_camera.rgb` 就是这个形状 |
| `observation.state` | float32 | (21,) | `agent.qpos`(9) + `agent.qvel`(9) + `extra.goal_pos`(3) |
| `action` | float32 | (7,) | `pd_ee_delta_pose`：平移 3 + 旋转 3 + 夹爪 1 |

所以这**不是纯视觉策略**。准确的说法是：以 RGB 为主、带本体感受与目标坐标的策略。

**`goal_pos` 必须保留，它不是特权信息。** `PickCubeEnv` 里
`self._hidden_objects.append(self.goal_site)`，而 `sapien_env.py` 第 601 行在
`capture_sensor_data()` 之前对所有 `_hidden_objects` 调 `hide_visual()`——目标球只在人看的
viewer 里显示，**相机观测里看不见**。所以拿掉 `goal_pos` 策略就无从知道要把方块放哪，任务变无解。
这条写下来是为了拦住"把 goal_pos 当特权信息做消融"这个看起来合理但会把任务判死的建议。

**刻意排除的特权变量**，理由写进 manifest：

| 排除项 | 理由 |
| --- | --- |
| `extra.is_grasped` | 与成功判定同源，给了等于泄题 |
| `extra.obj_pose` | 方块的精确位姿，给了等于替策略把感知做了 |
| `extra.tcp_to_obj_pos` | 直接告诉策略该朝哪走去抓 |
| `extra.obj_to_goal_pos` | 直接告诉策略抓到后该朝哪走去放 |
| `extra.tcp_pose` | 可由 `qpos` 推出，冗余 |

**这五项正是 PPO 教师用的东西。** 教师的输入语义是 `obs_mode='state'` 对应的 42 维，实测逐项为
`qpos`(9) + `qvel`(9) + `is_grasped`(1) + `tcp_pose`(7) + `goal_pos`(3) + `obj_pose`(7) +
`tcp_to_obj_pos`(3) + `obj_to_goal_pos`(3) = 42，与 PPO actor 第一层 `(256, 42)` 严丝合缝。
也就是说教师连"该往哪个方向走"的向量都是直接拿到的，这就是它中位 13 步完成的原因——它不需要感知。

实现注意：ManiSkill 3.0.1 的 `obs_mode="state"` API 返回展平后的 `[B,42]` Tensor，不能按
`observation["extra"]` 读取字段。指纹审计和 PPO 参照实现使用 `obs_mode="state_dict"` 取得命名的
`agent`/`extra` 字段，再按上述顺序拼接 42 维；输入语义和 checkpoint 的 42 维顺序不变。

**所以 M0 实际在做的是把一个特权专家蒸馏成一个视觉学生。** 这是标准做法，但必须提前认下后果：
基座成功率天然明显低于 PPO 教师。这一条要写进 manifest，见第 9 步。

**物体位姿一律不进 state。** 等到 M2 选中 PegInsertionSide 之类需要朝向的任务时，再单独决定
要不要给、要不要做消融——那是那时候的选择，不是现在的结论。

#### 6.3 Full 与 Trim 两份

**Full**：保留完整轨迹，1013 条 / 50,360 帧。

**Trim**：第 i 条截到 `min(T_i, t_i^首次成功 + 5)`。也就是首次成功后只留 5 帧稳定行为。

预期规模：首次成功中位 13，加 5 帧约 18 帧，全库大约 **18,000–20,000 帧**，是 Full 的四成。

**为什么 5 帧是安全的。** PickCube 的成功条件本身就含 `is_robot_static`，所以首次成功那一刻
手臂已经停稳了，后面的帧是纯填充。而实测 984/1013（97.1%）的 episode 成功一旦达成就保持到结尾，
说明成功极少抖动，在首次成功处截断不会切掉"还会失败"的部分。留 5 帧是为了让策略学会停住而不是
继续漂。

**5 这个数是工程选择不是原理，但必须在看到第 7 步结果之前定死。** 若 Trim 明显更好，扫 k 这件事
留给 M2，M0 不做。

两份都转，目录分开：`pickcube_rl_7d_full/` 与 `pickcube_rl_7d_trim/`。

转换器默认只把原生 `success` 的 episode 写进训练数据。源数据 1013 条全成功，所以这里不筛掉任何
条目；但这条约束要写进实现，供后续 M3 采策略 rollout 时复用。

### 7. Full vs Trim smoke 对照，同时验收官方转换器

**这一步同时干三件事**：验收 6.1 的官方转换器、定下 Full 还是 Trim、实测耗时反推全量时长。

两份数据集各训一个短 smoke，**除数据集外全部相同**：种子、模型结构、更新步数、batch size、
动作归一化。步数取正式训练量的 10%–20%。

```bash
for ds in full trim; do
conda run --no-capture-output -n robot-practice lerobot-train \
  --dataset.repo_id=local/maniskill_pickcube_rl_7d_${ds} \
  --dataset.root=datasets/lerobot/maniskill/pickcube_rl_7d_${ds} \
  --policy.type=diffusion --policy.device=cuda --policy.push_to_hub=false \
  --policy.n_obs_steps=2 --policy.horizon=16 --policy.n_action_steps=8 \
  --output_dir=outputs/maniskill/m0_probe/smoke/train_${ds} \
  --job_name=m0_pickcube_${ds}_smoke --seed=1000 \
  --batch_size=64 --num_workers=4 \
  --steps=2000 --save_freq=1000 --log_freq=200 --eval_freq=0 --wandb.enable=false
done
```

然后在**同一批 development 状态**上闭环评测两者。smoke 阶段用 development 的前 50 条即可
（这里只是定方向，不用撑住硬阈值）。

判断：

- **成功率差异 < 5 个百分点** → 用 **Full**，理由是最简单
- **Trim 明显更好** → 正式训练用 Trim，并在结论里记录"原始 RL 数据含 73% 成功后填充帧，
  截断后闭环性能提升 X 个百分点"

若 Full/Trim 差异极大（比如 Trim 高出 20 个百分点以上），再考虑第三份 **Balanced**：成功前全留，
成功后按固定比例随机保留。**只有在 Full 与 Trim 差距很大时才做**，否则不做。

耗时也在这一步落盘：转换 1013 条要多久、训 2000 步要多久、评 50 条要多久。据此反推第 8 步的
`--steps`。**不许按纸面数字直接开全量。**

### 8. 训练正式基线

Full/Trim 选择完成后，正式训练固定使用 Full。LeRobot 0.4.4 的
`lerobot-train` 通过 Accelerate 识别分布式环境，本计划用薄的仓库 launcher 把显式的
`--num-gpus` 转成 8 个训练进程；不要把 `--num-gpus` 直接传给 `lerobot-train`。

```bash
bash scripts/experiments/maniskill/train_formal.sh \
  --num-gpus=8 \
  --dataset-variant=full \
  --output-dir=outputs/maniskill/m0_probe/train_main \
  --job-name=m0_pickcube_main \
  --batch-size=8 \
  --steps=20000 \
  --save-freq=10000 \
  --log-freq=200 \
  --seed=1000 \
  --num-workers=4
```

这里 `--batch-size` 是每张卡的 batch；8 卡下全局 batch 为 `8 × 8 = 64`，与 smoke 的全局
batch 保持一致。`--steps=20000` 是正式训练的 optimizer step 预算；如果 smoke 重新测得的
耗时或显存约束发生变化，只调整这个预算或每卡 batch，不改变数据划分和评测口径。

训练前可只检查最终命令，不会创建输出：

```bash
bash scripts/experiments/maniskill/train_formal.sh --num-gpus=8 --print-command
```

`--eval_freq=0` 是因为 LeRobot 的在线评测走它自己的 env 层，而本纲要明确不走那一层（LeRobot 的
LIBERO wrapper 里 `if terminated: self.reset()` 正是 Prompt Swap bug 的成因）。闭环评测在第 9 步
自己做。

`n_obs_steps=2`、`horizon=16`、`n_action_steps=8` 与 PushT 那套一致（`e2_rollout.json` 里
`n_action_steps = 8`），便于跨环境对照。

**只训一个种子。** M0 是基础设施闸门不是正式算法结果，多种子留给 M1 之后。

**选模只用 development 状态。** 禁止用 test 状态选 checkpoint，禁止看了 test 再回去改训练。

### 9. 闭环评测：三个参照系统

需要新写 `src/robot_practice/experiments/maniskill/eval_closed_loop.py`。要点：

- 用 `gym.make("PickCube-v1", num_envs=N, obs_mode="rgb", sim_backend="physx_cuda",
  control_mode="pd_ee_delta_pose")`，**不套 LeRobot 的 env 层**
- **按 3.2 的四条约束 reset**：seed 以列表传入且首元素是该集合的首 seed、一次 reset 的环境数等于
  集合条数、后端固定 `physx_cuda`、状态比对容差 `1e-4`。**列表里第一个 seed 之外的值不起作用**
  （3.1 第一条），所以拆批跑会静默换掉状态集合
- **成功判定只用 `info["success"]`，不要自己写谓词。** 同时记录
  `is_grasped` / `is_obj_placed` / `is_robot_static` 三个原生阶段信号

同一批状态上跑三个系统：

| 系统 | 输入 | 角色 |
| --- | --- | --- |
| **随机动作** | 无 | 确认评测器的下界不是异常的。数字要实测，不许假定是 0 |
| **特权 PPO** | 官方 42 维 state | **特权教师参照** |
| **视觉基座** | RGB + qpos/qvel + goal_pos | 本探针的产物 |

**PPO 那一栏不要叫上限。** 它与视觉 diffusion 在观测、结构、优化目标、策略类上全都不同，视觉策略
原则上可以超过它。它的角色只是"知道全部状态时这个任务能做到什么程度"的一个参照点。

**两者的差值也不要解释成纯感知代价。** 差值里同时含 BC 对 RL 的目标差异、chunk 结构、数据量限制、
以及第 6.2 节列的五项特权变量。只报差值，不做因果归因。

#### 9.1 要记录的指标

**成功率分两个记，不要合并：**

- `success_once`：episode 中是否曾经 `success = 1`
- `success_at_end`：最后一步是否 `success = 1`

分开的理由是 PickCube 的成功条件含 `is_robot_static`，可能抖动；而 Prompt Swap 那个 bug 的教训
正是"在什么时刻读成功"能把一整列清零。

其余指标：

- **首次成功步**：成功 episode 的均值、中位、p90。可以直接看出视觉学生比 PPO 教师慢多少
- **阶段达成率**：`grasped_rate`、`placed_rate`、`static_success_rate`，用来区分
  没抓住 → 抓住但没搬到 → 放到了但不稳 → 成功
- **控制指标**：平移动作范数、旋转动作范数、动作抖动、夹爪翻转次数、episode 长度
- **延迟**：策略前向 p50/p95、环境步进耗时

产出 `metrics.json`、`episodes.parquet`（逐条记录，供后续配对比较）、`run_manifest.json`
（策略路径、数据集路径、控制模式、种子集合、后端、state 字段构成、**"示范者是特权专家、学生是
视觉策略，两者成功率不可直接比较"这条说明**）。

#### 9.2 评测规模

- 选模与 Full/Trim 对照：development 200 条
- 正式评测：test 400 条，**只跑一次**

纯仿真成本 400 × 50 / 5716 ≈ 3.5 秒，策略前向是主要开销，由第 7 步 smoke 实测后填进来。

### 10. 回归测试

新增 `tests/test_maniskill_convert_demos.py` 与 `tests/test_maniskill_eval_closed_loop.py`：

- 转换器：features 形状与声明一致；帧数等于源 episode 长度（Trim 那份等于截断后长度）；
  state 里**不含** `is_grasped`、`obj_pose`、`tcp_to_obj_pos`、`obj_to_goal_pos`
- 划分：划分文件里记了首 seed、条数、后端、容差四项；三个集合的首 seed 互不相同
- 划分语义：同一首 seed 下第 i 行的状态跨批大小稳定（用 b 与 2b 各 reset 一次比对，容差 `1e-4`）；
  改列表里非首位的 seed 不改变任何环境的状态
- 泄漏审计：能在构造出的"评测状态与训练状态完全相同"的输入上报出重复
- 评测器：成功判定读的是 `info["success"]`，不存在自写谓词；`success_once` 与 `success_at_end`
  分开落盘；随机基线也被评测并落盘

**测试不许用 `SimpleNamespace` 假 env 冒充真实环境链。** 现有的
`tests/test_smolvla_libero_target_protocol_evaluation.py:193` 就是这么写的，结果只证明了 patch
能作用在玩具对象上，覆盖不到真实 env，Prompt Swap 的 bug 因此漏过测试。新测试要么用真实
ManiSkill env（CPU 单环境足够快），要么明确标注它只覆盖纯函数部分。

```bash
conda run --no-capture-output -n robot-practice python -m pytest \
  tests/test_maniskill_convert_demos.py tests/test_maniskill_eval_closed_loop.py -q
conda run --no-capture-output -n robot-practice ruff check \
  src/robot_practice/experiments/maniskill tests/test_maniskill_*.py
conda run --no-capture-output -n robot-practice ruff format --check \
  src/robot_practice/experiments/maniskill tests/test_maniskill_*.py
```

### 11. motionplanning 数据的三个用途

不删除，但**不进主线**。

**一、控制模式转换诊断。** `pd_joint_pos` → `pd_ee_delta_pose` 是官方支持的转换路径
（`conversion.py` 的 `from_pd_joint_pos_to_ee` 同时支持 4 维 `root_translation` 与 7 维
`root_translation:root_aligned_body_rotation`）。想知道 ManiSkill 的转换管路有多可靠就跑它，
但结果只作诊断，不进 M0 的主结论：

```bash
traj="datasets/raw/maniskill_demos/PickCube-v1/motionplanning/trajectory.h5"
conda run --no-capture-output -n robot-practice \
  python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "${traj}" --use-first-env-state \
  -c pd_ee_delta_pose -o rgb -b physx_cpu \
  --save-traj --record-rewards --allow-failure --max-retry 3 \
  --num-envs 1 --count 20
```

**二、主线的两个退路。** 两种情况会切到它：rl 数据训出的基座候选多样性不足（M1 需要）；或者第 4 步
判定 rl 那份跨版本动力学不兼容。切过去时必须把**评测 horizon 从 50 放宽到 100**（它首次成功中位
73 步、p90 84 步、最大 99 步，100 能覆盖几乎全部），并把这个偏离写进结论——**不能和 PickCube-50 的
主结果混在一起**。

**三、指纹提取实现的交叉验证。** 它的 `reset_kwargs.seed` 是 0..1003，且在 CPU 单环境下录的，
所以 `reset(seed=k)` **有可能**复现录制状态。**这是假设，未验证**——rl 那份四种试法全 0 命中
（3.1 第三条），所以不要预设 motionplanning 一定行。真验证方法：随机取 20 条，在 `physx_cpu`、
`num_envs=1`、seed 按列表传的条件下 reset，与录制首帧按容差 `1e-4` 比对。**若它也复现不出来，
说明 seed 在本机对任何一份官方演示都不是可用的状态标识，那就只剩指纹这条路**，而 3.3 的实现改用
自己 reset 出来的状态两两比对来自检（同一首 seed 同一行号跑两次应逐值相同）。

## 判断标准

**执行前写死，看到数字之后不得修改。**

| 判据 | 阈值 | 不满足时怎么办 |
| --- | --- | --- |
| 演示清单 | 与第 1 步实测表一致 | 不一致说明下错了或版本变了，停下来查 |
| 成功后填充帧比例 | 已实测并落盘（当前 73.0%） | 无阈值，如实记录。它决定第 6.3 步的 Trim 是否必要 |
| Replay 试通（10 条） | 10/10 通过 | 停止。查后端、状态恢复、`episode_seed` 与 `--use-first-env-state` 的先后关系 |
| **Replay 存活率（全量）** | **≥ 95%** | **管路判定失败**，不要往训练走。两条归因要分开查：一是 GPU 并行路径的状态恢复（`episode_seed` 复现不出状态，靠 `--use-first-env-state` 兜），二是**跨版本动力学**（演示生成于 `il-baselines` 等 commit，本机是 3.0.1）。若判定为跨版本动力学不兼容，改用 `motionplanning/` 那份（`main` 分支生成，更接近 release），并按第 11 步用途二把评测 horizon 放宽到 100，同时把这个偏离写进结论 |
| 初始状态重复 | 容差 `1e-4` 下无重复，且近邻分布已落盘 | 有重复则**换一个新的首 seed 重取整块**，不要挑单条剔除（行号与首 seed 绑定，剔一条会打乱后面的行号）。换用的首 seed 与原因写进 manifest |
| 旋转通道是否有效 | 三个旋转维度的绝对值中位数 ≥ 平移三维中位数的 10% | **不算 M0 失败。** 记录"PickCube 的 7 维实际是 4 维"，并在进 W/R 之前把任务换成 PegInsertionSide-v1 或 PlugCharger-v1 |
| 官方转换器可用性 | 已实测：产物能否被 `lerobot-train` 直接吃下去 | 不能则记录失败原因，再自己写转换器 |
| Full vs Trim | 差异 < 5 个百分点用 Full；Trim 明显更好则用 Trim | 无失败分支，只是选一个 |
| 随机基线 | 已在同一批状态上实测并落盘 | 补测。不许假定它是 0 |
| 特权 PPO 参照 | 已在同一批状态上实测并落盘 | 补测。它是解读视觉基座数字的必要参照 |
| **视觉基座闭环成功率（development 200 条）** | **≥ 50%** | **不算管路失败**，见下面"两个分支" |
| 成功判定来源 | 只用 `info["success"]`，代码中不存在自写成功谓词 | 改实现。这条是 RF7 与 Prompt Swap 两个 bug 换来的 |
| 测试与 ruff | 全绿 | 补测试 |

那个 10% 的旋转阈值是**工程门槛不是公认标准**，作用只是把"旋转通道是死的"这件事变成一个执行前
就定好的二值判断，避免看到数字之后临时解释。

### 两个必须分开的失败分支

**管路失败与学不会是两件事，不要混。** 原来的"成功率 <50% 就是管路失败"这条判据是错的，已删。

**分支一：replay 存活率 < 95%。** 这才是真正的管路失败。演示数据都没法在同一后端上重新仿真出来，
后面全部无意义。走上游计划的退路：回 PushT 只做 W0/W1 机制核查，并明确承认 W2 结论是弱证据。

**分支二:replay 正常，但视觉基座成功率 < 50%,同时特权 PPO 明显更高。** 这**不是**管路失败,
是视觉行为克隆基线还没达到可用水平。因为示范者是特权专家（见 6.2），学生看不到方块位姿，性能差
可能来自感知或 BC 本身。依次检查：Full vs Trim、动作归一化、图像预处理、观测时序对齐、
执行 horizon、chunk 长度、表征、选模。查完仍不达标才判 M0 不通过。

### 成功率过高也要停下来

若视觉基座成功率 **≥ 95%**，管路是通的，但 PickCube 太容易，**M1 会没有可选择空间**（对照：
PushT 的 single-sample 0.55 对 Oracle@10 0.90 才有 35 个百分点可挑）。

这不算 M0 失败，但必须在进 M1 之前把 WAM 主环境换成 StackCube-v1（实测 8871 步/秒，多阶段且会
倒塌）。这条分支现在写下来，免得到时候临时改判据。

**注意 rl 数据本身就抬高了这个风险。** PPO 稠密奖励收敛出的是接近最优的窄行为，中位 13 步、
50 步预算下就解完了。如果基座学像了，没有可选择空间是因为**没东西可改**——是天花板效应。
这与"候选动作是否多样"是两个不同的机制，不要混为一谈；后者是 M1 要测的经验问题，M0 不预判。

**不预设成功率数值。** PickCube 是这套 benchmark 里最简单的任务，但演示来自特权专家，
和视觉策略能学到什么是两件事。照实记录。

## 产物路径

- 演示原始数据：`datasets/raw/maniskill_demos/PickCube-v1/{rl,motionplanning,teleop}/`
- 原始数据审计：`outputs/maniskill/m0_probe/raw_data_audit.json`、
  `outputs/maniskill/m0_probe/success_tail_stats.json`
- 状态划分：`configs/experiments/maniskill/split.json`
- 初始状态审计：`outputs/maniskill/m0_probe/initial_state_fingerprints.parquet`、
  `outputs/maniskill/m0_probe/split_leakage_report.json`
- 存活率：`outputs/maniskill/m0_probe/replay_survival.json`
- 动作通道统计：`outputs/maniskill/m0_probe/action_channel_stats.json`
- 转换后数据集：`datasets/lerobot/maniskill/pickcube_rl_7d_full/`、
  `datasets/lerobot/maniskill/pickcube_rl_7d_trim/`（最终只保留选中的那份作正式数据）
- smoke：`outputs/maniskill/m0_probe/smoke/{train_full,train_trim,eval_full,eval_trim}/`
- 正式训练：`outputs/maniskill/m0_probe/train_main/`
- 正式评测：`outputs/maniskill/m0_probe/eval/{random,ppo_privileged,diffusion_visual}/` 与
  `outputs/maniskill/m0_probe/eval/comparison.json`，每个至少含
  `metrics.json`、`episodes.parquet`、`run_manifest.json`
- 新增代码：`src/robot_practice/experiments/maniskill/{audit_demos.py,convert_demos.py,
  eval_closed_loop.py}`、`scripts/experiments/maniskill/{convert_demos.sh,eval_closed_loop.sh,
  train_formal.sh}`
- 测试：`tests/test_maniskill_convert_demos.py`、`tests/test_maniskill_eval_closed_loop.py`、
  `tests/test_maniskill_training_launcher.py`

## 不在本计划范围

- M1 的候选空间审计与 Oracle@8 headroom 探针。M0 只产出冻结基线，不做分支采样
- WAM、Residual RL、LIBERO 相关的一切
- 除 PickCube-v1 之外的任务。StackCube-v1 与 PegInsertionSide-v1 等 M0 通过后再接
- 建仓库风格的 ManiSkill 训练入口。M0 直接用 `lerobot-train`，薄一点好
- **任务选型。** M0 只验 PickCube 这一条管路。四个候选任务的横向选型是 M2，见上游计划
- **物体位姿要不要进 `observation.state`。** 到 M2 选中需要朝向的任务时再定
- **Trim 的 k 值扫描。** k 定在 5，若 Trim 明显更好，扫 k 留给 M2
- **与官方基准公开数字的对比。** 官方对 PickCube 用 4 维，本计划用 7 维，口径不同
- LeRobot 的 ManiSkill env 适配。本纲要明确不走 LeRobot 的 env 层

## 风险与停止条件

- **最可能卡在第 4 步的 replay 试通，而且有两个独立成因。** 演示里没有观测，图像全靠重新仿真加渲染
  生成。成因一：GPU 并行 replay 路径用 `episode_seed` 做 reset，那个 seed 复现不出录制状态
  （3.1 已实测四种试法全 0 命中），要靠 `--use-first-env-state` 盖过去，但这个先后关系没验过。
  成因二：**跨版本动力学**——三份演示分别生成于 `il-baselines`、`main`、`new-object-building-api`
  三个 commit，本机是 3.0.1 release。初始状态分布已验为同分布，但"同一串动作在本机版本上会不会走到
  同一个地方"源码读不出来。**先跑 10 条，别跑 1013 条**
- **不要为了定位 seed 复现失败的机制去装第二套 ManiSkill。** 3.1 第三条已列出五条被排除的解释，
  剩下的候选（别处抽样次数、`BatchedRNG` 后端、torch 版本、CUDA RNG offset 记账、生成脚本自己的
  seed 处理）即使复现成功也定位不到具体哪一条。而它不影响本计划任何决定：隔离已改成比对实际状态，
  分布已验一致，动力学兼容由 10 条试通判定
- **GPU PhysX 一个进程只能启一次。** 实测同进程先建 CPU env 再建 GPU env 会报
  `GPU PhysX can only be enabled once`。第 2 步的审计（CPU 读 h5）、3.3 的指纹提取、
  第 9 步的 GPU 评测**必须拆成独立进程**，不要写成一个大脚本
- **CPU 后端不支持 `num_envs > 1`。** 实测报
  `Cannot set the sim backend to 'cpu' and have multiple environments`。要并行只能走 GPU
- **一批状态只由列表首个 seed 与行号决定，其余 seed 不起作用**（3.1 第一条已实测：改 seed[1]
  两个环境都不变，改 seed[0] 全变）。所以**不许把一个集合拆成小批跑**——拆成两个 100 的批只会把
  前 100 行跑两遍，而且不会报错。装不下就换首 seed 另起一块，见 3.2
- **同一 seed 在 CPU 与 GPU 上是不同状态**（3.1 已实测）。划分文件必须把后端写死
- **不要用 `--use-env-states` 配控制模式转换。** 代码里有断言禁止：转换会改变达成同一状态所需的
  动作步数
- **不要把特权变量放进 state。** 教师用的 42 维里有五项是特权的（6.2 有清单），学生一项都不该拿。
  但 `goal_pos` 不在其中，目标球在相机里被隐藏，拿掉任务就无解
- **不要自己写成功谓词。** ManiSkill 的 `info["success"]` 和三个阶段信号都是原生的，直接用。
  RF7 的旋转参考角用了 0 弧度而非 `π/4`（唯一成功那条报 45.83°、改对后 0.83°），Prompt Swap 在
  环境重置**之后**才评估把一整列清零成 0/10——两次都是自写判据出的事
- **不要沿用 LeRobot 的 env wrapper 语义。** 那个 `if terminated: self.reset()` 的自动重置就是
  Prompt Swap bug 的机制。ManiSkill 原生接口本身向量化，不需要它
- **不要把评测规模砍到 50 条。** 判断标准里的 50% 与后续 M1 的 10 个百分点都撑不住那个样本量
  （3.2 有对照表），而在 ManiSkill 上样本量几乎免费
- **不要把视觉基座与 PPO 教师的差值当成感知代价**，也不要把 PPO 叫成上限。只报差值
- **`--overwrite` 类操作要小心。** 仓库里 `train_b0.py:74` 的 `_clear_run_artifacts` 直接
  `shutil.rmtree` 整个 run 目录，B0 的背景 test 与选模结果已经因此丢过一次。新写的 ManiSkill 入口
  **不要复制这个行为**，已存在的输出目录默认拒绝而不是删除
- **成功率过高也要停下来处理**，见判断标准。带着一个没有可选择空间的基线进 M1 是白跑

## 收尾

- [ ] 把本文件 `状态` 改为 `done`
- [ ] 更新 `plans/README.md` 索引
- [ ] 在上游计划里把阶段零 M0 勾掉，或按退路记录失败
- [ ] M0 通过后进 M1（候选空间审计），M1 通过后进 M2（任务选型闸门），见上游计划
- [ ] 结论按 `reports/AGENTS.md` 登记；**replay 存活率、视觉基座闭环成功率、特权 PPO 参照值**
      是必须写进结论的三个数
