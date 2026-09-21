# DM05-libero 在 Square 数据上的零样本推理

## 实验目标

验证现有 DM05-libero 权重能否处理本地 Square 数据，并观察未经 Square FT 的离线
动作误差与一个示教初始场景的闭环行为。该实验是 `Offline / Smoke` 和
`Closed-loop / Smoke`，不作为正式 Square benchmark 或模型能力上限。

## 实验思路

- NPZ 包含 100 条示教、27214 帧，原始 HDF5 包含 300 条示教。
- NPZ 只有主相机；按完整动作序列匹配 HDF5，恢复状态并补取真实腕部图像。
- 每个采样帧核验仿真状态、归一化 proprio 和主相机像素误差。
- 4 条轨迹各取 8 个完整 10-step 窗口，共 32 个样本；不跨 episode。
- 保留 LIBERO checkpoint 及其归一化；模型返回值与 Square 原始控制输入直接比较。
- 加入零动作和整个 NPZ 训练集均值动作基线，分开统计 EEF 与夹爪误差。
- 从首条匹配示教的 frame 0 运行一次最多 600 步的真实闭环；每 10 步重规划。
- 不执行训练，不重新下载权重，不根据此次结果自动调整 prompt 或统计量。

## 输入协议

| 项目 | 处理 |
| --- | --- |
| 主相机 | 离线用 NPZ 原始 `[3,96,96]` 图像，转 HWC；闭环用实时 agentview |
| 腕部相机 | 从对应 HDF5 场景恢复后真实渲染，不复制主相机 |
| 图像方向 | robosuite 原始渲染图像垂直翻转，与 Square NPZ 协议核验；不机械照搬 LIBERO 双轴翻转 |
| 状态 | 验证原始 9D position/quaternion/gripper 与 NPZ 一致，再把 quaternion 转 axis-angle 得到 8D |
| 状态条件 | 当前 LIBERO 模型 `add_state=False`，state 仅满足服务接口，不构成有效 token 条件 |
| 动作 | 7D delta OSC_POSE 控制输入，控制器内部平移/旋转缩放沿用数据元信息 |
| 归一化 | NPZ 的 min/max 仅用于匹配、恢复核验及均值基线；不替换 DM05 的 LIBERO q01/q99 |
| 任务指令 | `pick up the square nut and place it onto the square peg` |
| 精度与后端 | BF16 权重、默认 PyTorch 后端，沿用本地已验证加载入口 |

环境由已有 robot-practice 的 robosuite 直接创建，避免其旧 robomimic wrapper 对
mujoco_py 的依赖。仿真场景 XML 经当前 robosuite 的路径处理恢复。
版本及动态差异仍是后续正式评测需要控制的因素，恢复核验不替代完整专家动作 replay。

## 完整命令

在仓库根目录执行，前置环境、源码和 checkpoint 见 [DM0.5 本地指南](README.md)。
各命令重复运行会覆盖对应结果；如需对比新配置，改用新的 `--output` 目录。

### 1. 准备并核验样本

```bash
mkdir -p outputs/dm05/square_probe
set -o pipefail
PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  conda run --no-capture-output -n robot-practice python \
  src/robot_practice/experiments/dm05_square_probe.py \
  --mode prepare \
  --npz datasets/raw/dppo/robomimic/square-img/train.npz \
  --normalization datasets/raw/dppo/robomimic/square-img/normalization.npz \
  --hdf5 datasets/raw/robomimic/square/mh/demo_v141.hdf5 \
  --trajectory-ids 0,25,50,75 --frames-per-trajectory 8 --seed 7 \
  --output outputs/dm05/square_probe \
  2>&1 | tee outputs/dm05/square_probe/prepare.log
```

### 2. 在另一个终端启动模型服务

```bash
mkdir -p outputs/dm05/square_probe/server
set -o pipefail
PYTHONNOUSERSITE=1 outputs/dm05/env/bin/python \
  src/robot_practice/experiments/dm05_local.py \
  --opendm-dir outputs/dm05/opendm \
  --checkpoint outputs/dm05/checkpoints/DM05-libero \
  --serve --port 7891 --seed 7 --output outputs/dm05/square_probe/server \
  2>&1 | tee outputs/dm05/square_probe/server/run.log
```

### 3. 离线推理

```bash
set -o pipefail
PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice python \
  src/robot_practice/experiments/dm05_square_probe.py \
  --mode offline --url http://127.0.0.1:7891/v1/infer \
  --npz datasets/raw/dppo/robomimic/square-img/train.npz \
  --normalization datasets/raw/dppo/robomimic/square-img/normalization.npz \
  --prompt 'pick up the square nut and place it onto the square peg' \
  --seed 7 --output outputs/dm05/square_probe \
  2>&1 | tee outputs/dm05/square_probe/offline.log
```

每个请求使用 `seed + 样本序号`，保留完整模型输出，另外统计裁到 `[-1,1]` 的误差。

### 4. 单场景闭环

```bash
set -o pipefail
PYTHONNOUSERSITE=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  conda run --no-capture-output -n robot-practice python \
  src/robot_practice/experiments/dm05_square_probe.py \
  --mode rollout --url http://127.0.0.1:7891/v1/infer \
  --npz datasets/raw/dppo/robomimic/square-img/train.npz \
  --normalization datasets/raw/dppo/robomimic/square-img/normalization.npz \
  --hdf5 datasets/raw/robomimic/square/mh/demo_v141.hdf5 \
  --trajectory-ids 0 --max-steps 600 \
  --prompt 'pick up the square nut and place it onto the square peg' \
  --seed 7 --output outputs/dm05/square_probe \
  2>&1 | tee outputs/dm05/square_probe/rollout.log
```

仅初始化恢复一次示教状态；之后完全由模型控制，不进行逐帧 teacher forcing。
动作执行前裁到 `[-1,1]`，每个动作块使用 `seed + 请求序号`。结束后停止模型服务释放显存。

## 判断标准与停止条件

- 完整动作序列匹配必须唯一，归一化最大误差 ≤2e-6。
- 每个恢复帧：仿真状态最大误差 ≤1e-6、归一化 proprio 最大误差 ≤1e-3、主相机
  RGB MAE ≤10（像素 0–255）。任一失败则停止，不用错配图像继续推理。
- 接口通过：32/32 请求返回有限的 `(10,7)` 动作；出现 HTTP 错误、OOM 或非有限动作则停止。
- 离线比较控制输入单位下的 MSE/MAE，不将其当作闭环成功率；误差高于简单基线说明
  本抽样上没有展示出模仿优势。
- 闭环成功采用任意时刻 `env._check_success()`，最多 600 步。
  此口径不等于 DPPO 每个 4-step chunk 累计 reward 的成功口径，不直接合并统计。
- 只有 1 个训练示教初始场景，不能据此估计全任务泛化成功率。

## 结果落点

- `outputs/dm05/square_probe/preparation.json`：轨迹匹配与恢复核验。
- `samples.npz`、`predictions.npz`：离线样本与动作比较，均留在同一忽略目录中。
- `offline_metrics.json`：整体、EEF、夹爪、基线、时延与越界比例。
- `rollout_metrics.json`、`rollout.mp4`、`rollout_actions.npy`：单场景闭环结果与本地回放。
- `*_progress.log` 和对应运行日志：周期性 completed/total；不发布原始数据和媒体。

实测记录见 [Square 零样本推理报告](../../daily/2026-09-21-DM0.5在Square上的零样本推理.md)。
