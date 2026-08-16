# StackCube 训练样本可视化

状态：`Offline / Smoke`；Rerun 录制与 10-case 预览已生成。

## 实验目标

为 M1-W D_core / M2-W StackCube 训练样本提供与 LIBERO、PushT 一致的 Rerun 可视化：

- 同一套 Rerun viewer identity、时间轴和可选 signal panel；
- 同时查看 recorded base camera、hand camera、Panda qpos skeleton 和 cubeA/cubeB 轨迹；
- 允许从 train split 可复现地随机抽样 case；
- 明确区分 policy RGB/state 输入与仅供诊断的场景 overlay。

## 实验思路

读取冻结 D_core HDF5，不启动 ManiSkill、不执行动作、不加载 checkpoint。
每个 case 写一个 Rerun `.rrd`，保持 observation frame 与 action step 的时间对齐。
Panda 几何复用现有离线 `PandaKinematics`；cubeA/cubeB 使用 D_core 保存的场景状态绘制轨迹。
场景状态只进入可视化诊断面板，不进入 M2-W policy observation。
额外生成双相机 start/middle/end contact sheet，便于快速浏览 10 个 case。

## 执行命令

从仓库根目录执行：

```bash
cd "$(git rev-parse --show-toplevel)"
export PYTHONNOUSERSITE=1

bash scripts/experiments/maniskill/visualize_stackcube_train.sh \
  --d-core-h5=outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5 \
  --d-core-json=outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.json \
  --split-manifest=outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/split_manifest.json \
  --output-dir=outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10_next \
  --count=10 \
  --seed=20260813
```

输出目录必须为空；需要重新生成时换一个 `--output-dir`。若要固定查看指定训练 case，使用重复参数：

```bash
bash scripts/experiments/maniskill/visualize_stackcube_train.sh \
  --output-dir=outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10_explicit \
  --seed=20260813 \
  --episode=106 --episode=201 --episode=324 --episode=353 --episode=428 \
  --episode=434 --episode=575 --episode=719 --episode=878 --episode=949
```

## 打开 Rerun

```bash
conda run --no-capture-output -n robot-practice \
  robot-rerun --persist-state \
  outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10/episodes/*.rrd
```

Rerun 中每个 `.rrd` 是一个可选择的 train case，时间轴为 `episode_step`，频率为 20 Hz。
主视图包含：

- recorded base camera；
- recorded hand camera；
- Panda skeleton、TCP path、cubeA/cubeB path 及当前位置；
- 7D action、18D qpos/qvel 和距离/动作幅度诊断曲线；
- case metadata 与可视化语义说明。

## 本次抽样结果

使用 `numpy.default_rng(seed=20260813)`、不放回地从冻结 train split 抽取 10 个 case：

```text
106, 201, 324, 353, 428, 434, 575, 719, 878, 949
```

预览图：[contact_sheet.png](../../../outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10/preview/contact_sheet.png)

机器可读清单：[visualization_manifest.json](../../../outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10/visualization_manifest.json)

## 结果落点

```text
outputs/maniskill/m2_task_base/StackCube-v1/visualization_train10/
├── episodes/episode_<id>_success.rrd
├── preview/contact_sheet.png
└── visualization_manifest.json
```

## 判断标准与边界

- 选中的 case 必须全部属于 frozen train split，且不得读取 development/test split。
- 每个 case 的 RGB frame 数必须等于 action 数加一，双相机尺寸必须为 `128×128×3`。
- 每个 `.rrd` 必须可被 `robot-rerun` 打开，application id 为离线 Rerun viewer。
- contact sheet 只用于快速浏览；正式可视化以 `.rrd` 的时间轴和 geometry/signal panels 为准。
- cubeA/cubeB 场景 overlay 是离线诊断信息，不代表 M2-W policy 可见输入，也不构成闭环成功率结果。
