# 2026-07-16 LeRobot ACT 调试与 ALOHA 数据可视化

## 重要进展

- 确认了 `lerobot-train` 的 ACT 训练入口和安装包源码位置，并完成可复现的最小训练检查。
- 配置了 VS Code Python Debug，使训练流程可以从命令入口逐步跟进到 policy、dataset 和
  optimizer 代码。
- 检查了 ALOHA mobile cabinet 数据的图像、state、action 和视频组织方式，并确认本地数据
  完整性。
- 实现了同步的 Rerun 可视化，可逐帧查看三路相机、具名关节信号、双臂标称 3D 骨架和
  末端轨迹。
- 当天仅完成工具链和 smoke/debug 验证，没有可报告的正式 train loss、rollout success
  rate 或 checkpoint 结论。

## 今日结论

今天完成了 LeRobot ACT 训练链路的入口确认、VS Code Python Debug 配置、ALOHA
数据检查，以及一个仓库自有的同步可视化工具。现在可以在同一条 Rerun 时间轴上查看
三路相机、具名关节信号、双臂标称 3D 骨架和末端轨迹。

`lerobot/aloha_mobile_cabinet` 在 LeRobot 默认 root 中的重复下载已经完成：parquet、
六个 MP4 均存在，已无 `.incomplete` 文件。23:39 CST 复核时没有 ACT 训练进程，且
仓库内没有本次 3-step Debug 的 checkpoint 或训练日志，因此今天不记录 train loss、
成功率等正式实验指标。

## 环境与训练入口

已验证项目环境：

| 组件 | 当前版本或位置 |
|---|---|
| Conda 环境 | `robot-practice` |
| Python | `3.10.20` |
| PyTorch | `2.10.0+cu128` |
| LeRobot | `0.4.4` |
| Rerun | `0.26.2` |
| GPU | NVIDIA GeForce RTX 3090 |
| LeRobot 包 | `$CONDA_PREFIX/lib/python3.10/site-packages/lerobot` |

`./scripts/doctor.sh` 和 `./scripts/quickstart_info.sh` 已在项目环境中通过。项目继续使用
`PYTHONNOUSERSITE=1`，避免用户目录下的包污染 Conda 环境。

`lerobot-train` 的 console script 位于：

```text
$CONDA_PREFIX/bin/lerobot-train
```

它最终导入并调用：

```python
from lerobot.scripts.lerobot_train import main
```

因此训练主文件是：

```text
$CONDA_PREFIX/lib/python3.10/site-packages/lerobot/scripts/lerobot_train.py
```

## VS Code Debug

已添加本地 VS Code 配置：

- [`.vscode/settings.json`](../.vscode/settings.json)：固定使用 `robot-practice` Python；
- [`.vscode/launch.json`](../.vscode/launch.json)：定义
  `LeRobot: ACT train (3-step debug)`。

Debug 配置使用 Python module 方式启动 `lerobot.scripts.lerobot_train`，并设置：

```text
--dataset.repo_id=lerobot/aloha_mobile_cabinet
--policy.type=act
--steps=3
--batch_size=2
--num_workers=0
--policy.push_to_hub=false
--save_checkpoint=false
--eval_freq=0
--wandb.enable=false
--log_freq=1
```

`justMyCode=false` 允许进入 Conda 环境中的 LeRobot 源码。第一个推荐断点是
`lerobot_train.py:170` 的：

```python
cfg.validate()
```

此时 CLI 参数已经解析为 `TrainPipelineConfig`，但 dataset、policy 和 optimizer 尚未构造。
后续可继续观察 `make_dataset()`、`make_policy()`、optimizer 构造和
`update_policy()`。

详细设计见
[LeRobot VS Code Debug Design](../docs/superpowers/specs/2026-07-16-lerobot-vscode-debug-design.md)。

## ALOHA 数据集

本地 metadata 已验证：

| 字段 | 值 |
|---|---|
| Repo ID | `lerobot/aloha_mobile_cabinet` |
| Robot type | `aloha` |
| Episodes | 85 |
| Frames | 127,500 |
| Control frequency | 50 Hz |
| Episode 0 | 1,500 帧，约 30 秒 |
| State/action/effort | 各 14 维，左臂 7 维 + 右臂 7 维 |
| Cameras | `cam_high`、`cam_left_wrist`、`cam_right_wrist` |

Episode 0 的任务是：打开上层柜门，把锅放入柜中，然后关上柜门。

### 两套本地缓存

之前下载的完整 Hub snapshot 仍然存在：

```text
<lerobot_cache>/hub/
  datasets--lerobot--aloha_mobile_cabinet/
  snapshots/7a752b39f7e69de7e38aee485a6bab07528a061a
```

其中 `data/chunk-000/file-000.parquet` 约 8.7 MB，三路相机的六个 MP4 合计约
1.7 GB。snapshot 中的文件是指向同一 dataset cache 下 `blobs/` 的软链接。

VS Code Debug 使用的是 LeRobot 默认 dataset root：

```text
<lerobot_cache>/lerobot/aloha_mobile_cabinet
```

这解释了今天为什么看起来又下载了一次：完整 snapshot 和 LeRobot 的
`snapshot_download(local_dir=...)` 默认 root 是两个物理目录，不能自动跨目录复用。
截至今日收尾，默认 root 也已完整，占用约 1.7 GB。后续保持相同配置会直接复用它，
不需要再次下载。

## 同步数据与 3D 可视化

新增两个仓库自有模块，没有修改安装在 `site-packages` 中的 LeRobot：

- [aloha_kinematics.py](../src/robot_practice/aloha_kinematics.py)：NumPy 实现的
  双臂 ViperX 300S 标称正向运动学；
- [visualize_aloha.py](../src/robot_practice/visualize_aloha.py)：dataset 校验、批量读取、
  派生信号和 Rerun logging/blueprint。

可视化内容包括：

- 三路 RGB 相机；
- 14 个具名关节的 position、action 和 effort；
- 使用相邻真实 timestamp 计算的 velocity；
- `action - state`；
- 左右臂当前骨架、关节点、夹爪开合和末端坐标系；
- 左右末端执行器的完整 XYZ 轨迹；
- `frame_index` 与 elapsed `timestamp` 同步时间轴。

3D 只表示标称运动学，不包含柜子、锅、碰撞、动力学或接触力，也没有与 RGB 相机完成
外参标定或空间配准。

完整 Episode 0 已保存为：

- [aloha_mobile_cabinet_episode_0.rrd](../outputs/aloha_mobile_cabinet_episode_0.rrd)，
  约 186 MB；
- [Rerun 最终界面截图](../outputs/rerun_viewer_clean.png)。

直接打开 recording：

```bash
conda run -n robot-practice rerun \
  --port 9888 \
  --window-size 1600x1000 \
  --renderer vulkan \
  outputs/aloha_mobile_cabinet_episode_0.rrd
```

当前机器强制使用 OpenGL 会报 surface/adapter 不兼容，因此使用 Vulkan。Rerun 中可以用
Space 播放或暂停、Left/Right 逐帧查看，也可以拖动底部 `frame_index` 时间轴。

重新从 dataset 构建可视化：

```bash
conda activate robot-practice
python -m robot_practice.visualize_aloha \
  --repo-id lerobot/aloha_mobile_cabinet \
  --episode-index 0
```

只检查信号和 3D、不解码视频：

```bash
python -m robot_practice.visualize_aloha \
  --repo-id lerobot/aloha_mobile_cabinet \
  --episode-index 0 \
  --no-images \
  --max-frames 100
```

详细几何约定和数据流见
[可视化设计](../docs/superpowers/specs/2026-07-16-aloha-dataset-visualizer-design.md)。

## 验证结果

- `36` 个 pytest 测试通过；
- 零位姿、非零角度独立 MuJoCo golden pose、旋转正交性和夹爪映射均有覆盖；
- 覆盖 dataset feature/shape/joint order、timestamp、velocity 和 entity path 校验；
- 覆盖 `--no-images` 不触发 dataset 视频解码；
- 覆盖每帧两个 Rerun timeline 和末端完整轨迹的 static logging；
- 使用缓存中的三帧完成离线 `.rrd` 集成测试；
- 完整 1,500 帧 recording 已通过 `rerun rrd verify`；
- 人工检查确认三路相机、3D、曲线和 50 FPS 时间轴可以同步显示。

对应测试：

- [test_aloha_kinematics.py](../tests/test_aloha_kinematics.py)
- [test_visualize_aloha.py](../tests/test_visualize_aloha.py)

## 今日涉及的主要文件

| 文件 | 作用 |
|---|---|
| [`.vscode/settings.json`](../.vscode/settings.json) | 固定 Python interpreter |
| [`.vscode/launch.json`](../.vscode/launch.json) | ACT 三步训练 Debug 配置 |
| [README.md](../README.md) | 增加 ALOHA 可视化用法 |
| [aloha_kinematics.py](../src/robot_practice/aloha_kinematics.py) | 双臂标称 FK |
| [visualize_aloha.py](../src/robot_practice/visualize_aloha.py) | Rerun 可视化入口 |
| [test_aloha_kinematics.py](../tests/test_aloha_kinematics.py) | FK 单元测试 |
| [test_visualize_aloha.py](../tests/test_visualize_aloha.py) | 数据与 Rerun 测试 |

`.vscode/`、`outputs/` 和数据缓存都是本机内容，按仓库规则不提交生成物。当前 workspace
没有 `.git` 目录，因此今天无法创建 commit 或查看 Git diff。

## 下一步

1. 在 `cfg.validate()` 打断点，重新运行 `LeRobot: ACT train (3-step debug)`；数据已完整，
   这次应直接进入 dataset/policy 构建。
2. 依次进入 `make_dataset()`、`make_policy()` 和 `update_policy()`，记录实际 batch key、
   tensor shape、loss 和单步耗时。
3. 如果三步训练成功，将结果补充到 [experiment_log.md](experiment_log.md)，并按模板填写
   能够实际获得的实验字段；缺失的 closed-loop 指标保持明确未测。
4. 后续再决定是否保留两套 1.7 GB 数据。清理前先确认训练统一使用哪一个 root。
