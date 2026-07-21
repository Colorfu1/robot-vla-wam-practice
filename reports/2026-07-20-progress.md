# 2026-07-20 LeRobot ACT 实验与源码调试进展

## 今日结论

今天完成了 ACT A-Sanity 五轨迹过拟合实验的启动脚本，并确定了后续离线预测与
LIBERO 闭环评测的完整设计。同时，围绕 LeRobot 数据统计、Accelerate/BF16 训练路径
以及 ACT 模型内部的 VAE、位置编码和 attention 做了源码级梳理。

今天没有启动 20,000 step 的正式 A-Sanity 训练，因此尚无 checkpoint、train loss、
open-loop action error 或 closed-loop success rate 可以记录。两段参考调试主要执行
只读源码检查，没有修改安装在 `site-packages` 中的 LeRobot，也没有提交或推送代码。

## 总结范围

本报告参考两段本地源码调试记录。其中一段创建于 7 月 16 日并在今天继续。本报告只统计
2026-07-20 的对话和
仓库改动，不重复记录 7 月 16 日已经写入
[2026-07-16-progress.md](2026-07-16-progress.md) 的训练入口、VS Code Debug 和 ALOHA
可视化工作。

## ACT A-Sanity 实验启动器

新增了可复现的 A-Sanity 启动脚本：

- [a_sanity.sh](../scripts/experiments/act/a_sanity.sh)
- [ACT Experiment Launcher Design](../docs/superpowers/specs/2026-07-20-act-experiment-launchers-design.md)

实验使用 LIBERO Object task 8：`pick up the chocolate pudding and place it in the
basket`。从本地 `lerobot/libero` 数据集中选择五条属于同一任务的完整 demonstration：

```text
[823, 826, 834, 840, 841]
```

训练配置如下：

| 配置 | 值 |
|---|---|
| Dataset | `lerobot/libero` |
| Dataset task index | `29` |
| Dataset episodes | `[823,826,834,840,841]` |
| Policy | ACT |
| `chunk_size` | `20` |
| `n_action_steps` | `5` |
| `use_vae` | `false` |
| Batch size | `8` |
| Steps | `20000` |
| Seed | `1000` |
| Evaluation during training | disabled |
| W&B / Hub upload | disabled |
| Final checkpoint | step 20,000 |

启动器自动通过以下方式使用项目环境，不会改变调用者当前 shell：

```text
conda run --no-capture-output -n robot-practice lerobot-train ...
```

每次启动都会生成独立输出目录：

```text
outputs/act/a_sanity/<YYYYMMDD_HHMMSS>
```

时间戳用于保留每一次实验结果，避免后一次运行覆盖前一次运行。脚本还会检查 Conda、
`flock`、本地 LIBERO metadata 和输出目录，并使用文件锁禁止两个 A-Sanity 任务并发写入。

查看完整命令但不启动训练：

```bash
scripts/experiments/act/a_sanity.sh --print-command
```

正式启动：

```bash
scripts/experiments/act/a_sanity.sh
```

## ACT 离线与闭环评测设计

完成了
[ACT Offline and Closed-Loop Evaluation Design](../docs/superpowers/specs/2026-07-20-act-evaluation-design.md)，
明确将评测分为两个互补但不能混淆的部分。

### 离线 teacher-forced prediction

离线评测从 demonstration 读取每一帧观测，让 checkpoint 预测完整 action chunk。预测动作
不会影响下一帧输入，因此它可以衡量模仿误差和动作连续性，但不能判断机器人是否完成
任务。

计划记录的主要指标包括：

- `first_action_mae`：当前步预测动作与 demonstration 动作的平均绝对误差；
- `chunk_mae_by_horizon`：action chunk 每个预测 horizon 的误差；
- `delta_action_mae`：预测动作变化与 demonstration 动作变化的误差；
- `teacher_forced_action_tv`：相邻预测动作的平均变化量；
- `teacher_forced_action_jerk`：预测动作的二阶差分，用于反映突变和来回抖动；
- 推理延迟 p50/p95。

离线输出设计为 `run_config.json`、`metrics.json`、`predictions.parquet` 和
`prediction.rrd`。Rerun recording 将同步显示相机、目标动作、预测动作、误差和延迟，
并支持逐帧播放。

### LIBERO closed-loop rollout

闭环评测固定使用 `libero_object` task ID 8。策略预测的动作会真实送入模拟器，下一步
观测由执行结果产生，任务成功只采用 LIBERO 环境返回的 `is_success`。

计划记录的主要指标包括：

- `success_rate`：成功完成任务的 rollout 比例；
- reward 汇总；
- `action_tv`：相邻执行动作的平均变化量，越低通常越平滑；
- `action_jerk`：执行动作的二阶差分，反映动作突变和抖动；
- `gripper_toggles`：夹爪开合状态切换次数；
- `policy_step_latency`：完整 `select_action()` 调用耗时；
- `model_query_latency`：真正执行 `predict_action_chunk()` 的网络推理耗时。

闭环默认评测 20 个 episode、可视化 1 个 episode。可视化 episode 数量由参数覆盖；
`--perturbation` 接口保留，但第一版只接受默认值 `none`。计划输出 JSON、Parquet、Rerun
recording 和 MP4 视频。

昨天完成的是上述设计。对应 Python 模块、shell 入口和测试在 7 月 21 日才实现，不计入
本日已完成代码。

## LeRobot 数据统计分析

对 `ds_meta.stats` 的来源和含义进行了源码检查，得到以下结论：

- `min`、`max`、`mean`、`std` 和 `count` 是数据集级统计量，不是单帧数据；
- `observation.state`、`observation.effort` 和 `action` 使用全部 `127500` 帧；
- 这些范围描述 demonstration 中实际出现的数据，不等同于机械臂的物理关节限位；
- 图像统计只抽样部分帧，用于降低解码与像素统计成本；
- 对每个 1,500 帧 episode，LeRobot 估算抽样数为
  `int(1500 ** 0.75) = 241`；85 个 episode 合计 `20485` 个样本；
- 指数 `0.75` 是 LeRobot 为平衡统计精度与计算量采用的工程启发式，并非严格理论常数；
- 当前 ACT 图像归一化可以使用 ImageNet mean/std，因此三路相机显示相同的
  `[0.485, 0.456, 0.406]` 和 `[0.229, 0.224, 0.225]`。

同时确认可以使用 `policy.named_parameters()` 查看每个可训练参数的名称、形状和参数量，
而不只统计总参数数目。

## Accelerate 与 BF16 训练路径

梳理了 LeRobot `lerobot_train.py` 中 Hugging Face Accelerate 的职责，包括：

- 选择和管理 device；
- 包装 policy、optimizer、DataLoader 和 scheduler；
- 管理 autocast；
- 执行 backward 和梯度裁剪；
- 多进程同步、主进程输出以及模型 unwrap。

围绕 BF16 得到的关键结论：

- autocast 按算子选择 BF16 或 FP32，不是只转换整个模型的输入；
- 默认情况下模型参数仍保存为 FP32；
- linear、matmul、convolution 等符合规则的算子会临时使用 BF16 输入/权重参与计算；
- 数值敏感算子或不在 BF16 autocast 列表中的算子可以保持或提升到 FP32；
- backward 中间张量可能沿用 forward 选择的数据类型，但写入 FP32 参数的
  `param.grad` 通常最终为 FP32；
- gradient clipping 限制的是梯度幅值，并不负责 BF16 到 FP32 的类型转换；
- 与 `model.to(torch.bfloat16)` 相比，autocast 保留 FP32 参数和 optimizer update，通常
  更稳定、算子兼容性更好，但不会获得纯 BF16 参数存储的全部显存收益。

另外检查了一个外部 mmdet3d checkout 中的 `mmdet3d/apis/train_bf16.py`。该训练路径同样
在 forward 周围进入 `torch.cuda.amp.autocast(dtype=torch.bfloat16)`，没有把整个模型永久
转换为 BF16，也没有维护一份按 layer 名称配置的 BF16 白名单；具体精度由 PyTorch
autocast 的算子注册规则决定。

## ACT 模型源码调试

另一段调试集中分析了安装环境中的
`lerobot/policies/act/modeling_act.py`，主要结论如下。

### VAE encoder 与 latent distribution

- VAE encoder 输入 token 由 CLS、robot state 和 action chunk 组成；当
  `chunk_size=100` 时序列长度为 `1 + 1 + 100 = 102`；
- `vae_encoder_pos_enc` 是初始化时创建的固定一维 sinusoidal position embedding，以
  buffer 形式注册，不参与训练，但会随模型保存和迁移 device；
- encoder 输出经过线性投影后拆为 `mu` 和 `log_sigma_x2`，用于参数化对角高斯 posterior；
- 使用 log variance 可以保持方差为正，并方便重参数采样和 KL loss 计算；
- KL loss 将 `q(z | state, action)` 约束到标准高斯 `N(0, I)`，使训练时学到的 latent 在
  推理时可以从先验获得；
- 推理阶段没有 demonstration action，当前实现使用零 latent，即标准高斯的均值。

### Attention 张量与 LayerNorm

- 当前 `nn.MultiheadAttention` 使用默认 `batch_first=False`，因此输入布局为
  `(sequence_length, batch_size, model_dimension)`；
- `(102, 2, 512)` 表示 102 个 token、batch size 2、embedding dimension 512；
- `LayerNorm` 与 `Identity` 都保持输入输出 shape 不变；
- LayerNorm 对每个 token 最后的 512 维独立归一化，Identity 则直接返回原值；
- decoder self-attention 的 Q/K/V 都来自 decoder token，其中 Q/K 叠加 position
  embedding；cross-attention 的 Q 来自 decoder，K/V 来自 encoder 输出；
- attention 输出序列长度由 query 数量决定，所以 cross-attention 的输出仍对应 decoder
  action slots。

### 图像 backbone 与二维位置编码

- ACT 使用 torchvision ResNet18 backbone；启用预训练权重时参数来自
  `ResNet18_Weights.IMAGENET1K_V1`，不是从当前 LIBERO 或 ALOHA 数据集生成；
- `IntermediateLayerGetter` 只是包装并提取指定的中间 feature map，不负责产生预训练参数；
- 2D sinusoidal position embedding 通过 `not_mask.cumsum()` 构造有效像素的 x/y 坐标；
- 坐标归一化到 `2π`，再使用多组频率生成 sin/cos 编码；
- x 方向和 y 方向分别产生一半通道，拼接得到与 transformer `dim_model` 对齐的位置编码；
- 输出 shape 为 `(1, dim_model, height, width)`，在 batch 维广播使用。

## 今日实际修改的文件

| 文件 | 作用 |
|---|---|
| [2026-07-20-act-experiment-launchers-design.md](../docs/superpowers/specs/2026-07-20-act-experiment-launchers-design.md) | A-Sanity 启动器设计 |
| [a_sanity.sh](../scripts/experiments/act/a_sanity.sh) | 五轨迹 ACT 过拟合训练入口 |
| [2026-07-20-act-evaluation-design.md](../docs/superpowers/specs/2026-07-20-act-evaluation-design.md) | 离线与 LIBERO 闭环评测设计 |

参考调试中没有调用文件修改工具；ACT、LeRobot 和 mmdet3d 源文件均为只读检查。
该工作区当时也不是 Git working tree，因此没有本仓库 commit 或 push。

## 下一步

1. 先运行 `a_sanity.sh --print-command`，再次确认数据路径、五条 episode 和训练参数。
2. 启动 A-Sanity 20,000-step 训练，观察 loss 是否相对初始值下降 80% 以上。
3. 训练完成后使用完整 `pretrained_model` 目录进行离线评测，检查 action MAE、TV 和 jerk。
4. 在 LIBERO task 8 中执行 closed-loop rollout，以模拟器 `is_success` 作为成功标准。
5. 将正式训练和评测结果补充到 [experiment_log.md](experiment_log.md)，完整记录数据规模、
   observation keys、action horizon、train loss、open-loop error、success rate、latency 和
   failure cases。
