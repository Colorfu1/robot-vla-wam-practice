# M2 StackCube Diffusion Policy 一键训练指南

## 目标与边界

该入口把 M2 的 StackCube 数据适配成一份独立的 LeRobot 数据集，再调用安装环境里的原生
`lerobot.scripts.lerobot_train` 训练 Diffusion Policy。原始
`datasets/lerobot/maniskill/stackcube_dcore_7d`、已有 `train_formal.sh` 和 LeRobot 安装包均不修改。

适配数据只包含 M2 的 train/development：787/100 个 episode；test 不写入新数据，也不在适配和
训练过程中读取。两路相机使用 LeRobot 原生 `video` 特征，源帧必须是 `uint8 [H,W,C]` RGB，
使用 H.264 video-backed 存储；训练读取时由 LeRobot lazy decode 为 RGB `float32 [0,1]` 的
`[T,C,H,W]` tensor。这样 parquet 只保存状态、动作和帧索引，不把所有 PNG bytes 嵌入 Arrow。

## 一键正式训练

在仓库根目录执行：

```bash
bash scripts/experiments/maniskill/train_m2_diffusion.sh \
  --num-gpus=8
```

脚本第一次运行会生成：

```text
datasets/lerobot/maniskill/stackcube_dcore_7d_dp_rgb_video/
outputs/maniskill/m2_diffusion/configs/m2_stackcube_diffusion_seed1000.train_config.json
```

随后训练输出默认写入：

```text
outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000/
```

已有适配数据会先做 fingerprint 和原生 LeRobot 读取校验，然后复用；训练输出目录存在时脚本会
拒绝覆盖，请改用新的 `--output-dir`。

## 本地 one-step smoke

```bash
bash scripts/experiments/maniskill/train_m2_diffusion.sh \
  --num-gpus=1 \
  --batch-size=1 \
  --steps=1 \
  --save-freq=1 \
  --log-freq=1 \
  --num-workers=0 \
  --mixed-precision=no \
  --output-dir=/tmp/m2_stackcube_diffusion_smoke \
  --job-name=m2_stackcube_diffusion_smoke
```

只查看两阶段命令，不启动转换或训练：

```bash
bash scripts/experiments/maniskill/train_m2_diffusion.sh \
  --num-gpus=8 --print-command
```

输出中的第二条命令必须是：

```text
accelerate launch ... -m lerobot.scripts.lerobot_train --config_path=...
```

## 关键检查点

适配数据 manifest：

```text
datasets/lerobot/maniskill/stackcube_dcore_7d_dp_rgb_video/m2_diffusion_dataset_manifest.json
```

其中应看到 `storage=video`、`image_contract.backing=video`、`image_contract.decoded_color=RGB`、
train/development 为 787/100、test 为空，以及 `test_data_read=false`。训练 sidecar 位于 native train config 旁边的
`.adapter_manifest.json`（例如 `m2_stackcube_diffusion_seed1000.train_config.adapter_manifest.json`），
其中记录了数据 fingerprint 和原生入口；显式 train episode indices 在 native train config 中。

训练 checkpoint 由原生 LeRobot 保存到：

```text
<output-dir>/checkpoints/
```

## Development 选模

训练完成后，用固定的 M1 StackCube development state bank 做闭环选模。脚本会自动发现
`checkpoints/<step>/pretrained_model`，逐个调用现有 `eval_stackcube_gpu`，按下面顺序排序：

```text
success_at_end ↓
success_once ↓
is_cubeA_on_cubeB ↓
is_cubeA_static ↓
policy forward mean latency（小优先）
checkpoint step（小优先）
```

默认评测 200 个 development states、500 个控制步，使用 `n_action_steps=8`。示例：

```bash
bash scripts/experiments/maniskill/select_m2_diffusion_checkpoint.sh \
  --run-dir=outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000 \
  --checkpoint-steps=10000,20000,30000,40000,50000 \
  --num-gpus=8 \
  --num-envs=64
```

只对已经完成的评测重新排序：

```bash
bash scripts/experiments/maniskill/select_m2_diffusion_checkpoint.sh \
  --run-dir=outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000 \
  --selection-dir=outputs/maniskill/m2_diffusion/StackCube-v1/checkpoint_selection \
  --select-only
```

结果写入：

```text
outputs/maniskill/m2_diffusion/StackCube-v1/checkpoint_selection/
├── evaluations/checkpoint_<step>/development/metrics.json
├── checkpoint_results.json
├── selected_checkpoint.json
└── selected/pretrained_model/
```

`selected/pretrained_model` 只复制推理所需的 native LeRobot policy 文件，不复制每个 checkpoint
的 optimizer state；后续推理直接使用该目录。选模固定 `policy_seed=1000`，重复运行会复用相同的
随机采样流。脚本只读取 `development.jsonl`，不会访问 state-bank test 或 offline test。

适配器不向 `lerobot_train` 注入 monkey-patch，也不改变 LeRobot 的 Dataset、processor、policy、
optimizer 或 checkpoint pipeline；变化仅是输入数据根目录、显式 train episode 列表和 train-only
stats。
