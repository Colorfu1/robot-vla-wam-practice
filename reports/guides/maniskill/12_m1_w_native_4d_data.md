# M1-W StackCube 原生 4D 数据重生成

## 目的

这条路线不从冻结的 M1-W 7D 标签切片。它从原始
`pd_joint_pos` motion-planning demonstrations 重新执行 ManiSkill 官方 CPU
转换，得到原生 `pd_ee_delta_pos` 4D actions，再在 GPU 上闭环重放并采集训练数据。

## 固定流程

```text
原始 joint H5
  -> physx_cpu 官方 joint->pd_ee_delta_pos converter
  -> GPU native 4D baseline replay（记录 action/success/env_state）
  -> GPU 4D position-only online EE repair（只修 baseline 失败项）
  -> 按 start_state_fingerprint 合并 accepted episodes
  -> 固定 carrier layout，在同一次 GPU env.step 中保存 RGB + 4D action + qpos/qvel + env_state
```

转换产物与最终数据分开保存。最终正式数据只有 baseline 或 repair 在末帧成功的
episode；默认至少需要 900 条。身份只使用 `env_states[0]` 的 fingerprint，不使用
episode seed。

## 本地生成数据

数据生成必须在本地完成；远端不重复执行 converter、GPU replay 或 RGB capture。
本次本地实测结果如下：

| 阶段 | 结果 |
|---|---:|
| 官方 CPU joint→native 4D conversion | 1000/1000 |
| GPU native 4D baseline replay | 760/1000 |
| GPU position-only repair | 186/240 个失败项 |
| accepted merge | 946 条，超过 900 条闸门 |
| 最终实时 RGB replay | 946/946 success_once，946/946 success_at_end |

其中 54 条（240 条 baseline 失败中未被 repair 救回的部分）被排除，不进入训练集。
最终数据每条都通过成功闸门；RGB、action、qpos/qvel、env_state 来自同一次 GPU
`env.step` 执行。

生成命令：

```bash
bash scripts/experiments/maniskill/run_m1_wam_native_4d.sh \
  --phase=all --num-envs=100
```

如果中途失败，从对应阶段继续，不需要从头执行：

```bash
bash scripts/experiments/maniskill/run_m1_wam_native_4d.sh --phase=convert
bash scripts/experiments/maniskill/run_m1_wam_native_4d.sh --phase=gpu-replay --num-envs=100
bash scripts/experiments/maniskill/run_m1_wam_native_4d.sh --phase=gpu-repair --num-envs=100
bash scripts/experiments/maniskill/run_m1_wam_native_4d.sh --phase=merge
bash scripts/experiments/maniskill/run_m1_wam_native_4d.sh --phase=capture --num-envs=100
```

本次还固定了一个关键约束：baseline 的 carrier batch 必须复用 GPU replay
实际使用的“按轨迹长度排序后的 `traj_0, traj_1, ...`”顺序，不能按 source episode ID
重新排序；否则 CUDA batch composition 改变，会让原本成功的 replay 在 RGB capture
时失败。

## 产物

```text
outputs/maniskill/m1_wam_native_4d/StackCube-v1/
├── 01_convert_native_4d/replay.h5
├── 02_gpu_baseline/replay.h5
├── 03_gpu_repair/repair.h5
├── 04_accepted_merge/replay.h5
├── 05_frozen_layout/replay_layout.json
└── 06_final_live_rgb/replay.h5
```

`06_final_live_rgb/replay.h5` 才是可用于视觉训练的数据；其中 RGB 是执行对应 action
时由 GPU 环境直接返回的图像，不是从 state 事后物化的图像。

## 远端训练与 rollout

把本地的 `06_final_live_rgb/replay.h5` 及同名 `replay.json` 同步到远端后，使用独立的
`native4d` 数据类型；不要把它标成旧的
`m1w_projected`。先只做输入检查：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=native4d --phase=prepare --num-demos=787
```

确认 `00_prepare/adapter_manifest.json` 通过后，再在远端分开执行训练和评测，便于
从中间产物恢复：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=native4d --phase=train --num-demos=787 \
  --total-iters=50000 --eval-freq=10000 --save-freq=10000 \
  --dataloader-workers=4 \
  --gpus=0,1,2,3

bash scripts/experiments/maniskill/run_official_stackcube_18d4_ablation.sh \
  --dataset-kind=native4d --phase=eval --num-demos=787 \
  --gpus=0,1,2,3 --eval-split=development --eval-episodes=200
```

训练仍使用官方 Diffusion Policy、18D `qpos+qvel`、4D `pd_ee_delta_pos`，global
batch 为 256；`native4d` 只替换训练 H5，且不改变旧 `official`/`m1w` 产物。
