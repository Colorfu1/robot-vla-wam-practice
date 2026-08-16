# M1-W StackCube 数据与控制协议

## 重要进展

- 当前正式路线为 `CPU-converted 7D → GPU baseline → failed-only EE repair → 冻结 batch 布局 → live RGB`。
- 状态物化路线已废弃；旧的 `gpu_state_replay_*`、旧 `D_core` 和旧 LeRobot 数据不再作为正式数据。
- baseline 得到 805 条成功；repair 恢复 182/195 条，13 条失败被排除，正式集为 987 条。
- final 在真实 `env.step` 过程中同步保存 RGB、env state、实际发送的 action 和 success；只有该 H5
  能进入 trajectory audit、D_core 和 LeRobot 转换。
- state banks 已生成：train/development/test = 2000/200/400；split 内唯一、跨 split 重叠为 0。
- `finalize` 与 `validate-frozen` 均通过；M1-W StackCube 协议现已冻结。

## 1. 固定协议

| 项目 | 固定值 |
| --- | --- |
| task | `StackCube-v1` |
| backend | `physx_cuda` |
| control mode | `pd_ee_delta_pose` |
| action | 原生 7D |
| control frequency | 20 Hz |
| formal horizon | 500 |
| observation | 两路 `128×128×3` RGB + `qpos/qvel` 18D |
| identity | `env_states[0]` 的 `start_state_fingerprint`，容差 `1e-4` |
| formal episodes | 987（805 baseline + 182 repair） |

seed、HDF5 group 编号、执行顺序都不是跨文件 episode identity。

## 2. 冻结布局与正式数据生成

### 阶段 1：baseline live replay

输入是官方 joint demo 经 CPU converter 得到的 7D action。每个 episode 从 source 的初始
`env_state` 开始，实际调用
GPU `env.step(action)`，保存实际产生的 `success` 和 `env_states`，但不保存 RGB。该阶段的作用
只有一个：找出 replay 末尾失败的 episode，不能直接作为视觉训练数据。

产物：

```text
outputs/maniskill/stackcube_joint_to_ee_gpu_repair/03_gpu_replay_stable_none/replay.h5
outputs/maniskill/stackcube_joint_to_ee_gpu_repair/03_gpu_replay_stable_none/replay_manifest.json
```

### 阶段 2：failed-only actual-EE repair

只处理 baseline 失败的 episode。程序从原生 source 的状态路径恢复 TCP reference path，每一步
读取当前 GPU TCP pose，计算收敛到下一目标 pose 的 `pd_ee_delta_pose`，并实际执行该 action。

repair 过程中同步保存：

- 实际发送的修复后 7D EE action；
- 当前 GPU 环境实时生成的 RGB；
- 当前 GPU 环境实时记录的 `env_states`；
- success/reward 和 tracking diagnostics。

这里的 RGB 是 `env.step` 过程中由环境产生并写入的，不是由 state 事后物化得到的。

产物：

```text
outputs/maniskill/stackcube_joint_to_ee_gpu_repair/04_gpu_failed_ee_repair/repair.h5
outputs/maniskill/stackcube_joint_to_ee_gpu_repair/04_gpu_failed_ee_repair/repair_manifest.json
```

195 条失败中修复 182 条；其余 13 条记录在正式 manifest 并从训练集排除。

### 阶段 3：冻结 carrier 布局后 final live RGB replay

同一批 action 换并发顺序后，成功数会从 987 降为 957。因此不能将成功条目压紧重排，而是冻结：

- baseline：1000 次执行，805 条正式数据、195 条 carrier，按 source ID 分 10 批；
- repair：195 次执行，182 条正式数据、13 条 carrier，按原 repair 顺序分 2 批；
- carrier 只维持 GPU slot/batch，不写入训练集。

merge 产物只是 action/env-state 中间输入，不是最终视觉数据。随后对合并后的所有 action 再做
一次真实 GPU replay，设置 `obs_mode="rgb"`。每次 `env.step` 同步写出：

```text
RGB、qpos/qvel、完整 env_states、action、success、reward
```

采集器不用 `RecordEpisode`/merge，直接按 source ID 写 H5，避免 action/state 错配。实测正式
987 条 `success_once=987`、`success_at_end=987`，共 106,528 actions；H5 约 7.6 GB。

```text
outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/final_live_rgb_full/replay.h5
outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/final_live_rgb_full/replay.json
outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/final_live_rgb_full/replay_manifest.json
```

整个正式路线不调用 `gpu_state_materializer`；如果命令触发 `gpu_state_replay_*` 或
`materialization_manifest.json`，说明跑的是旧路线，应立即停止。

## 3. 运行正式采集

旧 `datasets/raw/.../StackCube-v1/gpu_native` 已删除，不再使用。冻结布局见
`outputs/maniskill/stackcube_joint_to_ee_gpu_repair/frozen_replay_layout.json`。重新采集使用：

```bash
bash scripts/experiments/maniskill/run_stackcube_live_rgb_pipeline.sh \
  --num-envs=100
```

## 4. audit、D_core 与 LeRobot

full pipeline 成功后再执行：

```bash
bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=trajectory-audit --task=StackCube-v1 \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=freeze-data --task=StackCube-v1 \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/convert_stackcube_dataset.sh \
  --replay-h5=outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5
```

转换器只读取 final live RGB replay 生成的 D_core，不创建环境，也不补生成图像。LeRobot
输入契约是双 RGB、18D `qpos+qvel`、7D action。

本次已完成并核对通过：D_core 为 987 条、106,528 个 action frame；LeRobot 产物位于
`datasets/lerobot/maniskill/stackcube_dcore_7d`，包含 987 episodes、106,528 frames、两路
128×128 RGB 视频、18D state 和 7D action。final H5 与 D_core 的每条 episode 均为 end-success，
每个 RGB 序列长度均为 action 数加一。

### 4.1 state banks 与 M1-W finalization（已完成）

state banks 是后续训练/闭环评测使用的固定初始状态集合，不是训练数据，也不是 RGB 数据。身份
使用 reset 后的 `start_state_fingerprint`，不把 seed 当作唯一 identity。正式结果为 train/
development/test = `2000/200/400`，每个 split 内唯一，跨 split 重叠为 0，`gate_pass=true`。
复现命令如下：

```bash
bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=download \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=source-audit --task=StackCube-v1 \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=environment-audit --task=StackCube-v1 \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=state-banks --task=StackCube-v1 \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit
```

通过后执行：

```bash
bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=finalize \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit

bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=validate-frozen \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --protocol=outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json
```

只有这一步完成，M1-W 计划才标记为 `done`。

## 5. 旧路线与规则

以下内容已经作废，不得作为正式结果引用：

- `gpu_state_replay_full` / `gpu_state_replay_smoke`；
- `rgb_materialization_validation`；
- 旧的 state-materialized `D_core` 和 `stackcube_dcore_7d`；
- 旧 `datasets/raw/maniskill_demos/StackCube-v1/gpu_native`；
- “从 env_states 逐帧 `get_obs()` 补 RGB”的数据生成方式。

今后新采集视觉数据时，必须在 action 执行过程中直接使用 `obs_mode="rgb"`，同步保存 RGB、
env state、action 和 success。本次之所以保留 baseline 的 `obs_mode="none"`，只是为了减少
失败筛选阶段的开销；repair 和 final 都已经按实时 RGB 规则采集。
