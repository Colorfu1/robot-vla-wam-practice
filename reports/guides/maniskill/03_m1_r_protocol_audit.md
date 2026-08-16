# ManiSkill M1-R Policy Decorator MS2/v2 协议预检指南

## 当前状态

- 状态：`blocked`（用户暂缓）；M1-W frozen protocol 已通过，M1-R runner、config、source audit、
  environment audit 和 50 条 smoke 已通过。986 条 full replay、D_core、state banks 和 protocol
  freeze 暂停，待切换到 `maniskill2-pidec` 环境后继续。
- 顺序 gate：
  [M1-W 计划](../../../plans/2026-08-12-M1-WAM数据与控制协议预检.md) 已为 `done`，且
  `outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json` 已通过验证。
- 当前不启动 full replay、D_core、state banks、训练或 R0/R1；已通过的小规模检查只作为
  M1-R 暂停前的审计结果保留。
- 正式执行计划为
  [2026-08-13 M1-R 计划](../../../plans/2026-08-13-M1-R-PolicyDecorator-MS2v2协议预检.md)，
  runner 为 `scripts/experiments/maniskill/run_m1_r_protocol_audit.sh`。

## 实验目标

M1-R 将冻结 Residual 路线自己的 task/data/controller/evaluation contract：ManiSkill2 0.5.3、
Policy Decorator pinned checkout、repo-local `PegInsertionSide-v2` runtime、metadata 为
`PegInsertionSide-v0` 的原生 7D `pd_ee_delta_pose` demonstrations、50D privileged state、
episode TimeLimit、split 与 exact replay gate，供 M2-R/R0/R1 消费。

M1-R 不复用 M1-W 的 ManiSkill3 package、GPU-native StackCube data、joint→EEF converter、StackCube
horizon、observation schema、state banks 或 output root。

## 实验思路

- 先用不可变 commit 和 H5/JSON SHA 验证 reference identity，再构造 MS2 环境。
- dataset task ID 固定 `PegInsertionSide-v0`；formal runtime 固定 Policy Decorator repo-local
  corrected `PegInsertionSide-v2`。`v2` 是环境 ID，不是另一份 demonstrations。
- actions 原样使用 native 7D `pd_ee_delta_pose`；禁止从 MS3 joint data 转换，也禁止跨版本
  action bridge 进入正式数据。
- 先复核 50 条 exact replay reference；通过后由 formal runner 扩为全 986 条并冻结数据。
- observation/prediction/action/execution contract 目标为 `2/16/4/4`；episode evaluation
  horizon 必须从 pinned source/runtime 解析，不继承 MS3 Peg@300。

## 冻结 identity

| 项目 | 固定值 |
| --- | --- |
| Python environment | 独立 `maniskill2-pidec` |
| ManiSkill | `mani-skill2==0.5.3` |
| Policy Decorator commit | `92ba9ba442587ae5989c286355ace2200a0537fb` |
| dataset H5 SHA-256 | `2483eb42a8995203b55e5c29795f5b3298c1724ab8fede3795de809636e7ef3b` |
| dataset JSON SHA-256 | `2cc855c7fb6173fc59f1c0fdec72ecde3b900a151772ad4171e745b13a90245d` |
| dataset env | `PegInsertionSide-v0` |
| formal runtime env | repo-local `PegInsertionSide-v2` |
| source controller/action | native `pd_ee_delta_pose`，7D normalized |
| public episode count | 986 |
| privileged observation | 50D state：`qpos(9)+qvel(9)+base_pose(7)+tcp_pose(7)+peg_pose(7)+peg_half_size(3)+box_hole_pose(7)+radius(1)` |
| temporal contract | observation/prediction/action/execution `2/16/4/4` |
| reward mode | reference metadata 未指定时默认 `normalized_dense`；formal replay/online residual 固定 `sparse` |

本指南统一使用以下 ignored、仓库相对路径；不要把临时目录或个人 checkout 路径写入报告：

```text
datasets/raw/policy_decorator/reference_repo/
datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.h5
datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/
```

## 结果分类

| 阶段 | 标签 | 含义 |
| --- | --- | --- |
| identity/source schema audit | `Offline` | 文件、metadata、action/state contract 检查 |
| 50 条 exact replay | `Smoke` | 已有 reference reconstruction 复核 |
| 986 条 exact replay | `Closed-loop` | expert-action replay；不是 learned-policy benchmark |
| frozen M1-R protocol | `Offline` | 只冻结 M2-R 输入，不产生 Residual RL 结果 |

## M1-W 顺序 gate

以下命令是 M1-R 的第一条命令。任何一项失败都必须停止：

```bash
cd "$(git rev-parse --show-toplevel)"

rg -n '状态：`done`' plans/2026-08-12-M1-WAM数据与控制协议预检.md
test -s outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice \
  python -m robot_practice.experiments.maniskill.m1_protocol_audit \
  validate-frozen \
  --config configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --protocol outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json
```

在 M1-W 仍为 `todo/in-progress/blocked` 时，不运行后续 MS2 命令，也不创建一个空的 M1-R
`frozen_protocol.json` 绕过顺序 gate。

## Reference identity 检查

M1-W gate 通过后，先检查已准备的 repo-relative reference：

```bash
test -d datasets/raw/policy_decorator/reference_repo/.git
test -s datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.h5
test -s datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.json

test "$(git -C datasets/raw/policy_decorator/reference_repo rev-parse HEAD)" = \
  "92ba9ba442587ae5989c286355ace2200a0537fb"

printf '%s  %s\n' \
  '2483eb42a8995203b55e5c29795f5b3298c1724ab8fede3795de809636e7ef3b' \
  'datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.h5' \
  '2cc855c7fb6173fc59f1c0fdec72ecde3b900a151772ad4171e745b13a90245d' \
  'datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.json' \
  | sha256sum --check --strict

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n maniskill2-pidec \
  python -c 'from importlib.metadata import version; assert version("mani-skill2") == "0.5.3"'
```

reference acquisition、固定 commit checkout 与文件 hash 已由 M1-R 子计划准备为上述
repo-relative 输入；本指南不允许从可变 `main` 或未记录 URL 临时下载替代文件。若本地缺少
这些输入，应先补齐固定 identity，再执行 audit，不得用临时替代文件继续。

## 已实现的 exact-replay reference 复核

### 50 条 smoke（legacy direct wrapper）

```bash
bash scripts/experiments/maniskill/run_policy_decorator_exact_replay.sh \
  --policy-decorator-root=datasets/raw/policy_decorator/reference_repo \
  --dataset-h5=datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.h5 \
  --dataset-json=datasets/raw/policy_decorator/PegInsertionSide-v0/trajectory.json \
  --count=50 \
  --sample-seed=20260811 \
  --output-dir=outputs/policy_decorator_ms2/m1_r_protocol_audit/reference_replay_50
```

gate 固定为 native `50/50`、strict `50/50`，且 initial H5 state reconstruction 最大绝对误差
为 0（或仅允许预注册的数值精度误差）。该命令只复核 reference，不代表 M1-R 已完成。正式阶段
使用下面的 runner，它会额外写入 `replay_manifest.json` 并执行 hard gate：

```bash
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=replay-smoke --count=50
```

### 全 986 条 replay（formal runner）

```bash
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=replay-full --count=986
```

formal runner 会调用固定 reference 输入并把 formal reward mode 固定为 `sparse`，产物写入独立的
`outputs/policy_decorator_ms2/m1_r_protocol_audit/reference_replay_full/`；native/strict 任一
失败都不会生成 gate-success manifest，也不会自动替换已有产物。

当前 formal smoke 已实际通过：native/strict 均 `50/50`，success-once 均 `50/50`，均为
terminated、无 truncation，native/strict 初始 state 最大误差均为 `0`。证据位于
`outputs/policy_decorator_ms2/m1_r_protocol_audit/reference_replay_smoke/`；这仍不是 full
986 结果。

## 正式 M1-R 阶段

执行计划和 runner 已为以下阶段提供独立、可恢复入口：

1. source H5/JSON、986 episode identity、50D state 字段顺序与 7D action stats audit；
2. v0 dataset reset/options 与 repo-local v2 success/reward/termination/TimeLimit contract audit；
3. full 986 native/strict exact replay、first-success 与 post-success-tail audit；
4. evaluation horizon decision；只允许从 pinned runtime/source contract 冻结；
5. `D_core` trim、offline train/development/test split 与 online-RL train/dev/test state banks；
6. leakage、chunk `2/16/4/4`、reward label 与 final protocol validation；
7. 为 R1 视觉迁移保留同 episode/action 的 deterministic rendering contract，但不在 M1-R
   提前生成视觉 student 结果。

这些阶段全部通过并生成 `frozen_protocol.json` 前，M1-R 保持 `in-progress`，不能进入 M2-R。

## 结果落点

当前已有 reference 复核只写：

```text
outputs/policy_decorator_ms2/m1_r_protocol_audit/reference_replay_50/
outputs/policy_decorator_ms2/m1_r_protocol_audit/reference_replay_full/
```

正式 M1-R 的目标根目录为：

```text
outputs/policy_decorator_ms2/m1_r_protocol_audit/source_audit.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/environment_contract.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/trajectory_audit.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/d_core/
outputs/policy_decorator_ms2/m1_r_protocol_audit/state_banks/
outputs/policy_decorator_ms2/m1_r_protocol_audit/split_leakage_report.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/frozen_protocol.json
```

所有 reference checkout、datasets、outputs、state banks 和 replay logs 都是 ignored artifact，
不得加入 git。

## 判断标准

- M1-W 为 `done` 且 frozen protocol 验证通过。
- Policy Decorator commit、H5/JSON SHA、ManiSkill2 版本与本指南完全一致。
- metadata task 恰为 v0，formal runtime 恰为 repo-local v2；不得混称“v2 数据”。
- 50 条 smoke native/strict 都为 50/50；full replay native/strict 都必须为 986/986。
- state reconstruction、controller frame/scale、50D observation、7D action bounds、success/reward/termination 和
  `2/16/4/4` contract 全部通过未来 M1-R manifest gate。
- evaluation horizon 有 pinned source/runtime 证据；不得继承 MS3 300 或看到结果后人工延长。
- offline/online split 无跨 split leakage；M2-R 只读取 M1-R frozen artifact。

## 停止条件

- M1-W 未完成：停止 M1-R full replay/state banks，这是当前顺序 gate。
- formal output schema 或 runner 失败：停止，不用旧的历史 P1–P3 目录冒充正式产物。
- identity/hash/version 不一致：停止；不换可变 checkout 或数据文件。
- native 与 strict replay 任一未过 gate：先修 reset/environment contract，不切回 MS3。
- 禁止把历史 P1–P3 cross-version diagnosis 当作 M1-R formal replay，也禁止将 smoke 写成
  `Formal` learned-policy 结果。
