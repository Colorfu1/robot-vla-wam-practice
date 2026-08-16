# 2026-08-13 M1-R Policy Decorator MS2/v2 协议预检

## 状态

`blocked`（用户暂缓）：M1-W state banks/frozen protocol、M1-R source/environment audit 与 50
条 native/strict replay smoke 已通过；986 条 full replay、D_core、state banks 和 protocol
freeze 尚未执行。用户要求先暂停，待切换到 `maniskill2-pidec` 环境后继续。

## 目标

冻结 Policy Decorator 的独立 Residual 输入协议：MS2 0.5.3、固定 checkout、v0 metadata、repo-local
v2 runtime、原生 7D EEF action、50D state、200 步 horizon、split/state banks 与 exact replay gate。
不训练 base，不运行 R0/R1，不把 MS2 结果与 MS3 M1-W 合并。

## 前置条件

1. M1-W 计划状态为 `done`，且 `outputs/maniskill/m1_wam_protocol_audit/frozen_protocol.json` 通过
   `validate-frozen`。
2. `datasets/raw/policy_decorator/reference_repo/` 的 HEAD 为
   `92ba9ba442587ae5989c286355ace2200a0537fb`。
3. Policy Decorator H5/JSON 的 SHA-256 与 config 完全一致。
4. `maniskill2-pidec` 中 `mani-skill2` 版本为 `0.5.3`。

## 执行步骤

### 1. 先收口 M1-W state banks

大规模 state reset 不在本地代理执行，由用户在目标机器运行：

```bash
bash scripts/experiments/maniskill/run_m1_protocol_audit.sh \
  --phase=state-banks --task=StackCube-v1 \
  --config=configs/experiments/maniskill/m1_wam_protocol_audit.json \
  --output-root=outputs/maniskill/m1_wam_protocol_audit
```

state-bank manifest 通过后，执行：

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

### 2. 验证 M1-R identity 和环境协议

```bash
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=validate-config
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=source-audit
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=environment-audit
```

### 3. exact replay

先运行 50 条 smoke：

```bash
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=replay-smoke --count=50
```

smoke 通过后运行全量 986 条：

```bash
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=replay-full --count=986
```

runner 会同时复放 dataset metadata 的 `PegInsertionSide-v0` 与固定 checkout 注册的
`PegInsertionSide-v2`，动作原样使用 H5 中的 7D action，不注入 H5 state。full 阶段产物独立落盘，
失败时不会伪造 manifest。

### 4. 冻结 D_core、offline split 和 online-RL state banks

```bash
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=trajectory-audit
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=freeze-data
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=state-banks
```

### 5. 最终冻结并验证

```bash
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh --phase=finalize
bash scripts/experiments/maniskill/run_m1_r_protocol_audit.sh \
  --phase=validate-frozen \
  --protocol=outputs/policy_decorator_ms2/m1_r_protocol_audit/frozen_protocol.json
```

## 判断标准

- source audit：986 条、固定 commit/H5/JSON hash、986 个唯一起始 state fingerprint、动作/观测
  shape 合法，7D action 无越界。
- environment audit：v0/v2 均为 7D action、50D state、20 Hz、TimeLimit 200；controller 固定为
  `pd_ee_delta_pose`、EE frame、delta mode、Panda arm position bound ±0.1、rotation bound 0.1；
  字段顺序为
  `qpos(9)+qvel(9)+base_pose(7)+tcp_pose(7)+peg_pose(7)+peg_half_size(3)+box_hole_pose(7)+radius(1)`；
  reference metadata 未指定 reward 时的默认值记录为 `normalized_dense`；formal replay/online
  residual contract 固定为 `sparse`。
- exact replay smoke：native/strict 均 50/50；full：native/strict 均 986/986；success_once 同样
  全通过；每条在 success 上 terminated、无 time-limit truncation；initial H5 state reconstruction
  最大绝对误差 `≤1e-6`。
- trajectory：全部轨迹长度 `≤200`，source success 全部存在，start-state fingerprint 唯一。
- D_core：每条保留首次成功后 5 帧以内的尾部；offline train/development/test 为 786/100/100，
  split 之间无 fingerprint overlap。
- state banks：train/development/test 为 2000/200/400；每个 split 内唯一，跨 split 重叠为 0。
- finalize：以上所有 manifest 的 `gate_pass` 均为 true，生成并通过 `frozen_protocol.json`。

## 产物路径

```text
outputs/policy_decorator_ms2/m1_r_protocol_audit/source_audit.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/environment_contract.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/reference_replay_50/
outputs/policy_decorator_ms2/m1_r_protocol_audit/reference_replay_full/
outputs/policy_decorator_ms2/m1_r_protocol_audit/trajectory_audit.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/d_core/
outputs/policy_decorator_ms2/m1_r_protocol_audit/d_core_manifest.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/state_banks/
outputs/policy_decorator_ms2/m1_r_protocol_audit/split_leakage_report.json
outputs/policy_decorator_ms2/m1_r_protocol_audit/frozen_protocol.json
```

## 风险与停止条件

- M1-W 未通过 frozen validation 时停止，不运行 M1-R full replay/state banks。
- commit、H5/JSON hash、环境版本、v0/v2 注册或 action/observation schema 任一不一致时停止。
- native 或 strict replay 任一失败时停止，不切换 MS3 数据、不注入 H5 state、不放宽 success gate。
- 任何阶段存在不完整目录或非 gate-success manifest 时，不覆盖原产物；更换明确的输出目录并保留
  失败证据后再诊断。
- 本计划只冻结协议和数据输入，不声称 learned policy 成功率，也不启动训练或 Residual RL。
