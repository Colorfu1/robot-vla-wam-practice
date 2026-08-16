# 官方 Diffusion Policy 接入 M1-W StackCube 7D 数据

## 状态

适配代码、单环境契约检查和多卡 exact-state 评测入口已完成；正式训练与评测仍由远端执行。
该入口保留官方 `train_rgbd.py` 的 Diffusion Policy、DDPM loss、EMA 和 checkpoint，只把数据集
与观测 wrapper 切换为冻结的 M1-W live-RGB 协议。

## 固定合同

- 数据：`outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5`。
- split：只读取 `split_manifest.json` 的 `train` 787 条，禁止把 development/test 混入训练。
- RGB：`base_camera`、`hand_camera` 两路实时保存的 `128×128×3`，训练时拼成 6 通道。
- state：`agent.qpos(9) + agent.qvel(9)`，18D；不读取 `extra` 或 success/env state。
- action：原生 `pd_ee_delta_pose`，7D；窗口为 `obs_horizon=2`、`pred_horizon=16`、执行 8 步。
- 评测环境：`StackCube-v1`、`sim_backend=gpu`（对应 M1-W 的 `physx_cuda`）、horizon 500。
- 运行时 contract：该数据标记为 `data_provenance=m1w`，评测器会拒绝 CPU rollout；结果中会写入
  `required_rollout_backend=physx_cuda` 和 `actual_rollout_backend=physx_cuda`。

官方读取器原本会将整份 H5 和预处理后的 observation 放入内存，且默认状态包含 `extra`。
本适配使用 `M1WLiveRGBDataset` 懒读窗口，并由每个 torchrun rank 独立打开只读 H5；因此
固定 `num_dataload_workers=0`，不会把约 10GB H5 复制到每张卡的显存。

## 一条命令完成训练与 exact-state development 评测

在远端仓库根目录执行。下面的命令使用 0–3 卡、global batch 256、50k iterations，每 10k
保存 checkpoint；训练结束后自动选 checkpoint（优先 `best_eval_success_once.pt`，否则最新
数字 checkpoint），再用同一组 4 张卡对 development state bank 的 200 条状态做 exact-state
闭环评测：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  --phase=all \
  --gpus=0,1,2,3 \
  --total-iters=50000 \
  --eval-freq=10000 \
  --save-freq=10000 \
  --eval-split=development \
  --eval-episodes=200 \
  --eval-envs=64 \
  --eval-steps=500
```

`--gpus` 同时控制训练和外部评测的 GPU 数量；外部评测由 torchrun 将 state bank 分片给各
rank，rank 0 最后合并结果，不会退回单卡。训练期间官方脚本自带的内部评测仍只在 rank 0
执行，它只是训练监控，不是最终报告。

使用 8 卡时只需替换 GPU 列表：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  --phase=all \
  --gpus=0,1,2,3,4,5,6,7 \
  --output-root=outputs/maniskill/official_dp_stackcube_m1w_7d_8gpu
```

也可以把训练预算改成 100k：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  --phase=all --gpus=0,1,2,3 \
  --total-iters=100000 --eval-freq=10000 --save-freq=10000 \
  --output-root=outputs/maniskill/official_dp_stackcube_m1w_7d_100k
```

正式 test 评测可以在训练完成后单独执行；它不会重新训练：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  --phase=eval \
  --gpus=0,1,2,3 \
  --eval-split=test \
  --eval-episodes=400 \
  --eval-envs=64 \
  --eval-steps=500
```

`all` 的 development 结果与 `eval` 的 test 结果都使用 frozen state bank 的
`start_state_fingerprint`，不使用 seed 作为身份。不要把训练期间 rank-0 的随机评测当作
正式 test 结果。

## 分阶段执行与产物

如果希望拆开执行（例如训练失败后不重新准备数据）：

```bash
bash scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  --phase=prepare --gpus=0,1,2,3

bash scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  --phase=train --gpus=0,1,2,3

bash scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  --phase=eval --gpus=0,1,2,3 \
  --eval-split=development --eval-episodes=200
```

关键产物：

```text
outputs/maniskill/official_dp_stackcube_m1w_7d/00_prepare/adapter_manifest.json
outputs/maniskill/official_dp_stackcube_m1w_7d/00_prepare/train_episode_ids.json
outputs/maniskill/official_dp_stackcube_m1w_7d/logs/00_prepare.log
outputs/maniskill/official_dp_stackcube_m1w_7d/logs/10_train.log
outputs/maniskill/official_dp_stackcube_m1w_7d/10_train/runs/official_dp_stackcube_m1w_7d_50000/
outputs/maniskill/official_dp_stackcube_m1w_7d/10_train/train_complete.json
outputs/maniskill/official_dp_stackcube_m1w_7d/20_eval/development/metrics.json
outputs/maniskill/official_dp_stackcube_m1w_7d/20_eval/development/workers/rank_*/metrics.json
```

训练日志只看：

```bash
tail -f outputs/maniskill/official_dp_stackcube_m1w_7d/logs/10_train.log
tail -f outputs/maniskill/official_dp_stackcube_m1w_7d/logs/20_eval_development.log
```

当前官方训练脚本没有 resume 参数；已有训练目录时 runner 会拒绝覆盖，请改用新的
`--output-root` 或 `--exp-name`。`configs/experiments/maniskill/official_dp_m1w_7d.json`
记录了同一份协议与默认超参。

## 远端同步

除冻结的 M1-W D_core 本身外，还要同步本地适配器、runner、配置，以及已经修改过的官方
`train_rgbd.py`：

```bash
REMOTE_PRACTICE_ROOT=/path/to/remote/practice

rsync -avP -e 'ssh -p 3333' \
  src/robot_practice/experiments/maniskill/official_m1w_adapter.py \
  src/robot_practice/experiments/maniskill/eval_official_stackcube_m1w.py \
  root@remote-host:"$REMOTE_PRACTICE_ROOT/src/robot_practice/experiments/maniskill/"

rsync -avP -e 'ssh -p 3333' \
  third_party/maniskill_official/examples/baselines/diffusion_policy/train_rgbd.py \
  root@remote-host:"$REMOTE_PRACTICE_ROOT/third_party/maniskill_official/examples/baselines/diffusion_policy/"

rsync -avP -e 'ssh -p 3333' \
  scripts/experiments/maniskill/run_official_stackcube_m1w_7d.sh \
  root@remote-host:"$REMOTE_PRACTICE_ROOT/scripts/experiments/maniskill/"

rsync -avP -e 'ssh -p 3333' \
  configs/experiments/maniskill/official_dp_m1w_7d.json \
  root@remote-host:"$REMOTE_PRACTICE_ROOT/configs/experiments/maniskill/"

rsync -avP -e 'ssh -p 3333' \
  reports/guides/maniskill/10_official_dp_m1w_7d.md \
  root@remote-host:"$REMOTE_PRACTICE_ROOT/reports/guides/maniskill/"
```

远端已有 M1-W H5 和 split manifest 时不需要重新生成数据；先执行 `--phase=prepare`，确认
`00_prepare/adapter_manifest.json` 的 787 条 train split 正确，再启动 `--phase=train`。
