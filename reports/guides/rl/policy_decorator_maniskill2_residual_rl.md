# Policy Decorator 在 ManiSkill2 的 Residual RL 复现指南

## 实验目标

在 ManiSkill2 `PegInsertionSide-v2` 上准备并审计 Policy Decorator 的 frozen Diffusion Policy，
用 smoke 验证本地接线，并在需要正式结果时经 pinned checkout 的官方脚本产出 frozen base 与
`+ residual SAC` 的对照。正式入口只认官方脚本，不修改官方实现。

本指南只描述如何运行、如何验收和如何记录，不把预期数字写成实验结果。

> **范围变更（2026-09-01）**：本指南原为三阶段（ManiSkill → Light-WAM/RoboTwin → RLinf）。
> Light-WAM 与 RLinf 两段已取消，Residual RL 的下游改由
> [2026-09-01 Residual RL 下游改接 robomimic + DPPO pixel](../../../plans/2026-09-01-Residual-RL下游改接robomimic-DPPO-pixel.md)
> 承担。本指南现在只覆盖 ManiSkill2 这一段。

## 实验思路

1. 先做 source/artifact/protocol audit，再做小规模 smoke；不把本地 evaluator 的输出当作官方结果。
2. 所有 revision、task、seed、action contract、训练预算和阈值在运行前冻结。
3. released checkpoint 只能做环境/权重 parity；主结论使用计划指定的 checkpoint。
4. 只报告 episode-level manifest、聚合指标、hash、吞吐、延迟和失败分类，不提交原始轨迹、视频、权重或日志。

结果标签必须区分 `Smoke`、`Offline`、`Closed-loop`、`Formal`。

`datasets/raw/policy_decorator/reference_repo/online/pi_dec_diffusion_maniskill2.py` 是唯一的官方
residual 训练/内置评测入口。它没有独立的 eval-only CLI：`best.pt` 是 frozen base，真正的 residual
结果必须由该脚本产生 `res_actor` checkpoint 并在同一官方 run 的周期评测中记录。没有明确批准的
正式训练时，只做下面的 audit/smoke；不要用 `policy_decorator_eval.py` 或 `validation_runner.py`
的自定义闭环数字冒充官方结果——这两个模块仅用于本地接口诊断/配对审计。

§0 的环境检查与 dummy smoke 是一条独立命令，应先从仓库根目录执行。进入 §1 后，下载和后续运行
代码块应在同一个 Bash 会话中依次执行，并先运行 §1.1 的 preamble；它启用未定义变量和 pipeline
失败传播，避免 `tee` 把训练失败伪装成成功。

所有本地 evaluator/runner 还必须遵守同一个 paired-evaluation 合同：先生成一次只读的
`episode_manifest.jsonl`（至少包含 `episode_key`、环境 seed/reset-state id、task config 和
executed action count），base 与 treatment 逐行消费相同 manifest；汇总器必须拒绝 key、reset-state
id 或 episode 数不一致的输入。仅传相同 seed 不足以证明配对。

## 固定合同与资源闸门

| 固定任务/基座 | 关键合同 |
| --- | --- |
| `PegInsertionSide-v2` + 官方 Diffusion Policy | state、`pd_ee_delta_pose` 7D、20 Hz、200 steps、obs/pred/action/execute = 2/16/4/4；smoke 不训练，正式论文对齐 run 使用 10 个 seed |

开始前必须确认：

- `maniskill2-pidec` 环境、Policy Decorator commit `92ba9ba442587ae5989c286355ace2200a0537fb`、
  ManiSkill2 `0.5.3` 和官方 checkpoint/`args.json` 均可审计。
- 所有长任务每 60 秒内向 stdout 和 `run.log` 写 `STATUS completed=<n> total=<N>`，结束写
  `DONE completed=<N> total=<N>`。
- 任何正式 run 的输出目录必须是新的或经过 audit 明确确认的同一 run；命令不得用 `rm -rf`、
  静默覆盖 checkpoint 或把不同协议的结果混写。

`PegInsertionSide-v2` 不是 `mani-skill2==0.5.3` wheel 自带的注册项。它是 pinned Policy Decorator
checkout 里的 `envs.maniskill_fixed` 注册的 repo-local corrected runtime；因此正式 run 必须把该
checkout 作为 `--reference-root`/`PYTHONPATH`，并显式导入该模块。镜像单独只保证原生
`PegInsertionSide-v0` 可用，不能把 `v0` 冒充正式的 `v2`。

## 0. 环境检查与 dummy smoke

先冻结以下本地入口，再启动任何正式实验：

```text
configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml
scripts/experiments/rl/run_three_stage_validation.sh
src/robot_practice/experiments/rl/
tests/experiments/rl/
```

配置与 runner 的文件名沿用历史命名（`three_stage_*`），因为
`src/robot_practice/experiments/rl/validation_runner.py` 与 `tests/experiments/rl/` 直接引用它们；
改名需要同步改代码与测试，本次范围变更不做。

下面这一条命令从仓库根目录依次完成实现文件检查、单元测试和 dummy preflight。它不下载或读取
真实数据、checkpoint、视频或权重，也不产生 Formal 结果：

```bash
set -Eeuo pipefail
repo_root="${PRACTICE_ROOT:-$PWD}"
repo_root="$(git -c "safe.directory=${repo_root}" -C "${repo_root}" rev-parse --show-toplevel)"
cd "${repo_root}"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/rl-policy-decorator-dummy.XXXXXX")"
required_paths=(
  configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml
  scripts/experiments/rl/run_three_stage_validation.sh
  scripts/experiments/rl/run_policy_decorator_paper_multigpu.sh
  scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh
  src/robot_practice/experiments/rl
  tests/experiments/rl
)
for required_path in "${required_paths[@]}"; do
  test -e "${required_path}" || {
    echo "missing required local implementation: ${required_path}" >&2
    exit 2
  }
done

PYTHONNOUSERSITE=1 conda run -n robot-practice \
  python -m pytest -q tests/experiments/rl

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=preflight --dummy \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root="${smoke_root}/preflight"

test -s "${smoke_root}/preflight/preflight.json"
```

若尚未实现上述入口，`missing required local implementation` 是预期的阻断，不得用空脚本、跳过
测试或直接运行 vendor 命令绕过。preflight 必须落盘 resolved config、所有 Git/Hugging Face
revision、包版本、GPU 信息、文件 hash、seed 列表和 train/development/test 无交集审计；分享摘要
不得包含机器名或绝对路径。任一 gate 失败都不能进入训练。

通过条件是 dummy preflight 通过，且目录内有持续的 `STATUS` 与最终 `DONE`。该 smoke 不读取真实
checkpoint，也不产生正式结果；它只覆盖接口和环境接线。checkpoint 下载、artifact audit 和官方
residual run 必须分别执行后续条目。

## 1. 下载与 artifact 准备

### 1.1 不改本机代理状态的应用层命令

下面的 preamble 只影响单个子进程，不修改 shell、Clash/TUN、系统路由或 Git 配置。所有后续代码块默认继承它；若重新打开终端，必须重新运行：

```bash
set -Eeuo pipefail
repo_root="${PRACTICE_ROOT:-$PWD}"
repo_root="$(git -c "safe.directory=${repo_root}" -C "${repo_root}" rev-parse --show-toplevel)"
cd "${repo_root}"
practice_root="${repo_root}"

direct_env=(
  env
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY
  -u http_proxy -u https_proxy -u all_proxy
  NO_PROXY='*' no_proxy='*'
)
# Keep the ownership exception scoped to these git subprocesses; do not modify global git config.
git_direct=("${direct_env[@]}" git -c 'safe.directory=*' -c http.proxy= -c https.proxy=)
```

当前处于 TUN 模式时，这只能绕过显式 `127.0.0.1:6789` 代理，不能强制 Mihomo 的透明流量选择 `DIRECT`。`hf` CLI 没有绑定物理网卡的选项；若必须物理直连，需要另行提供具有 direct egress 的机器或经批准的进程级网络隔离方案。不要用修改路由、关闭 TUN、改 Clash 配置来“临时下载”。

### 1.2 S1 官方 checkpoint 下载

S1 只下载 `PegInsertionSide` 的 Diffusion Policy checkpoint 和同目录的
`args.json`。不要使用 `gdown --folder`，否则会把 BET、其它任务和其它视觉配置一并下载。
下面使用固定的 Google Drive file id；当前 `best.pt` 约 34 MB，远小于 1 GB。下载完成后再执行 §1.3 的 artifact audit。

```bash
s1_artifact_root="outputs/rl_three_stage_20260828/artifacts/policy_decorator"
s1_model_root="${s1_artifact_root}/checkpoints/diffusion_PegInsertionSide"
s1_ckpt="${s1_model_root}/checkpoints/best.pt"
s1_args="${s1_model_root}/args.json"
mkdir -p "$(dirname "${s1_ckpt}")" "${s1_model_root}"

conda run --no-capture-output -n maniskill2-pidec \
  python -c 'import gdown' >/dev/null 2>&1 || {
  echo "gdown is missing; use the prebuilt maniskill2-pidec image" >&2
  exit 2
}

# diffusion_PegInsertionSide/checkpoints/best.pt, Google Drive file id is immutable.
"${direct_env[@]}" PYTHONNOUSERSITE=1 \
  conda run --no-capture-output -n maniskill2-pidec \
  python -m gdown --continue \
  'https://drive.google.com/uc?id=1neKVKAvuxBUasIPTR900G9xcy6cdLnnm' \
  -O "${s1_ckpt}"

# diffusion_PegInsertionSide/args.json, Google Drive file id is immutable.
"${direct_env[@]}" PYTHONNOUSERSITE=1 \
  conda run --no-capture-output -n maniskill2-pidec \
  python -m gdown --continue \
  'https://drive.google.com/uc?id=1XnirlwclhO5Mp3Cp_pskTnBVdPgn-QOp' \
  -O "${s1_args}"

test -s "${s1_ckpt}"
test -s "${s1_args}"
echo "S1_CHECKPOINT_DOWNLOAD=PASS"
```

### 1.3 S1 artifact audit

本节只检查本地 checkpoint、`args.json`、Policy Decorator checkout 和运行时合同，不下载数据、不运行 episode。可在下载段之后执行；重新打开终端时重新定义路径即可：

```bash
s1_artifact_root="outputs/rl_three_stage_20260828/artifacts/policy_decorator"
s1_model_root="${s1_artifact_root}/checkpoints/diffusion_PegInsertionSide"
s1_ckpt="${s1_model_root}/checkpoints/best.pt"
s1_args="${s1_model_root}/args.json"

test -s "${s1_ckpt}"
test -s "${s1_args}"
test -f datasets/raw/policy_decorator/reference_repo/envs/maniskill_fixed.py
python -m json.tool "${s1_args}" >/dev/null

# Load the artifact in the pinned S1 environment; do not check the demo_path here.
conda run --no-capture-output -n maniskill2-pidec \
  python - "${s1_ckpt}" "${s1_args}" <<'PY'
import json
import sys
from pathlib import Path

import torch

checkpoint_path = Path(sys.argv[1])
args_path = Path(sys.argv[2])
args = json.loads(args_path.read_text(encoding="utf-8"))
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
if not isinstance(checkpoint, dict) or not {"agent", "ema_agent"}.issubset(checkpoint):
    raise RuntimeError("S1 checkpoint must contain agent and ema_agent")
assert args["env_id"] == "PegInsertionSide-v2"
assert args["control_mode"] == "pd_ee_delta_pose"
assert args["obs_horizon"] == 2
assert args["pred_horizon"] == 16
assert args["act_horizon"] == 4
print("checkpoint_keys:", sorted(checkpoint))
print("args_env_id:", args["env_id"])
print("args_control_mode:", args["control_mode"])
print("CHECKPOINT_LOAD=PASS")
PY

sha256sum "${s1_ckpt}" "${s1_args}" \
  | tee "${s1_artifact_root}/s1_artifacts.sha256"

# Runtime/commit/resource audit. This does not download data and does not run episodes.
PYTHON_BIN=/opt/conda/envs/maniskill2-pidec/bin/python \
  bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s1-audit \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --checkpoint="${s1_ckpt}" \
  --reference-root=datasets/raw/policy_decorator/reference_repo \
  --output-root=outputs/rl_three_stage_20260828/stage1/audit

test -s outputs/rl_three_stage_20260828/stage1/audit/preflight.json
```

通过条件是：checkpoint 可由 `maniskill2-pidec` 加载，`args.json` 与 S1 的环境、控制模式和
时间合同一致，Policy Decorator 注册文件存在，SHA-256 已记录，并且
`stage1/audit/preflight.json` 的 `gate_pass` 为 `true`。该 audit 只证明 artifact/runtime/resource
合同，不等于 frozen-base 闭环成功率。`s1_ckpt` 和 `s1_args` 在同一个 Bash 会话中留给 §2
使用；重新打开终端后应从本节重新定义它们。


至少保留以下可审计文件；缺任一文件都不能进入汇总：

```text
<stage>/audit/resolved_config.yaml
<stage>/audit/revisions.json
<stage>/audit/package_versions.txt
<stage>/audit/gpu.json
<stage>/audit/hashes.json
<stage>/audit/split_audit.json
<run>/run.log
<run>/progress.jsonl
<run>/metrics.json
<eval>/episodes.jsonl
<eval>/episode_manifest.jsonl
```

`progress.jsonl` 必须在运行过程中持续追加 `STATUS`，而不是只在结束时生成；`episodes.jsonl` 和
manifest 只允许保存 episode key、seed/reset id、成功标记、计数和聚合所需指标，不保存原始
observation/action、视频或模型权重。

验收阈值与停止条件见
[Policy Decorator 复现计划](../../../plans/2026-08-28-PolicyDecorator-ManiSkill2-Residual-RL复现.md)
的「判断标准」与「产物路径」两节，本指南不重复。


## 2. S1：Policy Decorator / ManiSkill2

### 2.1 官方 residual run（需要正式 residual 结果时才执行）

官方 online 入口同时负责 residual SAC 训练和周期性闭环评测，没有独立的 eval-only 参数。
`best.pt` 是 frozen base；只有运行下面这个上游命令后，才会产生 `res_actor` checkpoint 和官方
评测曲线。命令直接调用 pinned checkout，不经过本仓库的 runner/evaluator；除 seed 和输出目录外，
只传官方 README 的任务参数。正式训练未获明确批准时不要执行这一节。

论文结果需要先做 frozen base 的十 seed 闭环，再做十个独立的官方 residual-SAC run。使用一条
wrapper 命令串行完成两阶段：base 全部成功且 checkpoint hash 不变后，才启动官方 SAC。每个
worker 只看到一张 GPU，不能改成 `torchrun`/DDP；GPU 数少于 10 时按批次运行。论文正文只公布
“十个 seed”，没有公布 canonical seed-id 列表，因此默认的 1--10 会写入 manifest。

```bash
bash scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh \
  --gpus=0,1,2,3 \
  --base-policy-ckpt=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt \
  --reference-root=datasets/raw/policy_decorator/reference_repo \
  --output-root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac
```

该 launcher 固定 `PegInsertionSide-v2` 的官方任务参数：`res_scale=0.1`、`prog_explore=30,000`、
`total_timesteps=4,000,000`、`num_envs=16`、每 50,000 steps 的 50-episode 评测、每 1,000,000
steps 保存 checkpoint；其余参数保持 pinned 官方默认值。论文将各评测点的 10 个 seed 结果求均值
并报告标准差；launcher 不自动修改或合并 TensorBoard event，汇总时必须按相同 `global_step`
读取十个 worker 的 `eval/success`。

官方输出目录中的 `args.json`、TensorBoard event、`checkpoints/4000000.pt` 和 stdout 是 S1
residual 主结果；不要再用本地 evaluator 重新跑一套闭环来替代它。若只需要 smoke，跳过本节。

### 2.2 本地接线 smoke（可选，不是官方结果）

```bash
s1_ckpt=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt
test -s "${s1_ckpt}"

smoke_root=outputs/rl_three_stage_20260828/stage1/smoke/residual
test ! -e "${smoke_root}" || {
  echo 'refusing to reuse S1 residual smoke output' >&2
  exit 1
}

conda run --no-capture-output -n maniskill2-pidec \
  bash scripts/experiments/rl/run_three_stage_validation.sh \
    --phase=s1-train --dummy --dummy-steps=4 \
    --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
    --seed=1 \
    --reference-root=datasets/raw/policy_decorator/reference_repo \
    --checkpoint="${s1_ckpt}" \
    --output-root="${smoke_root}"

test -s "${smoke_root}/metrics.json"
```

该 smoke 只验证 runner、zero-residual 组合动作、base 冻结标记和 STATUS/DONE 可观测性；
它不启动官方 online SAC、不执行环境采样/优化、不产生 checkpoint，也不能证明 residual
策略会提升成功率。runner 中保留的 formal `s1-train` 入口不属于本轮执行路径。

### 2.3 S1 结果边界

本轮默认只做 audit 和本地 smoke，不自动启动 §2.1 的 4M residual SAC；因此没有 S1 residual
formal 结果。若明确执行 §2.1，只报告官方脚本自身的周期评测和最终 checkpoint，并记录其
`args.json`、stdout、TensorBoard event 和 checkpoint hash。`policy_decorator_eval.py`、
`validation_runner.py` 生成的闭环数字只能标记为本地诊断，不能与官方结果或论文数字混用。

### 2.4 S1 base / SAC 闭环 rollout 可视化

需要查看两个策略的实际闭环行为时，重新执行一次配对的 S1 评测来采集 RGB 帧；已有
`episodes.jsonl` 或 TensorBoard 结果不能直接还原视频。`--record-rollout-dir` 在
`maniskill2-pidec` 环境中只写无 Rerun 依赖的 raw bundle，随后由 `robot-practice` 环境
生成 Rerun/MP4。S1 的观测仍是官方 50D state，画面是 `PegInsertionSide-v2` 的原生
`environment.render()`。

本节的默认边界是“远端 GPU 评测、本地 Rerun 可视化”：代码先从本地同步到远端 checkout，
base/residual 的闭环评测和 RGB 采集都在远端完成；只把 `eval_*` 和 `raw_*` 结果同步回本地，
再在本地 `robot-practice` 环境生成 `.rrd`/`.mp4`。这些 S1 可视化文件通常只有数 MB，不会
触发大于 1 GiB 的下载；命令不包含数据、模型或 checkpoint 下载。

base 与 residual 使用同一个 base checkpoint、同一个 `--seeds`、`--episodes-per-seed`、
`--ddim-steps` 和 `--act-horizon`。先跑 base，再把 base 的 `episode_manifest.jsonl`
作为 residual 的 `--manifest`，这样两边的 reset seed 逐 episode 对齐。对于 residual，
`--seeds` 还决定从 `--residual-runs/seed_<seed>` 读取哪个训练 seed 的 SAC actor；这里的
seed 不是额外的评测 seed。下面示例查看训练 seed 1 在 4M checkpoint 的一个 episode：

```bash
remote_repo="${REMOTE_REPO:?Set REMOTE_REPO to the remote checkout}"
remote_ssh='ssh -p 3333'

rsync -av --relative -e "${remote_ssh}" \
  src/robot_practice/experiments/rl/policy_decorator_eval.py \
  src/robot_practice/experiments/rl/policy_decorator_peg_visualization.py \
  scripts/experiments/rl/prepare_policy_decorator_peg_visualization_remote.sh \
  scripts/experiments/rl/visualize_policy_decorator_peg.sh \
  root@localhost:"${remote_repo}/"
```

然后登录远端，在远端仓库根目录执行评测。远端代码块如下：

```bash
s1_ckpt=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt
s1_args=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/args.json
s1_sac_root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/sac
s1_vis_root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/visualization_seed1_4m
s1_eval_seed=1
s1_checkpoint_step=4000000

for path in "${s1_vis_root}/eval_base" "${s1_vis_root}/raw_base" \
  "${s1_vis_root}/eval_residual" "${s1_vis_root}/raw_residual" \
  "${s1_vis_root}/rrd_base" "${s1_vis_root}/rrd_residual"; do
  test ! -e "${path}" || {
    echo "refusing to reuse existing S1 visualization output: ${path}" >&2
    exit 1
  }
done
mkdir -p "${s1_vis_root}"
```

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src conda run --no-capture-output -n maniskill2-pidec \
  python -m robot_practice.experiments.rl.policy_decorator_eval \
  --reference-root=datasets/raw/policy_decorator/reference_repo \
  --env-id=PegInsertionSide-v2 --base-policy-ckpt="${s1_ckpt}" \
  --base-policy-args="${s1_args}" --arm=base \
  --seeds="${s1_eval_seed}" --episodes-per-seed=1 \
  --ddim-steps=4 --act-horizon=4 \
  --record-rollout-dir="${s1_vis_root}/raw_base" --record-episodes=1 \
  --output-dir="${s1_vis_root}/eval_base"

PYTHONNOUSERSITE=1 PYTHONPATH=src conda run --no-capture-output -n maniskill2-pidec \
  python -m robot_practice.experiments.rl.policy_decorator_eval \
  --reference-root=datasets/raw/policy_decorator/reference_repo \
  --env-id=PegInsertionSide-v2 --base-policy-ckpt="${s1_ckpt}" \
  --base-policy-args="${s1_args}" --arm=residual \
  --residual-runs="${s1_sac_root}" --checkpoint-step="${s1_checkpoint_step}" \
  --manifest="${s1_vis_root}/eval_base/episode_manifest.jsonl" \
  --seeds="${s1_eval_seed}" --episodes-per-seed=1 \
  --ddim-steps=4 --act-horizon=4 \
  --record-rollout-dir="${s1_vis_root}/raw_residual" --record-episodes=1 \
  --output-dir="${s1_vis_root}/eval_residual"
```

评测完成后回到本地仓库根目录，只同步这四个结果目录：

```bash
remote_repo="${REMOTE_REPO:?Set REMOTE_REPO to the remote checkout}"
remote_ssh='ssh -p 3333'
s1_vis_root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/visualization_seed1_4m

for name in eval_base eval_residual raw_base raw_residual; do
  mkdir -p "${s1_vis_root}/${name}"
  rsync -av --info=progress2 -e "${remote_ssh}" \
    root@localhost:"${remote_repo}/${s1_vis_root}/${name}/" \
    "${s1_vis_root}/${name}/"
done
```

在本地 `robot-practice` 环境生成两个独立的 `.rrd`；加 `--write-video` 会同时生成 MP4：

```bash
bash scripts/experiments/rl/visualize_policy_decorator_peg.sh \
  --rollout-dir="${s1_vis_root}/raw_base" \
  --output-dir="${s1_vis_root}/rrd_base" \
  --episode-ids=0 --write-video

bash scripts/experiments/rl/visualize_policy_decorator_peg.sh \
  --rollout-dir="${s1_vis_root}/raw_residual" \
  --output-dir="${s1_vis_root}/rrd_residual" \
  --episode-ids=0 --write-video
```

然后在有桌面显示的两个终端分别打开两个 Rerun 窗口。文件名可能是 `success` 或
`failure`，用 `find` 避免硬编码结果标签：

```bash
# terminal A
rerun "$(find "${s1_vis_root}/rrd_base" -maxdepth 1 -type f -name '*.rrd' -print -quit)"

# terminal B
rerun "$(find "${s1_vis_root}/rrd_residual" -maxdepth 1 -type f -name '*.rrd' -print -quit)"
```

Rerun 面板包含原生场景
render、base action、scaled SAC residual、composed action、50D S1 state 及 reward/
success/terminal 信号；它们是诊断可视化，不会改变官方 S1 评测结果。上面的单 episode
命令只用于验证链路，不代表 base / SAC 的总体结果。

#### 2.4.1 多环境配置的正式可视化选择

S1 的环境 ID 固定为 `PegInsertionSide-v2`；这里的“多个环境配置”指多个独立的
`reset_seed`，而不是切换 control mode。要覆盖现有 paper run 的训练 seed 1--4 和每个
seed 的 50 个 reset 配置，先在远端只做 paired metrics（不录 RGB）。本小节是可独立复制
执行的完整代码块，不依赖前面的单 episode 小节：

如果两个 200-episode metrics 已经完成，推荐直接使用仓库脚本完成“检查 paired metrics →
挑选 episode → 远端录制 base/residual”这三步。脚本默认选择最多 12 个代表性 episode，
非空的 `recorded/` 目录不会被覆盖；空目录可以继续使用：

```bash
# 本地仓库根目录执行，只同步脚本本身
remote_repo="${REMOTE_REPO:?Set REMOTE_REPO to the remote checkout}"
remote_ssh='ssh -p 3333'
rsync -av --relative -e "${remote_ssh}" \
  scripts/experiments/rl/prepare_policy_decorator_peg_visualization_remote.sh \
  root@localhost:"${remote_repo}/"
```

```bash
# 远端仓库根目录执行
cd "${REMOTE_REPO:?Set REMOTE_REPO to the remote checkout}"
# 单卡：
bash scripts/experiments/rl/prepare_policy_decorator_peg_visualization_remote.sh --gpus=0

# 多卡（例如远端可见的 0--3 号 GPU）：
# bash scripts/experiments/rl/prepare_policy_decorator_peg_visualization_remote.sh --gpus=0,1,2,3
```

默认只使用 GPU 0；需要并行录制时传入物理 GPU ID，例如
`--gpus=0,1,2,3`。脚本按选中的 episode 分片，每张 GPU 同时运行一个 worker，先完成
base 再完成 residual；这不是 DDP，而是独立的单卡 rollout 进程，最终仍汇总为 `raw_base/`
和 `raw_residual/` 两个目录。

脚本完成后，选择结果位于
`outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/visualization_4seed50/recorded/`；
其下的 `raw_base` 和 `raw_residual` 是后续本地 Rerun 的输入。若需要重新录制，使用一个
新的 `--record-root`，不要覆盖已有目录。

本地后处理环境需要 `rerun-sdk==0.26.2` 和 `ffmpeg`。前者已写入仓库的
`requirements.txt`；若当前环境是旧环境，先执行：

```bash
PYTHONNOUSERSITE=1 python -m pip install --no-cache-dir 'rerun-sdk==0.26.2'
```

这只安装可视化依赖，不下载数据或模型。确认 `python -c 'import rerun'` 和
`command -v ffmpeg` 均成功后，再执行下面的本地转换。

如果 `eval_base/episodes.jsonl` 和 `eval_residual/episodes.jsonl` 已经各有 200 行，跳过下面
的 paired-metrics 代码块，直接执行上面的录制脚本；下面代码只用于尚未生成 200 集 metrics
的情况。

```bash
s1_ckpt=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt
s1_args=outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/args.json
s1_sac_root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/sac
# 可通过环境变量覆盖；空值也安全回退到已保存的 4M checkpoint。
s1_checkpoint_step="${s1_checkpoint_step:-4000000}"
s1_eval_seeds=1,2,3,4
s1_episodes_per_seed=50
s1_metrics_root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/visualization_4seed50
s1_record_root="${s1_metrics_root}/recorded"

test -s "${s1_ckpt}" || { echo "missing base checkpoint: ${s1_ckpt}" >&2; exit 1; }
test -s "${s1_args}" || { echo "missing base args: ${s1_args}" >&2; exit 1; }
test "${s1_checkpoint_step}" -gt 0 || {
  echo "s1_checkpoint_step must be a positive integer, got: ${s1_checkpoint_step@Q}" >&2
  exit 1
}
for s1_training_seed in 1 2 3 4; do
  s1_seed_checkpoint="$(find "${s1_sac_root}/seed_${s1_training_seed}" \
    -type f -path "*/checkpoints/${s1_checkpoint_step}.pt" -print -quit 2>/dev/null)"
  test -s "${s1_seed_checkpoint}" || {
    echo "missing residual checkpoint: seed=${s1_training_seed}, step=${s1_checkpoint_step}" >&2
    echo "searched under: ${s1_sac_root}/seed_${s1_training_seed}" >&2
    exit 1
  }
done

for path in "${s1_metrics_root}" "${s1_record_root}"; do
  test ! -e "${path}" || {
    echo "refusing to reuse S1 multi-config output: ${path}" >&2
    exit 1
  }
done
mkdir -p "${s1_metrics_root}"

PYTHONNOUSERSITE=1 PYTHONPATH=src conda run --no-capture-output -n maniskill2-pidec \
  python -m robot_practice.experiments.rl.policy_decorator_eval \
  --reference-root=datasets/raw/policy_decorator/reference_repo \
  --env-id=PegInsertionSide-v2 --base-policy-ckpt="${s1_ckpt}" \
  --base-policy-args="${s1_args}" --arm=base \
  --seeds="${s1_eval_seeds}" --episodes-per-seed="${s1_episodes_per_seed}" \
  --ddim-steps=4 --act-horizon=4 \
  --output-dir="${s1_metrics_root}/eval_base"

PYTHONNOUSERSITE=1 PYTHONPATH=src conda run --no-capture-output -n maniskill2-pidec \
  python -m robot_practice.experiments.rl.policy_decorator_eval \
  --reference-root=datasets/raw/policy_decorator/reference_repo \
  --env-id=PegInsertionSide-v2 --base-policy-ckpt="${s1_ckpt}" \
  --base-policy-args="${s1_args}" --arm=residual \
  --residual-runs="${s1_sac_root}" --checkpoint-step="${s1_checkpoint_step}" \
  --manifest="${s1_metrics_root}/eval_base/episode_manifest.jsonl" \
  --seeds="${s1_eval_seeds}" --episodes-per-seed="${s1_episodes_per_seed}" \
  --ddim-steps=4 --act-horizon=4 \
  --output-dir="${s1_metrics_root}/eval_residual"
```

远端录制脚本完成后回到本地仓库根目录，只同步选中的 raw bundle、对应评测摘要和 episode ID：

```bash
remote_repo="${REMOTE_REPO:?Set REMOTE_REPO to the remote checkout}"
remote_ssh='ssh -p 3333'
s1_record_root=outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/visualization_4seed50/recorded

mkdir -p "${s1_record_root}"
for name in eval_base eval_residual raw_base raw_residual; do
  mkdir -p "${s1_record_root}/${name}"
  rsync -av --info=progress2 -e "${remote_ssh}" \
    root@localhost:"${remote_repo}/${s1_record_root}/${name}/" \
    "${s1_record_root}/${name}/"
done
rsync -av -e "${remote_ssh}" \
  root@localhost:"${remote_repo}/outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/visualization_4seed50/recorded_episode_ids.txt" \
  "${s1_record_root}/../recorded_episode_ids.txt"

s1_record_ids="$(<"${s1_record_root}/../recorded_episode_ids.txt")"
test -n "${s1_record_ids}"
```

只在本地分别后处理两个 raw 目录。后处理后，
每个模型的窗口可以一次加载该模型的多个 `.rrd`：

```bash
for variant in base residual; do
  bash scripts/experiments/rl/visualize_policy_decorator_peg.sh \
    --rollout-dir="${s1_record_root}/raw_${variant}" \
    --output-dir="${s1_record_root}/rrd_${variant}" \
    --episode-ids="${s1_record_ids}" --write-video
done

# terminal A: base window
rerun "${s1_record_root}/rrd_base/"*.rrd

# terminal B: residual SAC window
rerun "${s1_record_root}/rrd_residual/"*.rrd
```
