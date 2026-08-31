# Residual RL 到 Light-WAM 与 RLinf 三阶段验证指南

## 实验目标

本指南把 [三阶段计划](../../../plans/2026-08-28-Residual-RL到Light-WAM与RLinf三阶段验证.md) 落成可执行流程：

1. 在 ManiSkill2 `PegInsertionSide-v2` 准备并审计 Policy Decorator 的 frozen Diffusion Policy，
   并用 smoke 验证本地接线；本轮不执行 residual SAC 正式训练或性能对照。S1 的正式入口只认
   pinned Policy Decorator checkout 的官方脚本，不修改官方实现。
2. 在 RoboTwin2.0 构建一个本地 Light-WAM，并比较同一 frozen base 与 `+ residual SAC`。
   另提供一条独立的 `PegInsertionSide-v2` + Light-WAM + residual SAC 适配链路，作为当前
   ManiSkill2 延伸实验；它不改变 RoboTwin S2 的既有合同。
3. 在 RoboTwin2.0 `place_empty_cup / demo_randomized` 上，用同一 OpenVLA-OFT SFT 基座复现 SFT、GRPO、PPO。

本指南只描述如何运行、如何验收和如何记录；当前计划仍为 `todo` 时，不得把预期数字写成实验结果。

## 实验思路

1. 三个阶段严格按 S1 → S2 → S3 推进；本轮 S1 只做 artifact/runtime audit 和本地接线
   smoke，S2 才做 residual paired 对照，S3 是同一 SFT base 的三臂对照。
2. 所有 revision、task、seed、action contract、训练预算和阈值在运行前冻结。
3. 先做 source/artifact/protocol audit，再做小规模 smoke；S1 不把本地 evaluator 的输出当作
   官方结果，也不在本轮自动启动 residual formal 训练。
4. released checkpoint 只能做环境/权重 parity；主结论使用计划指定的本地 checkpoint。
5. 只报告 episode-level manifest、聚合指标、hash、吞吐、延迟和失败分类，不提交原始轨迹、视频、权重或日志。

结果标签必须区分 `Smoke`、`Offline`、`Closed-loop`、`Formal`。

S1 的 `datasets/raw/policy_decorator/reference_repo/online/pi_dec_diffusion_maniskill2.py` 是唯一
的官方 residual 训练/内置评测入口。它没有独立的 eval-only CLI：`best.pt` 是 frozen base，真正的
residual 结果必须由该脚本产生 `res_actor` checkpoint 并在同一官方 run 的周期评测中记录。没有
明确批准的正式训练时，只做下面的 audit/smoke；不要用 `policy_decorator_eval.py` 或
`validation_runner.py` 的自定义闭环数字冒充官方结果。这两个模块仅用于本地接口诊断/配对审计。

§0 的环境检查与 dummy smoke 是一条独立命令，应先从仓库根目录执行。进入 §1 后，下载和后续运行代码块应在同一个 Bash 会话中依次执行，并先运行 §1.1 的 preamble；它启用未定义变量和 pipeline 失败传播，避免 `tee` 把训练失败伪装成成功。

所有本地 evaluator/runner 还必须遵守同一个 paired-evaluation 合同：先生成一次只读的 `episode_manifest.jsonl`（至少包含 `episode_key`、环境 seed/reset-state id、task config 和 executed action count），base 与 treatment 逐行消费相同 manifest；汇总器必须拒绝 key、reset-state id 或 episode 数不一致的输入。仅传相同 seed 不足以证明配对。

## 固定合同与资源闸门

| 阶段 | 固定任务/基座 | 关键合同 |
| --- | --- | --- |
| S1 | `PegInsertionSide-v2` + 官方 Diffusion Policy | state、`pd_ee_delta_pose` 7D、20 Hz、200 steps、obs/pred/action/execute = 2/16/4/4；smoke 不训练，正式论文对齐 run 使用 10 个 seed |
| S2 | `move_stapler_pad` + 本地 Light-WAM | 14D physical qpos target；residual 为 4×12 arm-joint，gripper 直通；1M low-level actions/seed、seeds 1/2/3 |
| S2-Peg（独立延伸） | `PegInsertionSide-v2` + 官方 Light-WAM | 两路 RGB 横向拼接、32D non-privileged proprio、7D `pd_ee_delta_pose`；base horizon 16、replan 4、residual 为每次执行的 4×7；base 与 residual 模块分离 |
| S3 | `place_empty_cup / demo_randomized` + OpenVLA-OFT SFT | Piper 14D、head RGB 224、25-action chunk、200 steps、128 fixed eval env；GRPO/PPO 官方配置不改算法/批量/env/seed |

开始前必须确认：

- S1 `maniskill2-pidec`、Policy Decorator commit `92ba9ba442587ae5989c286355ace2200a0537fb`、ManiSkill2 `0.5.3` 和官方 checkpoint/`args.json` 均可审计。
- S2 至少 4 张训练 GPU；下载 cache 时峰值磁盘至少 2.4 TB，现算路径约 200 GB；并行 env 方案已决定。
- S3 至少 8 张满足官方显存要求的 GPU，并能运行 `rlinf/rlinf:agentic-rlinf0.4-robotwin`。
- 所有长任务每 60 秒内向 stdout 和 `run.log` 写 `STATUS completed=<n> total=<N>`，结束写 `DONE completed=<N> total=<N>`。
- 任何正式 run 的输出目录必须是新的或经过 audit 明确确认的同一 run；命令不得用 `rm -rf`、静默覆盖 checkpoint 或把不同协议的结果混写。

S1 的 `PegInsertionSide-v2` 不是 `mani-skill2==0.5.3` wheel 自带的注册项。它是 pinned
Policy Decorator checkout 里的 `envs.maniskill_fixed` 注册的 repo-local corrected runtime；因此
正式 S1 必须把该 checkout 作为 `--reference-root`/`PYTHONPATH`，并显式导入该模块。镜像单独
只保证原生 `PegInsertionSide-v0` 可用，不能把 `v0` 冒充正式的 `v2`。

## 0. 环境检查与 dummy smoke

计划要求先冻结以下本地入口，再启动任何正式实验：

```text
configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml
scripts/experiments/rl/run_three_stage_validation.sh
scripts/experiments/robotwin/run_lightwam_residual_rl.sh
src/robot_practice/experiments/rl/
tests/experiments/rl/
tests/experiments/robotwin/
```

下面这一条命令从仓库根目录依次完成实现文件检查、单元测试、dummy preflight，以及 S1/S2/S3 的单 GPU smoke。它不下载或读取真实数据、checkpoint、视频或权重，也不产生 Formal 结果：

```bash
set -Eeuo pipefail
repo_root="${PRACTICE_ROOT:-$PWD}"
repo_root="$(git -c "safe.directory=${repo_root}" -C "${repo_root}" rev-parse --show-toplevel)"
cd "${repo_root}"
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/rl-three-stage-dummy.XXXXXX")"
required_paths=(
  configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml
  scripts/experiments/rl/run_three_stage_validation.sh
  scripts/experiments/rl/run_policy_decorator_paper_multigpu.sh
  scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh
  scripts/experiments/robotwin/run_lightwam_residual_rl.sh
  scripts/experiments/rl/run_lightwam_peg_residual.sh
  scripts/experiments/rl/train_lightwam_peg.sh
  configs/experiments/rl/lightwam_peg_insertion.yaml
  src/robot_practice/experiments/rl
  tests/experiments/rl
  tests/experiments/robotwin
)
for required_path in "${required_paths[@]}"; do
  test -e "${required_path}" || {
    echo "missing required local implementation: ${required_path}" >&2
    exit 2
  }
done

PYTHONNOUSERSITE=1 conda run -n robot-practice \
  python -m pytest -q tests/experiments/rl tests/experiments/robotwin

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=preflight --dummy \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root="${smoke_root}/preflight"

bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
  --phase=smoke-rollout --dummy --task=move_stapler_pad \
  --task-config=demo_randomized --episodes=2 --replan-steps=4 \
  --zero-residual=true --max-steps=8 \
  --output-root="${smoke_root}/s2_rollout"

bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
  --phase=smoke-train --dummy --task=move_stapler_pad \
  --task-config=demo_randomized --seed=1 --total-low-level-actions=64 \
  --learning-starts=8 --num-envs=2 --replan-steps=4 --batch-size=16 \
  --res-scale=0.1 --prog-explore=30000 \
  --output-root="${smoke_root}/s2_train"

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s3-audit --dummy \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root="${smoke_root}/s3_audit"

bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=evaluate --dummy --arm=base --device=cpu \
  --eval-seeds=1 --episodes-per-seed=1 --max-steps=4 \
  --output-dir="${smoke_root}/s2_peg_base"

bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=train --dummy --device=cpu --total-low-level-actions=8 \
  --learning-starts=1 --utd=1 --batch-size=2 --hidden-dim=32 \
  --log-every=4 --save-every=8 \
  --output-dir="${smoke_root}/s2_peg_residual"

test -s "${smoke_root}/preflight/preflight.json"
```

当前仓库若尚未实现上述入口，`missing required local implementation` 是预期的阻断，不得用空脚本、跳过测试或直接运行 vendor 命令绕过。preflight 必须落盘 resolved config、所有 Git/Hugging Face revision、包版本、GPU 信息、文件 hash、seed 列表和 train/development/test 无交集审计；分享摘要不得包含机器名或绝对路径。任一 gate 失败都不能进入训练。

通过条件是：S1 dummy preflight 通过；S2
`max_zero_residual_action_deviation=0`、`max_gripper_residual=0`、base hash 前后一致、
`base_parameters_frozen=true` 且 `sac_updates>0`；所有目录都有持续的 `STATUS`/最终 `DONE`。
该 smoke 不读取 S1 真实 checkpoint，也不产生 S1 正式结果；它只覆盖接口和环境接线。checkpoint
下载、artifact audit 和官方 residual run 必须分别执行后续条目。

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

### 1.4 S2 Light-WAM、RoboTwin 和数据

```bash
practice_root="$(pwd -P)"
test -d third_party/lightwam/Light-WAM/.git || \
  "${git_direct[@]}" clone https://github.com/L1ziang/Light-WAM.git third_party/lightwam/Light-WAM
"${git_direct[@]}" -C third_party/lightwam/Light-WAM fetch origin
"${git_direct[@]}" -C third_party/lightwam/Light-WAM checkout b2785f66e13fd9987e94ae1ecc1c441d5059c9ae

mkdir -p outputs/rl_three_stage_20260828/artifacts/lightwam
mkdir -p datasets/raw/lightwam datasets/lerobot/lightwam

"${direct_env[@]}" conda run --no-capture-output -n robot-practice hf download Wan-AI/Wan2.1-T2V-1.3B \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir outputs/rl_three_stage_20260828/artifacts/lightwam/Wan-AI/Wan2.1-T2V-1.3B

"${direct_env[@]}" conda run --no-capture-output -n robot-practice hf download l1ziang/lightwam-checkpoints \
  --revision 7cc8593fb95423a9cfbb93f82c95c2fa7d5357bd \
  --include "lightwam_robotwin/*" "config.yaml" \
  --local-dir outputs/rl_three_stage_20260828/artifacts/lightwam/released

"${direct_env[@]}" conda run --no-capture-output -n robot-practice hf download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --revision aac262c35d02cc71b2f6ef670bd65fd9f2bb2547 \
  --local-dir datasets/raw/lightwam/robotwin2.0-fastwam
```

若大分片中断，先只恢复对应分片，不要重跑全仓库，也不要传 `--force-download`。当前 8 个分片中 `part-06` 的远端大小是 `10737418240` bytes，`hf` 的 `local_dir` 续传状态位于目标目录下的 `.cache/huggingface/download/`：

```bash
target_dir=datasets/raw/lightwam/robotwin2.0-fastwam
find "${target_dir}/.cache/huggingface/download" -maxdepth 1 \
  -type f -name '*.incomplete' -printf '%s\t%p\n'

recovered=0
for attempt in $(seq 1 20); do
  echo "resume part-06 attempt ${attempt}/20"
  if "${direct_env[@]}" \
    HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=60 \
    HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 \
    conda run --no-capture-output -n robot-practice hf download \
      yuanty/robotwin2.0-fastwam robotwin2.0.tar.gz.part-06 \
      --repo-type dataset \
      --revision aac262c35d02cc71b2f6ef670bd65fd9f2bb2547 \
      --max-workers 1 \
      --local-dir "${target_dir}"; then
    recovered=1
    break
  fi
  partial_bytes="$(find "${target_dir}/.cache/huggingface/download" -maxdepth 1 \
    -type f -name '*.incomplete' -printf '%s\n' | sort -n | tail -n 1)"
  echo "download interrupted; current incomplete bytes=${partial_bytes:-0}" >&2
  sleep 5
done
test "${recovered}" -eq 1

test "$(stat -c '%s' "${target_dir}/robotwin2.0.tar.gz.part-06")" \
  -eq 10737418240
printf '%s  %s\n' \
  b69f2a619ff98447d334853be83f203be732b37d6370d36f32e4396562206ec9 \
  "${target_dir}/robotwin2.0.tar.gz.part-06" | sha256sum -c -
```

`HF_HUB_ENABLE_HF_TRANSFER=0` 是为了避免某些版本在启用 fast-transfer 时删除 `.incomplete` 后从零开始。
`HF_HUB_DISABLE_XET=1` 是**续传能生效的前提**：`huggingface_hub` 0.35.3 的
`file_download.py:_download_to_tmp_and_move` 只把 `resume_size = f.tell()` 传给经典分支
`http_get(..., resume_size=resume_size, ...)`；xet 分支调用的 `xet_get()` 完全不接收 `resume_size`，
而是把 `incomplete_path` 交给 `hf_xet.download_files`（`PyXetDownloadInfo(destination_path=...)`、
进度条 `initial=0`）自行写文件，因此 `.incomplete` 里的已有字节被忽略、每次重启都从零重下。
xet 侧唯一的复用是 `~/.cache/huggingface/xet/chunk-cache`，实测该缓存只有 30 MB 量级，对 10 GiB
分片等于没有断点。`is_xet_available()` 在 `HF_HUB_DISABLE_XET` 为真时直接返回 `False`，所以设为 `1`
即落到带 Range 的经典续传路径（日志会打印 `(resume from <n>/<total>)`，可用它确认续传真的生效）。
从 xet 路径切到经典路径前可先验证已有 `.incomplete` 是密集前缀而非稀疏乱序写入：
`stat -c'%s %b'` 的 `blocks*512/size ≈ 1.0` 且 `filefrag -v` 无 logical offset 空洞即可安全续传。
恢复完成后再按计划对所有分片做 SHA-256/artifact audit，确认无误才解包。

本次 `IncompleteRead → ChunkedEncodingError` 属于响应体提前结束；它不等同于目标文件已损坏，也不会自动把整个分片标记为完成。若某次重试仍退出，保留 `.incomplete` 并再次执行上面的单文件命令即可从最新偏移继续——但这一条只在 `HF_HUB_DISABLE_XET=1` 时成立，开着 xet 重试会静默从零开始（表现为 `.incomplete` 大小回落、日志没有 `resume from` 行）。

`lightwam-checkpoints` 必须保留 `--include`，只取 `lightwam_robotwin/` 和 `config.yaml`。解压原始数据：

```bash
mkdir -p datasets/raw/lightwam/robotwin2.0
test -z "$(find datasets/raw/lightwam/robotwin2.0 -mindepth 1 -print -quit)" || {
  echo 'refusing to extract into non-empty datasets/raw/lightwam/robotwin2.0' >&2
  exit 1
}
cat datasets/raw/lightwam/robotwin2.0-fastwam/robotwin2.0.tar.gz.part-* |
  tar --keep-old-files -xzf - -C datasets/raw/lightwam/robotwin2.0
```

RoboTwin assets 必须按对应 checkout 的下载清单处理；只有 `_download.py` 完全一致时才可复用另一 checkout 的已解压目录。否则在目标 checkout 中执行：

```bash
(
  cd third_party/lightwam/Light-WAM/third_party/RoboTwin
  "${direct_env[@]}" bash script/_download_assets.sh
)
```

### 1.5 latent cache（默认不下载）

只有 §3.1 的 VAE 现算/预编码 probe 冻结为“下载 cache”后，才运行：

```bash
"${direct_env[@]}" conda run --no-capture-output -n robot-practice hf download l1ziang/lightwam-offline-cache \
  --repo-type dataset \
  --revision 995bd27d4848cf5b0afefb4d33657ae127205076 \
  --include "latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded.tar.part-*" \
  --local-dir datasets/raw/lightwam/offline-cache-release

mkdir -p datasets/lerobot/lightwam/latent_cache_Wan2.1-T2V-1.3B
test -z "$(find datasets/lerobot/lightwam/latent_cache_Wan2.1-T2V-1.3B -mindepth 1 -print -quit)" || {
  echo 'refusing to extract into non-empty latent cache directory' >&2
  exit 1
}
cat datasets/raw/lightwam/offline-cache-release/latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded.tar.part-* |
  tar --keep-old-files -xf - -C datasets/lerobot/lightwam/latent_cache_Wan2.1-T2V-1.3B
```

### 1.6 S3 RLinf checkout、assets 和 released artifacts

```bash
test -d third_party/rlinf/RLinf/.git || \
  "${git_direct[@]}" clone https://github.com/RLinf/RLinf.git third_party/rlinf/RLinf
"${git_direct[@]}" -C third_party/rlinf/RLinf fetch origin
"${git_direct[@]}" -C third_party/rlinf/RLinf checkout 7ad72d1160858971293fec4d2cca232642320574

test -d third_party/rlinf/RoboTwin/.git || \
  "${git_direct[@]}" clone --branch RLinf_support https://github.com/RoboTwin-Platform/RoboTwin.git \
  third_party/rlinf/RoboTwin
"${git_direct[@]}" -C third_party/rlinf/RoboTwin fetch origin RLinf_support
"${git_direct[@]}" -C third_party/rlinf/RoboTwin checkout 0008ae6800df9f75fc8de7098bacb01735fd8fd2

(
  cd third_party/rlinf/RoboTwin
  "${direct_env[@]}" bash script/_download_assets.sh
)

mkdir -p outputs/rl_three_stage_20260828/artifacts/rlinf
"${direct_env[@]}" conda run --no-capture-output -n robot-practice hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-SFT-place_empty_cup \
  --revision 04150e05ded8b915ccf6adb7ea903ff13dc3aa27 \
  --local-dir outputs/rl_three_stage_20260828/artifacts/rlinf/sft
"${direct_env[@]}" conda run --no-capture-output -n robot-practice hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-RL-place_empty_cup \
  --revision 2984c5a20196b26d4e19ce3335e7caad80f1fcb6 \
  --local-dir outputs/rl_three_stage_20260828/artifacts/rlinf/grpo_released
"${direct_env[@]}" conda run --no-capture-output -n robot-practice hf download RLinf/RLinf-OpenVLAOFT-RoboTwin-PPO-place_empty_cup \
  --revision cf9c4730d06279d6eadc11b75946f96342959796 \
  --local-dir outputs/rl_three_stage_20260828/artifacts/rlinf/ppo_released
```

## 2. S1：Policy Decorator / ManiSkill2

### 2.1 官方 residual run（需要正式 residual 结果时才执行）

官方 online 入口同时负责 residual SAC 训练和周期性闭环评测，没有独立的 eval-only 参数。
`best.pt` 是 frozen base；只有运行下面这个上游命令后，才会产生 `res_actor` checkpoint 和官方
评测曲线。命令直接调用 pinned checkout，不经过本仓库的 runner/evaluator；除 seed 和输出目录外，
只传官方 README 的任务参数。正式训练未获明确批准时不要执行这一节。

论文结果需要先做 frozen base 的十 seed 闭环，再做十个独立的官方 residual-SAC run。使用一条
wrapper 命令串行完成两阶段：base 全部成功且 checkpoint hash 不变后，才启动官方 SAC。每个
worker 只看到一张 GPU，不能改成 `torchrun`/DDP；GPU 数少于 10 时按批次运行。论文正文只公布
“十个 seed”，没有公布 canonical seed-id 列表，因此默认的 1--10 会写入 manifest。S2 的三 seed
约定不受本节影响。

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

## 3. S2：本地 Light-WAM 与 residual SAC

### 3.1 环境与 VAE/cache 决策

训练环境和评测环境分开：`lightwam` 用于 precompute/训练，`lightwam-robotwin-eval` 只用于评测。

```bash
conda_env_exists() {
  conda env list --json |
    python -c 'import json, sys; envs=json.load(sys.stdin)["envs"]; name=sys.argv[1]; raise SystemExit(0 if any(p.rsplit("/", 1)[-1] == name for p in envs) else 1)' "$1"
}

# The RL three-stage Docker layer prebuilds both environments. Keep this fallback for a
# non-prebuilt host, but never recreate an existing environment: conda env create/clone is not
# idempotent and would otherwise destroy the image's pinned package contract.
if ! conda_env_exists lightwam; then
  conda create -n lightwam python=3.10 -y
  conda run --no-capture-output -n lightwam python -m pip install -U pip
  conda run --no-capture-output -n lightwam \
    python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
else
  echo "using prebuilt conda env: lightwam"
fi

if ! conda_env_exists lightwam-robotwin-eval; then
  conda env create -f third_party/lightwam/Light-WAM/scripts/robotwin_env.full.yml
else
  echo "using prebuilt conda env: lightwam-robotwin-eval"
fi

conda run --no-capture-output -n lightwam \
  python -m pip install -e third_party/lightwam/Light-WAM
conda run --no-capture-output -n lightwam-robotwin-eval \
  python -m pip install -e third_party/lightwam/Light-WAM --no-deps
```

先用固定 probe 子集做 cached vs online VAE 对照，记录去掉 warmup 后的 median/p95 `s/step`、dataloader 等待、peak VRAM、5 epoch 总步数和预编码体积。只有满足冻结决策表才下载全量 cache：

| 条件 | 决策 |
| --- | --- |
| online 总墙钟 `<=1.3×` cached 且 `BATCH_SIZE=16` 不 OOM | 现算，不下载全量 cache |
| online `>1.3×` cached 且磁盘峰值 `>=2.4 TB` | 下载 RoboTwin 分片 cache |
| online OOM 或必须改 batch/配方 | cache 为强制项，下载 |
| 磁盘不足且 online 可行 | 现算，记录墙钟代价 |

该 probe 的 loss/success 不是正式结果。

先审计 pinned checkout 的 `src/lightwam/` dataset/collate 与 `scripts/train_robotwin.sh`。注意：该 checkout 的 `scripts/precompute_robotwin.sh` 会把数据和 cache 路径硬编码到 `Light-WAM/data/...`，而 `scripts/train_robotwin.sh` 默认 `data.train.use_latent_cache=true`；仅设置 `LATENT_CACHE_DIR` 不能切换到现场 VAE。因此 probe 必须调用底层脚本并显式传 Hydra 路径/模式，不得把下面的环境变量误当成 wrapper 的配置。若 audit 发现本地 vendor 内容不是 pinned commit，先停止。

probe 采用 pinned dataset split 的约 1% validation episode（`val_set_proportion=0.99`、`is_training_set=false`、BaseLerobotDataset 默认 seed `42`），并在训练 A/B 中使用完全相同的 split。该 cache 只用于计时，不能拿去跑 formal 全量训练。两种模式使用同 GPU 数、`BATCH_SIZE=16`、同 seed；前 20 步 warmup，后 200 步计时。

```bash
practice_root="${repo_root}"
lightwam_root="${practice_root}/third_party/lightwam/Light-WAM"
robotwin_dir="${practice_root}/datasets/raw/lightwam/robotwin2.0/robotwin2.0"
text_cache_dir="${practice_root}/datasets/lerobot/lightwam/text_embeds_cache/robotwin"
probe_cache_dir="${practice_root}/datasets/lerobot/lightwam/latent_cache_probe_subset"
probe_root="${practice_root}/outputs/rl_three_stage_20260828/stage2/vae_cache_probe"
probe_gpu_ids="${PRECOMPUTE_GPU_IDS:-0,1,2,3}"
probe_num_processes="${PRECOMPUTE_NUM_PROCESSES:-4}"

test -d "${lightwam_root}/.git"
test -d "${robotwin_dir}"
test -d "${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam/Wan-AI/Wan2.1-T2V-1.3B"
test ! -e "${probe_root}" || {
  echo "refusing to reuse existing probe output root: ${probe_root}" >&2
  exit 1
}
test ! -e "${probe_cache_dir}" || {
  echo "refusing to reuse existing probe cache: ${probe_cache_dir}" >&2
  exit 1
}
for run_dir in "${probe_root}/cached" "${probe_root}/online"; do
  test ! -e "${run_dir}" || {
    echo "refusing to mix with existing probe run: ${run_dir}" >&2
    exit 1
  }
done
mkdir -p "${probe_root}" "${probe_cache_dir}"
cd "${lightwam_root}"

# 共享 text cache：显式写入仓库 artifact 路径，不使用 precompute_robotwin.sh 的硬编码目录。
CUDA_VISIBLE_DEVICES="${probe_gpu_ids}" PYTHONNOUSERSITE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
conda run --no-capture-output -n lightwam \
  torchrun --standalone --nproc_per_node="${probe_num_processes}" \
  scripts/precompute_text_embeds.py \
  "task=robotwin_uncond_3cam_384_1e-4" "overwrite=false" \
  "data.train.dataset_dirs=['${robotwin_dir}']" \
  "data.train.text_embedding_cache_dir=${text_cache_dir}" \
  "data.val.dataset_dirs=['${robotwin_dir}']" \
  "data.val.text_embedding_cache_dir=${text_cache_dir}" 2>&1 |
  tee "${probe_root}/precompute_text.log"

# A：probe split 预编码；不得把全量 cache 下载到本步骤。
CUDA_VISIBLE_DEVICES="${probe_gpu_ids}" PYTHONNOUSERSITE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
conda run --no-capture-output -n lightwam \
  torchrun --standalone --nproc_per_node="${probe_num_processes}" \
  scripts/precompute_video_latents.py \
  "task=robotwin_uncond_3cam_384_1e-4" "overwrite=true" \
  "model.video_backbone_type=wan2_1_t2v" \
  "model.video_backbone_name=Wan-AI/Wan2.1-T2V-1.3B" \
  "precompute_storage_format=sharded_v1" "precompute_video_only=true" \
  "precompute_shard_size=1024" "precompute_batch_size=16" \
  "precompute_num_workers=8" "precompute_cache_dtype=model" \
  "precompute_resume=false" "precompute_timing.enabled=true" \
  "precompute_timing.sync_cuda=true" "precompute_timing.log_every=10" \
  "data.train.dataset_dirs=['${robotwin_dir}']" \
  "data.train.latent_cache_dir=${probe_cache_dir}" \
  "data.train.val_set_proportion=0.99" "data.train.is_training_set=false" 2>&1 |
  tee "${probe_root}/precompute_subset.log"

# A：读取预编码 latent；extra overrides 使训练使用与 precompute 相同的 1% split。
PYTHONNOUSERSITE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
GPU_IDS="${probe_gpu_ids}" NUM_PROCESSES="${probe_num_processes}" \
RUN_ID=probe_cached MAX_STEPS=220 SAVE_EVERY=100000 EVAL_EVERY=0 \
BATCH_SIZE=16 GRAD_ACC=1 NUM_EPOCHS=5 \
ROBOTWIN_TRAIN_DIR="${robotwin_dir}" ROBOTWIN_VAL_DIR="${robotwin_dir}" \
ROBOTWIN_NORM_STATS="${practice_root}/datasets/raw/lightwam/robotwin2.0-fastwam/dataset_stats.json" \
TEXT_EMBED_CACHE_DIR="${text_cache_dir}" LATENT_CACHE_DIR="${probe_cache_dir}" \
OUTPUT_DIR="${probe_root}/cached" \
conda run --no-capture-output -n lightwam bash scripts/train_robotwin.sh \
  "data.train.val_set_proportion=0.99" "data.train.is_training_set=false" \
  "data.val.val_set_proportion=0.99" "data.val.is_training_set=false" \
  "data.train.use_latent_cache=true" "data.train.latent_cache_dir=${probe_cache_dir}" \
  2>&1 | tee "${probe_root}/cached.log"

# B：现场 VAE；必须显式关闭 cache，不能仅省略 LATENT_CACHE_DIR。
PYTHONNOUSERSITE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
GPU_IDS="${probe_gpu_ids}" NUM_PROCESSES="${probe_num_processes}" \
RUN_ID=probe_online MAX_STEPS=220 SAVE_EVERY=100000 EVAL_EVERY=0 \
BATCH_SIZE=16 GRAD_ACC=1 NUM_EPOCHS=5 \
ROBOTWIN_TRAIN_DIR="${robotwin_dir}" ROBOTWIN_VAL_DIR="${robotwin_dir}" \
ROBOTWIN_NORM_STATS="${practice_root}/datasets/raw/lightwam/robotwin2.0-fastwam/dataset_stats.json" \
TEXT_EMBED_CACHE_DIR="${text_cache_dir}" \
OUTPUT_DIR="${probe_root}/online" \
conda run --no-capture-output -n lightwam bash scripts/train_robotwin.sh \
  "data.train.val_set_proportion=0.99" "data.train.is_training_set=false" \
  "data.val.val_set_proportion=0.99" "data.val.is_training_set=false" \
  "data.train.use_latent_cache=false" "data.train.latent_cache_dir=null" \
  2>&1 | tee "${probe_root}/online.log"
cd "${practice_root}"
```

### 3.2 训练 smoke 与 formal 本地 base

3.1 的显式 text-cache 命令必须先成功，并把 `precompute_text.log` 纳入 audit。先为 pinned `eval_robotwin.sh` 准备它强制读取的 `checkpoints/Wan-AI` 路径，再执行 source、text-cache、dataset split、model path 和训练配置 audit；audit 失败不得启动任何 smoke。只允许创建指向已下载 artifact 的符号链接；发现已有真实目录、断链或指向其他 artifact 时必须停止，不能覆盖：

```bash
practice_root="${repo_root}"
lightwam_root="${practice_root}/third_party/lightwam/Light-WAM"
wan_artifact="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam/Wan-AI"
wan_expected="${lightwam_root}/checkpoints/Wan-AI"
test -d "${wan_artifact}/Wan2.1-T2V-1.3B"
mkdir -p "${lightwam_root}/checkpoints"
if [[ -e "${wan_expected}" || -L "${wan_expected}" ]]; then
  test -L "${wan_expected}" || {
    echo "refusing to replace real directory: ${wan_expected}" >&2
    exit 1
  }
  test "$(readlink -f "${wan_expected}")" = "$(readlink -f "${wan_artifact}")" || {
    echo "Wan-AI symlink points to a different artifact" >&2
    exit 1
  }
else
  ln -s "${wan_artifact}" "${wan_expected}"
fi
```

```bash
bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s2-audit \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root=outputs/rl_three_stage_20260828/stage2/audit
```

audit 通过后先做 released 5-episode smoke；5 episode 的成功率不是正式结果。`eval_robotwin.sh` 会把输出目录追加 checkpoint tag，所以实际目录名以日志中的 `EVALUATION.output_dir` 为准：

```bash
practice_root="${repo_root}"
lightwam_root="${practice_root}/third_party/lightwam/Light-WAM"
released_ckpt="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt"
test -f "${released_ckpt}"
released_smoke_root="${practice_root}/outputs/rl_three_stage_20260828/stage2/released_smoke"
test ! -e "${released_smoke_root}" || {
  echo "refusing to reuse Light-WAM released smoke root: ${released_smoke_root}" >&2
  exit 1
}
mkdir -p "${released_smoke_root}"
cd "${lightwam_root}"
for task_config in demo_clean demo_randomized; do
  mkdir -p "${released_smoke_root}/${task_config}"
  PYTHONNOUSERSITE=1 \
  DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
  EVAL_MODE=single ROBOTWIN_TASK_NAME=move_stapler_pad TASK_CONFIG="${task_config}" \
  EVAL_NUM_EPISODES=5 REPLAN_STEPS=24 SKIP_GET_OBS_WITHIN_REPLAN=true \
  CKPT="${released_ckpt}" \
  OUTPUT_DIR="${released_smoke_root}/${task_config}" \
  conda run --no-capture-output -n lightwam-robotwin-eval bash scripts/eval_robotwin.sh \
    2>&1 | tee "${released_smoke_root}/${task_config}/run.log"
done
cd "${practice_root}"
```

再固定 `VAE_MODE=online` 或 `VAE_MODE=cached`，用 `MAX_STEPS=20` 验证训练链路，最后从干净目录运行与 pinned task config 对齐的 5 epoch。`MAX_STEPS=null` 是有意传给 Hydra 的显式 null，不能省略；省略会触发 wrapper 的 `500000` 默认值并覆盖 epoch budget。`VAE_MODE=online` 也必须通过 extra override 显式关闭 latent cache。

```bash
practice_root="${repo_root}"
lightwam_root="${practice_root}/third_party/lightwam/Light-WAM"
robotwin_dir="${practice_root}/datasets/raw/lightwam/robotwin2.0/robotwin2.0"
text_cache_dir="${practice_root}/datasets/lerobot/lightwam/text_embeds_cache/robotwin"
full_cache_dir="${practice_root}/datasets/lerobot/lightwam/latent_cache_Wan2.1-T2V-1.3B/robotwin_3cam384_sharded"
stage2_root="${practice_root}/outputs/rl_three_stage_20260828/stage2"
vae_mode="${VAE_MODE:?set VAE_MODE=online or VAE_MODE=cached after the §3.1 probe}"

test -d "${lightwam_root}/.git"
test -d "${robotwin_dir}"
test -d "${text_cache_dir}"
case "${vae_mode}" in
  online)
    train_cache_args=(
      "data.train.use_latent_cache=false"
      "data.train.latent_cache_dir=null"
    )
    ;;
  cached)
    test -f "${full_cache_dir}/index.pt" || {
      echo "cached mode requires a complete sharded latent cache: ${full_cache_dir}/index.pt" >&2
      exit 1
    }
    train_cache_args=(
      "data.train.use_latent_cache=true"
      "data.train.latent_cache_dir=${full_cache_dir}"
    )
    ;;
  *)
    echo "VAE_MODE must be online or cached, got: ${vae_mode}" >&2
    exit 1
    ;;
esac

for run_dir in "${stage2_root}/train_smoke" "${stage2_root}/train_formal"; do
  test ! -e "${run_dir}" || {
    echo "refusing to mix with existing Light-WAM run: ${run_dir}" >&2
    exit 1
  }
done
mkdir -p "${stage2_root}/train_smoke" "${stage2_root}/train_formal"
cd "${lightwam_root}"

PYTHONNOUSERSITE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
GPU_IDS=0,1,2,3 NUM_PROCESSES=4 RUN_ID=smoke MAX_STEPS=20 SAVE_EVERY=20 \
EVAL_EVERY=0 NUM_WORKERS=8 NUM_EPOCHS=5 BATCH_SIZE=16 GRAD_ACC=1 \
ROBOTWIN_TRAIN_DIR="${robotwin_dir}" ROBOTWIN_VAL_DIR="${robotwin_dir}" \
ROBOTWIN_NORM_STATS="${practice_root}/datasets/raw/lightwam/robotwin2.0-fastwam/dataset_stats.json" \
TEXT_EMBED_CACHE_DIR="${text_cache_dir}" LATENT_CACHE_DIR="${full_cache_dir}" \
OUTPUT_DIR="${stage2_root}/train_smoke" \
conda run --no-capture-output -n lightwam bash scripts/train_robotwin.sh \
  "train_visualization.enabled=false" "${train_cache_args[@]}" \
  2>&1 | tee "${practice_root}/outputs/rl_three_stage_20260828/stage2/train_smoke/run.log"

PYTHONNOUSERSITE=1 \
DIFFSYNTH_MODEL_BASE_PATH="${practice_root}/outputs/rl_three_stage_20260828/artifacts/lightwam" \
GPU_IDS=0,1,2,3 NUM_PROCESSES=4 RUN_ID=formal NUM_EPOCHS=5 MAX_STEPS=null \
BATCH_SIZE=16 GRAD_ACC=1 NUM_WORKERS=8 EVAL_EVERY=500 SAVE_EVERY=2500 WARMUP_STEPS=null \
ROBOTWIN_TRAIN_DIR="${robotwin_dir}" ROBOTWIN_VAL_DIR="${robotwin_dir}" \
ROBOTWIN_NORM_STATS="${practice_root}/datasets/raw/lightwam/robotwin2.0-fastwam/dataset_stats.json" \
TEXT_EMBED_CACHE_DIR="${text_cache_dir}" LATENT_CACHE_DIR="${full_cache_dir}" \
OUTPUT_DIR="${stage2_root}/train_formal" \
conda run --no-capture-output -n lightwam bash scripts/train_robotwin.sh \
  "train_visualization.enabled=false" "${train_cache_args[@]}" \
  2>&1 | tee "${practice_root}/outputs/rl_three_stage_20260828/stage2/train_formal/run.log"
cd "${practice_root}"
```

local-final 必须是 5 epoch 最后一个完整 checkpoint；不能从 validation 曲线挑选。`MAX_STEPS=null`、`WARMUP_STEPS=null`、`NUM_WORKERS=8`、`EVAL_EVERY=500` 和 `train_visualization.enabled=false` 用于保持 pinned `robotwin_uncond_3cam_384_1e-4.yaml` 的行为；所有 resolved values 必须写入 audit。released 与 local-final 各自跑 `move_stapler_pad` 的 clean/randomized 100 episode、`replan_steps=24`，再对 local-final 跑 `replan_steps=4`，24-step 与 4-step 结果分开报告。

通过 smoke 后评估 released/local-final 的 base parity，并保留 `replan24` 和 `replan4` 的独立 manifest：

```bash
bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s2-eval-base --task=move_stapler_pad \
  --released-checkpoint=outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt \
  --local-run=outputs/rl_three_stage_20260828/stage2/train_formal \
  --settings=demo_clean,demo_randomized --episodes=100 \
  --official-replan-steps=24 --residual-replan-steps=4 --seed=42 \
  --output-root=outputs/rl_three_stage_20260828/stage2/eval_base
```

### 3.3 residual 接通 smoke、正式训练与评测

```bash
bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
  --phase=smoke-rollout --task=move_stapler_pad --task-config=demo_randomized \
  --base-checkpoint=outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt \
  --episodes=2 --replan-steps=4 --zero-residual=true \
  --output-root=outputs/rl_three_stage_20260828/stage2/residual_smoke/rollout

bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
  --phase=smoke-train --task=move_stapler_pad --task-config=demo_randomized \
  --base-checkpoint=outputs/rl_three_stage_20260828/artifacts/lightwam/released/lightwam_robotwin/checkpoints/weights/step_460000.pt \
  --seed=1 --total-low-level-actions=12000 --learning-starts=1024 \
  --num-envs=2 --replan-steps=4 \
  --res-scale=0.1 --prog-explore=30000 \
  --output-root=outputs/rl_three_stage_20260828/stage2/residual_smoke/train
```

smoke 使用 released checkpoint 仅验证接线，不产生主结果。它必须证明 zero residual 时组合动作逐步等于纯 base、base hash 不变、`requires_grad=False`、gripper residual 恒为 0，且 `sac_updates > 0`；同时测得 low-level actions/s、Light-WAM chunk latency、SAC update latency。`--total-low-level-actions` 按所有并行 env 实际执行的 low-level transition 总和计数，不能把 `num-envs` 再乘一次。若单 env 估算的 1M×3 时间不可接受，先解决并行 env，不得直接开跑。

```bash
for seed in 1 2 3; do
  bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
    --phase=train --task=move_stapler_pad --task-config=demo_randomized \
    --base-run=outputs/rl_three_stage_20260828/stage2/train_formal \
    --seed="${seed}" --total-low-level-actions=1000000 --learning-starts=8000 --num-envs=16 \
    --replan-steps=4 --res-scale=0.1 --prog-explore=30000 \
    --output-root="outputs/rl_three_stage_20260828/stage2/residual_train/seed_${seed}"
done

for setting in demo_clean demo_randomized; do
  bash scripts/experiments/robotwin/run_lightwam_residual_rl.sh \
    --phase=evaluate --task=move_stapler_pad --task-config="${setting}" \
    --base-run=outputs/rl_three_stage_20260828/stage2/train_formal \
    --residual-runs=outputs/rl_three_stage_20260828/stage2/residual_train \
    --seeds=1,2,3 --eval-seed=42 --episodes=100 --replan-steps=4 \
    --output-root="outputs/rl_three_stage_20260828/stage2/eval_residual/${setting}"
done

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s2-summarize \
  --base=outputs/rl_three_stage_20260828/stage2/eval_base/replan4 \
  --treatment=outputs/rl_three_stage_20260828/stage2/eval_residual \
  --output-root=outputs/rl_three_stage_20260828/stage2/summary
```

S2 residual 合同固定为：base 输出同一 replan chunk 的 14D physical qpos target `a_base`，actor 输出 `r ∈ R^(4×12)`，

```text
a_final[:, arm] = clip(a_base[:, arm] + 0.1 * action_std_arm * tanh(r), q_min_arm, q_max_arm)
a_final[:, gripper] = a_base[:, gripper]
```

其中 `arm` 是两只 Piper 的 12 个 arm-joint 维度，`gripper` 是剩余 2 维；`action_std_arm` 必须来自冻结的 `datasets/raw/lightwam/robotwin2.0-fastwam/dataset_stats.json` 中 action 的未归一化物理单位统计，具体 JSON key、维度映射和单位写入每个 run 的 resolved config，不能手填常数。actor 输入是与 frozen Light-WAM 相同的非 privileged camera/proprio observation、frozen adapter 的 pooled feature 和当前 14D proprio（按同一 stats 归一化）；不得读取 object pose、task-success 或其他 simulator privileged state。critic 输入为同一 observation 与最终组合的 physical qpos target；base backbone/adapter 参数全部 frozen，残差 actor/critic 才可训练。

SAC 固定 `gamma=0.97 / UTD=0.25 / batch=1024 / actor=q lr=1e-4 / tau=0.01`，每 seed 的正式预算是所有 env 合计 1M low-level actions，`learning_starts=8000`。`arm residual saturation` 定义为所有 formal train/eval action decisions 中 `abs(tanh(r)) >= 0.99` 的 arm 元素占比；`jerk` 定义为执行 qpos command 的三阶有限差分 `||Δ³q/dt³||₂`，以同一 paired episode manifest 的 p95 比较 base 与 treatment。不得 sweep、换 base 或用 released 权重替代 local-final 主结论。

### 3.4 独立延伸：PegInsertionSide-v2 上的 Light-WAM + residual SAC

本节是当前 ManiSkill2 目标的独立 pipeline，不修改上面的 RoboTwin S2 实现。官方 Light-WAM
代码只保留在 `third_party/lightwam/Light-WAM`，本地新增代码只做 observation、normalization、
 action-chunk 和 SAC 的适配。固定合同为：`obs_mode=rgbd`、`base_camera + hand_camera` 横向拼接、
`agent.qpos + agent.qvel + agent.base_pose + extra.tcp_pose` 的 32D non-privileged proprio、
7D `pd_ee_delta_pose`、base action horizon 16、每 4 个 low-level actions 重新规划；residual
actor 只输出当前执行窗口的 4×7 correction，后 12 步 base action 留给下一次 replan。

#### 3.4.1 Light-WAM checkpoint 与 artifact 前置

本 pipeline 不自动下载数据、Wan 权重、Light-WAM checkpoint 或 text embedding。准备 artifact
后只检查路径；如果某个下载预计超过 1 GB，应先单独确认，不能把下载塞进下面的命令。正式
base checkpoint 必须是官方 Light-WAM `save_checkpoint()` 产生的 payload（含 `mot` 或兼容的
`dit`），并且训练配置/`dataset_stats.json` 描述本节的 2-camera、32D、7D contract。S1 的
Policy Decorator `best.pt`（`agent`/`ema_agent`）和 RoboTwin 14D checkpoint 都不能复用。
下面两个入口默认设置 `DIFFSYNTH_SKIP_DOWNLOAD=true`，缺少 Wan 本地文件时会直接失败而不会
自行下载。先把 `DIFFSYNTH_MODEL_BASE_PATH` 指向已经准备好的官方 Wan 文件目录；只有确认过
下载量后，才显式执行 `export DIFFSYNTH_SKIP_DOWNLOAD=false` 再运行相关入口。

```bash
peg_base_weights_dir=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/checkpoints/weights
peg_base_ckpt="$(find "${peg_base_weights_dir}" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1)"
peg_base_config=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/config.yaml
peg_base_stats=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/dataset_stats.json
peg_text_cache=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/text_embeds

test -n "${peg_base_ckpt}" || {
  echo "no official Light-WAM checkpoint found under ${peg_base_weights_dir}" >&2
  exit 1
}
for artifact in "${peg_base_ckpt}" "${peg_base_config}" "${peg_base_stats}"; do
  test -s "${artifact}" || {
    echo "missing Peg Light-WAM artifact: ${artifact}" >&2
    exit 1
  }
done
test -d "${peg_text_cache}"
test -d third_party/lightwam/Light-WAM/src/lightwam
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-outputs/rl_three_stage_20260828/artifacts/lightwam}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
```

#### 3.4.2 环境检查与 dummy smoke（一条独立命令）

该命令不读取真实数据或模型，只验证两路 camera 预处理、7D action composition、SAC update、
checkpoint、frozen base hash 和 `STATUS/DONE`。它不是闭环结果。正式 Peg residual 运行需要
镜像中的 `lightwam-peg` 组合环境；该环境保留 Light-WAM 的 Torch/CUDA 版本，并将 SAPIEN
固定为 2.2.2、NumPy 固定为 ManiSkill2 0.5.3 所需的 1.23.5；为避免与该 NumPy ABI 冲突，
该组合环境不安装 RobotWin-only 的 CuRobo/zarr 依赖，`lightwam` 和
`lightwam-robotwin-eval` 仍保留完整 RobotWin/CuRobo 栈。`lightwam` 仍只用于官方 base 训练，
`maniskill2-pidec` 仍只用于 S1，不要在它们之间混装包：

```bash
peg_smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/lightwam-peg-smoke.XXXXXX")"

bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=evaluate --dummy --arm=base --device=cpu \
  --eval-seeds=1 --episodes-per-seed=1 --max-steps=4 \
  --output-dir="${peg_smoke_root}/base"

bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=train --dummy --device=cpu --total-low-level-actions=8 \
  --learning-starts=1 --utd=1 --batch-size=2 --hidden-dim=32 \
  --log-every=4 --save-every=8 \
  --output-dir="${peg_smoke_root}/residual"

test -s "${peg_smoke_root}/base/metrics.json"
test -s "${peg_smoke_root}/residual/checkpoints/latest.pt"
```

使用重新构建的 RL 镜像时，正式 residual 命令会自动优先选择
`/opt/conda/envs/lightwam-peg/bin/python`；也可以显式写出：

```bash
PYTHON_BIN=/opt/conda/envs/lightwam-peg/bin/python \
  bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=evaluate --dummy --arm=base --device=cpu \
  --eval-seeds=1 --episodes-per-seed=1 --max-steps=4 \
  --output-dir="${peg_smoke_root}/base_explicit_env"
```

#### 3.4.3 Light-WAM base 训练（独立于 residual SAC）

base 训练入口只调用官方 `lightwam.runtime.run_training()` 和官方
`HDF5RobotVideoDataset`；`--data-root` 指向已经准备好的官方 HDF5 目录，不在此步骤下载。
入口会要求该目录直接包含 `episode_*.hdf5`，并默认打开
`FASTWAM_STRICT_DATASET_ERRORS=true`，避免官方 dataset 在坏样本上自动换随机样本掩盖格式错误。
`--dry-run` 只写 resolved config，不构造模型。正式训练完成后，使用最后一个完整的
`checkpoints/weights/step_*.pt`，不要从 validation 曲线挑选中间 checkpoint：

```bash
peg_data_root=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/data
peg_base_root=outputs/rl_three_stage_20260828/artifacts/lightwam_peg
peg_text_cache=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/text_embeds

bash scripts/experiments/rl/train_lightwam_peg.sh \
  --data-root="${peg_data_root}" \
  --output-dir="${peg_base_root}/dry_run" \
  --text-embedding-cache-dir="${peg_text_cache}" \
  --dry-run

bash scripts/experiments/rl/train_lightwam_peg.sh \
  --data-root="${peg_data_root}" \
  --output-dir="${peg_base_root}/train" \
  --text-embedding-cache-dir="${peg_text_cache}" \
  --batch-size=2 --num-workers=8 --num-epochs=10 \
  --save-every=2000 --eval-every=200
```

结果落点：`config.yaml`、`dataset_stats.json`、`run.log` 和
`checkpoints/weights/step_*.pt`。base 训练结果只能标为 `Offline`/`Formal`（取决于是否使用
真实数据和完整预算），不能把 `dry-run` 或 smoke 标为正式结果。

#### 3.4.4 residual SAC 训练

先用 frozen base 做单独闭环评测，再启动 residual 训练。Residual actor/critic 只接收
non-privileged proprio 和 frozen base action chunk；base 参数不进入 SAC optimizer。训练预算按
实际执行的 low-level action 计数，第一版固定单 GPU、单 env，避免把尚未实现的 batch rgbd
适配误当成并行训练：

当前决策的 residual 为 `r ∈ R^(4×7)`，只组合 `a_base[:4]`；`a_base[4:]` 不会被本次
decision 执行，下一次 replan 会重新从 frozen base 获取新的 16-step chunk。

```bash
peg_base_weights_dir=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/train/checkpoints/weights
peg_base_ckpt="$(find "${peg_base_weights_dir}" -maxdepth 1 -type f -name 'step_*.pt' | sort | tail -n 1)"
peg_base_config=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/train/config.yaml
peg_base_stats=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/train/dataset_stats.json
peg_text_cache=outputs/rl_three_stage_20260828/artifacts/lightwam_peg/text_embeds

bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=evaluate --arm=base \
  --base-checkpoint="${peg_base_ckpt}" \
  --base-config="${peg_base_config}" --base-stats="${peg_base_stats}" \
  --text-embedding-cache-dir="${peg_text_cache}" --eval-seeds=1,2,3 \
  --episodes-per-seed=50 --max-steps=200 \
  --output-dir=outputs/rl_three_stage_20260828/stage2_peg/eval_base

bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=train --seed=1 --num-envs=1 \
  --base-checkpoint="${peg_base_ckpt}" \
  --base-config="${peg_base_config}" --base-stats="${peg_base_stats}" \
  --text-embedding-cache-dir="${peg_text_cache}" --total-low-level-actions=4000000 \
  --learning-starts=8000 --replan-steps=4 --res-scale=0.1 \
  --prog-explore=30000 --utd=0.25 \
  --output-dir=outputs/rl_three_stage_20260828/stage2_peg/residual_train/seed_1
```

Residual checkpoint 落点为 `residual_train/seed_1/checkpoints/latest.pt` 和带 low-level action
计数的 checkpoint；`metrics.json` 必须同时记录 base hash 前后相同、`sac_updates`、实际
low-level actions 和 contract。需要多个 training seed 时复制最后一条命令并只改变 `--seed`
和 output directory；这些 seed 是训练 seed，不是 evaluation seed。

#### 3.4.5 base / residual 闭环评测与配对汇总

评测使用相同 `--eval-seeds`、`episodes-per-seed` 和 reset-seed 规则分别跑 base 与 residual；其中
实现的确定性规则为 `reset_seed = eval_seed * episodes_per_seed + episode_index`，从而避免
不同 seed 组在多 episode 评测时碰撞。Light-WAM 的扩散采样也使用同一 episode 的
`reset_seed * 1000000 + decision_step` 作为 inference seed，因此 base 与 residual 两个独立
进程会得到相同的 frozen-base action chunk；该 scheme 会写入每条 manifest；
两边的 `episode_manifest.jsonl` 必须逐行匹配后才能汇总。`--arm=base` 的 composition 是
zero residual，`--arm=residual` 从 SAC checkpoint 读取 actor，最终动作均在 ManiSkill action
space 内 clip。两者的 `episodes.jsonl` 和 `metrics.json` 都标为 `Closed-loop`；dummy 只能
标为 `Smoke`：

```bash
peg_residual_ckpt=outputs/rl_three_stage_20260828/stage2_peg/residual_train/seed_1/checkpoints/latest.pt

bash scripts/experiments/rl/run_lightwam_peg_residual.sh \
  --phase=evaluate --arm=residual \
  --base-checkpoint="${peg_base_ckpt}" \
  --base-config="${peg_base_config}" --base-stats="${peg_base_stats}" \
  --text-embedding-cache-dir="${peg_text_cache}" \
  --residual-checkpoint="${peg_residual_ckpt}" \
  --eval-seeds=1,2,3 --episodes-per-seed=50 --max-steps=200 \
  --output-dir=outputs/rl_three_stage_20260828/stage2_peg/eval_residual_seed_1

bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s2-summarize \
  --base=outputs/rl_three_stage_20260828/stage2_peg/eval_base \
  --treatment=outputs/rl_three_stage_20260828/stage2_peg/eval_residual_seed_1 \
  --output-root=outputs/rl_three_stage_20260828/stage2_peg/summary_seed_1 \
  --formal
```

当前新增入口的代码和测试分别位于：
`src/robot_practice/experiments/rl/lightwam_maniskill.py`（ManiSkill/Light-WAM adapter）、
`src/robot_practice/experiments/rl/maniskill_lightwam_residual.py`（独立 SAC 与闭环）、
`src/robot_practice/experiments/rl/lightwam_train.py`（官方 base-training launcher）、
`tests/experiments/rl/test_lightwam_maniskill.py` 和
`tests/experiments/rl/test_maniskill_lightwam_residual.py`。该延伸链路目前尚未产生正式成功率
结果；正式结果的前提是同 contract 的 Peg Light-WAM checkpoint、stats/context 和真实数据已
分别审计通过。

## 4. S3：RLinf SFT、GRPO、PPO

### 4.1 released parity

先做 S3 source/config/artifact audit，再复评三个 released artifact；audit 失败不得启动容器：

```bash
stage3_root="${repo_root}/outputs/rl_three_stage_20260828/stage3"
for run_root in "${stage3_root}/audit" "${stage3_root}/released_eval" \
  "${stage3_root}/train_smoke" "${stage3_root}/train_formal"; do
  test ! -e "${run_root}" || {
    echo "refusing to reuse existing RLinf output root: ${run_root}" >&2
    exit 1
  }
done
mkdir -p "${stage3_root}/released_eval" "${stage3_root}/train_smoke" "${stage3_root}/train_formal"
bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s3-audit \
  --config=configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml \
  --output-root=outputs/rl_three_stage_20260828/stage3/audit
```

在 8-GPU 镜像中复评三个 released artifact；这一步只验证环境和权重 parity。host 侧先创建 `tee` 的父目录：

```bash
mkdir -p outputs/rl_three_stage_20260828/stage3/released_eval
for arm in sft grpo_released ppo_released; do
  docker run --rm --gpus all --shm-size 32g --network host \
    -v "$(pwd):/workspace/practice" \
    -w /workspace/practice/third_party/rlinf/RLinf \
    -e ROBOTWIN_PATH=/workspace/practice/third_party/rlinf/RoboTwin \
    -e ROBOT_PLATFORM=ALOHA \
    rlinf/rlinf:agentic-rlinf0.4-robotwin \
    bash -lc "source switch_env openvla-oft && \
      bash evaluations/run_eval.sh robotwin robotwin_place_empty_cup_openvlaoft_eval \
      rollout.model.model_path=/workspace/practice/outputs/rl_three_stage_20260828/artifacts/rlinf/${arm} \
      env.eval.assets_path=/workspace/practice/third_party/rlinf/RoboTwin \
      env.eval.total_num_envs=128 env.eval.rollout_epoch=1 \
      env.eval.use_fixed_reset_state_ids=true env.eval.max_episode_steps=200 \
      env.eval.max_steps_per_rollout_epoch=200 \
      runner.logger.log_path=/workspace/practice/outputs/rl_three_stage_20260828/stage3/released_eval/${arm}" \
    2>&1 | tee "outputs/rl_three_stage_20260828/stage3/released_eval/${arm}.run.log"
done
```

released parity 未落在 `75.78/94.53/92.97 ±5pp` 前，停止，不进入 1000 epoch。

### 4.2 训练 smoke 与 formal

GRPO/PPO 各先用 `runner.max_epochs=1` smoke；通过后只把它改回官方 `1000`，不改 algorithm、batch、env、seed 或 component placement。训练命令必须把 `actor.model.model_path` 固定到同一个 SFT artifact，输出写入 `stage3/train_smoke` 或 `stage3/train_formal`，并保留 `run.log`。

```bash
mkdir -p outputs/rl_three_stage_20260828/stage3/train_smoke
for algo in grpo ppo; do
  docker run --rm --gpus all --shm-size 32g --network host \
    -v "$(pwd):/workspace/practice" \
    -w /workspace/practice/third_party/rlinf/RLinf \
    -e ROBOTWIN_PATH=/workspace/practice/third_party/rlinf/RoboTwin \
    -e ROBOT_PLATFORM=ALOHA \
    rlinf/rlinf:agentic-rlinf0.4-robotwin \
    bash -lc "source switch_env openvla-oft && \
      export REPO_PATH=\$(pwd) EMBODIED_PATH=\$(pwd)/examples/embodiment \
      PYTHONPATH=\$(pwd):\${ROBOTWIN_PATH}:\${PYTHONPATH} HYDRA_FULL_ERROR=1 && \
      python examples/embodiment/train_embodied_agent.py \
      --config-path \$(pwd)/examples/embodiment/config \
      --config-name robotwin_place_empty_cup_${algo}_openvlaoft \
      runner.max_epochs=1 \
      runner.logger.log_path=/workspace/practice/outputs/rl_three_stage_20260828/stage3/train_smoke/${algo} \
      actor.model.model_path=/workspace/practice/outputs/rl_three_stage_20260828/artifacts/rlinf/sft \
      env.train.assets_path=/workspace/practice/third_party/rlinf/RoboTwin \
      env.eval.assets_path=/workspace/practice/third_party/rlinf/RoboTwin" \
    2>&1 | tee "outputs/rl_three_stage_20260828/stage3/train_smoke/${algo}.run.log"
done
```

通过 smoke 后，以下是完整的 formal 命令；每个输出目录必须不存在，禁止在旧 run 上续写或用 released RL 权重作为 actor 初始值：

```bash
for algo in grpo ppo; do
  formal_dir="outputs/rl_three_stage_20260828/stage3/train_formal/${algo}"
  test ! -e "${formal_dir}" || {
    echo "refusing to mix with existing RLinf formal run: ${formal_dir}" >&2
    exit 1
  }
  mkdir -p "${formal_dir}"
  docker run --rm --gpus all --shm-size 32g --network host \
    -v "$(pwd):/workspace/practice" \
    -w /workspace/practice/third_party/rlinf/RLinf \
    -e ROBOTWIN_PATH=/workspace/practice/third_party/rlinf/RoboTwin \
    -e ROBOT_PLATFORM=ALOHA \
    rlinf/rlinf:agentic-rlinf0.4-robotwin \
    bash -lc "source switch_env openvla-oft && \
      export REPO_PATH=\$(pwd) EMBODIED_PATH=\$(pwd)/examples/embodiment \
      PYTHONPATH=\$(pwd):\${ROBOTWIN_PATH}:\${PYTHONPATH} HYDRA_FULL_ERROR=1 && \
      python examples/embodiment/train_embodied_agent.py \
      --config-path \$(pwd)/examples/embodiment/config \
      --config-name robotwin_place_empty_cup_${algo}_openvlaoft \
      runner.max_epochs=1000 \
      runner.logger.log_path=/workspace/practice/outputs/rl_three_stage_20260828/stage3/train_formal/${algo} \
      actor.model.model_path=/workspace/practice/outputs/rl_three_stage_20260828/artifacts/rlinf/sft \
      env.train.assets_path=/workspace/practice/third_party/rlinf/RoboTwin \
      env.eval.assets_path=/workspace/practice/third_party/rlinf/RoboTwin" \
    2>&1 | tee "outputs/rl_three_stage_20260828/stage3/train_formal/${algo}.run.log"
done
```

GRPO 保持 group size 8、action-level reward、token-level logprob、actor-only；PPO 保持 group size 1、GAE、actor-critic/value head。正式判断使用预定 validation 点和 1000 epoch final，不延长预算。训练和评测的 episode manifest 必须固定 reset-state id/episode key；不能用不同 rollout 结果直接比较。

### 4.3 S3 汇总

```bash
bash scripts/experiments/rl/run_three_stage_validation.sh \
  --phase=s3-summarize \
  --released-eval=outputs/rl_three_stage_20260828/stage3/released_eval \
  --training-runs=outputs/rl_three_stage_20260828/stage3/train_formal \
  --output-root=outputs/rl_three_stage_20260828/stage3/summary
```

不得用 released RL 权重的成功率冒充本地训练结果；S3 没有相同 held-out action label 时，`open-loop action error` 必须明确写 `N/A` 及原因。

## 5. 验收阈值与停止条件

| Gate | 通过条件 | 不通过处理 |
| --- | --- | --- |
| S1 protocol | commit/version/hash/schema 和 50/50 replay smoke 全匹配 | 停止，先修协议 |
| S1 headroom | pooled base success 在 `[10%, 90%]` | 停止，检查 artifact/protocol |
| S1 residual | 本轮不训练 residual；接线 smoke 通过即可，性能提升门槛为 `N/A` | 不生成 residual treatment 结果；若 smoke 失败先修接线 |
| S2 released parity | clean/randomized 与官方 26%/34% 各自差 `<=10pp` | 停止，修 evaluator/权重/环境 |
| S2 local base | 5 epoch 完成、无 NaN，local-final 相对 released 各设置不低于 `-10pp` | 不进入 residual RL |
| S2 residual | 平均提升 `>=+10pp`、三 seed 同向、CI 下界 `>0`，任一设置不回退超过 2pp | 记录负迁移，继续 S3，不 sweep |
| S2 safety | arm residual saturation `<10%`、jerk P95 增幅 `<=20%`、gripper residual 恒 0、base hash 不变 | 作废或判不安全 |
| S3 parity | SFT/GRPO/PPO 分别为 `75.78/94.53/92.97 ±5pp` | 停止训练复现 |
| S3 training | GRPO 至少一个 validation `>=89.53%`；PPO 至少一个 `>=87.97%`；final 不低于 SFT `-2pp` | 记录负复现，不追加预算 |
| observability | 每 60 秒 `STATUS`，结束 `DONE` | 停止任务，先补进度 |

以下情况立即停止对应 seed：连续 50k low-level actions 无成功、NaN/Inf、residual saturation `>=10%`、base hash 改变、OOM 重启三次或 simulator crash rate `>1%`。

## 6. 结果落点与记录

结果只写入以下目录：

```text
outputs/rl_three_stage_20260828/{preflight,artifacts,stage1,stage2,stage3}/
datasets/raw/lightwam/
datasets/lerobot/lightwam/
```

每个 arm 完成后立即追加 `reports/experiment_log.md`，必须填写：task suite、dataset size、observation keys、action space、action horizon、control frequency、train loss、open-loop action error、closed-loop success rate、latency、failure cases。每个阶段结束后创建日期化报告，区分 released parity、本地训练、Smoke 与 Formal，并只使用仓库相对路径。

每个阶段至少保留以下可审计文件；缺任一文件都不能进入汇总：

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

`progress.jsonl` 必须在运行过程中持续追加 `STATUS`，而不是只在结束时生成；`episodes.jsonl` 和 manifest 只允许保存 episode key、seed/reset id、成功标记、计数和聚合所需指标，不保存原始 observation/action、视频或模型权重。

三阶段全部完成后才可：

1. 把计划状态改为 `done`；
2. 更新 `plans/README.md` 索引；
3. 创建 `reports/YYYY-MM-DD-Residual-RL到Light-WAM与RLinf三阶段验证.md`。

在此之前，计划状态保持 `todo` 或 `in-progress`，不得声称“已复现”。
