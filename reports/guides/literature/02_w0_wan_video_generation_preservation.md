# W0 Wan 视频生成能力保持性快速验证

## 实验目标

回答 `FW-V`（连 Wan2.2 一起 tune）相对 `FW-FZ`（冻结/不训练视频分支）是否保留、改善或破坏视频生成能力。该指南只做配对的离线快速验证，不替代之前的 W0 闭环成功率评测。

## 实验思路

1. 用 pinned VBench-I2V 元数据固定 21 条 prompt、15 张 image，比较 9 个官方 dimension。
2. 用四个现有 LIBERO `_no_noops_lerobot` 训练集各取 10 个 task，构成 40 条 W0 future-video 对照。
3. 两个 checkpoint 共享 manifest、首帧、文本、seed、VAE/text encoder、帧数、分辨率和采样步数。
4. `check-protocol` 先尝试 Wan2.2 TI2V strict-load 与两个 checkpoint 的 81 帧 smoke；失败则两者统一使用 FastWAM-native。
5. W0 只传首帧与原始语言命令，固定 `action=None`、`proprio=None`；不宣称验证 action-conditioned generation。
6. 结果只做直接配对比较：PSNR/SSIM/VBench 正差值代表 FW-V 更好，LPIPS 用 FW-FZ−FW-V。

实验标签：VBench-I2V-mini-21 是 `Offline/Smoke` 小样本筛查；W0-Video-40 是 `Offline` 的训练分布内匹配检查，不是 held-out 或 `Closed-loop` 正式结果。

## 前置条件与一次性准备

所有命令从仓库根目录执行。远端无网：安装、下载和 Git checkout 在本地完成，再同步到远端。

### W0-Video-40 必需项

```bash
bash <<'BASH'
set -u -o pipefail

repo_root="$(pwd -P)"
printf 'repo_root: %s\n' "${repo_root}"

vendor_root="${W0_VENDOR_ROOT:-third_party/w0_fastwam}"
failed=0

check_file() {
  if [[ -f "$1" ]]; then
    printf 'OK file: %s\n' "$1"
  else
    printf 'MISSING file: %s\n' "$1" >&2
    failed=1
  fi
}

check_dir() {
  if [[ -d "$1" ]]; then
    printf 'OK dir: %s\n' "$1"
  else
    printf 'MISSING dir: %s\n' "$1" >&2
    failed=1
  fi
}

check_command() {
  if command -v "$1" >/dev/null 2>&1; then
    printf 'OK command: %s -> %s\n' "$1" "$(command -v "$1")"
  else
    printf 'MISSING command: %s\n' "$1" >&2
    failed=1
  fi
}

check_file outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt
check_file outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt
check_dir "${vendor_root}/FastWAM"
check_command ffmpeg
check_command ffprobe
check_dir pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B
check_dir pretrained/w0_fastwam/models/Wan-AI/Wan2.1-T2V-1.3B
check_dir pretrained/w0_fastwam/models/DiffSynth-Studio/Wan-Series-Converted-Safetensors
check_file pretrained/w0_fastwam/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
if python - "${vendor_root}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1]) / "FastWAM/configs/model/fastwam.yaml"
if not path.is_file():
    raise SystemExit(f"missing config: {path}")

values = []
for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.split("#", 1)[0].strip()
    if line.startswith("action_conditioned:"):
        values.append(line)

if values != ["action_conditioned: false"]:
    raise SystemExit(
        f"expected exactly one action_conditioned: false, got {values}"
    )
print(f"OK: {path}: {values[0]}")
PY
then
  :
else
  failed=1
fi
for suite in spatial object goal 10; do
  dataset=pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2/libero_${suite}_no_noops_lerobot
  check_file "${dataset}/meta/info.json"
  check_file "${dataset}/meta/episodes.jsonl"
  check_file "${dataset}/meta/tasks.jsonl"
done

if (( failed )); then
  printf 'PRECHECK FAILED; current shell was not modified.\n' >&2
else
  printf 'PRECHECK PASSED.\n'
fi
exit "$failed"
BASH
```

缺失项补齐后再继续。

### VBench-I2V 可选项

以下只在执行 VBench-I2V-mini-21 时需要；W0-Video-40 不需要 VBench checkout、crop 数据或 VBench 权重。

固定 VBench 与 Wan2.2 checkout。远端只做下面的只读校验；本地准备后同步完整目录（包括 `.git`）：

```bash
rsync -a --info=progress2 \
  third_party/vbench/ REMOTE_HOST:/path/to/practice/third_party/vbench/
rsync -a --info=progress2 \
  third_party/wan22/ REMOTE_HOST:/path/to/practice/third_party/wan22/
```

本地和远端执行同一段校验：

```bash
(
set -euo pipefail

repo_root="$(pwd -P)"

check_checkout() {
  local name="$1"
  local tree="$2"
  local expected="$3"

  test -d "${tree}/.git"

  local actual
  actual="$(git -c "safe.directory=${tree}" -C "${tree}" rev-parse HEAD)"
  [[ "${actual}" == "${expected}" ]] || {
    printf '%s: expected %s, got %s\n' "${name}" "${expected}" "${actual}" >&2
    exit 2
  }

  local worktree_status
  worktree_status="$(git -c "safe.directory=${tree}" -C "${tree}" status --porcelain)"
  [[ -z "${worktree_status}" ]] || {
    printf '%s checkout is dirty:\n%s\n' "${name}" "${worktree_status}" >&2
    exit 2
  }

  printf 'OK %s: %s, clean\n' "${name}" "${actual}"
}

check_checkout \
  VBench \
  "${repo_root}/third_party/vbench" \
  fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490

check_checkout \
  Wan2.2 \
  "${repo_root}/third_party/wan22" \
  81e67351b81197330c9533e7f345233e7702b44a

printf 'Git checkout verification: PASSED\n'
)
```

本地下载 crop image suite，完成后同步 `pretrained/vbench_i2v/crop/` 到远端：

```bash
(
set -euo pipefail
mkdir -p pretrained/vbench_i2v
conda run --no-capture-output -n robot-practice \
  python -c 'import gdown; print("gdown: OK")'
conda run --no-capture-output -n robot-practice \
  gdown --id 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ \
  --output pretrained/vbench_i2v/crop.zip
unzip -q pretrained/vbench_i2v/crop.zip -d pretrained/vbench_i2v
test -d pretrained/vbench_i2v/crop/16-9
)
```

仅跑 VBench 时，换镜像或环境后确认评分依赖：

```bash
export W0_VIDEO_SCORE_ENV="${W0_VIDEO_SCORE_ENV:-robot-practice}"
conda run --no-capture-output -n "${W0_VIDEO_SCORE_ENV}" \
  python -c 'import clip, cv2, decord, dreamsim, imageio.v3, lpips, pyiqa; print("VBench/W0/LPIPS evaluator dependencies: OK")'
```

镜像缓存检查：

```bash
( set -euo pipefail; hub="${TORCH_HOME:-${HOME}/.cache/torch}/hub"; test -f "${hub}/checkpoints/alexnet-owt-7be5be79.pth"; test -f "${hub}/checkpoints/cotracker2.pth"; test -f "${hub}/facebookresearch_co-tracker_main/hubconf.py" )
```

VBench 的 dreamsim 缓存放在仓库根目录 `models/`；需要运行 VBench score 时检查：

```bash
( set -euo pipefail; test -f models/ensemble_lora/adapter_model.safetensors; test -f models/dino_vitb16_pretrain.pth; test -f models/clip_vitb16_pretrain.pth.tar; test -f models/open_clip_vitb16_pretrain.pth.tar; test -f models/checkpoints/dino_vitbase16_pretrain.pth; test -d models/facebookresearch_dino_main )
```

正式 VBench score 还需要以下本地权重；不要用 `--allow-vbench-downloads`：

```bash
(
set -euo pipefail
cache=pretrained/vbench_i2v
test -f "${cache}/dino_model/dino_vitbase16_pretrain.pth"
test -d "${cache}/dino_model/facebookresearch_dino_main"
test -f "${cache}/clip_model/ViT-B-32.pt"
test -f "${cache}/clip_model/ViT-L-14.pt"
test -f "${cache}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
test -f "${cache}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
test -f "${cache}/amt_model/amt-s.pth"
test -f "${cache}/raft_model/models/raft-things.pth"
)
```

## 执行清单（先跑 W0-Video-40）

### 当前 checkpoint 不可加载时的 pipeline smoke

如果两个正式 checkpoint 尚未同步，不能用随机权重冒充 VBench 结果；可以使用仓库内明确标记的
`--dummy-model` 只验证「manifest → 条件图 → 视频写出」链路。它生成确定性的输入图像平移视频，
不加载 FastWAM/Wan2.2，不代表模型质量，也不能生成正式指标。Docker 中运行时仍需挂载仓库并传
`--gpus all`；`prepare-vbench` 的 Git pinned-checkout 校验已兼容 bind mount 的 owner 差异。

```bash
docker run --rm --gpus all \
  -v "${PWD}:/repo" -w /repo \
  test-lab-instance-cn-beijing.cr.volces.com/lidar-wwd/practice:w0-fastwam-20260825-videoeval \
  bash -lc '
    export PYTHONPATH=/repo/src
    python -m robot_practice.experiments.literature.w0_video_generation_eval \
      --mode=prepare-vbench \
      --vbench-root=third_party/vbench \
      --cache-root=pretrained/vbench_i2v \
      --output=outputs/literature/w0_fastwam/video_generation/vbench/assets.json
    python -m robot_practice.experiments.literature.w0_video_generation_eval \
      --mode=prepare-vbench \
      --selection=mini-21 \
      --protocol=configs/experiments/literature/w0_video_generation_dummy_protocol.json \
      --output=outputs/literature/w0_fastwam/video_generation/vbench/dummy_manifest.json
    python -m robot_practice.experiments.literature.w0_video_generation_eval \
      --mode=generate --dataset=vbench-mini-21 --variant=fw_v \
      --manifest=outputs/literature/w0_fastwam/video_generation/vbench/dummy_manifest.json \
      --output-root=outputs/literature/w0_fastwam/video_generation/vbench/dummy_fw_v \
      --device=cuda:0 --dummy-model --overwrite
  '
```

通过条件是输出 21 个 MP4，并且每个视频满足 dummy protocol 的 320×180、16 帧约束。这个 smoke
通过后仍需恢复真实 checkpoint，并补齐 VBench evaluator 权重，才能执行正式 `score`。

远端无外网，所以顺序固定为「评测环境与 LPIPS 权重验证 → smoke → 正式」。VBench-I2V 的完整评测
另外需要六份 VBench 权重与 dreamsim 的 `models/`，本清单不含。所有命令在仓库根目录执行。

**1. 验证评测环境与 LPIPS 权重。** 下面是一条可直接执行的 Bash 命令：它检查评分环境中的依赖，
从已安装的 `lpips` 包复制标定权重，并确认输出文件非空；不联网、不安装任何包：

```bash
(
set -euo pipefail

score_env="${W0_VIDEO_SCORE_ENV:-robot-practice}"
conda run --no-capture-output -n "${score_env}" \
  python -c 'import cv2, imageio.v3, lpips; print("W0-Video-40 evaluator dependencies: OK")'

mkdir -p pretrained/vbench_i2v/lpips
conda run --no-capture-output -n "${score_env}" \
  python - <<'PY'
from pathlib import Path
import shutil

import lpips

src = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
dst = Path("pretrained/vbench_i2v/lpips/alex.pth")
if not src.is_file():
    raise SystemExit(f"missing LPIPS weight: {src}")
shutil.copyfile(src, dst)
print(f"copied LPIPS weight: {src} -> {dst}")
PY
test -s pretrained/vbench_i2v/lpips/alex.pth
printf 'LPIPS evaluator weight: %s bytes\n' "$(wc -c < pretrained/vbench_i2v/lpips/alex.pth)"
)
```

**2. 单 suite smoke。** 下面这个 gate 仍按 `prepare-w0 → generate → score` 三步执行，但会自动检查
manifest、视频数量/尺寸/帧数和 LPIPS 指标。原始日志写入 `smoke_spatial/logs/`，终端只显示进度和
`PASS/FAIL`，不需要手动检查字段或数值：

```bash
(
set -euo pipefail

smoke_root=outputs/literature/w0_fastwam/video_generation/smoke_spatial
log_root="${smoke_root}/logs"
manifest="${smoke_root}/manifest.json"
video_root="${smoke_root}/fw_v/videos"
metrics="${smoke_root}/fw_v/metrics.json"
video_device="${W0_VIDEO_DEVICE:-cuda:0}"
mkdir -p "${log_root}"

run_step() {
  local label="$1"
  local log="$2"
  shift 2
  printf 'RUN %s\n' "${label}"
  set +e
  "$@" 2>&1 | tee "${log}" | grep --line-buffered -E '^(STATUS|DONE)'
  local step_status="${PIPESTATUS[0]}"
  set -e
  if [[ "${step_status}" == 0 ]]; then
    printf 'PASS %s\n' "${label}"
  else
    printf 'FAIL %s (see %s)\n' "${label}" "${log}" >&2
    return "${step_status}"
  fi
}

check_step() {
  local label="$1"
  local log="$2"
  shift 2
  if "$@" >"${log}" 2>&1; then
    printf 'PASS %s\n' "${label}"
  else
    printf 'FAIL %s (see %s)\n' "${label}" "${log}" >&2
    return 1
  fi
}

check_manifest() {
  conda run --no-capture-output -n robot-practice \
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["dataset"]=="W0-Video-40" and len(d["samples"])==10' \
    "${manifest}"
}

check_videos() {
  local count
  count="$(find "${video_root}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  [[ "${count}" -eq 10 ]]
  for video in "${video_root}"/*.mp4; do
    local dimensions
    dimensions="$(ffprobe -v error -select_streams v:0 -count_frames \
      -show_entries stream=width,height,nb_read_frames \
      -of csv=p=0:s=, "${video}")"
    [[ "${dimensions}" == "448,224,9" ]]
  done
}

check_metrics() {
  conda run --no-capture-output -n robot-practice \
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["metric_availability"]["lpips"] is True and d["summary"]["macro"]["count"]==10' \
    "${metrics}"
}

run_step prepare-w0 "${log_root}/prepare-w0.log" \
  bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=prepare-w0 \
  --dataset-root=pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2 \
  --suites=libero_spatial \
  --selection=one-per-task-motion-p75 \
  --raw-frames=33 \
  --video-stride=4 \
  --base-seed=42000 \
  --output="${manifest}"
check_step prepare-w0-artifacts "${log_root}/prepare-w0-check.log" check_manifest

run_step generate-fw-v "${log_root}/generate-fw-v.log" \
  bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=generate \
  --dataset=w0-video-40 \
  --variant=fw_v \
  --model-cache-root=pretrained/w0_fastwam \
  --checkpoint=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt \
  --manifest="${manifest}" \
  --device="${video_device}" \
  --output-root="${smoke_root}/fw_v"
check_step generate-fw-v-artifacts "${log_root}/generate-fw-v-check.log" check_videos

run_step score-fw-v "${log_root}/score-fw-v.log" \
  bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=score \
  --dataset=w0-video-40 \
  --variant=fw_v \
  --manifest="${manifest}" \
  --videos="${video_root}" \
  --lpips-weights=pretrained/vbench_i2v/lpips/alex.pth \
  --device="${video_device}" \
  --output="${metrics}"
check_step score-fw-v-artifacts "${log_root}/score-fw-v-check.log" check_metrics

printf 'PASS W0-Video-40 FW-V smoke\n'
)
```

**3. 删除 smoke 产物**，避免与正式结果混淆：

```bash
rm -rf outputs/literature/w0_fastwam/video_generation/smoke_spatial
```

**4. 正式跑四 suite / 40 条。** 执行本文后面“实验命令”中的
“3. W0-Video-40 manifest、生成与指标”小节：先固定 40 条 manifest，再分别生成并 score `FW-FZ`
和 `FW-V`，最后执行该小节末尾的“配对差值”命令。

## 实验命令

执行顺序：“3. W0-Video-40 manifest、生成与指标”**可以独立先跑**，它只需要 LPIPS 权重；前面的
“1. 固定协议与 VBench-I2V-mini-21 manifest”及“2. VBench-I2V-mini-21 生成与打分”另外需要 DINO、
CLIP ViT-B-32/ViT-L-14、Aesthetic、MUSIQ、AMT、RAFT 六份本地权重与 dreamsim 的 `models/`；最后的
“4. 配对汇总与固定可视化”需要四份 metrics 齐全才能跑。

`--vendor-root` 默认 `third_party/w0_fastwam`，而该路径是 gitignored 的、不会被 rsync 同步。
训练镜像自带的 vendor 树在 `/opt/robot-practice/w0-vendor`，远端缺这棵树时给 `generate` 加
`--vendor-root=/opt/robot-practice/w0-vendor`。上面那段预检会把缺失路径全部列出。

先做本地实现检查：

```bash
conda run --no-capture-output -n robot-practice env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python -m pytest tests/test_w0_video_generation_eval.py -q
conda run --no-capture-output -n robot-practice \
  python -m ruff check src/robot_practice/experiments/literature/w0_video_generation_eval.py \
  tests/test_w0_video_generation_eval.py
bash -n scripts/experiments/literature/run_w0_video_generation_eval.sh
```

### 1. 固定协议与 VBench-I2V-mini-21 manifest

```bash
(
set -euo pipefail

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=prepare-vbench \
  --vbench-root=third_party/vbench \
  --cache-root=pretrained/vbench_i2v \
  --output=outputs/literature/w0_fastwam/video_generation/vbench/assets.json

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=check-protocol \
  --wan-root=third_party/wan22 \
  --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B \
  --fw-v=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt \
  --fw-fz=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt \
  --output=outputs/literature/w0_fastwam/video_generation/vbench/protocol.json

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=prepare-vbench \
  --vbench-root=third_party/vbench \
  --cache-root=pretrained/vbench_i2v \
  --selection=mini-21 \
  --selection-seed=20260824 \
  --base-seed=52000 \
  --protocol=outputs/literature/w0_fastwam/video_generation/vbench/protocol.json \
  --output=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json
)
```

通过条件：manifest 为 21 条，subject/background/camera 各 7 条，15 张 unique image，7 个 camera prompt 各 1 条；两个 checkpoint 的协议 JSON 完全相同。`protocol.json` 的 `kind` 若为 `fastwam-native` 是预定回退，不是失败。

### 2. VBench-I2V-mini-21 生成与打分

```bash
(
set -euo pipefail

export W0_VIDEO_MODEL_ENV="${W0_VIDEO_MODEL_ENV:-fastwam-upstream}"
export W0_VIDEO_SCORE_ENV="${W0_VIDEO_SCORE_ENV:-robot-practice}"

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=generate --dataset=vbench-mini-21 --variant=fw_fz \
  --model-cache-root=pretrained/w0_fastwam \
  --vbench-root=third_party/vbench --wan-root=third_party/wan22 \
  --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B \
  --checkpoint=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt \
  --manifest=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json \
  --output-root=outputs/literature/w0_fastwam/video_generation/vbench/fw_fz

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=generate --dataset=vbench-mini-21 --variant=fw_v \
  --model-cache-root=pretrained/w0_fastwam \
  --vbench-root=third_party/vbench --wan-root=third_party/wan22 \
  --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B \
  --checkpoint=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt \
  --manifest=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json \
  --output-root=outputs/literature/w0_fastwam/video_generation/vbench/fw_v

# Docker 评分入口会自动把仓库挂载到 `/opt/robot-practice`，并将仓库根目录的
# `models/`（DreamSim ensemble）以只读方式挂载到容器内的同一路径。镜像默认自动选择
# 本地的 `*w0-fastwam*videoeval` 镜像；如有多个镜像，可设置 `W0_VIDEO_EVAL_IMAGE`。
for variant in fw_fz fw_v; do
  scripts/experiments/literature/run_w0_video_generation_eval_docker.sh \
    --mode=score --dataset=vbench-mini-21 --variant="${variant}" \
    --vbench-root=third_party/vbench \
    --cache-root=pretrained/vbench_i2v \
    --manifest=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json \
    --videos="outputs/literature/w0_fastwam/video_generation/vbench/${variant}/videos" \
    --output="outputs/literature/w0_fastwam/video_generation/vbench/${variant}/metrics.json"
done
)
```

固定报告 9 个 dimension：`i2v_subject`、`subject_consistency`、`motion_smoothness`、`dynamic_degree`、`i2v_background`、`background_consistency`、`aesthetic_quality`、`imaging_quality`、`camera_motion`。21 条子集不计算伪造的完整 VBench Total Score。

### 3. W0-Video-40 manifest、生成与指标

这一节**不依赖前面的两个 VBench-I2V 小节**，也不需要 DINO、CLIP、Aesthetic、MUSIQ、AMT、RAFT 与 dreamsim：
`--dataset=w0-video-40` 的 score 走 `score_w0`，只用 PSNR/SSIM/LPIPS，LPIPS 的两个权重都已
就位（AlexNet 主干在镜像的 `$TORCH_HOME/hub`，`alex.pth` 从 `lpips` 包复制）。所以在只准备了
LPIPS 的情况下可以先单独跑完这一节。

资源约定：`prepare-w0` 只做 CPU 视频解码和 manifest 选择，不使用 GPU。`generate` 和 `score` 现在
支持 `--num-gpus=N`：父进程按固定 manifest 的 sample 顺序做 round-robin 分片，每个 worker 独占一张
可见 GPU，生成完成后自动合并视频，score 完成后自动按原 manifest 顺序合并 metrics。正式比较时
FW-FZ 与 FW-V 必须使用相同的固定 manifest、sample ID 和 seed。

多卡时不要传 `--device=cuda:1` 之类的单卡参数；通过 `CUDA_VISIBLE_DEVICES` 指定物理卡，worker 内部
统一使用 `cuda:0`。下面默认使用 4 张可见 GPU；如果调度器已经设置了 `CUDA_VISIBLE_DEVICES`，会保留
调度器的设置，但必须确保其中至少有 4 张卡。多卡中间产物位于对应 variant 目录下的隐藏
`.multigpu/`，最终视频和 metrics 仍写入下面的正式路径。

注意：W0-Video-40 这条链路读取已有 LIBERO 视频、调用 FastWAM 做视频生成并计算 PSNR/SSIM/LPIPS，
不启动 MuJoCo/EGL，也不读取 `W0_EGL_DEVICE_MAP`、`MUJOCO_EGL_DEVICE_ID` 或
`MUJOCO_GL`。因此这里的 `CUDA_VISIBLE_DEVICES` 只负责模型/LPIPS 的 CUDA 绑定；`W0_EGL_DEVICE_MAP`
只适用于另一个会实际创建 MuJoCo 渲染上下文的闭环评测入口。

```bash
export W0_VIDEO_NUM_GPUS="${W0_VIDEO_NUM_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
```

下面按参数分行并使用反斜杠续行，直接整段粘贴即可；`generate` 用 `W0_VIDEO_MODEL_ENV`（默认
`fastwam-upstream`），`score` 用 `W0_VIDEO_SCORE_ENV`（默认 `robot-practice`），由 wrapper 自行分派。

**若终端在粘贴时吞掉续行反斜杠**，首行会带着不完整参数执行并报
`generate is missing: manifest, output_root, checkpoint`，其余行则作为独立命令报
`--single-camera=front: command not found` 与 `--checkpoint=...: No such file or directory`。
遇到这种情况改用下面两个短入口，长命令都在脚本里：

| 命令 | 做什么 |
| --- | --- |
| `bash scripts/experiments/literature/run_w0_video_40.sh` | **正式协议**：40 条、双相机 448×224；两个变体依次 generate + score，最后打印配对读数。可选参数 `[both\|fw_v\|fw_fz] [num_gpus]`；`overwrite=1` 强制重生成，默认跳过已存在的视频 |
| `bash scripts/experiments/literature/probe_w0_single_camera.sh` | **诊断，不是协议**：3 条、单相机 224×224；两个变体加对比图，用于定位 `FW-FZ` 崩掉的主因。可选参数 `[front\|wrist]` |

两者不会互相污染：诊断脚本写在 `probe3/`，其 `protocol.kind` 带 `-single-camera-diagnostic`
后缀，且 224×224 与协议的 448×224 形状不符，正式 `score` 会直接报错而不是静默混算。

固定 manifest：

```bash
bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=prepare-w0 \
  --dataset-root=pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2 \
  --suites=libero_spatial,libero_object,libero_goal,libero_10 \
  --selection=one-per-task-motion-p75 \
  --raw-frames=33 \
  --video-stride=4 \
  --base-seed=42000 \
  --output=outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json
```

`one-per-task-motion-p75` 会把每个 episode 的两路视频**整段解码**来算 motion score，单进程实测约
449 frames/s，一个 suite（`libero_spatial` 434 episodes）约 7 分钟，四个 suite 合计约半小时。它没有
多进程路径，建议 `nohup` 起，用 `run.log` 的 `STATUS stage=prepare-w0:<suite> completed=/total=` 跟进度。

生成 `FW-FZ`：

```bash
bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=generate \
  --dataset=w0-video-40 \
  --variant=fw_fz \
  --model-cache-root=pretrained/w0_fastwam \
  --checkpoint=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt \
  --manifest=outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json \
  --num-gpus="${W0_VIDEO_NUM_GPUS}" \
  --output-root=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_fz
```

生成 `FW-V`：

```bash
bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=generate \
  --dataset=w0-video-40 \
  --variant=fw_v \
  --model-cache-root=pretrained/w0_fastwam \
  --checkpoint=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt \
  --manifest=outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json \
  --num-gpus="${W0_VIDEO_NUM_GPUS}" \
  --output-root=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_v
```

打分 `FW-FZ`：

```bash
bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=score \
  --dataset=w0-video-40 \
  --variant=fw_fz \
  --manifest=outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json \
  --videos=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_fz/videos \
  --lpips-weights=pretrained/vbench_i2v/lpips/alex.pth \
  --num-gpus="${W0_VIDEO_NUM_GPUS}" \
  --output=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_fz/metrics.json
```

打分 `FW-V`：

```bash
bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=score \
  --dataset=w0-video-40 \
  --variant=fw_v \
  --manifest=outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json \
  --videos=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_v/videos \
  --lpips-weights=pretrained/vbench_i2v/lpips/alex.pth \
  --num-gpus="${W0_VIDEO_NUM_GPUS}" \
  --output=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_v/metrics.json
```

W0 只计算 future frame `[1..8]`，先拆分 `224x448` 的 front/wrist，再对 frame 与 camera 等权平均输出 PSNR、SSIM、LPIPS；同时写 per-task、per-suite 和 4-suite macro。缺少 LPIPS 时不要加 `--allow-missing-lpips` 做正式比较；该 flag 仅用于标记为不完整的 PSNR/SSIM smoke。

`--mode=compare`（“4. 配对汇总与固定可视化”）要求四份 metrics 齐全，只跑本节时会以
`compare is missing: vbench-fw-fz, vbench-fw-v` 退出。在补齐“2. VBench-I2V-mini-21 生成与打分”之前，
可以用下面这条命令读取 W0 配对差值；它只是读数，不做“4. 配对汇总与固定可视化”中的
manifest/protocol SHA 校验，因此不能替代 `comparison.json`：

```bash
W0_DIR=outputs/literature/w0_fastwam/video_generation/w0_video_40 conda run --no-capture-output -n "${W0_VIDEO_SCORE_ENV:-robot-practice}" python -c 'import json, os; d = os.environ["W0_DIR"]; L = [json.load(open(f"{d}/{v}/metrics.json")) for v in ("fw_fz", "fw_v")]; m = [x["summary"]["macro"] for x in L]; [print("%-6s n=%d  PSNR %7.3f  SSIM %.4f  LPIPS %s" % (x["variant"], y["count"], y["psnr"], y["ssim"], "unavailable" if y["lpips"] is None else "%.4f" % y["lpips"])) for x, y in zip(L, m)]; print("FW-V minus FW-FZ:  dPSNR %+.3f  dSSIM %+.4f  dLPIPS(FZ-V) %s" % (m[1]["psnr"] - m[0]["psnr"], m[1]["ssim"] - m[0]["ssim"], "unavailable" if None in (m[0]["lpips"], m[1]["lpips"]) else "%+.4f" % (m[0]["lpips"] - m[1]["lpips"]))); [print("  %-15s PSNR %7.3f -> %7.3f   SSIM %.4f -> %.4f" % (s, L[0]["summary"]["per_suite"][s]["psnr"], L[1]["summary"]["per_suite"][s]["psnr"], L[0]["summary"]["per_suite"][s]["ssim"], L[1]["summary"]["per_suite"][s]["ssim"])) for s in sorted(L[0]["summary"]["per_suite"])]'
```

正差值代表 `FW-V` 更好；LPIPS 报的是 `FW-FZ − FW-V`，方向与 PSNR/SSIM 一致。

#### 已定案：`FW-FZ` 的未来帧是噪声，且这不是管路问题

首轮 `FW-FZ` 生成的 frame 1..8 是饱和彩条，PSNR 只有 7.4–15.6 dB，且**每一帧都与 `gt[0]` 最相似**
而不是与自己那一帧最相似。三条候选原因已逐条排除，不要重复排查：

| 候选 | 排除依据 |
| --- | --- |
| checkpoint 没完整加载 | vendor 的 `load_checkpoint` 用 `strict=False`，本仓库已加 `_assert_checkpoint_actually_loaded` 在键集不匹配时抛错；生成能跑完即代表键集完全匹配 |
| 采样步数太少 | 同一 `FW-V` checkpoint 在 10 / 20 / 50 步下 frame 1 分别为 25.81 / 25.48 / 25.29 dB，差 <0.6 dB 且目视一致；步数更多反而略差，上游的 `eval_num_inference_steps: 10` 够用 |
| 生成管路或归一化错 | `FW-V` 在同一管路、同一 10 步下 frame 1..8 为 21.3–25.8 dB，逐帧时间对齐正确 |

其余可能性也已在代码层排除：输入归一化与上游 `runtime.py` 逐字相同；`sigma_shift: null` 两侧同取
`shift=5.0`；`text_cfg_scale` 在 `infer_joint` 函数体内从未被引用，该路径没有实现 CFG；
`save_checkpoint` 存整个 `mot`，12.04 GB 与 6.02 B bf16 相符；trainer 存盘前已
`accelerator.unwrap_model`，无 `module.` 前缀。frame 0 往返 34–36 dB 说明 VAE 健康。

所以 `FW-FZ` 的未来帧确实预测不出来。机制上也不意外，但**原因不是 action expert 的干扰**：
`_build_mot_attention_mask`（`fastwam.py:386-407`）把 `mask` 初始化为全 `False`，只填三块——
video→video、action→action、action→首帧 video。`mask[:video_seq_len, video_seq_len:]`（video 查询
action 键）**从未被赋值，恒为 `False`**，所以 action expert 对视频生成的影响是零。

真正的原因是 `FW-FZ` 的 video expert 是**未被训练过的 Wan2.2 权重**（`lambda_video=0` 且被冻结补丁
锁住），却被放在一个与其预训练设置差很远的工作点上。至少四个变量同时偏离，本地无法从现有产物中
分离：

| 偏离项 | FastWAM / W0 设置 | Wan2.2 TI2V 预训练设置 |
| --- | --- | --- |
| 图像布局 | front 与 wrist 各 `224x224` 横向拼接 | 单一自然画面，没有并排接缝 |
| 分辨率 | `448x224` → 每帧 `14x28 = 392` 个 latent token | `check-protocol` 的官方 smoke 用 `1280x720` |
| 帧数 | 9（latent T=3） | 81 |
| video 自注意力 mask | `first_frame_causal`（`configs/model/fastwam.yaml:30`） | DiT 类默认 `bidirectional` |

`first_frame_causal` 只限制首帧 token 的查询行，未来帧的行仍是全 `True`；但首帧表征会逐层参与未来
帧的注意力，所以它不是 no-op。

**因此 W0-Video-40 的 `FW-V − FW-FZ` 差值接近同义反复**，它复述的是训练设置（一组训练了 video
expert、另一组没有），而不是回答「video 目标是教会动力学，还是只是保护预训练表征」。报告时
`FW-V` 的 W0-Video-40 绝对值可以作为「FW-V 的视频分支确实学到了域内动力学」的证据，但不要把这个
差值写成机制结论。真正能测「保持性」的是 §2 的 VBench-I2V-mini-21：那里 `FW-FZ` 等于未受损的
Wan2.2，是保持度的上界，`FW-V` 相对它的差值才衡量 co-training 破坏还是保住了通用生成能力，且方向
无法事先预测。

逐帧核查用：

```bash
python scripts/experiments/literature/compare_w0_video_frames.py --manifest outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json --videos fw_fz=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_fz/videos --videos fw_v=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_v/videos --output outputs/literature/w0_fastwam/video_generation/w0_video_40/frame_compare.png --samples libero_spatial_task00,libero_object_task00
```

#### 指标口径与读法

三个指标都算在 future frame `[1..8]` 上（frame 0 是给定的条件帧，排除），先把 `448x224` 拆成
front/wrist 两个 `224x224`，再对 8 帧与 2 相机等权平均。

| 指标 | 方向 | 实现 | 它衡量什么，盲点在哪 |
| --- | --- | --- | --- |
| PSNR | **越高越好**，dB | `10*log10(255**2 / MSE)` | 逐像素均方误差换对数刻度，每 +6 dB 约等于误差减半。**奖励模糊**：在多解的未来预测里，输出「所有可能未来的平均」比清晰但位置略偏的猜测得分更高，所以它一部分在衡量模型有多保守 |
| SSIM | **越高越好**，`[-1, 1]` | `_ssim_global`：整幅图逐通道的均值/方差/协方差，标准 `C1=0.01**2`、`C2=0.03**2` | **是 global 而非 11x11 高斯窗局部 SSIM**，比窗口版粗得多，只看整幅图的亮度/对比度/协方差是否吻合，因此**不能直接与论文里的 SSIM 数字比较**。协方差为负时结果为负，代表输出与目标反相关，即无关而不是更差 |
| LPIPS | **越低越好** | lpips 0.1.4 AlexNet + v0.1 线性校准权重（`alex.pth`，6009 字节） | 深层特征距离，用人类两选一判断训练而来，与主观感受相关性最好，不像 PSNR 那样偏爱模糊 |

LPIPS 方向相反，所以本 guide 统一把它的差值报成 `FW-FZ − FW-V`，使三个指标**正值都代表
`FW-V` 更好**。

**三者共同的盲点**：它们只回答「像不像这一条真值未来」，不回答「是不是一个物理上合理的未来」。
一个合理但不同的未来会被三个指标同时惩罚。真值只有一条而未来本来多解，这是评价世界模型时的真实
局限，也是本节指标只作后验描述、不作 W0 主结论的原因。

读数时用本管路自己的天花板做锚点，比引用通用经验值可靠：frame 0 是条件帧走一遍
VAE 编码→解码→mp4 的往返，实测 **33.8–35.7 dB**，这就是本管路能达到的上限。

#### 实测结果（40 条，后验描述）

两个变体共用同一 manifest（sha256 `a6676f659cc7bc12…`）与同一 protocol，各 40/40、每 suite 10 条，
`metric_availability.lpips=true`。

| 变体 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| `FW-FZ` | 9.963 | 0.2040 | 0.6822 |
| `FW-V` | **24.705** | **0.9231** | **0.0983** |
| `FW-V − FW-FZ` | **+14.742** | **+0.7190** | **+0.5839**（FZ−V） |

per-suite PSNR（`FW-FZ` → `FW-V`）：`libero_spatial` 10.289→25.874、`libero_goal` 9.734→25.456、
`libero_10` 10.172→24.982、`libero_object` 9.658→22.509。

分布比 macro 更能说明性质：`FW-FZ` 的 PSNR 落在 `6.851–15.483`（sd 1.802），40 条里只有 1 条
>15 dB；SSIM 落在 `−0.042–0.593`（sd 0.152），29/40 低于 0.3、**4 条为负**；四个 suite 的 macro
PSNR 极差仅 0.63 dB，说明失败与任务内容无关，是整体性的。`FW-V` 的 PSNR 落在
`17.459–28.236`（sd 2.699），35/40 条 SSIM >0.85，最差三条集中在 `libero_object`。

**可支持的结论**：`FW-V` 的视频分支确实学到了域内动力学，没有塌缩——24.705 dB 距本管路上限约
10 dB，且分布集中。

**不可支持的结论**：不能读成「video co-training 提升了视频生成能力」。`FW-FZ` 的 video expert 是
未训练过的 Wan2.2 权重，且运行在上表四个同时偏离预训练的设置上，所以该差值把「学过这个任务」与
「跑在有效工作点上」混在一起，按构造接近同义反复。要回答通用生成能力的保持性，只能靠 §2 的
VBench-I2V-mini-21。

另需注意样本窗口取自训练 demonstration，属分布内拟合，不是 held-out 泛化。

一个过程教训：先前用 3 条做的诊断给出 `FW-V` macro 21.980 dB，比全量的 24.705 低 2.7 dB，因为那
3 条里含全量最差的 `libero_object_task00`（17.46）。**小样本诊断可以定方向，不能报数值。**

上表对应的逐帧对比图（每条带三行：GT / `fw_fz` / `fw_v`，横向 frame 0→8，frame 0 是给定的条件帧
所以三行相同）：

![W0-Video-40 双相机拼接逐帧对比](../../../docs/assets/w0_video_generation/w0_video_40_frame_compare.jpg)

该图由上面那条 `compare_w0_video_frames.py` 命令生成。诊断用的全分辨率 PNG（3 个样本）写在
`outputs/` 下，属 gitignored 产物；README 与本 guide 嵌入的是**已提交**的 JPEG 版本，放在
`docs/assets/w0_video_generation/`，这样在任何地方渲染文档都能看到图。两张图各约 300 KB，是本仓库
唯一超过几 KB 的跟踪图片，例外理由是它们承载结论本身——只有指标表而没有图，读者无法判断
`FW-FZ` 的失败是「更模糊」还是「无关」。重建命令（`--output` 用 `.jpg` 即走 JPEG 编码，
`--quality` 默认 85）：

```bash
python scripts/experiments/literature/compare_w0_video_frames.py --manifest outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json --videos fw_fz=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_fz/videos --videos fw_v=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_v/videos --samples libero_spatial_task00,libero_10_task00 --limit 2 --quality 85 --output docs/assets/w0_video_generation/w0_video_40_frame_compare.jpg
```

#### 单相机对照：双相机拼接布局**不是** `FW-FZ` 崩溃的原因

`FW-FZ` 的 video expert 是未训练过的 Wan2.2 权重，而自然视频里不存在并排双画面，所以「布局 OOD」
是最直觉的解释。用 `probe_w0_single_camera.sh` 把条件输入换成单个 `224x224` front 视角后，这个解释
被否掉了。为可比，拼接组这里只取每帧左半（front），与单相机同口径，同样是 frames `[1..8]`、
同 3 条样本：

| 变体 | 拼接 `448x224`（仅 front 半幅） | 单相机 `224x224` | 变化 |
| --- | ---: | ---: | ---: |
| `FW-FZ` PSNR / SSIM | 10.338 / 0.1462 | 9.975 / 0.1859 | **−0.363 dB / +0.0397** |
| `FW-V` PSNR / SSIM | 26.098 / 0.9480 | 22.011 / 0.9112 | **−4.087 dB / −0.0367** |

![单相机逐帧对比](../../../docs/assets/w0_video_generation/single_camera_front_compare.jpg)

该图的重建命令（`--camera=front` 必须与生成时一致，否则 GT 布局不匹配）：

```bash
python scripts/experiments/literature/compare_w0_video_frames.py --manifest outputs/literature/w0_fastwam/video_generation/probe3/manifest.json --camera=front --videos fz_front=outputs/literature/w0_fastwam/video_generation/probe3/fw_fz_front/videos --videos v_front=outputs/literature/w0_fastwam/video_generation/probe3/fw_v_front/videos --samples libero_spatial_task00,libero_10_task00 --limit 2 --quality 85 --output docs/assets/w0_video_generation/single_camera_front_compare.jpg
```

两条读法：

- **去掉接缝对 `FW-FZ` 没有帮助**：仍是约 10 dB、SSIM 约 0.19，图上仍是饱和彩带。所以拼接布局
  可以从 `FW-FZ` 崩溃的候选原因中排除。
- **同一个布局变化让 `FW-V` 掉了 4.1 dB，但它依然完好**（22.011 dB / SSIM 0.911）。这是本对照
  最有力的部分：一次真实的分布偏移，对训练过的 expert 只值 4 dB 且不改变可用性，因此它不可能
  解释 `FW-FZ` 那 15 dB 量级的崩塌。顺带说明 `FW-V` 对相机布局有一定跨设置能力，但它确实是在
  拼接格式上训练的，单相机对它也是偏移。

注意单相机同时把每帧 latent token 从 `14x28=392` 降到 `14x14=196`，所以它隔离了布局但**没有**
隔离分辨率。`FW-FZ` 在两种分辨率下都停在约 10 dB，已在地板上，无法从中读出分辨率的作用。

剩下的候选是分辨率（官方 smoke 用 `1280x720`）、帧数（9 对 81）、以及
`video_attention_mask_mode: first_frame_causal`（DiT 类默认 `bidirectional`）。区分它们需要把
`FW-FZ` 的 video expert 放回 Wan2.2 原生工作点，即 §1 的 `check-protocol`——它走官方
`WanTI2V.generate`、`1280x720`、81 帧，且不使用 FastWAM 的 mask 模式。更便宜的前置检查是直接把
`FW-FZ` 的 video expert 张量与 Wan2.2 预训练权重逐一比对：若逐字节相同，则「`FW-FZ` 的视频分支等于
未受损的 Wan2.2」被证实，崩塌就只能归因于工作点，不需要再生成任何视频。

### 4. 配对汇总与固定可视化

```bash
bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=compare \
  --vbench-fw-fz=outputs/literature/w0_fastwam/video_generation/vbench/fw_fz/metrics.json \
  --vbench-fw-v=outputs/literature/w0_fastwam/video_generation/vbench/fw_v/metrics.json \
  --w0-fw-fz=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_fz/metrics.json \
  --w0-fw-v=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_v/metrics.json \
  --output=outputs/literature/w0_fastwam/video_generation/comparison.json
```

## 监控与结果落点

每个长任务都把 `STATUS completed/total` 同时写到 stdout 和对应 output root 的 `run.log`；另一个 shell 可直接查看：

```bash
tail -f outputs/literature/w0_fastwam/video_generation/vbench/fw_fz/run.log
rg '^STATUS|^DONE' outputs/literature/w0_fastwam/video_generation -g 'run.log'
```

主要产物：

- `outputs/literature/w0_fastwam/video_generation/vbench/assets.json`
- `outputs/literature/w0_fastwam/video_generation/vbench/protocol.json`
- `outputs/literature/w0_fastwam/video_generation/vbench/manifest.json`
- `outputs/literature/w0_fastwam/video_generation/vbench/{fw_fz,fw_v}/run_config.json`
- `outputs/literature/w0_fastwam/video_generation/vbench/{fw_fz,fw_v}/metrics.json`
- `outputs/literature/w0_fastwam/video_generation/w0_video_40/manifest.json`
- `outputs/literature/w0_fastwam/video_generation/w0_video_40/{fw_fz,fw_v}/run_config.json`
- `outputs/literature/w0_fastwam/video_generation/w0_video_40/{fw_fz,fw_v}/metrics.json`
- `outputs/literature/w0_fastwam/video_generation/w0_video_40/comparison_videos/`
- `outputs/literature/w0_fastwam/video_generation/w0_video_40/montages/`
- `outputs/literature/w0_fastwam/video_generation/comparison.json`

`metrics.json` 必须能追溯到 checkpoint SHA-256、manifest SHA-256、实际 protocol、依赖版本和 evaluator 权重信息。

## 判断标准与停止条件

| 检查项 | 通过标准 | 不满足时 |
| --- | --- | --- |
| VBench 固定集 | 21 prompts / 15 images / 三组各 7 / 七种 camera instruction 各 1 | 停止，不生成 |
| VBench 生成 | 每个 variant 21/21，帧数正确，无空视频、缺帧、NaN | 修复后只重跑受影响 variant |
| VBench 指标 | 两个 variant 都有完整 9 项 score 和 checkpoint provenance | 不得宣称通用视频能力保持 |
| W0 固定集 | 40 tasks，每 suite 10，每 task 1，raw window 连续 33，无 padding | 停止，不生成 |
| W0 指标 | 两个 variant 都有 40/40、per-task/per-suite/macro 的 PSNR/SSIM/LPIPS | 不得只看 montage 下结论 |
| 配对一致性 | 两边 manifest/protocol SHA 相同，正差值方向符合 guide 定义 | 停止比较，先修 provenance |
| 结论边界 | 分开报告 VBench 9 项与 W0 3 项；写“指标分化”而不是合成总分 | 不追加显著性、非劣 margin 或完整 VBench 总分 |

即使所有命令完成，也只能说明两个 checkpoint 在这两个离线协议上的相对视频生成表现：VBench-I2V-mini-21 不是完整 VBench，W0-Video-40 来自训练 demonstration，不是 held-out，也不回答闭环成功率是否提升。

官方实现与数据说明：[VBench-I2V pinned README](https://github.com/Vchitect/VBench/tree/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/vbench2_beta_i2v)、[Wan2.2 pinned repository](https://github.com/Wan-Video/Wan2.2/tree/81e67351b81197330c9533e7f345233e7702b44a)。
