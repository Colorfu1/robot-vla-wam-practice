# W0 Wan 视频生成能力保持性快速验证

## 实验目标

回答 `FW-V`（连 Wan2.2 一起 tune）相对 `FW-FZ`（冻结/不训练视频分支）是否保留、改善或破坏视频生成能力。该指南只做配对的离线快速验证，不替代之前的 W0 闭环成功率评测。

## 实验思路

1. 主实验使用四个现有 LIBERO `_no_noops_lerobot` 训练集的固定样本，直接走 pinned Wan2.2
   `WanTI2V.generate`，输入 LIBERO 首帧和 task command。
2. 官方 Wan 档位与 FastWAM 分辨率档位只改变空间分辨率；两者都固定 81 帧、24 FPS、50 步、
   `shift=5`、UniPC、`guide_scale=5`。
3. `FW-FZ` 与 `FW-V` 共享同一 manifest、首帧、文本、seed、VAE/text encoder 和 Wan 推理路径。
4. VBench-I2V 仍可作为独立的公开图像/prompt 补充评测，但不与 LIBERO-Wan 结果混用。
5. W0 原生轨迹只传首帧与原始语言命令，固定 `action=None`、`proprio=None`；不宣称验证
   action-conditioned generation。

实验标签：VBench-I2V-mini-21 是 `Offline/Smoke` 小样本筛查；W0-Video-40 是 `Offline` 的训练分布内匹配检查，不是 held-out 或 `Closed-loop` 正式结果。

## 前置条件与一次性准备

从仓库根目录执行。远端无网；Git checkout 和数据准备在本地完成后同步到远端。

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

### VBench-I2V（mini-21 或 full，可选）

只跑 W0-Video-40 时跳过本小节；mini-21 与 full 共用 pinned checkout、crop、evaluator weights、
Torch Hub cache 和 DreamSim cache。

本地准备 pinned checkout 和 crop 数据，再把目录原样同步到远端（包括 `.git`）：

```bash
rsync -a --info=progress2 \
  third_party/vbench/ REMOTE_HOST:/path/to/practice/third_party/vbench/
rsync -a --info=progress2 \
  third_party/wan22/ REMOTE_HOST:/path/to/practice/third_party/wan22/
```

本地、远端均执行下面的只读校验：

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

完整 VBench-I2V 的 crop、六份 VBench 权重、DINO repo、Torch Hub cache 和 DreamSim cache
都在有网的本地机器准备；下面命令会显式清掉常见 proxy 变量，**不要在无网远端执行**。
已存在的文件不会重复下载：

```bash
(
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
cache=pretrained/vbench_i2v
mkdir -p "${cache}"/{clip_model,aesthetic_model/emb_reader,pyiqa_model,amt_model,raft_model,dino_model}

if [[ ! -d "${cache}/crop/16-9" ]]; then
  conda run --no-capture-output -n robot-practice \
    python -c 'import gdown' 2>/dev/null || \
    conda run --no-capture-output -n robot-practice python -m pip install --no-deps gdown==5.2.0
  conda run --no-capture-output -n robot-practice \
    python -m gdown --id 1Y_JnYnyJ3a6QhiranoX0MQVZFcTDPekZ \
    --output "${cache}/crop.zip" --continue
  unzip -q "${cache}/crop.zip" -d "${cache}"
fi

if [[ ! -d "${cache}/dino_model/facebookresearch_dino_main" ]]; then
  git -c http.proxy= -c https.proxy= clone --depth=1 https://github.com/facebookresearch/dino.git \
    "${cache}/dino_model/facebookresearch_dino_main"
fi

download() {
  local url="$1"
  local target="$2"
  mkdir -p "$(dirname "${target}")"
  [[ -s "${target}" ]] || wget --no-proxy -c -O "${target}" "${url}"
}

download https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth \
  "${cache}/dino_model/dino_vitbase16_pretrain.pth"
download https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
  "${cache}/clip_model/ViT-B-32.pt"
download https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt \
  "${cache}/clip_model/ViT-L-14.pt"
download https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth \
  "${cache}/aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"
download https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth \
  "${cache}/pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"
download https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth \
  "${cache}/amt_model/amt-s.pth"

if [[ ! -s "${cache}/raft_model/models/raft-things.pth" ]]; then
  download https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip \
    "${cache}/raft_model/models.zip"
  unzip -q -o "${cache}/raft_model/models.zip" -d "${cache}/raft_model"
fi

torch_home="${PWD}/${cache}/torch"
mkdir -p "${torch_home}"
conda run --no-capture-output -n robot-practice \
  env TORCH_HOME="${torch_home}" \
  python docker/training/warm_w0_video_eval_caches.py \
  --stage=torch-hub --skip-lpips --vbench-cache-root="${cache}"

conda run --no-capture-output -n robot-practice \
  python docker/training/warm_w0_video_eval_caches.py \
  --stage=dreamsim --dreamsim-target="${PWD}/models"

test -d "${cache}/crop/16-9"
test -f "${cache}/dino_model/dino_vitbase16_pretrain.pth"
test -f "${cache}/clip_model/ViT-B-32.pt"
test -f "${cache}/clip_model/ViT-L-14.pt"
test -f "${cache}/raft_model/models/raft-things.pth"
test -f "${cache}/torch/hub/checkpoints/cotracker2.pth"
test -f "${cache}/torch/hub/checkpoints/dino_vitbase16_pretrain.pth"
printf 'Local no-proxy VBench-I2V assets: PASSED\n'
)
```

换镜像或环境后确认 VBench 评分依赖：

```bash
export W0_VIDEO_SCORE_ENV="${W0_VIDEO_SCORE_ENV:-robot-practice}"
conda run --no-capture-output -n "${W0_VIDEO_SCORE_ENV}" \
  python -c 'import clip, cv2, decord, dreamsim, imageio.v3, lpips, pyiqa; print("VBench/W0/LPIPS evaluator dependencies: OK")'
```

VBench 的 `camera_motion` 需要离线 Torch Hub cache；`models/` 只放 DreamSim，不是这个 cache。
如果评测镜像没有预置，在有网本地准备后同步 `pretrained/vbench_i2v/torch/` 到远端：

```bash
(
set -euo pipefail
torch_home="${PWD}/pretrained/vbench_i2v/torch"
mkdir -p "${torch_home}"
conda run --no-capture-output -n robot-practice \
  env TORCH_HOME="${torch_home}" \
  python docker/training/warm_w0_video_eval_caches.py \
  --stage=torch-hub \
  --skip-lpips \
  --vbench-cache-root=pretrained/vbench_i2v
test -f "${torch_home}/hub/facebookresearch_co-tracker_main/hubconf.py"
test -f "${torch_home}/hub/checkpoints/cotracker2.pth"
test -f "${torch_home}/hub/checkpoints/dino_vitbase16_pretrain.pth"
)
```

检查当前 score 将使用的 Torch Hub cache：

```bash
( set -euo pipefail; hub="${PWD}/pretrained/vbench_i2v/torch/hub"; test -f "${hub}/checkpoints/alexnet-owt-7be5be79.pth"; test -f "${hub}/checkpoints/cotracker2.pth"; test -f "${hub}/checkpoints/dino_vitbase16_pretrain.pth"; test -f "${hub}/facebookresearch_co-tracker_main/hubconf.py" )
```

确认仓库根目录的 DreamSim 缓存：

```bash
( set -euo pipefail; test -f models/ensemble_lora/adapter_model.safetensors; test -f models/dino_vitb16_pretrain.pth; test -f models/clip_vitb16_pretrain.pth.tar; test -f models/open_clip_vitb16_pretrain.pth.tar; test -f models/checkpoints/dino_vitbase16_pretrain.pth; test -d models/facebookresearch_dino_main )
```

确认 VBench 本地权重（正式 score 不使用 `--allow-vbench-downloads`）：

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

## 执行清单（先跑 LIBERO-Wan-mini-21）

### 正式跑法：四组，两个单变量

`FW-FZ` 的 video expert 是未改动的 Wan2.2 预训练权重，所以它在任何工作点上的表现就是那个工作点
的保持度上界。要把「`FW-V` 丧失了能力」与「这个工作点本身就跑不动 Wan2.2」分开，必须让工作点之间
只差一个变量。下面四组就是为此设计的：A 与 D 只差分辨率，B 与 C 只差代码路径。

| 组 | 数据 | 变体 | 生成代码 | 尺寸 / 帧 | 指标 | 产物 |
| --- | --- | --- | --- | --- | --- | --- |
| A | VBench-I2V mini-21（自然图像） | fw_fz + fw_v | 官方 `WanTI2V.generate` | `1248x704` / 81 | VBench 九维 | `vbench/native/` |
| B | LIBERO w40（40 条） | fw_fz + fw_v | 官方 `WanTI2V.generate` | `448x224` / 9 | PSNR/SSIM/LPIPS | `libero_wan/fastwam-resolution-only/both/all40/` |
| C | LIBERO w40 | fw_fz | **FastWAM `infer_joint`** | `448x224` / 9 | PSNR/SSIM/LPIPS | `w0_video_40/` |
| D | VBench-I2V mini-21 | fw_fz + fw_v | 官方 `WanTI2V.generate` | **`448x224`** / 81 | VBench 九维 | `vbench/native-fastwam-point/` |

```bash
bash scripts/experiments/literature/run_w0_vbench_native.sh mini21 8
```

```bash
bash scripts/experiments/literature/run_w0_libero_wan.sh all40 8 fastwam-resolution-only
```

```bash
bash scripts/experiments/literature/run_w0_video_40.sh fw_fz 8
```

```bash
bash scripts/experiments/literature/run_w0_vbench_native.sh mini21 8 fastwam-resolution-only
```

四条命令依次对应 A、B、C、D。都会命中选样缓存，`prepare` 秒过；A、D 是 81 帧（A 的 seq_len 18018，
D 只有 2058），B、C 是 9 帧，都很快。产物按操作点分树，互不覆盖。

**A 与 D 的差值回答**：分辨率单独是否足以摧毁未微调的 Wan2.2。若 `FW-FZ` 在 D 也崩，则它在 LIBERO
上崩溃与「LIBERO 域」「双相机拼接」都无关；若 D 仍干净，崩溃就得归因到域或布局。

**B 与 C 的差值回答**：FastWAM 的推理管路本身是否是崩溃原因。同 40 条、同分辨率、同 9 帧、同真值，
只差生成入口。

**A 与 B 不构成单变量**（分辨率、帧数、图像域同时变），只作参考。

`wan22-official` 档位下 LIBERO-Wan 也可跑（`1248x704` / 9 帧），那是分辨率消融的第三点，不在这四组内。

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

**主实验先跑 LIBERO-Wan-mini-21。** 执行本文“实验命令”中的
“2. LIBERO-Wan-mini-21：Wan 官方推理 + LIBERO 输入”小节：先固定 manifest，再分别生成
`FW-FZ` 和 `FW-V`。该路径生成 81 帧视频，不使用旧的 W0-Video-40 9 帧 reference score。

## 实验命令

主流程是 **LIBERO-Wan-mini-21**；VBench-I2V 和 W0-Video-40 是互相独立的补充实验。
LIBERO-Wan 只生成 Wan 路径下的配对视频，不调用旧的 9 帧 W0 reference score。

`--vendor-root` 默认 `third_party/w0_fastwam`，而该路径是 gitignored 的、不会被 rsync 同步。
训练镜像自带的 vendor 树在 `/opt/robot-practice/w0-vendor`，远端缺这棵树时给 `generate` 加
`--vendor-root=/opt/robot-practice/w0-vendor`。上面那段预检会把缺失路径全部列出。

### 1. 固定协议与 VBench-I2V-mini-21 manifest（可选补充）

**执行顺序不可跳步，因为 `generate` 的协议是从 manifest 读的（`manifest["protocol"]`），不是从
`protocol.json` 读的：**

```
check-protocol  ->  protocol.json    选定 wan22-ti2v-compatible 还是 fastwam-native
prepare-vbench  ->  manifest.json    通过 --protocol 把该协议嵌进 manifest
generate        ->  videos           使用 manifest["protocol"]
score           ->  metrics.json
```

跳过前两步会重演 2026-08-25 那次失效：manifest 仍带旧的 `fastwam-native` 协议，`generate` 又默认
跳过已存在的视频，于是整轮只是把 `448x224` / 9 帧的旧视频重新打了一遍分，而两个变体都停在
`FW-FZ` 从未训练过的工作点上。

**正式跑法是这一个短入口**，它按上面四步依次执行，并在协议没升级到 `wan22-ti2v-compatible` 时
拒绝继续：

```bash
bash scripts/experiments/literature/run_w0_vbench_native.sh
```

默认是 `probe` 模式：只跑 1 条样本并打印 generate 墙钟，用来估算全量成本。确认耗时可接受后，按需要的
样本量选档位：

```bash
bash scripts/experiments/literature/run_w0_vbench_native.sh mini21 8
```

```bash
bash scripts/experiments/literature/run_w0_vbench_native.sh vbench-full 8
```

**smoke 通过后可以直接跳过 mini-21 走全量，前提是墙钟允许。** 两个档位的差别不是流程而是统计功效：

| 档位 | 样本 | 每 dimension 的 n | 生成次数（两个变体） | 输出子树 |
| --- | ---: | --- | ---: | --- |
| `mini21` | 21 | 7（quality 类 14） | 42 | `vbench/native/` |
| `vbench-full` | 1118 | `i2v_background` 109 至 `camera_motion` 763 | **2236** | `vbench/native-full/` |

`mini21` 只是筛查子集：n=7 时一个 ±0.2857 的差值恰好等于**两条视频**在二元指标上的翻转，无法支撑结论。
`vbench-full` 才是能回答保持性问题的规模。所以如果算力允许，直接跑 `vbench-full`；`mini21` 的用途是
在不确定成本时先拿一组端到端结果。

`full` 这个词有歧义（既可指 mini-21 的全部 21 条，也可指 VBench-I2V 的全部 1118 条），脚本会拒绝它并
提示改用 `mini21` 或 `vbench-full`。

**成本提醒**：`wan22-ti2v-compatible` 是 `1280x720` × 81 帧 × 50 步，而 `fastwam-native` 是
`448x224` × 9 帧 × 10 步——经 2×2 patchify 后注意力序列长度为 18,900 对 294 个 token（`((81-1)//4+1) × 45 × 80 // 4`），且注意力对它是二次的。
所以必须先跑 `probe` 看墙钟，再决定要不要跑 42 次生成。

两种协议的产物写在不同子树（`vbench/native/` 与 `vbench/fallback/`）：旧的 `fastwam-native` 视频在
不同分辨率与帧数下无法复用，覆盖掉它们也会毁掉「那轮为什么无效」的证据。

需要单步执行或核对时，下面是脚本实际调用的命令。`--selection` 与 `--protocol` 必须同时给，否则
`prepare-vbench` 会以 `--protocol is required when --selection is set` 退出：

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
  --output=outputs/literature/w0_fastwam/video_generation/vbench/native/manifest.json
)
```

通过条件：manifest 为 21 条，subject/background/camera 各 7 条，15 张 unique image，7 个 camera prompt 各 1 条；两个 checkpoint 的协议 JSON 完全相同。

#### 信任 VBench 结果前必须先看 `protocol.json` 的 `kind`

`protocol.json` 的 `kind` 决定这一半能回答什么问题，两种取值不是「细节差异」而是**结论级差异**：

| `kind` | 生成路径 | 这一半能回答什么 |
| --- | --- | --- |
| `wan22-ti2v-compatible` | 官方 `WanTI2V.generate`、`1280x720`、81 帧、24 FPS、50 步、UniPC | **能**。`FW-FZ` 在此等于未受损的 Wan2.2，构成保持度上界 |
| `fastwam-native` | `FastWAM.infer_joint`、`448x224`、9 帧、10 步、`first_frame_causal` | **不能**。两个变体都跑在 `FW-FZ` 从未训练过的工作点上，与 W0-Video-40 是同一个混淆，只换了图像域 |

2026-08-25 的那轮 VBench 就落在 `fastwam-native` 上，因此**不能作为保持性证据**。原因不是权重不兼容：
`protocol.json` 的 `checks` 记录 `fw_fz` 的 `strict_load: true`（825 个张量，说明冻结权重与官方 Wan2.2
键名完全兼容），而 `fw_v` 是
`strict_load: false, reason "ModuleNotFoundError: No module named 'ftfy'"`——`check-protocol` 跑在
`W0_VIDEO_MODEL_ENV`（`fastwam-upstream`），而该环境缺 `ftfy`，官方
`third_party/wan22/wan/modules/tokenizers.py` 是无保护的 `import ftfy`。回退规则是「任一变体 strict
load 失败即用 fastwam-native」，于是整轮生成静默地跑在了错的工作点上。

两处已修：镜像 `w0-fastwam-20260826-wan22deps`（`latest` 同 digest）向 `fastwam-upstream` 补装
`ftfy==6.3.1`、`decord==0.6.0`、`librosa==0.11.0`——这三个是把 `from wan import WanTI2V` 跑通所需的
全部缺口，逐个加包实测确定；`check-protocol` 现在遇到 `ModuleNotFoundError`/`ImportError` 形态的
strict-load 失败会**直接抛错而不是回退**，因为缺包不构成关于 checkpoint 的任何结论
（确需记录退化协议时用 `W0_ALLOW_ENVIRONMENT_FALLBACK=1`）。`protocol.json` 的 `kind` 若为 `fastwam-native` 是预定回退，不是失败。

### VBench-I2V-mini-21 smoke（正式评测前必跑）

这条命令只用 1 张卡，从现有 21 条 manifest 自动选 3 条样本，覆盖全部 9 个 VBench dimension。它会实际加载所有评分模型和 CoTracker，并验证 Torch Hub cache 没有被 DreamSim 改回 `./models`。失败时不要启动下面的 8 卡正式 score。

```bash
(
set -euo pipefail

export W0_VIDEO_SCORE_ENV="${W0_VIDEO_SCORE_ENV:-robot-practice}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=smoke-vbench \
  --dataset=vbench-mini-21 \
  --variant=fw_fz \
  --vbench-root=third_party/vbench \
  --cache-root=pretrained/vbench_i2v \
  --manifest=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json \
  --videos=outputs/literature/w0_fastwam/video_generation/vbench/fw_fz/videos \
  --num-gpus=1 \
  --output=outputs/literature/w0_fastwam/video_generation/vbench/smoke/fw_fz/metrics.json

test -s outputs/literature/w0_fastwam/video_generation/vbench/smoke/fw_fz/metrics.json
printf 'VBench-I2V-mini-21 smoke: PASSED\n'
)
```

### 2. LIBERO-Wan-mini-21：Wan 官方推理 + LIBERO 输入

这是本轮要回答“FT 是否破坏 Wan 视频生成能力”的主实验。它不使用 VBench 图像或 prompt，
而是对同一条 LIBERO 样本，把首帧和 task command 送入 pinned Wan2.2 的 `WanTI2V.generate`。
`FW-FZ` 和 `FW-V` 只替换 Wan video expert 的 checkpoint，其余输入、随机种子和推理参数相同。

两档协议如下；`fastwam-resolution-only` 的含义是只把空间分辨率改为 FastWAM 工作点，不能再
改成 9 帧或 10 步：

| 档位 | 分辨率 | 帧数 | FPS | 推理步数 | shift | sampler | guide scale |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `wan22-official` | 1280×720 | 81 | 24 | 50 | 5 | UniPC | 5 |
| `fastwam-resolution-only` | 448×224 | 81 | 24 | 50 | 5 | UniPC | 5 |

默认使用 front+wrist 横向拼接后的 LIBERO 首帧，随后按目标分辨率做固定的中心裁剪和 resize；
两档必须使用同一个第四参数。若要只用 front camera，两档都传 `front`，不能只改一边。

先跑 1 条 probe：

```bash
bash scripts/experiments/literature/run_w0_libero_wan.sh probe 1 fastwam-resolution-only
```

确认显存和墙钟后，跑 21 条 LIBERO 配对生成（8 卡）：

```bash
bash scripts/experiments/literature/run_w0_libero_wan.sh mini21 8 fastwam-resolution-only
```

如需官方分辨率对照，使用完全相同的样本选择、seed 和 camera：

```bash
bash scripts/experiments/literature/run_w0_libero_wan.sh mini21 8 wan22-official
```

产物分别在：

```text
outputs/literature/w0_fastwam/video_generation/libero_wan/fastwam-resolution-only/both/mini21/fw_fz/videos
outputs/literature/w0_fastwam/video_generation/libero_wan/fastwam-resolution-only/both/mini21/fw_v/videos
outputs/literature/w0_fastwam/video_generation/libero_wan/wan22-official/both/mini21/fw_fz/videos
outputs/literature/w0_fastwam/video_generation/libero_wan/wan22-official/both/mini21/fw_v/videos
```

每个目录的 `run_config.json` 记录实际 checkpoint、manifest SHA-256 和协议。当前这条 LIBERO-Wan
入口只负责生成配对视频；不要把 81 帧视频交给旧的 W0-Video-40 score，因为旧 score 固定比较
9 帧双相机 reference，时间轴并不匹配。

### 3. VBench-I2V-mini-21 生成与打分

这是独立的公开 VBench 图像/prompt 补充评测，不使用 LIBERO 输入。正式 Wan2.2 档位固定为
`1280x720`、81 帧、24 FPS、50 步、`shift=5`、UniPC、`guide_scale=5`。本节不再推荐旧的
`fastwam-resolution-only` VBench 命令；需要分辨率对照时使用上一节的 LIBERO-Wan 入口。

先跑 1 条 probe（可选）：

```bash
bash scripts/experiments/literature/run_w0_vbench_native.sh probe 1 wan22-official
```

确认耗时和显存后，跑 mini-21 正式比较（8 卡）：

```bash
bash scripts/experiments/literature/run_w0_vbench_native.sh mini21 8 wan22-official
```

该入口会依次执行 `check-protocol -> prepare-vbench -> generate FW-FZ/FW-V -> score`，并拒绝
`fastwam-native` 回退协议。结果在：

```text
outputs/literature/w0_fastwam/video_generation/vbench/native/fw_fz/metrics.json
outputs/literature/w0_fastwam/video_generation/vbench/native/fw_v/metrics.json
```

不要把 VBench 结果与 LIBERO-Wan 结果混在一起。

手工执行时有两处必须改：`--manifest` 与 `--output-root` 都要指向 §1 选定协议对应的子树
（`wan22-ti2v-compatible` 是 `vbench/native/`，回退协议是 `vbench/fallback/`），**不要**再用旧的
`vbench/manifest.json` 与 `vbench/{fw_fz,fw_v}/`——那是 2026-08-25 那轮 `fastwam-native` 的产物。
另外 `generate` 默认跳过已存在的视频，重跑同一路径时要加 `--overwrite`，否则它会静默什么都不做。

下面默认使用 8 张 GPU；`W0_VIDEO_NUM_GPUS` 必须不大于 `CUDA_VISIBLE_DEVICES` 中的可见卡数。
score wrapper 会自动将 `TORCH_HOME` 和 Torch Hub 内部目录固定到 `pretrained/vbench_i2v/torch/hub`，不需要手工 `export TORCH_HOME`。

```bash
(
set -euo pipefail

export W0_VIDEO_MODEL_ENV="${W0_VIDEO_MODEL_ENV:-fastwam-upstream}"
export W0_VIDEO_SCORE_ENV="${W0_VIDEO_SCORE_ENV:-robot-practice}"
export W0_VIDEO_NUM_GPUS="${W0_VIDEO_NUM_GPUS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=generate --dataset=vbench-mini-21 --variant=fw_fz \
  --model-cache-root=pretrained/w0_fastwam \
  --vbench-root=third_party/vbench --wan-root=third_party/wan22 \
  --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B \
  --checkpoint=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt \
  --manifest=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json \
  --num-gpus="${W0_VIDEO_NUM_GPUS}" \
  --output-root=outputs/literature/w0_fastwam/video_generation/vbench/fw_fz

bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
  --mode=generate --dataset=vbench-mini-21 --variant=fw_v \
  --model-cache-root=pretrained/w0_fastwam \
  --vbench-root=third_party/vbench --wan-root=third_party/wan22 \
  --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B \
  --checkpoint=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt \
  --manifest=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json \
  --num-gpus="${W0_VIDEO_NUM_GPUS}" \
  --output-root=outputs/literature/w0_fastwam/video_generation/vbench/fw_v

# 当前命令在评测容器内执行，直接调用 score wrapper；不要在容器内再启动 Docker。
for variant in fw_fz fw_v; do
  bash scripts/experiments/literature/run_w0_video_generation_eval.sh \
    --mode=score --dataset=vbench-mini-21 --variant="${variant}" \
    --vbench-root=third_party/vbench \
    --cache-root=pretrained/vbench_i2v \
    --manifest=outputs/literature/w0_fastwam/video_generation/vbench/manifest.json \
    --videos="outputs/literature/w0_fastwam/video_generation/vbench/${variant}/videos" \
    --num-gpus="${W0_VIDEO_NUM_GPUS}" \
    --output="outputs/literature/w0_fastwam/video_generation/vbench/${variant}/metrics.json"
done
)
```

固定报告 9 个 dimension：`i2v_subject`、`subject_consistency`、`motion_smoothness`、`dynamic_degree`、`i2v_background`、`background_consistency`、`aesthetic_quality`、`imaging_quality`、`camera_motion`。21 条子集不计算伪造的完整 VBench Total Score。

### 4. 完整 VBench-I2V：1118 条、10 个 dimension

下面的 `full` 使用 pinned `vbench2_i2v_full_info.json` 的全部 1118 条 metadata row、355 张 unique image，覆盖官方 I2V 的 10 个 dimension（额外包含 `temporal_flickering`）。它与 mini-21 使用同一份 protocol、模型输入和多卡分片逻辑；正式比较时两种 checkpoint 共生成 2236 个视频。

先重新生成并锁定协议；`kind` 必须是 `wan22-ti2v-compatible`，否则停止，不要把旧的 `fastwam-native` 9 帧视频当作正式 full 结果：

```bash
(
set -euo pipefail

export W0_VIDEO_MODEL_ENV="${W0_VIDEO_MODEL_ENV:-fastwam-upstream}"
bash scripts/experiments/literature/run_w0_video_generation_eval.sh --mode=check-protocol --wan-root=third_party/wan22 --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B --fw-v=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt --fw-fz=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt --output=outputs/literature/w0_fastwam/video_generation/vbench/protocol.json
conda run --no-capture-output -n "${W0_VIDEO_MODEL_ENV}" python -c 'import json; p="outputs/literature/w0_fastwam/video_generation/vbench/protocol.json"; x=json.load(open(p)); assert x["kind"]=="wan22-ti2v-compatible", x["kind"]; print("Wan2.2-compatible protocol: PASSED")'
)
```

先固定完整 manifest，并让命令只输出一个判定：

```bash
(
set -euo pipefail

bash scripts/experiments/literature/run_w0_video_generation_eval.sh --mode=prepare-vbench --vbench-root=third_party/vbench --cache-root=pretrained/vbench_i2v --selection=full --base-seed=52000 --protocol=outputs/literature/w0_fastwam/video_generation/vbench/protocol.json --output=outputs/literature/w0_fastwam/video_generation/vbench/full/manifest.json

conda run --no-capture-output -n robot-practice python -c 'import json; from collections import Counter; p="outputs/literature/w0_fastwam/video_generation/vbench/full/manifest.json"; m=json.load(open(p)); assert m["dataset"]=="VBench-I2V-full"; assert len(m["samples"])==1118; assert m["selected_image_count"]==355; assert m["evaluation_dimensions"]==["i2v_subject","subject_consistency","motion_smoothness","dynamic_degree","i2v_background","background_consistency","aesthetic_quality","imaging_quality","camera_motion","temporal_flickering"]; assert Counter(d for s in m["samples"] for d in s["dimensions"]) == Counter({"camera_motion":763,"aesthetic_quality":355,"imaging_quality":355,"temporal_flickering":355,"i2v_subject":246,"subject_consistency":246,"motion_smoothness":246,"dynamic_degree":246,"i2v_background":109,"background_consistency":109}); print("VBench-I2V-full manifest: PASSED")'
)
```

然后用 8 张卡生成两个 variant，先对 FW-FZ 做 3 条 all-dimension smoke，再用 8 张卡正式 score。不要加 `--overwrite`，中断后可以从已有视频继续：

```bash
(
set -euo pipefail

export W0_VIDEO_MODEL_ENV="${W0_VIDEO_MODEL_ENV:-fastwam-upstream}"
export W0_VIDEO_SCORE_ENV="${W0_VIDEO_SCORE_ENV:-robot-practice}"
export W0_VIDEO_NUM_GPUS="${W0_VIDEO_NUM_GPUS:-8}"
export W0_VENDOR_ROOT="${W0_VENDOR_ROOT:-/opt/robot-practice/w0-vendor}"
video_gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export CUDA_VISIBLE_DEVICES="${video_gpu_list}"

bash scripts/experiments/literature/run_w0_video_generation_eval.sh --mode=generate --dataset=vbench-i2v-full --variant=fw_fz --model-cache-root=pretrained/w0_fastwam --vendor-root="${W0_VENDOR_ROOT}" --wan-root=third_party/wan22 --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B --checkpoint=outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt --manifest=outputs/literature/w0_fastwam/video_generation/vbench/full/manifest.json --num-gpus="${W0_VIDEO_NUM_GPUS}" --output-root=outputs/literature/w0_fastwam/video_generation/vbench/full/fw_fz

bash scripts/experiments/literature/run_w0_video_generation_eval.sh --mode=generate --dataset=vbench-i2v-full --variant=fw_v --model-cache-root=pretrained/w0_fastwam --vendor-root="${W0_VENDOR_ROOT}" --wan-root=third_party/wan22 --wan-model-root=pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B --checkpoint=outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt --manifest=outputs/literature/w0_fastwam/video_generation/vbench/full/manifest.json --num-gpus="${W0_VIDEO_NUM_GPUS}" --output-root=outputs/literature/w0_fastwam/video_generation/vbench/full/fw_v

export CUDA_VISIBLE_DEVICES="${video_gpu_list%%,*}"
bash scripts/experiments/literature/run_w0_video_generation_eval.sh --mode=smoke-vbench --dataset=vbench-i2v-full --variant=fw_fz --vbench-root=third_party/vbench --cache-root=pretrained/vbench_i2v --manifest=outputs/literature/w0_fastwam/video_generation/vbench/full/manifest.json --videos=outputs/literature/w0_fastwam/video_generation/vbench/full/fw_fz/videos --num-gpus=1 --output=outputs/literature/w0_fastwam/video_generation/vbench/full/smoke/fw_fz/metrics.json

test -s outputs/literature/w0_fastwam/video_generation/vbench/full/smoke/fw_fz/metrics.json
printf 'VBench-I2V-full smoke: PASSED\n'

export CUDA_VISIBLE_DEVICES="${video_gpu_list}"
for variant in fw_fz fw_v; do
  bash scripts/experiments/literature/run_w0_video_generation_eval.sh --mode=score --dataset=vbench-i2v-full --variant="${variant}" --vbench-root=third_party/vbench --cache-root=pretrained/vbench_i2v --manifest=outputs/literature/w0_fastwam/video_generation/vbench/full/manifest.json --videos="outputs/literature/w0_fastwam/video_generation/vbench/full/${variant}/videos" --num-gpus="${W0_VIDEO_NUM_GPUS}" --output="outputs/literature/w0_fastwam/video_generation/vbench/full/${variant}/metrics.json"
done
)
```

正式结果在 `outputs/literature/w0_fastwam/video_generation/vbench/full/fw_fz/metrics.json` 和同目录的 `fw_v/metrics.json`；每个 variant 的 `run.log`、`.multigpu/` worker 日志也保留在对应目录。完整 VBench 只报告这 10 个已实现的 I2V dimension，不合成未经官方定义的总分。

### 5. W0-Video-40 manifest、生成与指标

先做 W0 本地实现检查：

```bash
conda run --no-capture-output -n robot-practice env PYTHONNOUSERSITE=1 PYTHONPATH=src \
  python -m pytest tests/test_w0_video_generation_eval.py -q
conda run --no-capture-output -n robot-practice \
  python -m ruff check src/robot_practice/experiments/literature/w0_video_generation_eval.py \
  tests/test_w0_video_generation_eval.py
bash -n scripts/experiments/literature/run_w0_video_generation_eval.sh
```

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
| `bash scripts/experiments/literature/run_w0_libero_wan.sh` | **本轮主实验**：LIBERO 首帧+command 进入官方 Wan2.2 `WanTI2V.generate`；显式固定 81 帧、24 FPS、50 步，支持 `mini21 8` 和两档分辨率 |
| `bash scripts/experiments/literature/run_w0_vbench_native.sh` | **VBench-I2V 一半**：check-protocol → prepare-vbench → generate → score 四步依次执行，协议未升级到 `wan22-ti2v-compatible` 时拒绝继续。默认 `probe`（1 条样本、打印墙钟）；确认成本后用 `mini21 8`（21 条）或 `vbench-full 8`（1118 条）|
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

`--mode=compare`（“6. 配对汇总与固定可视化”）要求四份 metrics 齐全，只跑本节时会以
`compare is missing: vbench-fw-fz, vbench-fw-v` 退出。在补齐“3. VBench-I2V-mini-21 生成与打分”之前，
可以用下面这条命令读取 W0 配对差值；它只是读数，不做“6. 配对汇总与固定可视化”中的
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

`FW-FZ` 在 `448x224` 上崩到约 10 dB 的原因是**分辨率**，不是 LIBERO 域、双相机拼接或 FastWAM 代码
路径。三者的排除靠跨分辨率对照：同一批 LIBERO 样本、同官方 `WanTI2V.generate`、同 9 帧，只换分辨率，
`FW-FZ` 从 `448x224` 的 9.243 dB 升到 `1248x704` 的 15.960 dB（四 suite 全在 14.6–16.9）；代码路径靠
FastWAM `infer_joint`（9.963 dB）与官方 `WanTI2V.generate`（9.243 dB）在同工作点同样崩来排除。已提交
的四宫格 `docs/assets/w0_video_generation/libero_resolution_quad.jpg` 展示了这条对角线。重建命令：

```bash
python scripts/experiments/literature/make_w0_video_review_reel.py --videos "FW-FZ 448x224=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_fz/videos" --videos "FW-V 448x224=outputs/literature/w0_fastwam/video_generation/w0_video_40/fw_v/videos" --videos "FW-FZ 1248x704=outputs/literature/w0_fastwam/video_generation/libero_wan/wan22-official/both/mini21/fw_fz/videos" --videos "FW-V 1248x704=outputs/literature/w0_fastwam/video_generation/libero_wan/wan22-official/both/mini21/fw_v/videos" --out-sheet docs/assets/w0_video_generation/libero_resolution_quad.jpg --sheet-orientation column --sheet-frames 5 --panel-width 250 --samples libero_spatial_task01,libero_object_task00,libero_goal_task00
```

`1248x704` 上 PSNR/SSIM 判 `FW-V` 略高属**指标失效**（它抹平成低频糊、MSE 代价小），LPIPS、后续帧
对比度 `ret`（1.01 对 0.77）与目视都判 `FW-FZ` 更好；以 LPIPS 与目视为准。

### 6. 配对汇总与固定可视化

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
