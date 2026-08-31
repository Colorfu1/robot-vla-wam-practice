#!/usr/bin/env bash
# Generate the same LIBERO image+command samples with the pinned official Wan2.2 TI2V path.
#
# The third argument selects the only intended generation difference:
#   wan22-official          1280x720
#   fastwam-resolution-only 448x224
# Both use 81 frames, 24 FPS, 50 UniPC steps, shift=5, and guide_scale=5.
#
# Usage:
#   bash scripts/experiments/literature/run_w0_libero_wan.sh probe 1 fastwam-resolution-only
#   bash scripts/experiments/literature/run_w0_libero_wan.sh mini21 8 fastwam-resolution-only
#   bash scripts/experiments/literature/run_w0_libero_wan.sh mini21 8 wan22-official
#   bash scripts/experiments/literature/run_w0_libero_wan.sh all40 8 fastwam-resolution-only

set -euo pipefail

scope="${1:-probe}"
num_gpus="${2:-1}"
operating_point="${3:-fastwam-resolution-only}"
input_camera="${4:-both}"

case "${scope}" in
  probe | mini21 | all40) ;;
  *)
    echo "usage: $0 [probe|mini21|all40] [num_gpus] [wan22-official|fastwam-resolution-only|front-square-hires] [both|front|wrist]" >&2
    exit 2
    ;;
esac
case "${operating_point}" in
  wan22-official | fastwam-resolution-only | front-square-hires) ;;
  *)
    echo "unsupported operating point: ${operating_point}" >&2
    exit 2
    ;;
esac
case "${input_camera}" in
  both | front | wrist) ;;
  *)
    echo "unsupported input camera: ${input_camera}" >&2
    exit 2
    ;;
esac
if ! [[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_gpus must be a positive integer, got ${num_gpus}" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "${repo_root}"

runner="scripts/experiments/literature/run_w0_video_generation_eval.sh"
dataset_root="pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2"
model_root="pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B"
fw_v_checkpoint="outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt"
fw_fz_checkpoint="outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt"
root="outputs/literature/w0_fastwam/video_generation/libero_wan/${operating_point}/${input_camera}"
protocol="${root}/protocol.json"

for path in "${fw_v_checkpoint}" "${fw_fz_checkpoint}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing checkpoint: ${path}" >&2
    exit 1
  fi
done

if [[ "${num_gpus}" -gt 1 ]]; then
  visible="$(python -c 'import os; s=os.environ.get("CUDA_VISIBLE_DEVICES"); print(len([x for x in s.split(",") if x.strip() and x.strip() != "-1"]) if s is not None else __import__("torch").cuda.device_count())')"
  if [[ "${visible}" -lt "${num_gpus}" ]]; then
    echo "num_gpus=${num_gpus} requested but only ${visible} CUDA device(s) visible" >&2
    exit 1
  fi
  echo "multi-GPU: ${num_gpus} of ${visible} visible devices"
fi

echo "=== 1/4 check-protocol (${operating_point})"
bash "${runner}" \
  --mode=check-protocol \
  --wan-root=third_party/wan22 \
  --wan-model-root="${model_root}" \
  --fw-v="${fw_v_checkpoint}" \
  --fw-fz="${fw_fz_checkpoint}" \
  --operating-point="${operating_point}" \
  --frames=9 \
  --output="${protocol}"

kind="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["kind"])' "${protocol}")"
if [[ "${kind}" != "wan22-ti2v-compatible" ]]; then
  echo "LIBERO-Wan requires the official Wan2.2 path; got ${kind}" >&2
  exit 1
fi

if [[ "${scope}" == "all40" ]]; then
  selection="all-40"
  dataset="libero-wan-40"
  selection_dir="all40"
else
  selection="mini-21"
  dataset="libero-wan-mini-21"
  selection_dir="mini21"
fi
work="${root}/${selection_dir}"
manifest="${work}/manifest.json"
mkdir -p "${work}"

echo "=== 2/4 prepare-libero-wan (${selection}, camera=${input_camera})"
bash "${runner}" \
  --mode=prepare-libero-wan \
  --dataset-root="${dataset_root}" \
  --suites=libero_spatial,libero_object,libero_goal,libero_10 \
  --selection="${selection}" \
  --selection-seed=20260824 \
  --base-seed=52000 \
  --raw-frames=33 \
  --video-stride=4 \
  --input-camera="${input_camera}" \
  --protocol="${protocol}" \
  --output="${manifest}"

if [[ "${scope}" == "probe" ]]; then
  python -c 'import json,sys; p=sys.argv[1]; m=json.load(open(p, encoding="utf-8")); m["samples"]=m["samples"][:1]; m["selection"]=m["selection"]+"-1sample-probe"; json.dump(m, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2); print("probe manifest: 1 sample")' "${manifest}"
fi

gpu_args=()
if [[ "${num_gpus}" -gt 1 ]]; then
  gpu_args=(--num-gpus="${num_gpus}")
fi

for variant in fw_fz fw_v; do
  if [[ "${variant}" == "fw_fz" ]]; then
    checkpoint="${fw_fz_checkpoint}"
  else
    checkpoint="${fw_v_checkpoint}"
  fi
  echo "=== 3/4 generate ${variant} (${scope}, ${operating_point})"
  bash "${runner}" \
    --mode=generate \
    --dataset="${dataset}" \
    --variant="${variant}" \
    --model-cache-root=pretrained/w0_fastwam \
    --vendor-root="${W0_VENDOR_ROOT:-third_party/w0_fastwam}" \
    --wan-root=third_party/wan22 \
    --wan-model-root="${model_root}" \
    --checkpoint="${checkpoint}" \
    --manifest="${manifest}" \
    --output-root="${work}/${variant}" \
    "${gpu_args[@]}"
done

echo "LIBERO-Wan ${scope} generation: PASSED"

# The reference frames are the same LIBERO future frames that W0-Video-40 scores, so PSNR/SSIM/LPIPS
# here are directly comparable to that run at the same resolution -- which is the whole point of the
# official-code arm. Without this stage the code-path comparison would be eyeball-only.
for variant in fw_fz fw_v; do
  if [[ "${variant}" == "fw_fz" ]]; then
    checkpoint="${fw_fz_checkpoint}"
  else
    checkpoint="${fw_v_checkpoint}"
  fi
  echo "=== 4/4 score ${variant} (${scope}, ${operating_point})"
  bash "${runner}" \
    --mode=score \
    --dataset="${dataset}" \
    --variant="${variant}" \
    --manifest="${manifest}" \
    --videos="${work}/${variant}/videos" \
    --output="${work}/${variant}/metrics.json" \
    --checkpoint="${checkpoint}" \
    "${gpu_args[@]}"
done

echo "=== paired read-out"
W0_LIBERO_WAN_DIR="${work}" python "${script_dir}/read_w0_libero_wan_metrics.py"

echo "LIBERO-Wan ${scope} generation+score: PASSED"
echo "manifest: ${manifest}"
echo "videos: ${work}/fw_fz/videos and ${work}/fw_v/videos"
echo "metrics: ${work}/fw_fz/metrics.json and ${work}/fw_v/metrics.json"
