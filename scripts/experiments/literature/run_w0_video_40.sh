#!/usr/bin/env bash
# Run the full W0-Video-40 protocol: generate then score, for one or both variants.
#
# Why this exists as a script: each generate/score invocation is 300-400 characters, and a
# terminal paste that inserts newlines runs the tail as separate commands, which fails as
# "generate is missing: manifest, output_root, checkpoint" followed by
# "--checkpoint=...: No such file or directory". Everything long lives here.
#
# This is the frozen protocol from reports/guides/literature/02_w0_wan_video_generation_preservation.md
# section 3: 40 samples, both cameras concatenated into 448x224, 9 frames, 10 sampling steps.
# It is NOT the single-camera diagnostic -- that is probe_w0_single_camera.sh, which uses 3
# samples at 224x224 and marks its protocol as a deviation.
#
# generate skips samples whose video already exists, so re-running is cheap and idempotent;
# pass overwrite=1 in the environment to force regeneration.
#
# Usage:
#   bash scripts/experiments/literature/run_w0_video_40.sh                 # both variants, 1 GPU
#   bash scripts/experiments/literature/run_w0_video_40.sh fw_v            # one variant
#   bash scripts/experiments/literature/run_w0_video_40.sh both 4          # both, 4 GPUs
#   bash scripts/experiments/literature/run_w0_video_40.sh fw_v 4

set -euo pipefail

selection="${1:-both}"
num_gpus="${2:-1}"

case "${selection}" in
  both) variants=(fw_fz fw_v) ;;
  fw_v | fw_fz) variants=("${selection}") ;;
  *)
    echo "usage: $0 [both|fw_v|fw_fz] [num_gpus]" >&2
    exit 2
    ;;
esac
if ! [[ "${num_gpus}" =~ ^[1-9][0-9]*$ ]]; then
  echo "num_gpus must be a positive integer, got ${num_gpus}" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "${repo_root}"

work="outputs/literature/w0_fastwam/video_generation/w0_video_40"
manifest="${work}/manifest.json"
runner="scripts/experiments/literature/run_w0_video_generation_eval.sh"
lpips_weights="pretrained/vbench_i2v/lpips/alex.pth"

fw_v_checkpoint="outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt"
fw_fz_checkpoint="outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt"

if [[ ! -f "${manifest}" ]]; then
  echo "missing ${manifest}; run --mode=prepare-w0 first (guide section 3)" >&2
  exit 1
fi
if [[ ! -s "${lpips_weights}" ]]; then
  echo "missing ${lpips_weights}; copy it from the lpips package (guide prerequisites)" >&2
  exit 1
fi

gpu_args=()
if [[ "${num_gpus}" -gt 1 ]]; then
  gpu_args=(--num-gpus="${num_gpus}")
fi
overwrite_args=()
if [[ "${overwrite:-0}" != "0" ]]; then
  overwrite_args=(--overwrite)
fi

for variant in "${variants[@]}"; do
  if [[ "${variant}" == "fw_fz" ]]; then
    checkpoint="${fw_fz_checkpoint}"
  else
    checkpoint="${fw_v_checkpoint}"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi

  echo "=== generate ${variant} (40 samples, 448x224 two-camera, num_gpus=${num_gpus})"
  bash "${runner}" \
    --mode=generate \
    --dataset=w0-video-40 \
    --variant="${variant}" \
    --model-cache-root=pretrained/w0_fastwam \
    --checkpoint="${checkpoint}" \
    --manifest="${manifest}" \
    --output-root="${work}/${variant}" \
    "${gpu_args[@]}" \
    "${overwrite_args[@]}"

  echo "=== score ${variant}"
  bash "${runner}" \
    --mode=score \
    --dataset=w0-video-40 \
    --variant="${variant}" \
    --manifest="${manifest}" \
    --videos="${work}/${variant}/videos" \
    --lpips-weights="${lpips_weights}" \
    --output="${work}/${variant}/metrics.json" \
    "${gpu_args[@]}"
done

if [[ "${selection}" == "both" ]]; then
  echo
  echo "=== paired read-out (not a substitute for --mode=compare, which also checks provenance)"
  W0_DIR="${work}" python - <<'PY'
import json
import os

root = os.environ["W0_DIR"]
loaded = [json.load(open(f"{root}/{v}/metrics.json", encoding="utf-8")) for v in ("fw_fz", "fw_v")]
macro = [item["summary"]["macro"] for item in loaded]
for item, values in zip(loaded, macro):
    lpips = "unavailable" if values["lpips"] is None else f"{values['lpips']:.4f}"
    print(
        f"{item['variant']:<6} n={values['count']}  PSNR {values['psnr']:7.3f}  "
        f"SSIM {values['ssim']:.4f}  LPIPS {lpips}"
    )
delta_lpips = (
    "unavailable"
    if None in (macro[0]["lpips"], macro[1]["lpips"])
    else f"{macro[0]['lpips'] - macro[1]['lpips']:+.4f}"
)
print(
    f"FW-V minus FW-FZ:  dPSNR {macro[1]['psnr'] - macro[0]['psnr']:+.3f}  "
    f"dSSIM {macro[1]['ssim'] - macro[0]['ssim']:+.4f}  dLPIPS(FZ-V) {delta_lpips}"
)
for suite in sorted(loaded[0]["summary"]["per_suite"]):
    left = loaded[0]["summary"]["per_suite"][suite]
    right = loaded[1]["summary"]["per_suite"][suite]
    print(
        f"  {suite:<15} PSNR {left['psnr']:7.3f} -> {right['psnr']:7.3f}   "
        f"SSIM {left['ssim']:.4f} -> {right['ssim']:.4f}"
    )
PY
fi

echo
echo "DONE W0-Video-40 for: ${variants[*]}"
echo "  metrics: ${work}/<variant>/metrics.json"
echo "  the FW-V minus FW-FZ delta is an architecture comparison, not a video co-training"
echo "  ablation; see the guide section 3 note before quoting it."
