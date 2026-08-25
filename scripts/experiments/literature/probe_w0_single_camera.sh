#!/usr/bin/env bash
# Single-camera diagnostic for the W0 video-generation comparison.
#
# Why this exists as a script: the full generate invocation is far too long to paste into a
# terminal in one piece, and a paste that inserts newlines runs the tail as separate commands
# ("--single-camera=front: command not found"). Everything long lives here instead.
#
# What it measures: the frozen protocol conditions on front and wrist concatenated into one
# 448x224 image. FW-FZ's video expert carries unmodified Wan2.2 weights, and no natural video
# contains a side-by-side camera pair, so that layout is out of distribution for it. Running
# both variants on a single 224x224 view separates "cannot predict this domain" from "cannot
# read this layout".
#
# Caveat worth keeping in mind when reading the result: a single view also drops the token
# count per frame from 14x28=392 to 14x14=196, so this changes resolution as well as layout.
# FW-V is included precisely because it was trained on the concatenated layout too, so it is
# the control for "is a single view simply out of distribution for both".
#
# Usage:
#   bash scripts/experiments/literature/probe_w0_single_camera.sh            # front
#   bash scripts/experiments/literature/probe_w0_single_camera.sh wrist

set -euo pipefail

camera="${1:-front}"
case "${camera}" in
  front | wrist) ;;
  *)
    echo "usage: $0 [front|wrist]" >&2
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)"
cd "${repo_root}"

video_root="outputs/literature/w0_fastwam/video_generation"
probe_root="${video_root}/probe3"
manifest="${probe_root}/manifest.json"
full_manifest="${video_root}/w0_video_40/manifest.json"
runner="scripts/experiments/literature/run_w0_video_generation_eval.sh"
compare="scripts/experiments/literature/compare_w0_video_frames.py"

fw_v_checkpoint="outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt"
fw_fz_checkpoint="outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt"

# The 3-sample probe manifest is derived from the frozen 40-sample one, so the samples, seeds
# and frame windows are identical to the formal run rather than a fresh selection.
if [[ ! -f "${manifest}" ]]; then
  if [[ ! -f "${full_manifest}" ]]; then
    echo "missing ${full_manifest}; run --mode=prepare-w0 first" >&2
    exit 1
  fi
  mkdir -p "${probe_root}"
  python - "${full_manifest}" "${manifest}" <<'PY'
import json
import sys

source, target = sys.argv[1], sys.argv[2]
payload = json.load(open(source, encoding="utf-8"))
wanted = ("libero_spatial_task00", "libero_object_task00", "libero_10_task00")
samples = [item for item in payload["samples"] if item["sample_id"] in wanted]
if len(samples) != len(wanted):
    raise SystemExit(f"expected {len(wanted)} samples, found {len(samples)}")
payload["samples"] = samples
json.dump(payload, open(target, "w", encoding="utf-8"), indent=2)
print(f"wrote {target} with {len(samples)} samples")
PY
fi

for variant in fw_fz fw_v; do
  if [[ "${variant}" == "fw_fz" ]]; then
    checkpoint="${fw_fz_checkpoint}"
  else
    checkpoint="${fw_v_checkpoint}"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
  output_root="${probe_root}/${variant}_${camera}"
  echo "=== generate ${variant} camera=${camera} -> ${output_root}"
  bash "${runner}" \
    --mode=generate \
    --dataset=w0-video-40 \
    --variant="${variant}" \
    --single-camera="${camera}" \
    --model-cache-root=pretrained/w0_fastwam \
    --checkpoint="${checkpoint}" \
    --manifest="${manifest}" \
    --output-root="${output_root}"
done

sheet="${probe_root}/compare_${camera}.png"
echo "=== compare -> ${sheet}"
python "${compare}" \
  --manifest "${manifest}" \
  --camera "${camera}" \
  --videos "fz_${camera}=${probe_root}/fw_fz_${camera}/videos" \
  --videos "v_${camera}=${probe_root}/fw_v_${camera}/videos" \
  --output "${sheet}" \
  --limit 3

echo
echo "DONE single-camera probe camera=${camera}"
echo "  sheet: ${sheet}"
echo "  rows are GT / fz_${camera} / v_${camera}, columns are frame 0..8"
