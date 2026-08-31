#!/usr/bin/env bash
# Run the VBench-I2V half end to end, refusing to proceed on a downgraded protocol.
#
# The order matters and is easy to get wrong, because `generate` reads its protocol from the
# *manifest*, not from protocol.json:
#
#   check-protocol   -> protocol.json      decides wan22-ti2v-compatible vs fastwam-native
#   prepare-vbench   -> manifest.json      embeds that protocol via --protocol
#   generate         -> videos             uses manifest["protocol"]
#   score            -> metrics.json
#
# Skipping either of the first two steps reproduces the 2026-08-25 mistake: the manifest keeps
# the old fastwam-native protocol, `generate` skips the videos that already exist, and the run
# silently re-scores 448x224 / 9-frame videos. Both variants then sit at an operating point the
# frozen one was never trained for, so the result cannot answer the preservation question.
#
# Cost warning: the default `wan22-official` profile generates 1280x720 x 81 frames at 50
# steps and 24 FPS. The `fastwam-resolution-only` profile generates 448x224 with the same
# 81-frame, 24-FPS, 50-step Wan2.2 schedule; resolution is the only profile difference. Run
# `probe` first and read the wall clock before committing to 42 generations. This wrapper still
# uses VBench images/prompts; use run_w0_libero_wan.sh for the LIBERO image+command experiment.
#
# Usage:
#   bash scripts/experiments/literature/run_w0_vbench_native.sh                         # 1 sample, timed
#   bash scripts/experiments/literature/run_w0_vbench_native.sh mini21 8 wan22-official
#   bash scripts/experiments/literature/run_w0_vbench_native.sh mini21 8 fastwam-resolution-only
#   bash scripts/experiments/literature/run_w0_vbench_native.sh vbench-full 8 wan22-official
#   W0_ACCEPT_FALLBACK_PROTOCOL=1 bash ... mini21 8 wan22-official  # knowingly degraded
#
# mini21 is a screening subset: 7 samples per dimension, which makes a +-0.2857 delta exactly
# two videos on a binary metric. vbench-full is the set that can actually support a conclusion
# (109-763 samples per dimension) and is 1118 prompts per variant, so 2236 generations. Which
# one is affordable is a wall-clock question, not a taste question -- run probe first.

set -euo pipefail

scope="${1:-probe}"
num_gpus="${2:-1}"
operating_point="${3:-wan22-official}"
case "${scope}" in
  probe | mini21 | vbench-full) ;;
  full)
    echo "ambiguous scope 'full': use 'mini21' for the 21-prompt subset or 'vbench-full' for" >&2
    echo "all 1118 VBench-I2V prompts (2236 generations across both variants)" >&2
    exit 2
    ;;
  *)
    echo "usage: $0 [probe|mini21|vbench-full] [num_gpus] [operating_point]" >&2
    exit 2
    ;;
esac
case "${operating_point}" in
  wan22-official | fastwam-resolution-only) ;;
  front-square-hires)
    # 896x896 exists for the single 224x224 LIBERO front view. VBench-I2V ships 16-9 crops, so a
    # square target would discard 43.8% of every conditioning image -- not a resolution ablation.
    echo "front-square-hires is for the square LIBERO front view, not VBench's 16-9 images" >&2
    exit 2
    ;;
  *)
    echo "unsupported operating point '${operating_point}'; use wan22-official or" >&2
    echo "fastwam-resolution-only" >&2
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
vbench_root="outputs/literature/w0_fastwam/video_generation/vbench"
# One protocol file per operating point. They are both wan22-ti2v-compatible, so a single shared
# file would leave protocol.json describing whichever arm ran last while the videos of the other
# arm sit next to it.
if [[ "${operating_point}" == "wan22-official" ]]; then
  protocol="${vbench_root}/protocol.json"
else
  protocol="${vbench_root}/protocol-${operating_point}.json"
fi
fw_v_checkpoint="outputs/literature/w0_fastwam/causal_10ep/with_video/checkpoints/weights/step_021700.pt"
fw_fz_checkpoint="outputs/literature/w0_fastwam/causal_10ep_frozen/no_video/checkpoints/weights/step_021700.pt"
wan_model_root="pretrained/w0_fastwam/models/Wan-AI/Wan2.2-TI2V-5B"

for path in "${fw_v_checkpoint}" "${fw_fz_checkpoint}"; do
  if [[ ! -f "${path}" ]]; then
    echo "missing checkpoint: ${path}" >&2
    exit 1
  fi
done
if [[ ! -f "${vbench_root}/assets.json" ]]; then
  echo "missing ${vbench_root}/assets.json; run --mode=prepare-vbench first (guide section 1)" >&2
  exit 1
fi

# Validate the GPU count up front. _require_multi_gpu_devices only runs once generation starts,
# which is after check-protocol has loaded two 12 GB checkpoints and prepare-vbench has written
# a manifest -- a slow way to learn that --num-gpus exceeds what is visible.
if [[ "${num_gpus}" -gt 1 ]]; then
  visible="$(python - <<'PY'
import os

configured = os.environ.get("CUDA_VISIBLE_DEVICES")
if configured is None:
    try:
        import torch

        print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
    except ImportError:
        print(-1)
else:
    tokens = [token.strip() for token in configured.split(",") if token.strip()]
    print(0 if tokens in ([], ["-1"]) else len(tokens))
PY
)"
  if [[ "${visible}" == "-1" ]]; then
    echo "cannot count CUDA devices: torch is not importable in this shell's python" >&2
    exit 1
  fi
  if [[ "${visible}" -lt "${num_gpus}" ]]; then
    echo "num_gpus=${num_gpus} requested but only ${visible} CUDA device(s) visible" >&2
    echo "  set CUDA_VISIBLE_DEVICES, or lower num_gpus" >&2
    exit 1
  fi
  echo "    multi-GPU: ${num_gpus} of ${visible} visible devices, applied to generate and score"
fi

echo "=== 1/5 check-protocol"
bash "${runner}" \
  --mode=check-protocol \
  --wan-root=third_party/wan22 \
  --wan-model-root="${wan_model_root}" \
  --fw-v="${fw_v_checkpoint}" \
  --fw-fz="${fw_fz_checkpoint}" \
  --operating-point="${operating_point}" \
  --output="${protocol}"

kind="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['kind'])" "${protocol}")"
echo "    protocol kind: ${kind}"
if [[ "${kind}" != "wan22-ti2v-compatible" ]]; then
  python - "${protocol}" <<'PY'
import json
import sys

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = protocol.get("wan22_candidate", {})
print(f"    fallback reason: {candidate.get('reason')}")
print(f"    smoke: {candidate.get('smoke')}")
for name, check in (protocol.get("checks") or {}).items():
    print(f"    {name}: strict_load={check.get('strict_load')} {check.get('reason', '')}")
PY
  if [[ "${W0_ACCEPT_FALLBACK_PROTOCOL:-0}" == "0" ]]; then
    cat >&2 <<'EOF'

Refusing to continue on the fastwam-native protocol.

Both variants would run at 448x224 / 9 frames, which is the operating point the frozen video
expert was never trained for, so the VBench half would measure "who works at the FastWAM
operating point" rather than how much general video ability survived fine-tuning -- the same
confound as the W0-Video-40 half, with only the image domain changed.

Fix the reason printed above, or set W0_ACCEPT_FALLBACK_PROTOCOL=1 to record a deliberately
degraded protocol.
EOF
    exit 1
  fi
  echo "    W0_ACCEPT_FALLBACK_PROTOCOL=1 set; continuing on a degraded protocol"
fi

if [[ "${operating_point}" == "fastwam-resolution-only" && "${kind}" != "wan22-ti2v-compatible" ]]; then
  echo "fastwam-resolution-only requires the official Wan2.2 generator; got ${kind}" >&2
  exit 1
fi

# Keep protocol variants in separate trees. The old fastwam-native videos are not reusable at a
# different generator, resolution, frame count, or schedule, and overwriting them would destroy
# the evidence for why that run was invalid.
suffix="$([[ "${kind}" == "wan22-ti2v-compatible" ]] && echo native || echo fallback)"
if [[ "${operating_point}" == "fastwam-resolution-only" ]]; then
  suffix="${suffix}-fastwam-point"
fi
if [[ "${scope}" == "vbench-full" ]]; then
  selection="full"
  dataset="vbench-i2v-full"
  work="${vbench_root}/${suffix}-full"
else
  selection="mini-21"
  dataset="vbench-mini-21"
  work="${vbench_root}/${suffix}"
fi
manifest="${work}/manifest.json"
mkdir -p "${work}"

echo "=== 2/5 prepare-vbench (${selection}, protocol ${kind})"
bash "${runner}" \
  --mode=prepare-vbench \
  --vbench-root=third_party/vbench \
  --cache-root=pretrained/vbench_i2v \
  --selection="${selection}" \
  --selection-seed=20260824 \
  --base-seed=52000 \
  --protocol="${protocol}" \
  --output="${manifest}"

sample_total="$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['samples']))" "${manifest}")"
echo "    manifest samples: ${sample_total} (x2 variants = $((sample_total * 2)) generations)"

if [[ "${scope}" == "probe" ]]; then
  python - "${manifest}" <<'PY'
import json
import sys

path = sys.argv[1]
manifest = json.load(open(path, encoding="utf-8"))
manifest["samples"] = manifest["samples"][:1]
manifest["selection"] = f"{manifest.get('selection', 'mini-21')}-1sample-probe"
json.dump(manifest, open(path, "w", encoding="utf-8"), indent=2)
print(f"    probe mode: manifest reduced to 1 sample ({manifest['samples'][0]['sample_id']})")
PY
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
  echo "=== 3/5 generate ${variant} (${scope}, ${kind})"
  started="$(date +%s)"
  bash "${runner}" \
    --mode=generate \
    --dataset="${dataset}" \
    --variant="${variant}" \
    --model-cache-root=pretrained/w0_fastwam \
    --vbench-root=third_party/vbench \
    --wan-root=third_party/wan22 \
    --wan-model-root="${wan_model_root}" \
    --checkpoint="${checkpoint}" \
    --manifest="${manifest}" \
    --output-root="${work}/${variant}" \
    "${gpu_args[@]}"
  echo "    ${variant} generate wall clock: $(( $(date +%s) - started ))s"
done

if [[ "${scope}" == "probe" ]]; then
  echo
  echo "PROBE DONE. Multiply the wall clock above by 21 for mini21, or by 1118 for vbench-full."
  echo "  videos: ${work}/{fw_fz,fw_v}/videos"
  exit 0
fi

for variant in fw_fz fw_v; do
  echo "=== 4/5 score ${variant}"
  bash "${runner}" \
    --mode=score \
    --dataset="${dataset}" \
    --variant="${variant}" \
    --vbench-root=third_party/vbench \
    --cache-root=pretrained/vbench_i2v \
    --manifest="${manifest}" \
    --videos="${work}/${variant}/videos" \
    --output="${work}/${variant}/metrics.json" \
    "${gpu_args[@]}"
done

echo "=== 5/5 paired read-out"
W0_VBENCH_DIR="${work}" python - <<'PY'
import json
import os

root = os.environ["W0_VBENCH_DIR"]
loaded = {
    variant: json.load(open(f"{root}/{variant}/metrics.json", encoding="utf-8"))
    for variant in ("fw_fz", "fw_v")
}
left = loaded["fw_fz"]["vbench"]["dimensions"]
right = loaded["fw_v"]["vbench"]["dimensions"]
print("%-24s %8s %8s %9s %5s" % ("dimension", "FW-FZ", "FW-V", "V - FZ", "n"))
for name in left:
    print(
        "%-24s %8.4f %8.4f %+9.4f %5d"
        % (name, left[name]["score"], right[name]["score"],
           right[name]["score"] - left[name]["score"], left[name]["sample_count"])
    )
print()
print(f"protocol: {loaded['fw_v']['protocol']['kind']}")
print("A per-dimension n of 7 makes a +-0.2857 delta exactly two videos on a binary metric;")
print("read those as noise, not as a result.")
PY
