#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

mode=""
for argument in "$@"; do
  case "${argument}" in
    --mode=*) mode="${argument#*=}" ;;
    --help|-h)
      cat <<'EOF'
Usage:
  run_w0_video_generation_eval.sh --mode=prepare-vbench ...
  run_w0_video_generation_eval.sh --mode=prepare-w0 ...
  run_w0_video_generation_eval.sh --mode=check-protocol ...
  run_w0_video_generation_eval.sh --mode=generate ... [--num-gpus=N]
  run_w0_video_generation_eval.sh --mode=score ... [--num-gpus=N]
  run_w0_video_generation_eval.sh --mode=compare ...

Environment selection:
  W0_VIDEO_PREP_ENV    default: robot-practice
  W0_VIDEO_MODEL_ENV   default: fastwam-upstream (prepare-w0/generate/check-protocol)
  W0_VIDEO_SCORE_ENV   default: robot-practice (set to the env with VBench/LPIPS deps)
  W0_VIDEO_COMPARE_ENV default: robot-practice

This wrapper does not clone, install, download, or modify model/data artifacts.
All remaining arguments are forwarded to the repository-local evaluator. For W0-Video-40,
--num-gpus=N launches one isolated worker per visible GPU and merges the fixed-manifest shards.
EOF
      exit 0
      ;;
  esac
done

if [[ -z "${mode}" ]]; then
  echo "error: --mode is required" >&2
  exit 2
fi

case "${mode}" in
  prepare-vbench) env_name="${W0_VIDEO_PREP_ENV:-robot-practice}" ;;
  prepare-w0|generate|check-protocol) env_name="${W0_VIDEO_MODEL_ENV:-fastwam-upstream}" ;;
  score) env_name="${W0_VIDEO_SCORE_ENV:-robot-practice}" ;;
  compare) env_name="${W0_VIDEO_COMPARE_ENV:-robot-practice}" ;;
  *) echo "error: unsupported --mode=${mode}" >&2; exit 2 ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec conda run --no-capture-output -n "${env_name}" \
  python -m robot_practice.experiments.literature.w0_video_generation_eval "$@"
