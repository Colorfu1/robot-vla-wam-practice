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
  run_w0_video_generation_eval_docker.sh --mode=score ...

The score path automatically mounts:
  repository root -> /opt/robot-practice
  models/         -> /opt/robot-practice/models:ro

Environment:
  W0_VIDEO_EVAL_IMAGE   Docker image; otherwise auto-detect *w0-fastwam*videoeval
  W0_VIDEO_DOCKER_GPUS  Docker --gpus value; default: all

For VBench-I2V-mini-21, pass --num-gpus=N to launch one score worker per visible GPU.
EOF
      exit 0
      ;;
  esac
done

if [[ "${mode}" != "score" ]]; then
  echo "error: this Docker wrapper currently supports --mode=score only" >&2
  exit 2
fi

if [[ ! -d "models" ]]; then
  echo "error: missing local DreamSim cache directory: models/" >&2
  exit 2
fi

image="${W0_VIDEO_EVAL_IMAGE:-}"
if [[ -z "${image}" ]]; then
  image="$(docker image ls --format '{{.Repository}}:{{.Tag}}' | rg 'w0-fastwam.*videoeval' | head -1 || true)"
fi
if [[ -z "${image}" ]]; then
  echo "error: no W0 video-eval Docker image found; set W0_VIDEO_EVAL_IMAGE" >&2
  exit 2
fi

docker_args=(
  run --rm
  --gpus "${W0_VIDEO_DOCKER_GPUS:-all}"
  -v "${practice_root}:/opt/robot-practice"
  -v "${practice_root}/models:/opt/robot-practice/models:ro"
  -w /opt/robot-practice
  -e "W0_VIDEO_PREP_ENV=${W0_VIDEO_PREP_ENV:-robot-practice}"
  -e "W0_VIDEO_MODEL_ENV=${W0_VIDEO_MODEL_ENV:-fastwam-upstream}"
  -e "W0_VIDEO_SCORE_ENV=${W0_VIDEO_SCORE_ENV:-robot-practice}"
  -e PYTHONNOUSERSITE=1
  -e PYTHONUNBUFFERED=1
  --entrypoint /bin/bash
  "${image}"
  scripts/experiments/literature/run_w0_video_generation_eval.sh
)

exec docker "${docker_args[@]}" "$@"
