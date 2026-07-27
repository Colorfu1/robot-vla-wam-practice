#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: verify_training_image.sh --image=IMAGE_REF [options]

Options:
  --image=IMAGE_REF   Required local image tag.
  --repo=PATH         Repository to mount read-only (default: current repo).
  --require-gpus=N    Expected visible GPU count (default: nvidia-smi count).
  --skip-headless     Skip PushT, MuJoCo, LIBERO, and ManiSkill checks.
  --nccl              Run NCCL when at least two GPUs are required.
EOF
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image_ref=""
repo_path="$repo_root"
required_gpus=""
headless=true
run_nccl=false

for argument in "$@"; do
  case "$argument" in
    --image=*) image_ref="${argument#*=}" ;;
    --repo=*) repo_path="${argument#*=}" ;;
    --require-gpus=*) required_gpus="${argument#*=}" ;;
    --skip-headless) headless=false ;;
    --nccl) run_nccl=true ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $argument" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ -z "$image_ref" ]]; then
  echo "--image=IMAGE_REF is required" >&2
  exit 64
fi
repo_path="$(cd "$repo_path" && pwd)"
if [[ ! -f "$repo_path/pyproject.toml" ]]; then
  echo "Invalid repository path: $repo_path" >&2
  exit 66
fi
if [[ -z "$required_gpus" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    required_gpus="$(nvidia-smi -L | wc -l)"
  else
    required_gpus=0
  fi
fi
if ! [[ "$required_gpus" =~ ^[0-9]+$ ]]; then
  echo "--require-gpus must be a non-negative integer" >&2
  exit 64
fi
if [[ "$required_gpus" -eq 0 ]]; then
  echo "GPU verification requires at least one visible GPU" >&2
  exit 69
fi

verify_arguments=(
  --require-gpus "$required_gpus"
  --repo "/mnt/robot practice"
)
if [[ "$headless" == true ]]; then
  verify_arguments+=(--headless)
fi
if [[ "$run_nccl" == true ]]; then
  if [[ "$required_gpus" -lt 2 ]]; then
    echo "--nccl requires at least two GPUs" >&2
    exit 64
  fi
  verify_arguments+=(--nccl-world-size "$required_gpus")
fi

docker run --rm \
  --gpus all \
  --ipc=host \
  --mount "type=bind,src=$repo_path,dst=/mnt/old-practice,readonly" \
  --mount "type=bind,src=$repo_path,dst=/mnt/robot practice,readonly" \
  "$image_ref" \
  bash -c '
    set -euo pipefail
    robot-practice-init /mnt/old-practice
    ln -s "/mnt/robot practice" /tmp/robot-practice-repo-link
    robot-practice-init /tmp/robot-practice-repo-link
    exec robot-practice-verify "$@"
  ' bash "${verify_arguments[@]}"
