#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="${script_dir}"

# Override these when the image tag or visible GPU set differs on the target host.
image="${W0_FASTWAM_IMAGE:-practice:latest}"
gpus="${W0_DOCKER_GPUS:-all}"
container_name="${W0_DOCKER_NAME:-practice-w0-debug}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Start a background Docker container for the W0 Fast-WAM VSCode debug profile.

Usage:
  ./run_w0_fastwam_debug_docker.sh

Environment overrides:
  W0_FASTWAM_IMAGE  Docker image (default: practice:latest local image)
  W0_DOCKER_GPUS    Docker --gpus value (default: all)
  W0_DOCKER_NAME    Container name (default: practice-w0-debug)

The repository is mounted at /workspace. The container stays alive in the background
so VSCode can attach to it as practice-w0-debug.  On startup it idempotently applies
the tracked W0 Fast-WAM memory-loading patch to the mounted vendor checkout used by
the VSCode launch configuration. NVIDIA device nodes are also pinned into the Docker
device policy so a host systemd daemon-reload does not revoke CUDA from this container.
The image must already exist locally; this launcher never pulls it.
EOF
  exit 0
fi

mapfile -t available_gpu_devices < <(compgen -G '/dev/nvidia[0-9]*' | sort -V)
if ((${#available_gpu_devices[@]} == 0)) || [[ ! -c /dev/nvidiactl ]]; then
  echo "error: NVIDIA device nodes are unavailable on the host" >&2
  exit 3
fi

# Match Docker's common --gpus forms closely enough to pin only the selected GPU
# nodes. Control/capability nodes are shared by the selected devices.
normalized_gpus="${gpus//\"/}"
selected_gpu_devices=()
if [[ "${normalized_gpus}" == "all" ]]; then
  selected_gpu_devices=("${available_gpu_devices[@]}")
elif [[ "${normalized_gpus}" =~ ^[1-9][0-9]*$ ]]; then
  requested_count="${normalized_gpus}"
  if ((requested_count > ${#available_gpu_devices[@]})); then
    echo "error: requested ${requested_count} GPUs, found ${#available_gpu_devices[@]}" >&2
    exit 3
  fi
  selected_gpu_devices=("${available_gpu_devices[@]:0:requested_count}")
elif [[ "${normalized_gpus}" == device=* ]]; then
  IFS=',' read -r -a requested_devices <<< "${normalized_gpus#device=}"
  for requested_device in "${requested_devices[@]}"; do
    requested_device="${requested_device//[[:space:]]/}"
    if [[ "${requested_device}" =~ ^[0-9]+$ ]]; then
      nvidia_device="/dev/nvidia${requested_device}"
    elif [[ "${requested_device}" == GPU-* ]]; then
      nvidia_device=""
      while IFS=',' read -r gpu_index gpu_uuid; do
        gpu_index="${gpu_index//[[:space:]]/}"
        gpu_uuid="${gpu_uuid//[[:space:]]/}"
        if [[ "${gpu_uuid}" == "${requested_device}" ]]; then
          nvidia_device="/dev/nvidia${gpu_index}"
          break
        fi
      done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)
    else
      echo "error: unsupported W0_DOCKER_GPUS selector: ${requested_device}" >&2
      exit 3
    fi
    if [[ -z "${nvidia_device}" || ! -c "${nvidia_device}" ]]; then
      echo "error: selected NVIDIA device is unavailable: ${requested_device}" >&2
      exit 3
    fi
    selected_gpu_devices+=("${nvidia_device}")
  done
else
  echo "error: unsupported W0_DOCKER_GPUS value: ${gpus}" >&2
  echo 'hint: use all, a GPU count, or device=0,1' >&2
  exit 3
fi

nvidia_devices=("${selected_gpu_devices[@]}")
for nvidia_device in \
  /dev/nvidiactl \
  /dev/nvidia-modeset \
  /dev/nvidia-uvm \
  /dev/nvidia-uvm-tools \
  /dev/nvidia-caps/nvidia-cap*; do
  [[ -c "${nvidia_device}" ]] && nvidia_devices+=("${nvidia_device}")
done

container_bootstrap='set -euo pipefail
bash /workspace/scripts/experiments/literature/apply_w0_fastwam_memory_patch.sh \
  --vendor-root=/workspace/third_party/w0_fastwam
exec tail -f /dev/null'

docker_args=(
  run --rm -d --pull=never
  --name "${container_name}"
  --gpus "${gpus}"
)

for nvidia_device in "${nvidia_devices[@]}"; do
  docker_args+=(--device "${nvidia_device}:${nvidia_device}:rwm")
done

docker_args+=(
  --ipc=host
  --shm-size=32g
  -v "${practice_root}:/workspace"
  -w /workspace
  -e PYTHONNOUSERSITE=1
  -e PYTHONUNBUFFERED=1
  --entrypoint /bin/bash
  "${image}"
  -lc "${container_bootstrap}"
)

if (($# != 0)); then
  echo "error: this launcher only starts a background container; unexpected arguments: $*" >&2
  echo "hint: use --help for usage" >&2
  exit 2
fi

container_id="$(docker "${docker_args[@]}")"
printf 'container=%s name=%s status=background\n' "${container_id}" "${container_name}"
