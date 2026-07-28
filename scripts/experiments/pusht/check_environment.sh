#!/usr/bin/env bash
# Purpose: run Phase-0 PushT interface, explicit-noise, and fixed-seed replay checks.
# Entry point: python -m robot_practice.experiments.pusht_diffusion.interface_checks
# Arguments: --device=cpu|cuda, --output-dir=DIR, --overwrite, --print-command.
# Example: bash scripts/experiments/pusht/check_environment.sh
# Output: outputs/pusht_diffusion/metadata (existing output is refused unless --overwrite).

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: check_environment.sh [--device=cpu|cuda] [--output-dir=DIR] [--overwrite] [--print-command]
EOF
}

device="cuda"
output_dir="outputs/pusht_diffusion/metadata"
overwrite=false
print_command=false
seen=""
for argument in "$@"; do
  key="${argument%%=*}"
  if [[ " ${seen} " == *" ${key} "* ]]; then
    echo "error: duplicate argument: ${key}" >&2
    exit 64
  fi
  case "${argument}" in
    --device=cpu|--device=cuda) device="${argument#*=}"; seen+=" ${key}" ;;
    --output-dir=*) output_dir="${argument#*=}"; seen+=" ${key}" ;;
    --overwrite) overwrite=true; seen+=" ${key}" ;;
    --print-command) print_command=true; seen+=" ${key}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: ${argument}" >&2; usage >&2; exit 64 ;;
  esac
done
[[ -n "${output_dir}" ]] || { echo "error: --output-dir must not be empty" >&2; exit 64; }

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
[[ "${output_dir}" == /* ]] || output_dir="${practice_root}/${output_dir}"
command=(
  conda run --no-capture-output -n robot-practice
  python -m robot_practice.experiments.pusht_diffusion.interface_checks
  "--device=${device}"
  "--dataset-root=${practice_root}/datasets/lerobot/lerobot/pusht"
  "--output-dir=${output_dir}"
)
[[ "${overwrite}" == true ]] && command+=(--overwrite)
if [[ "${print_command}" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
