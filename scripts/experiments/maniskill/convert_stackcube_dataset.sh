#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

replay_h5="outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5"
output_dir="datasets/lerobot/maniskill/stackcube_dcore_7d"
repo_id="local/maniskill_stackcube_dcore_7d"
fps="20"
max_episodes=""

usage() {
  cat <<'EOF'
Usage:
  convert_stackcube_dataset.sh [options]

Options:
  --replay-h5=PATH       StackCube D_core HDF5 input
  --output-dir=PATH      LeRobot dataset output directory
  --repo-id=ID           LeRobot local repo id
  --fps=N                Dataset FPS (default: 20)
  --max-episodes=N       Optional conversion smoke limit
EOF
}

for argument in "$@"; do
  case "${argument}" in
    --replay-h5=*) replay_h5="${argument#*=}" ;;
    --output-dir=*) output_dir="${argument#*=}" ;;
    --repo-id=*) repo_id="${argument#*=}" ;;
    --fps=*) fps="${argument#*=}" ;;
    --max-episodes=*) max_episodes="${argument#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: ${argument}" >&2; usage >&2; exit 2 ;;
  esac
done

command=(
  python -m robot_practice.experiments.maniskill.convert_stackcube
  --replay-h5 "${replay_h5}"
  --output-dir "${output_dir}"
  --repo-id "${repo_id}"
  --fps "${fps}"
  --verbose
)
if [[ -n "${max_episodes}" ]]; then
  command+=(--max-episodes "${max_episodes}")
fi

env PYTHONNOUSERSITE=1 conda run --no-capture-output -n robot-practice "${command[@]}"
