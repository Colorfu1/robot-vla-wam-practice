#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
phase=""
config="${repo_root}/configs/experiments/rl/residual_robomimic_dppo_square.yaml"
output_root="${repo_root}/outputs/rl_residual_robomimic_20260901"
dummy=0
smoke=0
max_iterations=""
time_stages=0
residual_mode="learned"
seed=""
residual_root=""
manifest=""

while (($#)); do
  case "$1" in
    --phase=*) phase="${1#*=}" ;;
    --config=*) config="${1#*=}" ;;
    --output-root=*) output_root="${1#*=}" ;;
    --dummy) dummy=1 ;;
    --smoke) smoke=1 ;;
    --max-iterations=*) max_iterations="${1#*=}" ;;
    --time-stages) time_stages=1 ;;
    --residual-mode=*) residual_mode="${1#*=}" ;;
    --seed=*) seed="${1#*=}" ;;
    --residual-root=*) residual_root="${1#*=}" ;;
    --manifest=*) manifest="${1#*=}" ;;
    -h|--help)
      sed -n '1,80p' "$0"
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

[[ -n "$phase" ]] || { echo "--phase is required" >&2; exit 2; }
mkdir -p "$output_root"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${repo_root}/src:${repo_root}/third_party/dppo/dppo${PYTHONPATH:+:${PYTHONPATH}}"

args=("--phase=$phase" "--config=$config" "--output-root=$output_root" "--residual-mode=$residual_mode")
((dummy)) && args+=(--dummy)
((smoke)) && args+=(--smoke)
((time_stages)) && args+=(--time-stages)
[[ -n "$max_iterations" ]] && args+=("--max-iterations=$max_iterations")
[[ -n "$seed" ]] && args+=("--seed=$seed")
[[ -n "$residual_root" ]] && args+=("--residual-root=$residual_root")
[[ -n "$manifest" ]] && args+=("--manifest=$manifest")

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx dppo; then
  exec conda run --no-capture-output -n dppo python -m robot_practice.experiments.rl.robomimic_dppo_residual "${args[@]}"
fi
if [[ "${ROBOT_PRACTICE_ALLOW_HOST_PYTHON:-0}" == 1 ]]; then
  exec python -m robot_practice.experiments.rl.robomimic_dppo_residual "${args[@]}"
fi
echo "dppo conda environment is missing; run this entrypoint inside the DPPO Docker image" >&2
exit 3
