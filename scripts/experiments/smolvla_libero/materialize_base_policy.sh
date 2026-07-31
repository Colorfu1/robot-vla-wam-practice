#!/usr/bin/env bash
#
# Materialize the cached SmolVLA base policy and its nested SmolVLM metadata as
# one self-contained local artifact. This command never downloads. The outer
# checkpoint already contains the VLM tensors, so only the nested VLM config,
# tokenizer, and processor files are copied (not its duplicate model weights).
#
# Usage:
#   bash scripts/experiments/smolvla_libero/materialize_base_policy.sh

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
destination="${practice_root}/policies/smolvla_or_pi0/smolvla_base"
vlm_destination="${destination}/vlm_assets"
base_repo_id="lerobot/smolvla_base"
vlm_repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
vlm_files=(
  added_tokens.json
  chat_template.json
  config.json
  generation_config.json
  merges.txt
  preprocessor_config.json
  processor_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
)

if ! conda_bin="$(command -v conda)"; then
  echo "error: conda is not available on PATH" >&2
  exit 69
fi

offline_environment=(
  env
  HF_HUB_OFFLINE=1
  TRANSFORMERS_OFFLINE=1
  HF_DATASETS_OFFLINE=1
)

resolve_cached_snapshot() {
  local repo_id="$1"
  SMOLVLA_REPO_ID="${repo_id}" \
    "${offline_environment[@]}" \
    "${conda_bin}" run --no-capture-output -n robot-practice python -c \
    'import os; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=os.environ["SMOLVLA_REPO_ID"], local_files_only=True))'
}

validate_base_policy() {
  local candidate="$1"
  SMOLVLA_BASE_CANDIDATE="${candidate}" \
    "${offline_environment[@]}" \
    "${conda_bin}" run --no-capture-output -n robot-practice python -c \
    'import os; from robot_practice.experiments.smolvla_libero.b0_training import validate_local_base_policy; validate_local_base_policy(os.environ["SMOLVLA_BASE_CANDIDATE"])'
}

validate_vlm_assets() {
  local candidate="$1"
  SMOLVLA_VLM_CANDIDATE="${candidate}" \
    "${offline_environment[@]}" \
    "${conda_bin}" run --no-capture-output -n robot-practice python -c \
    'import os; from robot_practice.experiments.smolvla_libero.b0_training import validate_local_vlm_assets; validate_local_vlm_assets(os.environ["SMOLVLA_VLM_CANDIDATE"])'
}

staging=""
cleanup() {
  if [[ -n "${staging}" && -d "${staging}" ]]; then
    case "${staging}" in
      "${destination}".incomplete.*|"${destination}"/.vlm_assets.incomplete.*)
        rm -rf -- "${staging}"
        ;;
      *)
        echo "error: refusing to clean unexpected staging path: ${staging}" >&2
        ;;
    esac
  fi
}
trap cleanup EXIT

if [[ -e "${vlm_destination}" ]]; then
  validate_base_policy "${destination}"
  echo "Local SmolVLA base policy is already complete: ${destination}"
  exit 0
fi

vlm_source="$(resolve_cached_snapshot "${vlm_repo_id}")"
if [[ ! -d "${vlm_source}" ]]; then
  echo "error: cached ${vlm_repo_id} snapshot is unavailable: ${vlm_source}" >&2
  exit 66
fi
for filename in "${vlm_files[@]}"; do
  if [[ ! -f "${vlm_source}/${filename}" ]]; then
    echo "error: cached VLM snapshot is missing ${filename}: ${vlm_source}" >&2
    exit 66
  fi
done

if [[ -e "${destination}" ]]; then
  if [[ ! -d "${destination}" ]]; then
    echo "error: base policy path is not a directory: ${destination}" >&2
    exit 17
  fi
  for filename in \
    config.json model.safetensors policy_preprocessor.json policy_postprocessor.json; do
    if [[ ! -f "${destination}/${filename}" ]]; then
      echo "error: existing base policy is missing ${filename}: ${destination}" >&2
      exit 66
    fi
  done
  if [[ -n "$(find "${destination}" -type l -print -quit)" ]]; then
    echo "error: existing base policy contains symlinks: ${destination}" >&2
    exit 66
  fi
  staging="$(mktemp -d "${destination}/.vlm_assets.incomplete.XXXXXX")"
  for filename in "${vlm_files[@]}"; do
    rsync -aL --copy-links "${vlm_source}/${filename}" "${staging}/${filename}"
  done
  validate_vlm_assets "${staging}"
  mv -- "${staging}" "${vlm_destination}"
  staging=""
else
  source_path="$(resolve_cached_snapshot "${base_repo_id}")"
  if [[ ! -d "${source_path}" ]]; then
    echo "error: cached ${base_repo_id} snapshot is unavailable: ${source_path}" >&2
    exit 66
  fi
  mkdir -p -- "$(dirname -- "${destination}")"
  staging="$(mktemp -d "${destination}.incomplete.XXXXXX")"
  rsync -aL --copy-links --info=progress2 "${source_path}/" "${staging}/"
  mkdir -- "${staging}/vlm_assets"
  for filename in "${vlm_files[@]}"; do
    rsync -aL --copy-links \
      "${vlm_source}/${filename}" "${staging}/vlm_assets/${filename}"
  done
  validate_base_policy "${staging}"
  mv -- "${staging}" "${destination}"
  staging=""
fi

validate_base_policy "${destination}"
echo "Materialized offline SmolVLA base policy: policies/smolvla_or_pi0/smolvla_base"
