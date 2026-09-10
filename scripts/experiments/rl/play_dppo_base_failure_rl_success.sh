#!/usr/bin/env bash
# Select paired DPPO episodes and load a folder of Rerun recordings.
#
# Pair mode (default) creates two lightweight symlink folders containing only
# Base-failure/RL-success pairs. Folder mode loads every .rrd in one supplied
# folder into one Rerun viewer, so the viewer's recording selector can switch
# between scenes.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd -P)"
cd -- "${practice_root}"

export PYTHONNOUSERSITE=1

exp_root="${EXP_ROOT:-outputs/rl_residual_robomimic_20260901}"
vis_root="${VIS_ROOT:-${exp_root}/official_dppo_visualization}"
base_root="${BASE_ROOT:-${vis_root}/rerun_base}"
rl_root="${RL_ROOT:-${vis_root}/rerun_rl}"
selection_root="${SELECTION_ROOT:-${vis_root}/base_failure_rl_success}"
folder=""
pair_index=""
max_pairs=""
rerun_bin="${RERUN_BIN:-auto}"
folder_port="${RERUN_PORT:-9877}"
base_port="${BASE_RERUN_PORT:-9877}"
rl_port="${RL_RERUN_PORT:-9878}"
folder_port_explicit=0
base_port_explicit=0
rl_port_explicit=0
list_only=0
prepare_only=0
replace_selection=0
detach=0
include_merged=0

usage() {
  cat <<'EOF'
Usage:
  # Pair mode: select Base failure -> RL success, create two folders, open both
  play_dppo_base_failure_rl_success.sh [options]

  # Folder mode: load every .rrd in one folder into one Rerun viewer
  play_dppo_base_failure_rl_success.sh --folder PATH [options]

Pair mode options:
  --base-root PATH       Base Rerun directory
  --rl-root PATH         RL Rerun directory
  --selection-root PATH  output root for filtered base/rl folders
  --pair-index N         keep only one zero-based candidate
  --max-pairs N          keep the first N candidates (default: all)
  --replace-selection    remove existing generated symlink entries first
  --prepare-only         create/check filtered folders without opening viewers

Folder mode options:
  --folder PATH          load all .rrd files directly below this folder
  --port N               gRPC port for this folder's new Rerun viewer (default: 9877)
  --include-merged       also load the combined archive, if present
  --list-only            list candidates/files without launching viewers

Common options:
  --rerun PATH           rerun or robot-rerun executable (default: auto)
  --base-port N          pair mode Base viewer port (default: 9877)
  --rl-port N             pair mode RL viewer port (default: 9878)
  --detach               return after opening viewer windows
  -h, --help             show this help

Environment overrides:
  EXP_ROOT, VIS_ROOT, BASE_ROOT, RL_ROOT, SELECTION_ROOT, RERUN_BIN,
  RERUN_PORT, BASE_RERUN_PORT, RL_RERUN_PORT
EOF
}

while (($# > 0)); do
  case "$1" in
    --base-root=*)
      base_root="${1#*=}"
      shift
      ;;
    --base-root)
      base_root="$2"
      shift 2
      ;;
    --rl-root=*)
      rl_root="${1#*=}"
      shift
      ;;
    --rl-root)
      rl_root="$2"
      shift 2
      ;;
    --selection-root=*)
      selection_root="${1#*=}"
      shift
      ;;
    --selection-root)
      selection_root="$2"
      shift 2
      ;;
    --folder=*)
      folder="${1#*=}"
      shift
      ;;
    --folder)
      folder="$2"
      shift 2
      ;;
    --pair-index=*)
      pair_index="${1#*=}"
      shift
      ;;
    --pair-index)
      pair_index="$2"
      shift 2
      ;;
    --max-pairs=*)
      max_pairs="${1#*=}"
      shift
      ;;
    --max-pairs)
      max_pairs="$2"
      shift 2
      ;;
    --rerun=*)
      rerun_bin="${1#*=}"
      shift
      ;;
    --rerun)
      rerun_bin="$2"
      shift 2
      ;;
    --port=*)
      folder_port="${1#*=}"
      folder_port_explicit=1
      shift
      ;;
    --port)
      folder_port="$2"
      folder_port_explicit=1
      shift 2
      ;;
    --base-port=*)
      base_port="${1#*=}"
      base_port_explicit=1
      shift
      ;;
    --base-port)
      base_port="$2"
      base_port_explicit=1
      shift 2
      ;;
    --rl-port=*)
      rl_port="${1#*=}"
      rl_port_explicit=1
      shift
      ;;
    --rl-port)
      rl_port="$2"
      rl_port_explicit=1
      shift 2
      ;;
    --list-only)
      list_only=1
      shift
      ;;
    --prepare-only)
      prepare_only=1
      shift
      ;;
    --replace-selection)
      replace_selection=1
      shift
      ;;
    --include-merged)
      include_merged=1
      shift
      ;;
    --detach)
      detach=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

validate_port() {
  local option_name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]] || ((value > 65535)); then
    echo "error: $option_name must be an integer in [1, 65535]: $value" >&2
    exit 2
  fi
}

validate_port '--port/ RERUN_PORT' "$folder_port"
validate_port '--base-port/ BASE_RERUN_PORT' "$base_port"
validate_port '--rl-port/ RL_RERUN_PORT' "$rl_port"

port_is_busy() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -H -ltn 2>/dev/null | awk -v suffix=":$port" '$4 ~ (suffix "$") {found=1} END {exit !found}'
    return $?
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

find_free_port() {
  local candidate="$1"
  while port_is_busy "$candidate"; do
    candidate=$((candidate + 1))
    if ((candidate > 65535)); then
      echo "error: no free TCP port at or above $1" >&2
      exit 69
    fi
  done
  printf '%s\n' "$candidate"
}

ensure_port_available() {
  local viewer_name="$1"
  local port="$2"
  if port_is_busy "$port"; then
    echo "error: $viewer_name port=$port is already listening; choose another port" >&2
    exit 69
  fi
}

if [[ -n "$pair_index" ]] && ! [[ "$pair_index" =~ ^[0-9]+$ ]]; then
  echo "error: --pair-index must be a non-negative integer: $pair_index" >&2
  exit 2
fi
if [[ -n "$max_pairs" ]] && ! [[ "$max_pairs" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --max-pairs must be a positive integer: $max_pairs" >&2
  exit 2
fi

resolve_path() {
  local value="$1"
  if [[ "$value" == /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s/%s\n' "$practice_root" "$value"
  fi
}

select_rerun_binary() {
  if [[ "$rerun_bin" == auto ]]; then
    if command -v robot-rerun >/dev/null 2>&1; then
      rerun_bin=robot-rerun
    elif command -v rerun >/dev/null 2>&1; then
      rerun_bin=rerun
    else
      echo "error: neither robot-rerun nor rerun is installed" >&2
      exit 69
    fi
  elif ! command -v "$rerun_bin" >/dev/null 2>&1; then
    echo "error: Rerun executable is unavailable: $rerun_bin" >&2
    exit 69
  fi
}

load_folder() {
  local folder_arg="$1"
  local folder_root
  if ! folder_root="$(cd -- "$folder_arg" 2>/dev/null && pwd -P)"; then
    echo "error: Rerun folder is missing: $folder_arg" >&2
    exit 2
  fi

  mapfile -t recordings < <(
    find -L "$folder_root" -mindepth 1 -maxdepth 1 -type f -name '*.rrd' -print | sort
  )
  if ((include_merged == 0)); then
    filtered_recordings=()
    for recording in "${recordings[@]}"; do
      [[ "$(basename -- "$recording")" == "robomimic_dppo_visualization.rrd" ]] && continue
      filtered_recordings+=("$recording")
    done
    recordings=("${filtered_recordings[@]}")
  fi
  if ((${#recordings[@]} == 0)); then
    echo "error: no individual .rrd recordings found in: $folder_root" >&2
    exit 3
  fi

  echo "folder=$folder_root"
  echo "recordings=${#recordings[@]}"
  if ((list_only)); then
    printf '%s\n' "${recordings[@]}"
    return 0
  fi
  if ((prepare_only)); then
    echo "prepare-only: not launching viewer"
    return 0
  fi

  if ((folder_port_explicit == 0)); then
    folder_port="$(find_free_port "$folder_port")"
  else
    ensure_port_available "Rerun" "$folder_port"
  fi
  select_rerun_binary
  echo "Launching Rerun with ${#recordings[@]} recordings on port=$folder_port: $rerun_bin"
  "$rerun_bin" --port "$folder_port" "${recordings[@]}" &
  viewer_pid=$!
  if ((detach)); then
    echo "viewer_started pid=$viewer_pid"
    return 0
  fi
  set +e
  wait "$viewer_pid"
  viewer_status=$?
  set -e
  if ((viewer_status != 0)); then
    echo "warning: viewer exit status=$viewer_status" >&2
  fi
}

if [[ -n "$folder" ]]; then
  load_folder "$folder"
  exit 0
fi

base_root_arg="$base_root"
if ! base_root="$(cd -- "$base_root_arg" 2>/dev/null && pwd -P)"; then
  echo "error: Base Rerun directory is missing: $base_root_arg" >&2
  exit 2
fi
rl_root_arg="$rl_root"
if ! rl_root="$(cd -- "$rl_root_arg" 2>/dev/null && pwd -P)"; then
  echo "error: RL Rerun directory is missing: $rl_root_arg" >&2
  exit 2
fi
base_manifest="$base_root/visualization_manifest.json"
rl_manifest="$rl_root/visualization_manifest.json"
test -s "$base_manifest" || {
  echo "error: Base visualization manifest is missing: $base_manifest" >&2
  exit 2
}
test -s "$rl_manifest" || {
  echo "error: RL visualization manifest is missing: $rl_manifest" >&2
  exit 2
}
command -v jq >/dev/null 2>&1 || {
  echo "error: jq is required to pair Base/RL visualization manifests" >&2
  exit 69
}

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/dppo-base-failure-rl-success.XXXXXX")"
candidate_tsv="$tmp_root/candidates.tsv"
cleanup() {
  if [[ -n "${tmp_root:-}" && -d "$tmp_root" ]]; then
    rm -f "$candidate_tsv"
    rmdir "$tmp_root" 2>/dev/null || true
  fi
}
trap cleanup EXIT

jq -nr \
  --slurpfile base "$base_manifest" \
  --slurpfile rl "$rl_manifest" '
  ($base[0].episodes // [])
  | map({key: ("\(.policy_seed):\(.episode_id)"), value: .})
  | from_entries as $base_by_key
  | ($rl[0].episodes // [])
  | map({key: ("\(.policy_seed):\(.episode_id)"), value: .})
  | from_entries as $rl_by_key
  | $base_by_key
  | to_entries[]
  | .key as $key
  | .value as $base_episode
  | ($rl_by_key[$key] // null) as $rl_episode
  | select($rl_episode != null)
  | select($base_episode.success == false and $rl_episode.success == true)
  | [
      $key,
      ($base_episode.policy_seed | tostring),
      ($base_episode.episode_id | tostring),
      ($base_episode.recording // ""),
      ($rl_episode.recording // "")
    ]
  | @tsv
' > "$candidate_tsv"

mapfile -t candidates < "$candidate_tsv"
candidate_count="${#candidates[@]}"
if ((candidate_count == 0)); then
  echo "No paired episodes found where Base fails and RL succeeds."
  exit 4
fi

echo "Base failure -> RL success candidates: $candidate_count"
for index in "${!candidates[@]}"; do
  IFS=$'\t' read -r key seed episode base_recording rl_recording \
    <<< "${candidates[$index]}"
  printf '  [%d] seed=%s episode=%04d\n' "$index" "$seed" "$episode"
done

if ((list_only)); then
  exit 0
fi

selected_indices=()
if [[ -n "$pair_index" ]]; then
  if ((pair_index >= candidate_count)); then
    echo "error: --pair-index=$pair_index is out of range [0, $((candidate_count - 1))]" >&2
    exit 2
  fi
  selected_indices=("$pair_index")
else
  selected_count="$candidate_count"
  if [[ -n "$max_pairs" ]] && ((max_pairs < selected_count)); then
    selected_count="$max_pairs"
  fi
  for ((index = 0; index < selected_count; index++)); do
    selected_indices+=("$index")
  done
fi

selection_root="$(resolve_path "$selection_root")"
selected_base="$selection_root/base"
selected_rl="$selection_root/rl"

prepare_selection_dir() {
  local directory="$1"
  if [[ -e "$directory" && ! -d "$directory" ]]; then
    echo "error: selection path is not a directory: $directory" >&2
    exit 3
  fi
  mkdir -p "$directory"
  mapfile -t entries < <(find "$directory" -mindepth 1 -maxdepth 1 -print)
  if ((${#entries[@]} == 0)); then
    return 0
  fi
  if ((replace_selection == 0)); then
    echo "error: selection folder is non-empty: $directory" >&2
    echo "       use --replace-selection or choose a new --selection-root" >&2
    exit 3
  fi
  for entry in "${entries[@]}"; do
    if [[ ! -L "$entry" ]]; then
      echo "error: refusing to remove non-symlink selection entry: $entry" >&2
      exit 3
    fi
    rm -f -- "$entry"
  done
}

base_rrds=()
rl_rrds=()
prepare_selection_dir "$selected_base"
prepare_selection_dir "$selected_rl"
for ordinal in "${!selected_indices[@]}"; do
  index="${selected_indices[$ordinal]}"
  IFS=$'\t' read -r key seed episode base_recording rl_recording \
    <<< "${candidates[$index]}"
  if [[ -z "$base_recording" || -z "$rl_recording" ]]; then
    echo "error: candidate [$index] has no Rerun recording path in the manifest" >&2
    exit 3
  fi
  base_recording="$(resolve_path "$base_recording")"
  rl_recording="$(resolve_path "$rl_recording")"
  test -s "$base_recording" || {
    echo "error: Base Rerun recording is missing: $base_recording" >&2
    exit 3
  }
  test -s "$rl_recording" || {
    echo "error: RL Rerun recording is missing: $rl_recording" >&2
    exit 3
  }
  link_stem="$(printf '%03d_seed_%s_episode_%04d' "$ordinal" "$seed" "$episode")"
  base_link="$selected_base/${link_stem}_base_failure.rrd"
  rl_link="$selected_rl/${link_stem}_rl_success.rrd"
  ln -s -- "$base_recording" "$base_link"
  ln -s -- "$rl_recording" "$rl_link"
  base_rrds+=("$base_link")
  rl_rrds+=("$rl_link")
done

echo "selection_root=$selection_root"
echo "selected_pairs=${#selected_indices[@]}"
echo "base_folder=$selected_base"
echo "rl_folder=$selected_rl"
if ((prepare_only)); then
  exit 0
fi

if ((base_port_explicit == 0)); then
  base_port="$(find_free_port "$base_port")"
else
  ensure_port_available "Base viewer" "$base_port"
fi
if ((rl_port_explicit == 0)); then
  rl_port="$(find_free_port "$rl_port")"
else
  ensure_port_available "RL viewer" "$rl_port"
fi
if [[ "$base_port" == "$rl_port" ]]; then
  if ((rl_port_explicit)); then
    echo "error: Base and RL viewers cannot share port=$base_port; use different --base-port/--rl-port values" >&2
    exit 2
  fi
  rl_port="$(find_free_port "$((rl_port + 1))")"
fi

select_rerun_binary
echo "Launching Base viewer with ${#base_rrds[@]} recordings on port=$base_port: $rerun_bin"
"$rerun_bin" --port "$base_port" "${base_rrds[@]}" &
base_pid=$!
echo "Launching RL viewer with ${#rl_rrds[@]} recordings on port=$rl_port: $rerun_bin"
"$rerun_bin" --port "$rl_port" "${rl_rrds[@]}" &
rl_pid=$!

if ((detach)); then
  echo "viewers_started base_pid=$base_pid rl_pid=$rl_pid"
  exit 0
fi

set +e
wait "$base_pid"
base_status=$?
wait "$rl_pid"
rl_status=$?
set -e
if ((base_status != 0 || rl_status != 0)); then
  echo "warning: viewer exit status base=$base_status rl=$rl_status" >&2
fi
