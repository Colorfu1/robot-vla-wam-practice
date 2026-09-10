#!/usr/bin/env bash
# Verify the completed paired S1 metrics, choose diagnostic episodes, and
# record the same episodes for base and residual SAC on the remote GPU host.
#
# This script intentionally does not download data, models, or checkpoints.
# It is meant to run from the remote practice checkout, where the metrics and
# training artifacts already exist.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  prepare_policy_decorator_peg_visualization_remote.sh \
    [--metrics-root=PATH] [--record-root=PATH] [--checkpoint-step=N] \
    [--seeds=1,2,3,4] [--episodes-per-seed=50] [--max-record-episodes=12] \
    [--gpus=0]

The script reads completed paired eval_base/eval_residual metrics, selects
representative global 0-based episode IDs, then reruns both policies with the
same reset rows while recording only those IDs. Each selected episode is an
independent worker; --gpus=0,1 assigns at most one worker per listed GPU at a
time. The default is one worker on GPU 0.
EOF
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
practice_root="$(cd -- "${script_dir}/../../.." && pwd)"
cd -- "${practice_root}"

metrics_root="${practice_root}/outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/visualization_4seed50"
record_root=""
checkpoint_step="${S1_CHECKPOINT_STEP:-4000000}"
seeds_csv="${S1_EVAL_SEEDS:-1,2,3,4}"
episodes_per_seed=50
max_record_episodes=12
gpus_csv="${S1_GPUS:-0}"
base_policy_ckpt="${practice_root}/outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/checkpoints/best.pt"
base_policy_args="${practice_root}/outputs/rl_three_stage_20260828/artifacts/policy_decorator/checkpoints/diffusion_PegInsertionSide/args.json"
residual_runs="${practice_root}/outputs/rl_three_stage_20260828/stage1/paper10_base_then_sac/sac"
reference_root="${practice_root}/datasets/raw/policy_decorator/reference_repo"

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --metrics-root=*) metrics_root="${1#*=}"; shift ;;
    --record-root=*) record_root="${1#*=}"; shift ;;
    --checkpoint-step=*) checkpoint_step="${1#*=}"; shift ;;
    --seeds=*) seeds_csv="${1#*=}"; shift ;;
    --episodes-per-seed=*) episodes_per_seed="${1#*=}"; shift ;;
    --max-record-episodes=*) max_record_episodes="${1#*=}"; shift ;;
    --gpus=*) gpus_csv="${1#*=}"; shift ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 64 ;;
  esac
done

if [[ "${metrics_root}" != /* ]]; then
  metrics_root="${practice_root}/${metrics_root}"
fi
if [[ -z "${record_root}" ]]; then
  record_root="${metrics_root}/recorded"
elif [[ "${record_root}" != /* ]]; then
  record_root="${practice_root}/${record_root}"
fi

base_eval_root="${metrics_root}/eval_base"
residual_eval_root="${metrics_root}/eval_residual"
manifest="${base_eval_root}/episode_manifest.jsonl"
base_episodes="${base_eval_root}/episodes.jsonl"
residual_episodes="${residual_eval_root}/episodes.jsonl"
recorded_ids_file="${metrics_root}/recorded_episode_ids.txt"

[[ "${checkpoint_step}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --checkpoint-step must be a positive integer: ${checkpoint_step@Q}" >&2
  exit 64
}
[[ "${episodes_per_seed}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --episodes-per-seed must be a positive integer: ${episodes_per_seed@Q}" >&2
  exit 64
}
[[ "${max_record_episodes}" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: --max-record-episodes must be a positive integer: ${max_record_episodes@Q}" >&2
  exit 64
}
[[ -n "${gpus_csv}" ]] || { echo "error: --gpus is empty" >&2; exit 64; }
[[ -s "${base_policy_ckpt}" ]] || {
  echo "error: missing base checkpoint: ${base_policy_ckpt}" >&2
  exit 66
}
[[ -s "${base_policy_args}" ]] || {
  echo "error: missing base args: ${base_policy_args}" >&2
  exit 66
}
[[ -f "${reference_root}/online/pi_dec_diffusion_maniskill2.py" ]] || {
  echo "error: missing Policy Decorator checkout: ${reference_root}" >&2
  exit 66
}
[[ -s "${base_episodes}" && -s "${residual_episodes}" && -s "${manifest}" ]] || {
  echo "error: completed paired metrics or manifest are missing under: ${metrics_root}" >&2
  exit 66
}
if [[ -e "${record_root}" ]]; then
  [[ -d "${record_root}" ]] || {
    echo "error: recording output is not a directory: ${record_root}" >&2
    exit 73
  }
  existing_recording_entry="$(find "${record_root}" -mindepth 1 -print -quit)"
  [[ -z "${existing_recording_entry}" ]] || {
    echo "error: refusing to reuse non-empty recording output: ${record_root}" >&2
    exit 73
  }
  echo "[visualization] reusing empty recording directory: ${record_root}"
fi

IFS=',' read -r -a seeds <<< "${seeds_csv}"
(( ${#seeds[@]} > 0 )) || { echo "error: --seeds is empty" >&2; exit 64; }
IFS=',' read -r -a gpu_ids <<< "${gpus_csv}"
(( ${#gpu_ids[@]} > 0 )) || { echo "error: --gpus is empty" >&2; exit 64; }
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || {
    echo "error: GPU ids must be non-negative integers: ${gpu}" >&2
    exit 64
  }
  [[ -z "${seen_gpus[${gpu}]+x}" ]] || {
    echo "error: duplicate GPU id: ${gpu}" >&2
    exit 64
  }
  seen_gpus[${gpu}]=1
done
for seed in "${seeds[@]}"; do
  [[ "${seed}" =~ ^[0-9]+$ ]] || {
    echo "error: seeds must be non-negative integers: ${seed}" >&2
    exit 64
  }
  seed_root="${residual_runs}/seed_${seed}"
  mapfile -t checkpoints < <(
    find "${seed_root}" -type f -path "*/checkpoints/${checkpoint_step}.pt" -print 2>/dev/null | sort
  )
  if (( ${#checkpoints[@]} != 1 )); then
    echo "error: expected exactly one residual checkpoint for seed=${seed}, step=${checkpoint_step}" >&2
    echo "searched under: ${seed_root}; found ${#checkpoints[@]}" >&2
    exit 66
  fi
done

selection_python="${SELECTION_PYTHON:-python}"
command -v "${selection_python}" >/dev/null 2>&1 || {
  echo "error: selection Python is unavailable: ${selection_python}" >&2
  exit 69
}

expected_total=$(( ${#seeds[@]} * episodes_per_seed ))
recorded_ids="$(PYTHONNOUSERSITE=1 "${selection_python}" - \
  "${base_episodes}" "${residual_episodes}" "${max_record_episodes}" \
  "${expected_total}" <<'PY'
import json
import sys


def read(path):
    with open(path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows


base = read(sys.argv[1])
residual = read(sys.argv[2])
limit = int(sys.argv[3])
expected_total = int(sys.argv[4])

if len(base) != expected_total or len(residual) != expected_total:
    raise SystemExit(
        f"expected {expected_total} rows per variant, got "
        f"base={len(base)}, residual={len(residual)}"
    )
if len(base) != len(residual):
    raise SystemExit(f"paired episode count mismatch: {len(base)} != {len(residual)}")

for index, (base_row, residual_row) in enumerate(zip(base, residual)):
    for field in ("episode_key", "reset_seed", "reset_state_id"):
        if base_row.get(field) != residual_row.get(field):
            raise SystemExit(f"episode pairing mismatch at index {index}: field={field}")

categories = (
    (lambda b, r: not b["success"] and r["success"], 6),
    (lambda b, r: b["success"] and r["success"], 4),
    (lambda b, r: not r["success"], 2),
)
selected = []
seen = set()

for predicate, category_limit in categories:
    count = 0
    for index, (base_row, residual_row) in enumerate(zip(base, residual)):
        if index in seen or not predicate(base_row, residual_row):
            continue
        selected.append(index)
        seen.add(index)
        count += 1
        if count >= category_limit or len(selected) >= limit:
            break
    if len(selected) >= limit:
        break

for index in range(len(base)):
    if len(selected) >= limit:
        break
    if index not in seen:
        selected.append(index)
        seen.add(index)

if not selected:
    raise SystemExit("no episode was selected")
print(",".join(str(index) for index in selected))
PY
)"

printf '%s\n' "${recorded_ids}" > "${recorded_ids_file}"
recorded_count="$(awk -F, '{print NF}' <<<"${recorded_ids}")"
echo "[visualization] metrics_root=${metrics_root}"
echo "[visualization] paired_episodes=${expected_total}"
echo "[visualization] selected_episode_ids=${recorded_ids}"
echo "[visualization] selected_count=${recorded_count}"
echo "[visualization] gpus=${gpus_csv}"

mkdir -p "${record_root}"
for name in eval_base eval_residual raw_base raw_residual; do
  test ! -e "${record_root}/${name}" || {
    echo "error: refusing to reuse recording output: ${record_root}/${name}" >&2
    exit 73
  }
done

shard_manifest_root="${record_root}/manifests"
worker_root="${record_root}/.workers"
shard_info_file="${record_root}/shard_info.tsv"
mkdir -p "${shard_manifest_root}" "${worker_root}"

# Create one-row manifests so each worker executes only one selected reset.
# The evaluator still validates the original reset_state_id from the paired
# 200-episode manifest.
PYTHONNOUSERSITE=1 "${selection_python}" - \
  "${manifest}" "${recorded_ids}" "${shard_manifest_root}" \
  > "${shard_info_file}" <<'PY'
import json
import sys
from pathlib import Path


manifest_path = Path(sys.argv[1])
episode_ids = [int(value) for value in sys.argv[2].split(",") if value]
output_root = Path(sys.argv[3])
with manifest_path.open(encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]

for episode_id in episode_ids:
    if episode_id < 0 or episode_id >= len(rows):
        raise SystemExit(f"selected episode ID is outside manifest: {episode_id}")
    row = rows[episode_id]
    path = output_root / f"episode_{episode_id:04d}.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{episode_id}\t{int(row['policy_seed'])}\t{path}")
PY

declare -a selected_ids=()
declare -a selected_policy_seeds=()
declare -a selected_manifests=()
while IFS=$'\t' read -r global_id policy_seed shard_manifest; do
  [[ -n "${global_id}" && -n "${policy_seed}" && -n "${shard_manifest}" ]] || {
    echo "error: malformed shard info row" >&2
    exit 66
  }
  selected_ids+=("${global_id}")
  selected_policy_seeds+=("${policy_seed}")
  selected_manifests+=("${shard_manifest}")
done < "${shard_info_file}"
if (( ${#selected_ids[@]} != recorded_count )); then
  echo "error: shard count mismatch: ${#selected_ids[@]} != ${recorded_count}" >&2
  exit 66
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${practice_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_one_recording() {
  local arm="$1"
  local index="$2"
  local gpu="$3"
  local episode_id="${selected_ids[${index}]}"
  local policy_seed="${selected_policy_seeds[${index}]}"
  local shard_manifest="${selected_manifests[${index}]}"
  local worker_dir="${worker_root}/${arm}/episode_${episode_id}"
  local raw_dir="${worker_dir}/raw"
  local output_dir="${worker_dir}/eval"
  local log_file="${worker_dir}/run.log"
  local -a variant_args=()

  mkdir -p "${worker_dir}"
  if [[ "${arm}" == base ]]; then
    variant_args=(--arm=base)
  else
    variant_args=(
      --arm=residual
      --residual-runs="${residual_runs}"
      --checkpoint-step="${checkpoint_step}"
    )
  fi

  echo "[visualization] START arm=${arm} episode=${episode_id} seed=${policy_seed} gpu=${gpu}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    conda run --no-capture-output -n maniskill2-pidec \
    python -m robot_practice.experiments.rl.policy_decorator_eval \
    --reference-root="${reference_root}" \
    --env-id=PegInsertionSide-v2 \
    --base-policy-ckpt="${base_policy_ckpt}" \
    --base-policy-args="${base_policy_args}" \
    "${variant_args[@]}" \
    --manifest="${shard_manifest}" \
    --seeds="${policy_seed}" \
    --episodes-per-seed=1 \
    --ddim-steps=4 \
    --act-horizon=4 \
    --record-rollout-dir="${raw_dir}" \
    --record-episode-ids=0 \
    --record-episodes=1 \
    --record-fps=20 \
    --output-dir="${output_dir}"
  ) 2>&1 | tee "${log_file}"
  echo "[visualization] DONE arm=${arm} episode=${episode_id} gpu=${gpu}"
}

run_variant() {
  local arm="$1"
  local total="${#selected_ids[@]}"
  local gpu_count="${#gpu_ids[@]}"
  local start=0

  while (( start < total )); do
    local -a pids=()
    local batch_count=0
    while (( batch_count < gpu_count && start + batch_count < total )); do
      run_one_recording "${arm}" "$((start + batch_count))" "${gpu_ids[${batch_count}]}" &
      pids+=("$!")
      batch_count=$((batch_count + 1))
    done

    local failed=0
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        failed=1
      fi
    done
    (( failed == 0 )) || {
      echo "error: ${arm} recording batch failed" >&2
      return 1
    }
    start=$((start + batch_count))
  done
}

merge_variant() {
  local arm="$1"
  echo "[visualization] merging arm=${arm} workers"
  PYTHONNOUSERSITE=1 "${selection_python}" - \
    "${record_root}" "${arm}" "${recorded_ids}" <<'PY'
import json
import shutil
import sys
from pathlib import Path


root = Path(sys.argv[1])
arm = sys.argv[2]
episode_ids = [int(value) for value in sys.argv[3].split(",") if value]
raw_output = root / f"raw_{arm}"
eval_output = root / f"eval_{arm}"
raw_output.mkdir(parents=True, exist_ok=True)
eval_output.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


raw_template = None
raw_episodes = []
eval_manifest_rows = []
eval_episode_rows = []

for episode_id in episode_ids:
    worker = root / ".workers" / arm / f"episode_{episode_id}"
    worker_raw = worker / "raw"
    worker_eval = worker / "eval"
    worker_rollout_manifest = json.loads(
        (worker_raw / "rollout_manifest.json").read_text(encoding="utf-8")
    )
    worker_rollout_episodes = worker_rollout_manifest.get("episodes", [])
    if len(worker_rollout_episodes) != 1:
        raise SystemExit(f"expected one rollout episode in {worker_raw}")
    raw_record = dict(worker_rollout_episodes[0])
    source = worker_raw / str(raw_record["file"])
    target = raw_output / f"episode_{episode_id:04d}.npz"
    if not source.is_file():
        raise SystemExit(f"missing worker raw bundle: {source}")
    if target.exists():
        raise SystemExit(f"refusing to overwrite merged raw bundle: {target}")
    shutil.move(str(source), str(target))
    raw_record["episode_id"] = episode_id
    raw_record["global_episode_id"] = episode_id
    raw_record["file"] = target.name
    raw_episodes.append(raw_record)
    if raw_template is None:
        raw_template = dict(worker_rollout_manifest)

    manifest_rows = read_jsonl(worker_eval / "episode_manifest.jsonl")
    episode_rows = read_jsonl(worker_eval / "episodes.jsonl")
    if len(manifest_rows) != 1 or len(episode_rows) != 1:
        raise SystemExit(f"expected one eval row in {worker_eval}")
    manifest_row = dict(manifest_rows[0])
    episode_row = dict(episode_rows[0])
    manifest_row["global_episode_id"] = episode_id
    episode_row["episode_id"] = episode_id
    episode_row["global_episode_id"] = episode_id
    eval_manifest_rows.append(manifest_row)
    eval_episode_rows.append(episode_row)

if raw_template is None:
    raise SystemExit("no worker rollout manifest was found")
raw_template["episodes"] = sorted(raw_episodes, key=lambda row: int(row["episode_id"]))
raw_template["selected_episode_ids"] = episode_ids
raw_template["sharded_recording"] = True
write_json(raw_output / "rollout_manifest.json", raw_template)

with (eval_output / "episode_manifest.jsonl").open("w", encoding="utf-8") as stream:
    for row in sorted(eval_manifest_rows, key=lambda value: int(value["global_episode_id"])):
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
with (eval_output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
    for row in sorted(eval_episode_rows, key=lambda value: int(value["global_episode_id"])):
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

first_metrics = json.loads(
    (root / ".workers" / arm / f"episode_{episode_ids[0]}" / "eval" / "metrics.json")
    .read_text(encoding="utf-8")
)
first_metrics["episodes"] = len(eval_episode_rows)
first_metrics["success_rate"] = sum(bool(row["success"]) for row in eval_episode_rows) / len(eval_episode_rows)
first_metrics["recording_only"] = True
first_metrics["selected_episode_ids"] = episode_ids
first_metrics["sharded_recording"] = True
write_json(eval_output / "metrics.json", first_metrics)
write_json(
    eval_output / "resolved_config.json",
    {
        "arm": arm,
        "selected_episode_ids": episode_ids,
        "recording_only": True,
        "sharded_recording": True,
    },
)
print(f"merged arm={arm} episodes={len(episode_ids)} raw={raw_output}")
PY
}

run_variant base
merge_variant base
run_variant residual
merge_variant residual

echo "[visualization] DONE selected=${recorded_count} record_root=${record_root} gpus=${gpus_csv}"
