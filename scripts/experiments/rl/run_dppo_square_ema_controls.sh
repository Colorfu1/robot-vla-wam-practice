#!/usr/bin/env bash
# Two evaluation-only controls; no downloads, training, or changes to source checkpoints.
# Usage: GPU=0 bash scripts/experiments/rl/run_dppo_square_ema_controls.sh OFFICIAL.pt TRAINED.pt [--smoke]
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && "$3" != --smoke ) ]]; then
  echo 'Usage: run_dppo_square_ema_controls.sh OFFICIAL.pt TRAINED.pt [--smoke]' >&2
  exit 2
fi
official="$1"
trained="$2"
extra=()
smoke=false
if [[ "${3:-}" == --smoke ]]; then
  extra=(--smoke)
  smoke=true
fi
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
output="${OUTPUT_ROOT:-outputs/dppo_square_ema_controls_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -e "$output" ]]; then
  echo "Output already exists: $output" >&2
  exit 2
fi
# Verify both inputs before starting either evaluation. torch.load never permits arbitrary pickle.
conda run --no-capture-output -n dppo python -c '
import sys
import torch
from pathlib import Path
for name, path, expected in (("official", sys.argv[1], 4000), ("trained", sys.argv[2], 2000)):
    weights = torch.load(Path(path), map_location="cpu", weights_only=True)
    assert isinstance(weights.get("ema"), dict) and weights["ema"], f"{name}: missing EMA"
    if name == "official" or sys.argv[3] != "true":
        assert weights.get("epoch") == expected, f"{name}: wrong checkpoint epoch"
    print("VERIFIED", name, "epoch=", weights.get("epoch"), "variant=ema", flush=True)
' "$official" "$trained" "$smoke"
sha256sum --check <<'CHECKSUMS'
15eb8c018169a73dfaa8d108cbc692535215c65ed79a33bda29a3ca3ce25be62  datasets/raw/dppo/robomimic/square-img/normalization.npz
CHECKSUMS
mkdir -p "$output"
sha256sum "$official" "$trained" > "$output/checkpoints.sha256"
echo "RESULT_DIR=$output"
completed=0
for name in official_ema trained_ema; do
  checkpoint="$official"
  if [[ "$name" == trained_ema ]]; then
    checkpoint="$trained"
  fi
  echo "START control=$name completed=$completed total=2"
  conda run --no-capture-output -n dppo python -u \
    -m robot_practice.experiments.rl.dppo_square_official eval \
    --root=third_party/dppo_official --weight-variant=ema \
    --checkpoint="$checkpoint" --output="$output/$name" "${extra[@]}"
  completed=$((completed + 1))
  echo "STATUS phase=controls completed=$completed total=2"
done
conda run --no-capture-output -n dppo python -c '
import json
import sys
from pathlib import Path
import numpy as np
root = Path(sys.argv[1])
smoke = sys.argv[2] == "true"
summary = {"smoke": smoke, "weight_variant": "ema", "results": {}}
for name in ("official_ema", "trained_ema"):
    with np.load(root / name / "result.npz", allow_pickle=False) as result:
        row = {key: result[key].tolist() for key in result.files}
    expected = 1 if smoke else 80
    if int(row["num_episode"]) != expected:
        raise RuntimeError(f"{name}: expected {expected} completed episodes, got {row}")
    summary["results"][name] = row
    print(name, json.dumps(row), flush=True)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
' "$output" "$smoke"
echo "DONE phase=all completed=2 total=2 RESULT_DIR=$output"
