#!/usr/bin/env python3
"""Report the longest ground-truth window the frozen W0 selection can actually supply.

LIBERO-Wan reuses W0-Video-40's frozen selection: same episode, same start frame. The start frames
were chosen to sit at the motion p75 of the episode, so several land well past the middle -- and a
window that runs off the end cannot be fixed by moving the start without changing which samples are
selected. ``libero_spatial_task07`` starts at 61 in a 110-frame episode, so it can supply 49 future
frames and no more.

Wan2.2 needs ``4n+1`` frames, so the answer is the largest ``4n+1`` not exceeding the tightest
per-sample budget. Printed for the full 40 and for the mini-21 subset separately, because mini-21
may not contain the binding sample.

Usage:
  python scripts/experiments/literature/check_w0_window_feasibility.py
  python scripts/experiments/literature/check_w0_window_feasibility.py --manifest path/to/manifest.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

DEFAULT_DATASET_ROOT = "pretrained/w0_fastwam/dataset/data/libero_mujoco3.3.2"
DEFAULT_GLOB = "outputs/literature/w0_fastwam/video_generation/**/manifest.json"


def find_frozen_manifest(pattern: str) -> Path:
    for candidate in sorted(glob.glob(pattern, recursive=True)):
        try:
            payload = json.loads(Path(candidate).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if payload.get("dataset") == "W0-Video-40" and len(payload.get("samples", [])) == 40:
            return Path(candidate)
    raise SystemExit(f"no 40-sample W0-Video-40 manifest matched {pattern}")


def episode_lengths(dataset_root: Path, suites: list[str]) -> dict[tuple[str, int], int]:
    lengths: dict[tuple[str, int], int] = {}
    for suite in suites:
        path = dataset_root / f"{suite}_no_noops_lerobot" / "meta/episodes.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            lengths[(suite, int(row["episode_index"]))] = int(row["length"])
    return lengths


def mini21_ids(payload: dict) -> set[str]:
    """Reproduce prepare_libero_wan_manifest's mini-21 subset.

    Must match ``_libero_wan_selection_hash`` exactly: ``sha256(f"{seed}\\n{sample_id}")``.
    """
    seed = 20260824

    def key(sample: dict) -> str:
        return hashlib.sha256(f"{seed}\n{sample['sample_id']}".encode()).hexdigest()

    ranked = sorted(payload["samples"], key=key)
    return {str(sample["sample_id"]) for sample in ranked[:21]}


def largest_4n_plus_1(budget: int) -> int:
    return max(1, ((budget - 1) // 4) * 4 + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--glob", default=DEFAULT_GLOB)
    args = parser.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else find_frozen_manifest(args.glob)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(f"manifest: {manifest_path}")

    lengths = episode_lengths(Path(args.dataset_root), list(payload["suites"]))
    rows = []
    for sample in payload["samples"]:
        suite = str(sample["suite"])
        episode = int(sample["episode_index"])
        start = int(sample["start_frame"])
        length = lengths.get((suite, episode))
        if length is None:
            raise SystemExit(f"no length for {suite} episode {episode}")
        rows.append((str(sample["sample_id"]), start, length, length - start))

    subset = mini21_ids(payload)
    print(f"\n{'sample':<28} {'start':>6} {'length':>7} {'budget':>7}")
    for name, start, length, budget in sorted(rows, key=lambda item: item[3])[:8]:
        mark = " *mini21" if name in subset else ""
        print(f"{name:<28} {start:>6} {length:>7} {budget:>7}{mark}")

    for label, selected in (("all-40", rows), ("mini-21", [r for r in rows if r[0] in subset])):
        budgets = [budget for _name, _start, _length, budget in selected]
        tightest = min(budgets)
        binding = [name for name, _s, _l, budget in selected if budget == tightest]
        print(
            f"\n{label}: n={len(selected)} tightest budget={tightest} "
            f"({', '.join(binding)}) -> largest 4n+1 = {largest_4n_plus_1(tightest)}"
        )
        for candidate in (121, 81, 61, 49, 41, 33, 21, 9):
            fits = sum(1 for budget in budgets if budget >= candidate)
            flag = "OK" if fits == len(budgets) else f"{len(budgets) - fits}短"
            print(f"    frames={candidate:<4} 满足 {fits}/{len(budgets)}  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
