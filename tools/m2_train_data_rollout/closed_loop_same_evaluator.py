"""Run a train-data probe through the exact M2 development evaluator.

The sampled D_core start states are written to a temporary state-bank-shaped
JSONL named ``development.jsonl`` so the existing M2 evaluator can be reused
without changing its implementation. The outer result retains the source
training episode IDs for diagnosis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_practice.experiments.maniskill.eval_stackcube_action_flow import (
    run_development_rollout,
)

from .rollout_train_samples import (
    _load_sample_rows,
    _portable,
    _resolve_device,
    _rooted,
    _write_json,
)


def _write_train_state_bank(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            stream.write(
                json.dumps(
                    {
                        "index": index,
                        "source_episode_id": int(row["episode_id"]),
                        "state": row["state"],
                        "start_state_fingerprint": str(row["start_state_fingerprint"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def run_same_evaluator(
    *,
    checkpoint: Path,
    d_core_h5: Path,
    split_manifest: Path,
    output_dir: Path,
    sample_count: int,
    seed: int,
    max_steps: int,
    n_action_steps: int,
    num_envs: int,
    device: str,
) -> dict[str, Any]:
    output_dir = _rooted(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {_portable(output_dir)}")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    output_dir.mkdir(parents=True)
    rows = _load_sample_rows(
        d_core_h5=d_core_h5,
        split_manifest=split_manifest,
        sample_count=sample_count,
        seed=seed,
    )
    state_bank = output_dir / "development.jsonl"
    _write_train_state_bank(state_bank, rows)
    device_obj = _resolve_device(device)
    evaluator_output = output_dir / "same_evaluator"
    print(
        f"[M2 train-data same evaluator] sampled={len(rows)} num_envs={num_envs} "
        f"device={device_obj}",
        flush=True,
    )
    metrics = run_development_rollout(
        state_bank=state_bank,
        output_dir=evaluator_output,
        checkpoint=_rooted(checkpoint),
        n_action_steps=n_action_steps,
        max_episodes=len(rows),
        max_steps=max_steps,
        num_envs=num_envs,
        device=str(device_obj),
        worker_index=0,
        worker_count=1,
        smoke=False,
    )
    by_fingerprint = {str(row["start_state_fingerprint"]): row for row in rows}
    result_rows = []
    for item in metrics["rows"]:
        fingerprint = str(item["start_state_fingerprint"])
        source = by_fingerprint.get(fingerprint)
        if source is None:
            raise ValueError(f"Evaluator returned an unknown training-state fingerprint: {fingerprint}")
        result_rows.append(
            {
                "episode_id": int(source["episode_id"]),
                "source_group": str(source["source_group"]),
                "source_action_count": int(source["source_action_count"]),
                **item,
            }
        )
    result_rows.sort(key=lambda item: int(item["episode_id"]))
    summary = dict(metrics["summary"])
    summary["source_split"] = "train"
    summary["source_episode_ids"] = [int(row["episode_id"]) for row in result_rows]
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "train_data_closed_loop_probe_using_exact_development_evaluator",
        "source_split": "train",
        "checkpoint": _portable(checkpoint),
        "d_core_h5": _portable(d_core_h5),
        "offline_split_manifest": _portable(split_manifest),
        "state_bank": _portable(state_bank),
        "evaluator_output": _portable(evaluator_output),
        "sampling_seed": seed,
        "sample_count": sample_count,
        "sampled_episode_ids": [int(row["episode_id"]) for row in rows],
        "device": str(device_obj),
        "num_envs": num_envs,
        "max_steps": max_steps,
        "n_action_steps": n_action_steps,
        "test_access_count": 0,
    }
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in result_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in result_rows:
        _write_json(output_dir / f"episode_{int(row['episode_id']):04d}.json", row)
    _write_json(output_dir / "rollout_summary.json", {"manifest": manifest, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return {"manifest": manifest, "summary": summary, "rows": result_rows}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--d-core-h5",
        type=Path,
        default=Path("outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/trajectory.h5"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/d_core/split_manifest.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-action-steps", type=int, choices=(4, 8), default=8)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_same_evaluator(
        checkpoint=args.checkpoint,
        d_core_h5=args.d_core_h5,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        sample_count=args.sample_count,
        seed=args.seed,
        max_steps=args.max_steps,
        n_action_steps=args.n_action_steps,
        num_envs=args.num_envs,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
