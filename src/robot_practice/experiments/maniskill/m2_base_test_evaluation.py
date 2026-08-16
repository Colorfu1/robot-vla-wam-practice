"""Held-out test evaluation for a selected M2 Action-only Flow checkpoint.

This entry point is intentionally separate from development checkpoint
selection.  It reads the configured test split only when explicitly invoked,
uses exact replay states rather than seeds, and runs policy inference plus
GPU PhysX rollout in one worker per visible GPU.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .m2_base_action_flow import (
    _portable,
    _read_json,
    _rooted,
    _write_json,
    load_config,
    validate_config,
)
from .m2_base_checkpoint_selection import (
    _evaluate_gpu_task_batch,
    _load_split_rows,
    _prepare_selection_dir,
    _resolve_rollout_route,
    _summarize_rows,
    _visible_gpu_ids,
)

TASK_ID = "StackCube-v1"
DEFAULT_NUM_ENVS = 64
DEFAULT_MAX_STEPS = 200


def _load_selection(selection_root: str | Path) -> dict[str, Any]:
    target = _rooted(selection_root)
    if target.is_dir():
        target = target / "selected_checkpoint.json"
    if not target.is_file():
        raise FileNotFoundError(f"Selection manifest does not exist: {_portable(target)}")
    payload = _read_json(target)
    selected = payload.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError(f"Selection manifest has no selected checkpoint: {_portable(target)}")
    return payload


def _resolve_checkpoint(
    *,
    checkpoint: str | Path | None,
    selection_root: str | Path | None,
) -> tuple[Path, int | None, dict[str, Any] | None]:
    if checkpoint is not None and selection_root is not None:
        raise ValueError("Pass either --checkpoint or --selection-root, not both")
    if selection_root is not None:
        selection = _load_selection(selection_root)
        selected = selection["selected"]
        copied = selection.get("selected_checkpoint_copy")
        candidate = _rooted(copied) if copied else _rooted(selected["checkpoint"])
        if not candidate.is_file():
            candidate = _rooted(selected["checkpoint"])
        if not candidate.is_file():
            raise FileNotFoundError(f"Selected checkpoint does not exist: {_portable(candidate)}")
        return candidate, int(selected["n_action_steps"]), dict(selection)
    if checkpoint is None:
        raise ValueError("Pass --checkpoint or --selection-root")
    candidate = _rooted(checkpoint)
    if not candidate.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {_portable(candidate)}")
    return candidate, None, None


def evaluate_test(
    config_path: str | Path,
    *,
    checkpoint: str | Path | None,
    selection_root: str | Path | None,
    output_root: str | Path,
    num_gpus: int | None = None,
    num_envs: int = DEFAULT_NUM_ENVS,
    policy_device: str = "auto",
    max_steps: int = DEFAULT_MAX_STEPS,
    n_action_steps: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    validate_config(config, check_inputs=True)
    rollout_backend, resolved_device = _resolve_rollout_route(config, policy_device)
    if rollout_backend != "physx_cuda" or resolved_device != "cuda":
        raise ValueError("M2 base test evaluation currently requires protocol.backend=physx_cuda")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")
    gpu_count = int(num_gpus or int(config["training"]["world_size"]))
    if gpu_count <= 0:
        raise ValueError("num_gpus must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("M2 base test evaluation requires CUDA")
    visible_gpu_ids = _visible_gpu_ids()
    visible_device_count = torch.cuda.device_count()
    if len(visible_gpu_ids) < gpu_count or visible_device_count < gpu_count:
        raise RuntimeError(
            f"requested {gpu_count} visible GPUs for test evaluation, "
            f"but CUDA exposes {visible_device_count} devices "
            f"from mask {visible_gpu_ids}"
        )

    checkpoint_path, selected_n_action_steps, selection = _resolve_checkpoint(
        checkpoint=checkpoint,
        selection_root=selection_root,
    )
    action_steps = int(n_action_steps or selected_n_action_steps or 4)
    if action_steps not in (4, 8):
        raise ValueError("n_action_steps must be 4 or 8")
    rows = _load_split_rows(config, "test")
    output = _prepare_selection_dir(output_root, overwrite=overwrite)
    worker_count = min(gpu_count, len(rows))
    row_batches = [list(rows[index::worker_count]) for index in range(worker_count)]
    tasks: list[dict[str, Any]] = []
    for worker_index, batch_rows in enumerate(row_batches):
        physical_gpu_id = visible_gpu_ids[worker_index]
        worker_output = output / "workers" / f"gpu_{physical_gpu_id}"
        tasks.append(
            {
                "checkpoint": str(checkpoint_path),
                "rows": batch_rows,
                "n_action_steps": action_steps,
                "max_steps": max_steps,
                "output_dir": str(worker_output),
                "policy_device": "cuda",
                "rollout_backend": "physx_cuda",
                "num_envs": num_envs,
                "expected_action_dim": int(config["protocol"]["action_dim"]),
                "progress_label": "M2-base-test",
            }
        )

    print(
        f"[M2-base-test] split=test episodes={len(rows)} checkpoint={checkpoint_path.name} "
        f"n_action_steps={action_steps} gpu_workers={worker_count} num_envs={num_envs}",
        flush=True,
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    context = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
    ) as executor:
        future_to_worker = {
            executor.submit(_evaluate_gpu_task_batch, [task], visible_gpu_ids[index]): index
            for index, task in enumerate(tasks)
        }
        completed_workers = 0
        completed_episodes = 0
        for future in concurrent.futures.as_completed(future_to_worker):
            worker_index = future_to_worker[future]
            worker_result = future.result()[0]
            results.append(worker_result)
            completed_workers += 1
            completed_episodes += int(worker_result["summary"]["num_episodes"])
            summary = worker_result["summary"]
            print(
                f"[M2-base-test] workers={completed_workers}/{worker_count} "
                f"episodes={completed_episodes}/{len(rows)} "
                f"gpu={visible_gpu_ids[worker_index]} "
                f"success_once={summary['success_once']:.4f} "
                f"success_at_end={summary['success_at_end']:.4f}",
                flush=True,
            )

    all_rows = [row for result in results for row in result["rows"]]
    all_rows.sort(key=lambda row: int(row["episode_id"]))
    if len(all_rows) != len(rows):
        raise RuntimeError(f"Test evaluation returned {len(all_rows)} rows, expected {len(rows)}")
    summary = _summarize_rows(all_rows)
    checkpoint_step = None if selection is None else int(selection["selected"]["checkpoint_step"])
    payload = {
        "schema_version": 1,
        "task": TASK_ID,
        "split": "test",
        "checkpoint": _portable(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "n_action_steps": action_steps,
        "protocol": {
            "control_mode": config["protocol"]["control_mode"],
            "action_dim": int(config["protocol"]["action_dim"]),
            "sim_backend": "physx_cuda",
            "max_episode_steps": max_steps,
            "partial_reset": False,
            "identity": "start_state_fingerprint",
        },
        "policy_device": "cuda",
        "rollout_parallelism": {
            "backend": "physx_cuda",
            "num_gpus": worker_count,
            "num_envs": num_envs,
            "worker_count": worker_count,
            "physical_gpu_ids": visible_gpu_ids[:worker_count],
        },
        "summary": summary,
        "rows": all_rows,
        "test_access_count": len(rows),
        "selection_manifest": None if selection is None else _portable(selection_root or ""),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output / "metrics.json", payload)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "task": TASK_ID,
            "split": "test",
            "checkpoint": _portable(checkpoint_path),
            "n_action_steps": action_steps,
            "num_episodes": len(rows),
            "num_gpus": worker_count,
            "num_envs": num_envs,
            "test_access_count": len(rows),
        },
    )
    print(
        f"[M2-base-test] complete episodes={len(rows)} "
        f"success_once={summary['success_once']:.4f} "
        f"success_at_end={summary['success_at_end']:.4f} "
        f"elapsed={payload['elapsed_seconds']:.1f}s",
        flush=True,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selection-root", type=Path)
    group.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--policy-device", choices=("auto", "cuda"), default="auto")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate_test(
        args.config,
        checkpoint=args.checkpoint,
        selection_root=args.selection_root,
        output_root=args.output_root,
        num_gpus=args.num_gpus,
        num_envs=args.num_envs,
        policy_device=args.policy_device,
        max_steps=args.max_steps,
        n_action_steps=args.n_action_steps,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
