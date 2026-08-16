"""Compare M2 offline action reconstruction on train and development splits.

This diagnostic is intentionally separate from the M2 training and Diffusion
Policy integration code. It uses the same normalized Flow sampling path as the
existing development MSE check, but evaluates both allowed offline splits and
never opens the offline test split.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from robot_practice.evaluation.common import practice_root, write_json_atomic
from robot_practice.experiments.maniskill import m2_w_action_flow as m2
from robot_practice.policies.wam_tiny.action_flow import load_action_flow_checkpoint


def _rooted(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else practice_root() / value


def _portable(path: str | Path) -> str:
    value = _rooted(path).resolve()
    try:
        return value.relative_to(practice_root()).as_posix()
    except ValueError:
        return str(value)


def _read_json(path: str | Path) -> dict[str, Any]:
    with _rooted(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object at {_rooted(path)}")
    return value


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = _rooted(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(target, value)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def _split_mse(
    config: Mapping[str, Any],
    *,
    split: str,
    model: Any,
    stats: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> dict[str, Any]:
    if split not in ("train", "development"):
        raise ValueError("This diagnostic accepts only train and development splits")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers cannot be negative")
    dataset = m2.StackCubeWindowDataset(
        config,
        split=split,
        stats=stats,
        max_samples=max_samples,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    total_error = 0.0
    baseline_error = 0.0
    total_count = 0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            observations = {
                key: batch[key].to(device=device, dtype=torch.float32, non_blocking=True)
                for key in m2.IMAGE_KEYS + (m2.STATE_KEY,)
            }
            target = batch[m2.ACTION_KEY].to(device=device, dtype=torch.float32, non_blocking=True)
            padding = batch["action_is_pad"].to(device=device, non_blocking=True)
            sample_indices = [int(value) for value in batch["sample_index"].tolist()]
            fingerprints = [f"offline-{split}-sample-{value}" for value in sample_indices]
            prediction = model.sample_actions(
                observations,
                fingerprints=fingerprints,
                chunk_index=0,
            )
            mask = (~padding).unsqueeze(-1)
            total_error += float(((prediction - target).square() * mask).sum().cpu())
            baseline_error += float((target.square() * mask).sum().cpu())
            total_count += int(mask.sum().item()) * target.shape[-1]
            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"[M2 offline MSE] split={split} "
                    f"batches={batch_index} samples={min(batch_index * batch_size, len(dataset))}/"
                    f"{len(dataset)}",
                    flush=True,
                )
    if total_count <= 0:
        raise ValueError(f"Offline split {split!r} yielded no unpadded action targets")
    mse = total_error / total_count
    baseline = baseline_error / total_count
    print(
        f"[M2 offline MSE] split={split} complete samples={len(dataset)} "
        f"mse={mse:.6f} relative_gain={(baseline - mse) / max(baseline, 1e-12):.4f}",
        flush=True,
    )
    return {
        "split": split,
        "samples": len(dataset),
        "action_elements": total_count,
        "normalized_action_mse": mse,
        "zero_prediction_mse": baseline,
        "relative_gain": (baseline - mse) / max(baseline, 1e-12),
    }


def summarize_train_dev_gap(train: Mapping[str, Any], development: Mapping[str, Any]) -> dict[str, Any]:
    train_mse = float(train["normalized_action_mse"])
    development_mse = float(development["normalized_action_mse"])
    ratio = development_mse / max(train_mse, 1e-12)
    return {
        "development_minus_train_mse": development_mse - train_mse,
        "development_to_train_mse_ratio": ratio,
        "interpretation": (
            "large_train_dev_gap_candidate"
            if ratio >= 1.5
            else "no_large_train_dev_gap_by_1_5x_rule"
        ),
    }


def compare(
    *,
    config_path: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
) -> dict[str, Any]:
    config = m2.load_config(config_path)
    checkpoint_path = _rooted(checkpoint)
    if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
        raise FileNotFoundError(f"Checkpoint is not a regular file: {_portable(checkpoint_path)}")
    device_obj = _resolve_device(device)
    if device_obj.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device_obj)
    stats_path = checkpoint_path.parent / "normalization_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Checkpoint normalization stats are missing: {_portable(stats_path)}")
    stats = m2._load_stats(stats_path)
    model, checkpoint_payload = load_action_flow_checkpoint(
        checkpoint_path,
        device=device_obj,
        n_action_steps=4,
    )
    model.model.eval()
    train = _split_mse(
        config,
        split="train",
        model=model.model,
        stats=stats,
        device=device_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    development = _split_mse(
        config,
        split="development",
        model=model.model,
        stats=stats,
        device=device_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        max_samples=max_samples,
    )
    result = {
        "schema_version": 1,
        "purpose": "m2_train_vs_development_offline_mse_diagnostic",
        "checkpoint": _portable(checkpoint_path),
        "checkpoint_step": checkpoint_payload.get("step"),
        "normalization_stats": _portable(stats_path),
        "config": _portable(config_path),
        "device": str(device_obj),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "max_samples_per_split": max_samples,
        "test_access_count": 0,
        "splits": {"train": train, "development": development},
        "train_development_gap": summarize_train_dev_gap(train, development),
    }
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/maniskill/m2_w_action_only_flow.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per split for a quick diagnostic; omit for full train/dev splits.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    compare(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
