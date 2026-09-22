"""Cheap offline open-loop action-error diagnostic for Light-WAM Square.

Purpose
-------
Closed-loop Square success is low at both 26,350 and 52,500 optimizer steps
despite a very low training action loss.  This script measures, offline and on
a single GPU, whether the trained action expert reproduces the normalized 7D
action chunks from the robomimic NPZ when conditioned the same way as the
closed-loop evaluator: the current frame ``t`` predicts actions ``t..t+3``.

This is a descriptive, cheap control experiment, not a closed-loop quality
measure.  A good reading here is necessary but not sufficient for Square
success.  It reports per-dimension statistics, MSE, the fraction of raw
network output outside ``[-1, 1]`` (which the closed-loop adapter clips), and
two trivial baselines: an always-zero chunk and an always-mean chunk.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from robot_practice.experiments.rl.lightwam_base import (
    DEFAULT_ACTION_DIM,
    DEFAULT_ACTION_HORIZON,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_VIDEO_FRAMES,
    DummyLightWAMCore,
    LightWAMBaseAdapter,
    RobomimicLightWAMWindowDataset,
    load_context_cache,
    load_official_lightwam,
)

ROUTE_NAME = "lightwam_base_robomimic_square_open_loop_diag"
MODEL_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def _log(log_file, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}"
    print(line, flush=True)
    print(line, file=log_file, flush=True)


def _config_snapshot(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"experiment config must resolve to a mapping, got {type(payload)}")
    return payload


def _nested(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def _sample_window_indices(
    dataset_size: int,
    num_windows: int,
    *,
    seed: int,
    random_sample: bool,
) -> list[int]:
    if num_windows < 1:
        raise ValueError("--num-windows must be positive")
    if num_windows > dataset_size:
        raise ValueError(f"--num-windows={num_windows} exceeds the {dataset_size} dataset windows")
    if random_sample:
        rng = np.random.default_rng(seed)
        return sorted(int(index) for index in rng.choice(dataset_size, num_windows, replace=False))
    # Even spacing across the whole window list covers every trajectory in the
    # same order the trainer iterates them, which keeps the sample comparable
    # between checkpoints.
    indices = np.linspace(0, dataset_size - 1, num_windows, dtype=np.int64)
    return sorted({int(index) for index in indices})


def _cond_from_item(item: Mapping[str, Any]) -> dict[str, Any]:
    video = item["video"]
    proprio = item["proprio"]
    if video.ndim != 4 or video.shape[0] != 3:
        raise ValueError(f"dataset video must be [3,T,H,W], got {tuple(video.shape)}")
    if proprio.ndim != 2 or proprio.shape[0] < 1:
        raise ValueError(f"dataset proprio must be [T,D], got {tuple(proprio.shape)}")
    return {
        # The closed-loop evaluator conditions on the current frame only.
        # Dataset window ``t`` holds frames ``t..t+4`` and actions ``t..t+3``,
        # so the first frame is the correct conditioning input for its chunk.
        "rgb": video[:, 0],
        "state": proprio[0],
        "context": item["context"],
        "context_mask": item["context_mask"],
        "prompt": item["prompt"],
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape != y.shape or x.ndim != 1 or x.shape[0] < 2:
        return 0.0
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = float(np.sqrt((centered_x**2).sum() * (centered_y**2).sum()))
    if denominator <= 1.0e-12:
        return 0.0
    return float((centered_x * centered_y).sum() / denominator)


def _compute_metrics(
    target: np.ndarray,
    pred_raw: np.ndarray,
    *,
    action_dim: int,
    action_horizon: int,
) -> dict[str, Any]:
    if target.shape != pred_raw.shape:
        raise ValueError(
            f"target/prediction shape mismatch: {tuple(target.shape)} vs {tuple(pred_raw.shape)}"
        )
    if target.ndim != 3 or target.shape[1] != action_horizon or target.shape[2] != action_dim:
        raise ValueError(f"unexpected chunk shape {tuple(target.shape)}")

    pred_clipped = np.clip(pred_raw, -1.0, 1.0)
    target_flat = target.reshape(-1, action_dim)
    raw_flat = pred_raw.reshape(-1, action_dim)
    clipped_flat = pred_clipped.reshape(-1, action_dim)
    clip_mask = np.abs(pred_raw) > 1.0
    clip_mask_flat = clip_mask.reshape(-1, action_dim)

    per_dim: dict[str, Any] = {}
    for dim in range(action_dim):
        dim_target = target_flat[:, dim]
        dim_raw = raw_flat[:, dim]
        dim_clipped = clipped_flat[:, dim]
        dim_mean_target = float(dim_target.mean())
        per_dim[f"dim{dim}"] = {
            "target_mean": dim_mean_target,
            "target_std": float(dim_target.std()),
            "pred_raw_mean": float(dim_raw.mean()),
            "pred_raw_std": float(dim_raw.std()),
            "pred_clipped_mean": float(dim_clipped.mean()),
            "pred_clipped_std": float(dim_clipped.std()),
            "mse_raw": float(np.mean((dim_raw - dim_target) ** 2)),
            "mse_clipped": float(np.mean((dim_clipped - dim_target) ** 2)),
            "clip_rate": float(clip_mask_flat[:, dim].mean()),
            "corr_clipped": _pearson(dim_target, dim_clipped),
            "mean_baseline_mse": float(np.mean((dim_target - dim_mean_target) ** 2)),
            "zero_baseline_mse": float(np.mean(dim_target**2)),
        }
        denominator = float(np.mean((dim_target - dim_mean_target) ** 2))
        per_dim[f"dim{dim}"]["mse_clipped_over_mean_baseline"] = (
            per_dim[f"dim{dim}"]["mse_clipped"] / denominator if denominator > 1.0e-12 else None
        )
        zero_denominator = per_dim[f"dim{dim}"]["zero_baseline_mse"]
        per_dim[f"dim{dim}"]["mse_clipped_over_zero_baseline"] = (
            per_dim[f"dim{dim}"]["mse_clipped"] / zero_denominator
            if zero_denominator > 1.0e-12
            else None
        )
        target_std = per_dim[f"dim{dim}"]["target_std"]
        per_dim[f"dim{dim}"]["pred_std_ratio"] = (
            per_dim[f"dim{dim}"]["pred_clipped_std"] / target_std if target_std > 1.0e-12 else None
        )

    per_step_mse = [
        float(np.mean((pred_clipped[:, step] - target[:, step]) ** 2))
        for step in range(action_horizon)
    ]
    overall = {
        "mae_raw": float(np.mean(np.abs(raw_flat - target_flat))),
        "mae_clipped": float(np.mean(np.abs(clipped_flat - target_flat))),
        "mse_raw": float(np.mean((raw_flat - target_flat) ** 2)),
        "mse_clipped": float(np.mean((clipped_flat - target_flat) ** 2)),
        "clip_rate": float(clip_mask.mean()),
        "mean_baseline_mse": float(
            np.mean((target_flat - target_flat.mean(axis=0, keepdims=True)) ** 2)
        ),
        "zero_baseline_mse": float(np.mean(target_flat**2)),
        "non_finite_count": int((~np.isfinite(pred_raw)).sum()),
    }
    mean_denominator = overall["mean_baseline_mse"]
    overall["mse_clipped_over_mean_baseline"] = (
        overall["mse_clipped"] / mean_denominator if mean_denominator > 1.0e-12 else None
    )
    zero_denominator = overall["zero_baseline_mse"]
    overall["mse_clipped_over_zero_baseline"] = (
        overall["mse_clipped"] / zero_denominator if zero_denominator > 1.0e-12 else None
    )
    return {
        "windows": int(target.shape[0]),
        "overall": overall,
        "per_step_mse": per_step_mse,
        "per_dim": per_dim,
        "arm_mse_clipped": float(np.mean((clipped_flat[:, :6] - target_flat[:, :6]) ** 2)),
        "arm_mae_clipped": float(np.mean(np.abs(clipped_flat[:, :6] - target_flat[:, :6]))),
        "gripper_mse_clipped": float(np.mean((clipped_flat[:, -1] - target_flat[:, -1]) ** 2)),
        "gripper_sign_accuracy": float(
            np.mean(np.sign(clipped_flat[:, -1]) == np.sign(target_flat[:, -1]))
        ),
        "gripper_close_recall": (
            float(np.mean(clipped_flat[target_flat[:, -1] > 0, -1] > 0))
            if np.any(target_flat[:, -1] > 0)
            else None
        ),
        "gripper_target_close_count": int(np.sum(target_flat[:, -1] > 0)),
    }


def trajectory_windows(dataset, trajectory_ids):
    """Select complete windows only; IDs index NPZ trajectories, not HDF5 demo names."""
    ids = list(trajectory_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Trajectory IDs must be nonempty and unique")
    if min(ids) < 0 or max(ids) >= len(dataset.traj_lengths):
        raise ValueError("Trajectory ID out of range")
    ends = np.cumsum(dataset.traj_lengths)
    indices, records = [], []
    for index, (start, _) in enumerate(dataset.windows):
        trajectory = int(np.searchsorted(ends, start, side="right"))
        if trajectory in ids:
            offset = 0 if trajectory == 0 else int(ends[trajectory - 1])
            indices.append(index)
            records.append(
                {"trajectory_id": trajectory, "frame": int(start - offset), "window_index": index}
            )
    if {r["trajectory_id"] for r in records} != set(ids):
        raise ValueError("Selected trajectory has no complete windows")
    return indices, records


def _evaluate_model(
    *,
    model: torch.nn.Module,
    dataset: RobomimicLightWAMWindowDataset,
    indices: list[int],
    context: torch.Tensor,
    context_mask: torch.Tensor,
    action_horizon: int,
    action_dim: int,
    image_size: tuple[int, int],
    num_inference_steps: int,
    seed: int,
    label: str,
    output_dir: Path,
    log_file,
) -> dict[str, Any]:
    adapter = LightWAMBaseAdapter(
        model,
        action_horizon=action_horizon,
        action_dim=action_dim,
        image_size=image_size,
        context=context,
        context_mask=context_mask,
        num_inference_steps=num_inference_steps,
        seed=0,
        tiled=False,
        clip_actions=False,
    ).eval()

    started = time.perf_counter()
    targets: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for position, index in enumerate(indices, start=1):
        item = dataset[index]
        cond = _cond_from_item(item)
        action = adapter.predict_action(cond, deterministic=True, seed=seed)
        action = action.detach().squeeze(0)
        target = item["action"]
        if tuple(action.shape) != tuple(target.shape):
            raise ValueError(
                f"{label}: predicted {tuple(action.shape)} but target is {tuple(target.shape)}"
            )
        predictions.append(action.to(device="cpu", dtype=torch.float32).numpy())
        targets.append(target.to(device="cpu", dtype=torch.float32).numpy())
        if position % 32 == 0 or position == len(indices):
            elapsed = time.perf_counter() - started
            _log(
                log_file,
                f"STATUS model={label} completed={position} total={len(indices)} "
                f"elapsed_seconds={elapsed:.1f} "
                f"eta_seconds={elapsed / position * (len(indices) - position):.1f}",
            )

    target_array = np.stack(targets, axis=0)
    pred_array = np.stack(predictions, axis=0)
    metrics = _compute_metrics(
        target_array,
        pred_array,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    metrics["elapsed_seconds"] = float(time.perf_counter() - started)
    metrics["seconds_per_window"] = metrics["elapsed_seconds"] / len(indices)
    ends = np.cumsum(dataset.traj_lengths)
    trajectory_ids = np.asarray(
        [np.searchsorted(ends, dataset.windows[i][0], side="right") for i in indices]
    )
    metrics["per_trajectory"] = {
        str(int(t)): _compute_metrics(
            target_array[trajectory_ids == t],
            pred_array[trajectory_ids == t],
            action_dim=action_dim,
            action_horizon=action_horizon,
        )
        for t in np.unique(trajectory_ids)
    }
    metrics["trajectory_macro_mse_clipped"] = float(
        np.mean([row["overall"]["mse_clipped"] for row in metrics["per_trajectory"].values()])
    )
    prediction_path = output_dir / f"predictions_{label}.npz"
    np.savez_compressed(
        str(prediction_path),
        target=target_array,
        pred_raw=pred_array,
        pred_clipped=np.clip(pred_array, -1.0, 1.0),
        window_indices=np.asarray(indices, dtype=np.int64),
        trajectory_ids=trajectory_ids,
    )
    metrics["predictions_path"] = str(prediction_path)
    _log(log_file, f"DONE model={label} windows={len(indices)} path={prediction_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline open-loop action-error diagnostic for Light-WAM Square",
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/rl/lightwam_base_robomimic_square.yaml",
        help="Light-WAM Square experiment config",
    )
    parser.add_argument(
        "--checkpoints",
        default="",
        help="Comma-separated Light-WAM weight checkpoints to compare, e.g. "
        "checkpoints/weights/step_052500.pt",
    )
    parser.add_argument(
        "--include-untrained",
        action="store_true",
        help="Also evaluate a freshly constructed model with random-initialized "
        "adapters/action expert as a training-effect control",
    )
    parser.add_argument(
        "--dummy-model",
        action="store_true",
        help="Use DummyLightWAMCore instead of the official model; local CPU contract smoke only",
    )
    parser.add_argument("--num-windows", type=int, default=512)
    parser.add_argument(
        "--trajectory-ids",
        default=None,
        help="Comma-separated NPZ trajectory IDs; evaluate ALL their valid windows",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Sample windows uniformly at random instead of evenly across the dataset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        default="outputs/lightwam_base_robomimic_square/offline_open_loop_diag",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    config = _config_snapshot(Path(arguments.config))
    data_path = Path(_nested(config, "data.npz"))
    context_path = Path(_nested(config, "lightwam.context"))
    prompt = str(_nested(config, "lightwam.prompt"))
    vendor_root = Path(_nested(config, "lightwam.vendor_root"))
    model_config = Path(_nested(config, "lightwam.model_config"))
    local_model_root = Path(_nested(config, "lightwam.local_model_root"))
    action_horizon = int(_nested(config, "lightwam.action_horizon", DEFAULT_ACTION_HORIZON))
    video_frames = int(_nested(config, "lightwam.video_frames", DEFAULT_VIDEO_FRAMES))
    image_size = tuple(
        int(value) for value in _nested(config, "lightwam.image_size", list(DEFAULT_IMAGE_SIZE))
    )
    num_inference_steps = int(
        arguments.num_inference_steps
        if arguments.num_inference_steps is not None
        else _nested(config, "lightwam.inference_steps", 10)
    )
    model_dtype = MODEL_DTYPES[_nested(config, "lightwam.model_dtype", "bf16")]

    output_dir = Path(arguments.output_root)
    if output_dir.exists() and any(output_dir.iterdir()) and not arguments.overwrite:
        raise FileExistsError(
            f"refusing to overwrite non-empty output: {output_dir}; pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    with open(run_log, "a", encoding="utf-8") as log_file:
        checkpoints: list[tuple[str, Path | None]] = []
        if arguments.checkpoints:
            for raw in arguments.checkpoints.split(","):
                raw = raw.strip()
                if not raw:
                    continue
                checkpoint_path = Path(raw).expanduser().resolve()
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"checkpoint is missing: {checkpoint_path}")
                checkpoints.append((checkpoint_path.stem, checkpoint_path))
        if arguments.include_untrained:
            checkpoints.insert(0, ("untrained_random_init", None))
        if arguments.dummy_model:
            checkpoints = [("dummy_model", None)]
        if not checkpoints:
            raise ValueError("provide --checkpoints, --include-untrained, or --dummy-model")

        _log(log_file, f"START route={ROUTE_NAME} output={output_dir}")
        _log(log_file, f"STATUS phase=assets data={data_path} context={context_path}")
        dataset = RobomimicLightWAMWindowDataset(
            str(data_path),
            action_horizon=action_horizon,
            video_frames=video_frames,
            image_size=image_size,
            context_path=str(context_path),
            prompt=prompt,
            normalized=True,
        )
        context, context_mask = load_context_cache(context_path)
        indices = _sample_window_indices(
            len(dataset),
            arguments.num_windows,
            seed=arguments.seed,
            random_sample=arguments.random_sample,
        )
        if arguments.trajectory_ids is not None:
            import hashlib

            if arguments.random_sample:
                raise ValueError("--trajectory-ids cannot be combined with --random-sample")
            indices, records = trajectory_windows(
                dataset, [int(x) for x in arguments.trajectory_ids.split(",")]
            )
            digest = hashlib.sha256()
            with data_path.open("rb") as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            selection = {
                "dataset_sha256": digest.hexdigest(),
                "training_set_only": True,
                "trajectory_ids": sorted({r["trajectory_id"] for r in records}),
                "windows": records,
                "action_horizon": action_horizon,
                "video_frames": video_frames,
                "closed_loop_status": "blocked_missing_verified_demo_initial_state_mapping",
                "id_namespace": "NPZ trajectory order; not assumed equal to HDF5 demo IDs",
            }
            (output_dir / "trajectory_selection.json").write_text(
                json.dumps(selection, indent=2) + "\n", encoding="utf-8"
            )
        _log(
            log_file,
            f"STATUS phase=windows total={len(dataset)} sampled={len(indices)} "
            f"conditioning=first_window_frame action_horizon={action_horizon}",
        )

        model_results: dict[str, Any] = {}
        device = "cpu" if arguments.dummy_model else arguments.device
        for label, checkpoint in checkpoints:
            if arguments.dummy_model:
                model: torch.nn.Module = DummyLightWAMCore(
                    action_dim=DEFAULT_ACTION_DIM, device=device
                )
            else:
                _log(log_file, f"STATUS phase=load model={label} device={device}")
                model = load_official_lightwam(
                    vendor_root=vendor_root,
                    model_config=model_config,
                    local_model_root=local_model_root,
                    device=device,
                    model_dtype=model_dtype,
                    checkpoint=checkpoint,
                    allow_download=False,
                ).eval()
            model_results[label] = _evaluate_model(
                model=model,
                dataset=dataset,
                indices=indices,
                context=context,
                context_mask=context_mask,
                action_horizon=action_horizon,
                action_dim=DEFAULT_ACTION_DIM,
                image_size=image_size,
                num_inference_steps=num_inference_steps,
                seed=arguments.seed,
                label=label,
                output_dir=output_dir,
                log_file=log_file,
            )
            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        report = {
            "schema_version": 1,
            "route": ROUTE_NAME,
            "note": (
                "Descriptive offline open-loop action-fidelity control. Measures how well "
                "the action expert reproduces in-distribution normalized chunks when "
                "conditioned on the current frame, exactly as the closed-loop evaluator "
                "conditions it. It is not a closed-loop success measurement."
            ),
            "experiment_config": arguments.config,
            "dataset": {
                "path": str(data_path),
                "windows_total": len(dataset),
                "sampled_windows": len(indices),
                "action_horizon": action_horizon,
                "video_frames": video_frames,
                "image_size": list(image_size),
                "conditioning": "first_window_frame",
                "seed": arguments.seed,
                "random_sample": arguments.random_sample,
                "trajectory_ids": arguments.trajectory_ids,
                "training_set_only": True,
                "action_error_space": "normalized; raw means unclipped, not denormalized",
            },
            "models": model_results,
        }
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _log(log_file, f"DONE phase=all models={len(model_results)} metrics={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
