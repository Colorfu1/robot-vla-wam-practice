"""Launch official Light-WAM training with the PegInsertionSide data contract.

This is a thin configuration adapter.  Model construction and optimization
remain in the pinned Light-WAM checkout; this module only composes the
official config and replaces its dataset block with the local HDF5 dataset
contract used for PegInsertionSide.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

from .lightwam_maniskill import (
    PEG_ACTION_DIM,
    PEG_DEFAULT_VIDEO_SIZE,
    PEG_PROPRIO_DIM,
    add_lightwam_source,
)


class _Tee:
    def __init__(
        self,
        original: TextIO,
        log: TextIO,
        on_progress: Any | None = None,
    ) -> None:
        self.original = original
        self.log = log
        self.on_progress = on_progress
        self._buffer = ""
        self._last_progress = {"episode": -1, "step": -1}

    def write(self, value: str) -> int:
        self.original.write(value)
        self.log.write(value)
        if self.on_progress is not None:
            self._buffer += value
            while "\n" in self._buffer:
                line, self._buffer = self._buffer.split("\n", 1)
                progress_patterns = (
                    ("episode", r"\bStats progress: episode=(\d+)/(\d+)\b"),
                    ("step", r"\bstep=(\d+)(?:/(\d+))?\b"),
                )
                for kind, pattern in progress_patterns:
                    match = re.search(pattern, line)
                    if match is None:
                        continue
                    completed = int(match.group(1))
                    if completed <= self._last_progress[kind]:
                        continue
                    self._last_progress[kind] = completed
                    observed_total = None if match.group(2) is None else int(match.group(2))
                    self.on_progress(completed, observed_total)
                    break
        return len(value)

    def flush(self) -> None:
        self.original.flush()
        self.log.flush()


def _shape_meta(video_height: int, video_width: int) -> dict[str, Any]:
    del video_width
    return {
        "images": [
            {
                "key": "base_camera",
                "raw_shape": [3, 128, 128],
                "shape": [3, video_height, video_height],
            },
            {
                "key": "hand_camera",
                "raw_shape": [3, 128, 128],
                "shape": [3, video_height, video_height],
            },
        ],
        "action": [
            {"key": "default", "raw_shape": PEG_ACTION_DIM, "shape": PEG_ACTION_DIM}
        ],
        "state": [
            {"key": "default", "raw_shape": PEG_PROPRIO_DIM, "shape": PEG_PROPRIO_DIM}
        ],
    }


def _processor_config(video_height: int) -> dict[str, Any]:
    return {
        "_target_": "lightwam.datasets.lerobot.processors.lightwam_processor.LightWAMProcessor",
        "shape_meta": "${data.train.shape_meta}",
        "num_obs_steps": "${data.train.num_frames}",
        "num_output_cameras": 2,
        "action_output_dim": PEG_ACTION_DIM,
        "proprio_output_dim": PEG_PROPRIO_DIM,
        "action_state_transforms": None,
        "use_stepwise_action_norm": False,
        "norm_default_mode": "min/max",
        "norm_exception_mode": None,
        "delta_action_dim_mask": {
            "default": [True, True, True, True, True, True, False]
        },
        "action_state_merger": {
            "_target_": "lightwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign"
        },
        "train_transforms": [
            {"_target_": "lightwam.datasets.lerobot.transforms.image.ToTensor"},
            {"_target_": "torchvision.transforms.Resize", "size": [video_height, video_height]},
        ],
        "val_transforms": [
            {"_target_": "lightwam.datasets.lerobot.transforms.image.ToTensor"},
            {"_target_": "torchvision.transforms.Resize", "size": [video_height, video_height]},
        ],
    }


def _dataset_config(
    *,
    data_root: Path,
    video_height: int,
    video_width: int,
    text_cache: str | None,
    stats_path: str | None,
    seed: int,
    val_set_proportion: float,
    is_training_set: bool,
    override_instruction: str,
) -> dict[str, Any]:
    return {
        "_target_": "lightwam.datasets.lerobot.hdf5_robot_video_dataset.HDF5RobotVideoDataset",
        "dataset_dirs": [str(data_root)],
        "shape_meta": _shape_meta(video_height, video_width),
        # 16 action transitions / 4 action-video frequency = 4 video transitions.
        # This keeps the official training action horizon aligned with the
        # online Peg contract instead of inheriting the 32-action LIBERO default.
        "num_frames": 17,
        "global_sample_stride": 1,
        "action_video_freq_ratio": 4,
        "video_size": [video_height, video_width],
        "camera_key": None,
        "val_set_proportion": float(val_set_proportion),
        "is_training_set": bool(is_training_set),
        "seed": int(seed),
        "pretrained_norm_stats": stats_path,
        "concat_multi_camera": "horizontal",
        "override_instruction": override_instruction,
        "processor": _processor_config(video_height),
        "text_embedding_cache_dir": text_cache,
        "text_embedding_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "context_len": 128,
        "use_latent_cache": False,
        "latent_cache_dir": None,
    }


def build_peg_train_config(arguments: argparse.Namespace) -> Any:
    """Compose the official train config and replace only its data contract."""

    vendor_root = Path(arguments.vendor_root).expanduser().resolve()
    add_lightwam_source(vendor_root)

    from hydra import compose, initialize_config_dir
    from lightwam.utils.config_resolvers import register_default_resolvers
    from omegaconf import OmegaConf

    register_default_resolvers()
    config_dir = vendor_root / "configs"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Light-WAM configs directory is missing: {config_dir}")
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"task={arguments.official_task_config}"])

    if len(arguments.video_size) != 2 or any(int(value) <= 0 for value in arguments.video_size):
        raise ValueError("--video-size must contain two positive integers: HEIGHT WIDTH")
    if arguments.batch_size <= 0 or arguments.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers must be non-negative")
    if arguments.num_epochs <= 0:
        raise ValueError("--num-epochs must be positive")
    if arguments.max_steps is not None and arguments.max_steps <= 0:
        raise ValueError("--max-steps must be positive when provided")
    if arguments.log_every <= 0 or arguments.save_every <= 0 or arguments.eval_every < 0:
        raise ValueError("log-every/save-every must be positive and eval-every non-negative")
    if not 0.0 <= arguments.val_set_proportion < 1.0:
        raise ValueError("--val-set-proportion must be in [0, 1)")

    video_height, video_width = (int(arguments.video_size[0]), int(arguments.video_size[1]))
    train_data = _dataset_config(
        data_root=Path(arguments.data_root).expanduser().resolve(),
        video_height=video_height,
        video_width=video_width,
        text_cache=arguments.text_embedding_cache_dir,
        stats_path=arguments.stats_path,
        seed=int(arguments.seed),
        val_set_proportion=float(arguments.val_set_proportion),
        is_training_set=True,
        override_instruction=arguments.instruction,
    )
    cfg.data = OmegaConf.create({"train": train_data, "val": None})
    if arguments.val_data_root:
        cfg.data.val = OmegaConf.create(
            _dataset_config(
                data_root=Path(arguments.val_data_root).expanduser().resolve(),
                video_height=video_height,
                video_width=video_width,
                text_cache=arguments.text_embedding_cache_dir,
                stats_path=arguments.stats_path,
                seed=int(arguments.seed),
                val_set_proportion=0.0,
                is_training_set=False,
                override_instruction=arguments.instruction,
            )
        )
    cfg.output_dir = str(Path(arguments.output_dir).expanduser().resolve())
    cfg.batch_size = int(arguments.batch_size)
    cfg.num_workers = int(arguments.num_workers)
    cfg.learning_rate = float(arguments.learning_rate)
    cfg.num_epochs = int(arguments.num_epochs)
    cfg.max_steps = None if arguments.max_steps is None else int(arguments.max_steps)
    cfg.log_every = int(arguments.log_every)
    cfg.save_every = int(arguments.save_every)
    cfg.eval_every = int(arguments.eval_every)
    cfg.seed = int(arguments.seed)
    cfg.mixed_precision = str(arguments.mixed_precision)
    cfg.wandb.enabled = False
    if arguments.resume:
        cfg.resume = str(Path(arguments.resume).expanduser().resolve())
    OmegaConf.resolve(cfg)
    return cfg


def _write_resolved_config(path: Path, cfg: Any) -> None:
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, str(path), resolve=True)


def _validate_hdf5_root(path_value: str, *, label: str) -> Path:
    """Fail early instead of letting the official dataset silently retry bad data."""

    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must be an existing directory: {path}")
    if not any(path.glob("episode_*.hdf5")):
        raise FileNotFoundError(
            f"{label} has no episode_*.hdf5 files: {path}; "
            "Light-WAM HDF5RobotVideoDataset does not consume trajectory.h5 directly"
        )
    return path


def _write_progress(output: Path, completed: int, total: int) -> None:
    line = f"STATUS completed={int(completed)} total={int(total)}"
    print(line, flush=True)
    with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"event": "STATUS", "completed": int(completed), "total": int(total)},
                separators=(",", ":"),
            )
            + "\n"
        )


def run(arguments: argparse.Namespace) -> int:
    output = Path(arguments.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        raise FileExistsError(f"refusing to reuse non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if not arguments.dry_run and not arguments.text_embedding_cache_dir:
        raise ValueError(
            "formal Light-WAM training requires --text-embedding-cache-dir; "
            "the official model is configured without an online text encoder"
        )
    if not arguments.dry_run:
        _validate_hdf5_root(arguments.data_root, label="--data-root")
        if arguments.val_data_root:
            _validate_hdf5_root(arguments.val_data_root, label="--val-data-root")
    cfg = build_peg_train_config(arguments)
    _write_resolved_config(output / "config.yaml", cfg)
    if arguments.dry_run:
        print(f"DRY_RUN config={output / 'config.yaml'}", flush=True)
        return 0

    from lightwam.runtime import run_training

    with (output / "run.log").open("a", encoding="utf-8") as log:
        initial_total = int(arguments.max_steps or arguments.num_epochs)
        progress_state = {"completed": 0, "total": initial_total}

        def on_progress(completed: int, observed_total: int | None) -> None:
            total = int(observed_total or progress_state["total"])
            progress_state.update(completed=int(completed), total=total)
            _write_progress(output, completed, total)

        tee_out = _Tee(sys.stdout, log, on_progress=on_progress)
        tee_err = _Tee(sys.stderr, log, on_progress=on_progress)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            _write_progress(output, 0, initial_total)
            run_training(cfg)
            done_completed = int(progress_state["completed"] or initial_total)
            done_total = int(progress_state["total"])
            print(
                f"DONE completed={done_completed} total={done_total}",
                flush=True,
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--val-data-root")
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", required=True)
    parser.add_argument("--vendor-root", default="third_party/lightwam/Light-WAM")
    parser.add_argument(
        "--official-task-config",
        default="libero_uncond_2cam224_1e-4",
        help="official Light-WAM task config used as the model/trainer template",
    )
    parser.add_argument("--stats-path")
    parser.add_argument("--text-embedding-cache-dir")
    parser.add_argument("--instruction", default="insert the peg into the hole")
    parser.add_argument("--video-size", nargs=2, type=int, default=list(PEG_DEFAULT_VIDEO_SIZE))
    parser.add_argument("--val-set-proportion", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
