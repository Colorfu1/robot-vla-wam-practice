"""Select a formal ManiSkill visual checkpoint from frozen development metrics.

This module is intentionally offline.  It never creates a ManiSkill environment
and never loads policy weights.  The remote evaluation wrapper produces one
``metrics.json`` per candidate; this module validates the protocol and ranks
those completed results before test evaluation is allowed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

TASK_ID = "PickCube-v1"
SYSTEM = "visual"
SEED_SET = "development"
CONTROL_MODE = "pd_ee_delta_pose"
DEFAULT_RUN_DIR = Path("outputs/maniskill/m0_probe/train_main")
DEFAULT_SELECTION_DIR = Path("outputs/maniskill/m0_probe/checkpoint_selection")
DEFAULT_EXPECTED_EPISODES = 200
SELECTED_CHECKPOINT_DIRNAME = "selected"


def _practice_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _practice_root() / path
    return path.resolve()


def _portable_path(value: str | Path) -> str:
    path = _resolve_path(value)
    try:
        return path.relative_to(_practice_root()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required JSON is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_path(path: Path) -> None:
    """Remove one generated file or directory without following a symlink."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_selected_checkpoint(
    *,
    source_checkpoint_dir: Path,
    selected_dir: Path,
    selection: Mapping[str, Any],
) -> None:
    """Publish a self-contained selected checkpoint under the selection output.

    The source and output directories can live on different filesystems, so use
    a direct copy with symlinks disabled.  Never use a directory rename or
    replacement operation here.
    """

    source_checkpoint_dir = source_checkpoint_dir.resolve()
    selected_dir = Path(selected_dir)
    if not source_checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Selected checkpoint source is missing: {source_checkpoint_dir}")
    if source_checkpoint_dir == selected_dir.absolute():
        raise ValueError("Selected checkpoint source and destination must be different")

    selected_dir.parent.mkdir(parents=True, exist_ok=True)
    _remove_path(selected_dir)
    published = False
    try:
        shutil.copytree(source_checkpoint_dir, selected_dir, symlinks=False)
        _write_json_atomic(selected_dir / "selected_checkpoint.json", selection)
        published = True
    finally:
        if not published:
            _remove_path(selected_dir)


def parse_checkpoint_steps(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of unique non-negative checkpoint steps."""

    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(
            "--checkpoint-steps must be comma-separated non-negative integers"
        )
    steps = tuple(int(part) for part in parts)
    if not steps or len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("--checkpoint-steps must not contain duplicates")
    return tuple(sorted(steps))


def discover_checkpoint_steps(run_dir: str | Path) -> tuple[int, ...]:
    """Discover complete numeric LeRobot checkpoints in a formal run."""

    checkpoints_dir = _resolve_path(run_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Formal checkpoint directory is missing: {checkpoints_dir}")
    steps = []
    for candidate in checkpoints_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.isdigit():
            continue
        policy_dir = candidate / "pretrained_model"
        if (policy_dir / "config.json").is_file() and (policy_dir / "model.safetensors").is_file():
            steps.append(int(candidate.name))
    if not steps:
        raise FileNotFoundError(f"No complete numeric checkpoints found under {checkpoints_dir}")
    return tuple(sorted(steps))


def checkpoint_policy_path(run_dir: str | Path, step: int) -> Path:
    """Return the expected policy directory for one formal checkpoint."""

    path = _resolve_path(run_dir) / "checkpoints" / f"{step:06d}" / "pretrained_model"
    required = (path / "config.json", path / "model.safetensors")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {step:06d} is incomplete; missing {', '.join(missing)}"
        )
    return path


def candidate_metrics_path(evaluation_root: str | Path, step: int) -> Path:
    """Return the fixed metrics location emitted by the remote selector wrapper."""

    return _resolve_path(evaluation_root) / f"checkpoint_{step:06d}" / SEED_SET / "metrics.json"


def _rate(evaluation: Mapping[str, Any], name: str, *, path: Path) -> float:
    signal = evaluation.get(name)
    if not isinstance(signal, Mapping):
        raise TypeError(f"{path} is missing evaluation.{name}")
    if not isinstance(signal.get("rate"), (int, float)):
        raise TypeError(f"{path} is missing evaluation.{name}.rate")
    value = float(signal["rate"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{path} has invalid evaluation.{name}.rate={value}")
    return value


def _validate_run_manifest(
    metrics_path: Path,
    *,
    expected_policy_path: Path,
) -> dict[str, Any]:
    manifest_path = metrics_path.parent / "run_manifest.json"
    manifest = _read_json(manifest_path)
    actual = manifest.get("policy_path")
    if not isinstance(actual, str) or not actual:
        raise ValueError(f"{manifest_path} is missing policy_path")
    actual_path = _resolve_path(actual)
    if actual_path != expected_policy_path:
        raise ValueError(
            f"{metrics_path} belongs to {actual_path}, expected {expected_policy_path}"
        )
    return manifest


def load_candidate(
    *,
    step: int,
    policy_path: Path,
    metrics_path: Path,
    expected_episodes: int = DEFAULT_EXPECTED_EPISODES,
) -> dict[str, Any]:
    """Validate one completed visual development result and summarize it."""

    metrics = _read_json(metrics_path)
    if metrics.get("system") != SYSTEM:
        raise ValueError(f"{metrics_path} is not a visual evaluation")
    if metrics.get("task") != TASK_ID:
        raise ValueError(f"{metrics_path} is not a {TASK_ID} evaluation")
    if metrics.get("control_mode") != CONTROL_MODE:
        raise ValueError(f"{metrics_path} is not a {CONTROL_MODE} evaluation")
    if metrics.get("seed_set") != SEED_SET:
        raise ValueError(f"{metrics_path} must use seed_set={SEED_SET}")
    if metrics.get("num_episodes") != expected_episodes:
        raise ValueError(
            f"{metrics_path} has num_episodes={metrics.get('num_episodes')}, "
            f"expected {expected_episodes}"
        )
    evaluation = metrics.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise TypeError(f"{metrics_path} is missing evaluation")
    if evaluation.get("num_episodes") != expected_episodes:
        raise ValueError(
            f"{metrics_path} has evaluation.num_episodes={evaluation.get('num_episodes')}, "
            f"expected {expected_episodes}"
        )
    manifest = _validate_run_manifest(metrics_path, expected_policy_path=policy_path)
    return {
        "checkpoint_step": step,
        "policy_path": _portable_path(policy_path),
        "metrics_path": _portable_path(metrics_path),
        "success_at_end_rate": _rate(evaluation, "success_at_end", path=metrics_path),
        "success_once_rate": _rate(evaluation, "success_once", path=metrics_path),
        "first_success_step": evaluation.get("first_success_step"),
        "stage_signal_rates": evaluation.get("stage_signal_rates"),
        "latency": metrics.get("latency"),
        "run_manifest": _portable_path(metrics_path.parent / "run_manifest.json"),
        "policy_seed": metrics.get("policy_seed"),
        "max_episode_steps": metrics.get("max_episode_steps"),
        "sim_backend": manifest.get("sim_backend"),
    }


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank visual checkpoints using the frozen M0 development protocol."""

    if not candidates:
        raise ValueError("At least one checkpoint candidate is required")
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            -float(candidate["success_at_end_rate"]),
            -float(candidate["success_once_rate"]),
            int(candidate["checkpoint_step"]),
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    return ranked


def select_checkpoint(
    *,
    run_dir: str | Path,
    evaluation_root: str | Path,
    output_dir: str | Path,
    checkpoint_steps: Sequence[int] | None = None,
    expected_episodes: int = DEFAULT_EXPECTED_EPISODES,
) -> dict[str, Any]:
    """Validate, rank, and record completed development checkpoint evaluations."""

    if expected_episodes <= 0:
        raise ValueError("expected_episodes must be positive")
    selected_steps = (
        discover_checkpoint_steps(run_dir)
        if checkpoint_steps is None
        else tuple(sorted(int(step) for step in checkpoint_steps))
    )
    if not selected_steps or len(set(selected_steps)) != len(selected_steps):
        raise ValueError("checkpoint_steps must contain at least one unique step")
    candidates = []
    for step in selected_steps:
        policy_path = checkpoint_policy_path(run_dir, step)
        metrics_path = candidate_metrics_path(evaluation_root, step)
        candidates.append(
            load_candidate(
                step=step,
                policy_path=policy_path,
                metrics_path=metrics_path,
                expected_episodes=expected_episodes,
            )
        )
    ranked = rank_candidates(candidates)
    winner = ranked[0]
    output = _resolve_path(output_dir)
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"Selection output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selected_dir = output / SELECTED_CHECKPOINT_DIRNAME
    selected_policy_path = selected_dir / "pretrained_model"
    selection = {
        "schema_version": 2,
        "status": "complete",
        "selection_set": "maniskill_development",
        "task": TASK_ID,
        "system": SYSTEM,
        "seed_set": SEED_SET,
        "expected_episodes": expected_episodes,
        "primary_metric": "success_at_end_rate",
        "tie_breakers": ["success_once_rate_desc", "checkpoint_step_asc"],
        "selected_checkpoint_step": winner["checkpoint_step"],
        "source_policy_path": winner["policy_path"],
        "selected_checkpoint": _portable_path(selected_dir),
        "selected_policy_path": _portable_path(selected_policy_path),
        "selected_metrics_path": winner["metrics_path"],
        "run_dir": _portable_path(run_dir),
        "evaluation_root": _portable_path(evaluation_root),
        "ranking": ranked,
    }
    _copy_selected_checkpoint(
        source_checkpoint_dir=checkpoint_policy_path(run_dir, winner["checkpoint_step"]).parent,
        selected_dir=selected_dir,
        selection=selection,
    )
    _write_json_atomic(output / "checkpoint_results.json", {"ranking": ranked})
    _write_json_atomic(output / "selected_checkpoint.json", selection)
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--checkpoint-steps", type=parse_checkpoint_steps, default=None)
    parser.add_argument("--expected-episodes", type=int, default=DEFAULT_EXPECTED_EPISODES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    selection = select_checkpoint(
        run_dir=arguments.run_dir,
        evaluation_root=arguments.evaluation_root,
        output_dir=arguments.output_dir,
        checkpoint_steps=arguments.checkpoint_steps,
        expected_episodes=arguments.expected_episodes,
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
