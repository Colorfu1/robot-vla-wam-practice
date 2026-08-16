"""Evaluate and select native LeRobot Diffusion Policy checkpoints for M2.

The selector owns only the comparison orchestration.  Each candidate is
evaluated by the existing GPU StackCube evaluator on the fixed development
state bank, then candidates are ranked by closed-loop task performance.  The
test state bank is intentionally not an accepted selection input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from robot_practice.evaluation.common import iso_timestamp, practice_root, write_json_atomic

TASK_ID = "StackCube-v1"
SYSTEM = "visual"
SEED_SET = "development"
CONTROL_MODE = "pd_ee_delta_pose"
DEFAULT_RUN_DIR = Path("outputs/maniskill/m2_diffusion/StackCube-v1/formal_seed1000")
DEFAULT_DATASET_REPO_ID = "local/maniskill_stackcube_dcore_7d_dp_rgb_video"
DEFAULT_DATASET_ROOT = Path("datasets/lerobot/maniskill/stackcube_dcore_7d_dp_rgb_video")
DEFAULT_STATE_BANK_ROOT = Path(
    "outputs/maniskill/m1_wam_protocol_audit/StackCube-v1/state_banks"
)
DEFAULT_SELECTION_DIR = Path(
    "outputs/maniskill/m2_diffusion/StackCube-v1/checkpoint_selection"
)
DEFAULT_EXPECTED_EPISODES = 200
DEFAULT_MAX_STEPS = 500
DEFAULT_N_ACTION_STEPS = 8
DEFAULT_NUM_ENVS = 64
DEFAULT_POLICY_SEED = 1000
SELECTED_CHECKPOINT_DIRNAME = "selected"
SCHEMA_VERSION = 1


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = practice_root() / path
    return path.resolve()


def _portable_path(value: str | Path) -> str:
    path = _resolve_path(value)
    try:
        return path.relative_to(practice_root()).as_posix()
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
    """Discover complete native LeRobot checkpoints in a training run."""

    checkpoints_dir = _resolve_path(run_dir) / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory is missing: {checkpoints_dir}")
    steps = []
    for candidate in checkpoints_dir.iterdir():
        if not candidate.is_dir() or not candidate.name.isdigit():
            continue
        policy_dir = candidate / "pretrained_model"
        if (policy_dir / "config.json").is_file() and (policy_dir / "model.safetensors").is_file():
            steps.append(int(candidate.name))
    if not steps:
        raise FileNotFoundError(f"No complete native checkpoints found under {checkpoints_dir}")
    return tuple(sorted(steps))


def checkpoint_policy_path(run_dir: str | Path, step: int) -> Path:
    """Return the inference-ready policy directory for one checkpoint."""

    path = _resolve_path(run_dir) / "checkpoints" / f"{int(step):06d}" / "pretrained_model"
    required = (path / "config.json", path / "model.safetensors")
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Checkpoint {int(step):06d} is incomplete; missing {', '.join(missing)}"
        )
    return path


def evaluation_dir(selection_dir: str | Path, step: int) -> Path:
    return _resolve_path(selection_dir) / "evaluations" / f"checkpoint_{int(step):06d}" / SEED_SET


def build_evaluation_command(
    *,
    policy_path: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    state_bank_root: Path,
    output_dir: Path,
    expected_episodes: int,
    max_steps: int,
    n_action_steps: int,
    num_envs: int,
    num_gpus: int,
    policy_seed: int,
) -> list[str]:
    """Build the command for the existing StackCube GPU evaluator."""

    return [
        sys.executable,
        "-m",
        "robot_practice.experiments.maniskill.eval_stackcube_gpu",
        "--policy-path",
        str(policy_path),
        "--dataset-repo-id",
        dataset_repo_id,
        "--dataset-root",
        str(dataset_root),
        "--state-bank-root",
        str(state_bank_root),
        "--seed-set",
        SEED_SET,
        "--max-episodes",
        str(expected_episodes),
        "--max-steps",
        str(max_steps),
        "--n-action-steps",
        str(n_action_steps),
        "--num-envs",
        str(num_envs),
        "--num-gpus",
        str(num_gpus),
        "--policy-seed",
        str(policy_seed),
        "--device",
        "cuda",
        "--output-dir",
        str(output_dir),
    ]


def _rate(evaluation: Mapping[str, Any], name: str, *, path: Path) -> float:
    signal = evaluation.get(name)
    if not isinstance(signal, Mapping):
        raise TypeError(f"{path} is missing evaluation.{name}")
    value = signal.get("rate")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TypeError(f"{path} is missing finite evaluation.{name}.rate")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} has invalid evaluation.{name}.rate={result}")
    return result


def _optional_rate(stage_rates: Mapping[str, Any], name: str, *, path: Path) -> float:
    value = stage_rates.get(name, 0.0)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TypeError(f"{path} has invalid evaluation.stage_signal_rates.{name}={value!r}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{path} has invalid evaluation.stage_signal_rates.{name}={result}")
    return result


def _latency_mean(metrics: Mapping[str, Any], *, path: Path) -> float | None:
    latency = metrics.get("latency")
    if not isinstance(latency, Mapping):
        raise TypeError(f"{path} is missing latency")
    policy = latency.get("policy_forward")
    if not isinstance(policy, Mapping):
        raise TypeError(f"{path} is missing latency.policy_forward")
    value = policy.get("mean_ms")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
        raise TypeError(f"{path} has invalid latency.policy_forward.mean_ms={value!r}")
    return float(value)


def _validate_policy_manifest(metrics_path: Path, *, expected_policy_path: Path) -> dict[str, Any]:
    manifest_path = metrics_path.parent / "run_manifest.json"
    manifest = _read_json(manifest_path)
    actual = manifest.get("policy_path")
    if not isinstance(actual, str) or not actual:
        raise ValueError(f"{manifest_path} is missing policy_path")
    actual_path = _resolve_path(actual)
    if actual_path != expected_policy_path.resolve():
        raise ValueError(
            f"{metrics_path} belongs to {actual_path}, expected {expected_policy_path.resolve()}"
        )
    if manifest.get("task") != TASK_ID or manifest.get("control_mode") != CONTROL_MODE:
        raise ValueError(f"{manifest_path} is not a StackCube {CONTROL_MODE} evaluation")
    return manifest


def load_candidate(
    *,
    step: int,
    policy_path: Path,
    metrics_path: Path,
    expected_episodes: int,
    expected_policy_seed: int,
) -> dict[str, Any]:
    """Validate and flatten one completed StackCube development evaluation."""

    metrics = _read_json(metrics_path)
    if metrics.get("system") != SYSTEM:
        raise ValueError(f"{metrics_path} is not a visual evaluation")
    if metrics.get("task") != TASK_ID:
        raise ValueError(f"{metrics_path} is not a {TASK_ID} evaluation")
    if metrics.get("control_mode") != CONTROL_MODE:
        raise ValueError(f"{metrics_path} is not a {CONTROL_MODE} evaluation")
    if metrics.get("seed_set") != SEED_SET:
        raise ValueError(f"{metrics_path} must use seed_set={SEED_SET}")
    if metrics.get("policy_seed") != expected_policy_seed:
        raise ValueError(
            f"{metrics_path} has policy_seed={metrics.get('policy_seed')}, "
            f"expected {expected_policy_seed}"
        )
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
    stage_rates = evaluation.get("stage_signal_rates")
    if not isinstance(stage_rates, Mapping):
        raise TypeError(f"{metrics_path} is missing evaluation.stage_signal_rates")
    manifest = _validate_policy_manifest(metrics_path, expected_policy_path=policy_path)
    return {
        "checkpoint_step": int(step),
        "policy_path": _portable_path(policy_path),
        "metrics_path": _portable_path(metrics_path),
        "success_at_end_rate": _rate(evaluation, "success_at_end", path=metrics_path),
        "success_once_rate": _rate(evaluation, "success_once", path=metrics_path),
        "is_cubeA_grasped_rate": _optional_rate(stage_rates, "is_cubeA_grasped", path=metrics_path),
        "is_cubeA_on_cubeB_rate": _optional_rate(stage_rates, "is_cubeA_on_cubeB", path=metrics_path),
        "is_cubeA_static_rate": _optional_rate(stage_rates, "is_cubeA_static", path=metrics_path),
        "policy_forward_mean_ms": _latency_mean(metrics, path=metrics_path),
        "first_success_step": evaluation.get("first_success_step"),
        "termination_reasons": evaluation.get("termination_reasons"),
        "run_manifest": _portable_path(metrics_path.parent / "run_manifest.json"),
        "max_episode_steps": metrics.get("max_episode_steps"),
        "policy_seed": expected_policy_seed,
        "sim_backend": manifest.get("sim_backend"),
    }


def _latency_rank_value(candidate: Mapping[str, Any]) -> float:
    value = candidate.get("policy_forward_mean_ms")
    if value is None:
        return math.inf
    return float(value)


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rank candidates by fixed closed-loop development criteria."""

    if not candidates:
        raise ValueError("At least one checkpoint candidate is required")
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (
            -float(candidate["success_at_end_rate"]),
            -float(candidate["success_once_rate"]),
            -float(candidate["is_cubeA_on_cubeB_rate"]),
            -float(candidate["is_cubeA_static_rate"]),
            _latency_rank_value(candidate),
            int(candidate["checkpoint_step"]),
        ),
    )
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    return ranked


def _copy_selected_policy(*, source_policy_path: Path, selected_dir: Path, selection: Mapping[str, Any]) -> None:
    """Copy only inference files, excluding multi-GB optimizer state."""

    source_policy_path = source_policy_path.resolve()
    selected_dir = selected_dir.resolve()
    if not source_policy_path.is_dir():
        raise FileNotFoundError(f"Selected policy source is missing: {source_policy_path}")
    selected_policy_path = selected_dir / "pretrained_model"
    if source_policy_path == selected_policy_path:
        raise ValueError("Selected policy source and destination must be different")
    selected_dir.parent.mkdir(parents=True, exist_ok=True)
    if selected_dir.is_symlink() or selected_dir.is_file():
        selected_dir.unlink()
    elif selected_dir.is_dir():
        shutil.rmtree(selected_dir)
    published = False
    try:
        shutil.copytree(source_policy_path, selected_policy_path, symlinks=False)
        write_json_atomic(selected_dir / "selected_checkpoint.json", selection)
        published = True
    finally:
        if not published and selected_dir.exists():
            shutil.rmtree(selected_dir)


def select_completed_checkpoints(
    *,
    run_dir: str | Path,
    selection_dir: str | Path,
    checkpoint_steps: Sequence[int] | None,
    dataset_repo_id: str,
    dataset_root: str | Path,
    state_bank_root: str | Path,
    expected_episodes: int,
    policy_seed: int,
) -> dict[str, Any]:
    """Rank completed candidate metrics and publish a self-contained winner."""

    steps = (
        discover_checkpoint_steps(run_dir)
        if checkpoint_steps is None
        else tuple(sorted(int(step) for step in checkpoint_steps))
    )
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("checkpoint_steps must contain at least one unique step")
    selection_root = _resolve_path(selection_dir)
    evaluation_root = selection_root / "evaluations"
    candidates = []
    for step in steps:
        policy_path = checkpoint_policy_path(run_dir, step)
        metrics_path = evaluation_dir(selection_root, step) / "metrics.json"
        candidates.append(
            load_candidate(
                step=step,
                policy_path=policy_path,
                metrics_path=metrics_path,
                expected_episodes=expected_episodes,
                expected_policy_seed=policy_seed,
            )
        )
    ranked = rank_candidates(candidates)
    winner = ranked[0]
    selected_dir = selection_root / SELECTED_CHECKPOINT_DIRNAME
    selected_policy_path = selected_dir / "pretrained_model"
    selection = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "created_at": iso_timestamp(),
        "selection_set": "m2_diffusion_stackcube_development",
        "task": TASK_ID,
        "system": SYSTEM,
        "control_mode": CONTROL_MODE,
        "seed_set": SEED_SET,
        "expected_episodes": expected_episodes,
        "policy_seed": policy_seed,
        "primary_metric": "success_at_end_rate",
        "tie_breakers": [
            "success_once_rate_desc",
            "is_cubeA_on_cubeB_rate_desc",
            "is_cubeA_static_rate_desc",
            "policy_forward_mean_ms_asc",
            "checkpoint_step_asc",
        ],
        "selected_checkpoint_step": winner["checkpoint_step"],
        "source_policy_path": winner["policy_path"],
        "selected_checkpoint": _portable_path(selected_dir),
        "selected_policy_path": _portable_path(selected_policy_path),
        "selected_metrics_path": winner["metrics_path"],
        "run_dir": _portable_path(run_dir),
        "evaluation_root": _portable_path(evaluation_root),
        "dataset": {
            "repo_id": dataset_repo_id,
            "root": _portable_path(dataset_root),
        },
        "state_bank": {
            "root": _portable_path(state_bank_root),
            "seed_set": SEED_SET,
            "identity": "start_state_fingerprint",
            "test_access": False,
        },
        "ranking": ranked,
    }
    _copy_selected_policy(
        source_policy_path=checkpoint_policy_path(run_dir, winner["checkpoint_step"]),
        selected_dir=selected_dir,
        selection=selection,
    )
    _write_selection_outputs(
        selection_root,
        selection=selection,
        ranked=ranked,
    )
    return selection


def _write_selection_outputs(
    selection_root: Path,
    *,
    selection: Mapping[str, Any],
    ranked: Sequence[Mapping[str, Any]],
) -> None:
    selection_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        selection_root / "checkpoint_results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "selection_set": selection["selection_set"],
            "ranking": list(ranked),
        },
    )
    write_json_atomic(selection_root / "selected_checkpoint.json", selection)


def _run_candidate_evaluation(
    *,
    command: Sequence[str],
    output_dir: Path,
    log_path: Path,
    reuse_completed: bool,
) -> bool:
    metrics_path = output_dir / "metrics.json"
    if metrics_path.is_file():
        if reuse_completed:
            return False
        raise FileExistsError(
            f"Evaluation already exists: {metrics_path}; use --reuse-completed or --select-only"
        )
    if output_dir.exists():
        if output_dir.is_dir() and not output_dir.is_symlink() and not any(output_dir.iterdir()):
            # The first implementation created this empty directory before
            # launching eval_stackcube_gpu.  It is safe to remove exactly this
            # stale empty artifact so the fixed selector can be rerun directly.
            output_dir.rmdir()
        else:
            raise FileExistsError(
                f"Evaluation output already exists without metrics: {output_dir}; "
                "remove it before retrying"
            )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            list(command),
            cwd=practice_root(),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"StackCube checkpoint evaluation failed with status {result.returncode}; "
            f"inspect {log_path}"
        )
    if not metrics_path.is_file():
        raise RuntimeError(f"Evaluator exited successfully but did not write {metrics_path}")
    return True


def evaluate_and_select(
    *,
    run_dir: str | Path,
    selection_dir: str | Path,
    checkpoint_steps: Sequence[int] | None,
    dataset_repo_id: str,
    dataset_root: str | Path,
    state_bank_root: str | Path,
    expected_episodes: int,
    max_steps: int,
    n_action_steps: int,
    num_envs: int,
    num_gpus: int,
    policy_seed: int,
    reuse_completed: bool,
    select_only: bool,
) -> dict[str, Any]:
    """Evaluate all candidates once, then select from validated metrics."""

    if expected_episodes <= 0 or max_steps <= 0 or n_action_steps <= 0:
        raise ValueError("expected_episodes, max_steps, and n_action_steps must be positive")
    if num_envs <= 0 or num_gpus <= 0:
        raise ValueError("num_envs and num_gpus must be positive")
    if not select_only:
        if not _resolve_path(dataset_root).is_dir():
            raise FileNotFoundError(
                f"LeRobot dataset root does not exist: {_resolve_path(dataset_root)}"
            )
        if not _resolve_path(state_bank_root).is_dir():
            raise FileNotFoundError(
                f"State-bank root does not exist: {_resolve_path(state_bank_root)}"
            )
    steps = (
        discover_checkpoint_steps(run_dir)
        if checkpoint_steps is None
        else tuple(sorted(int(step) for step in checkpoint_steps))
    )
    if not steps or len(set(steps)) != len(steps):
        raise ValueError("checkpoint_steps must contain at least one unique step")
    selection_root = _resolve_path(selection_dir)
    logs_root = selection_root / "logs"
    if not select_only:
        for index, step in enumerate(steps, start=1):
            policy_path = checkpoint_policy_path(run_dir, step)
            output_dir = evaluation_dir(selection_root, step)
            command = build_evaluation_command(
                policy_path=policy_path,
                dataset_repo_id=dataset_repo_id,
                dataset_root=_resolve_path(dataset_root),
                state_bank_root=_resolve_path(state_bank_root),
                output_dir=output_dir,
                expected_episodes=expected_episodes,
                max_steps=max_steps,
                n_action_steps=n_action_steps,
                num_envs=num_envs,
                num_gpus=num_gpus,
                policy_seed=policy_seed,
            )
            print(f"[M2 checkpoint selection] evaluating {index}/{len(steps)} step={step:06d}", flush=True)
            _run_candidate_evaluation(
                command=command,
                output_dir=output_dir,
                log_path=logs_root / f"checkpoint_{step:06d}.log",
                reuse_completed=reuse_completed,
            )
    selection = select_completed_checkpoints(
        run_dir=run_dir,
        selection_dir=selection_root,
        checkpoint_steps=steps,
        dataset_repo_id=dataset_repo_id,
        dataset_root=dataset_root,
        state_bank_root=state_bank_root,
        expected_episodes=expected_episodes,
        policy_seed=policy_seed,
    )
    return selection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint-steps", type=parse_checkpoint_steps, default=None)
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--state-bank-root", type=Path, default=DEFAULT_STATE_BANK_ROOT)
    parser.add_argument("--expected-episodes", type=int, default=DEFAULT_EXPECTED_EPISODES)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--policy-seed", type=int, default=DEFAULT_POLICY_SEED)
    parser.add_argument(
        "--reuse-completed",
        action="store_true",
        help="Reuse a candidate metrics.json instead of refusing an existing evaluation.",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Skip rollout and rank already completed candidate evaluations.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print one evaluator command per checkpoint without running it.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.expected_episodes <= 0:
        raise SystemExit("--expected-episodes must be positive")
    if arguments.max_steps <= 0 or arguments.n_action_steps <= 0:
        raise SystemExit("--max-steps and --n-action-steps must be positive")
    if arguments.num_envs <= 0 or arguments.num_gpus <= 0:
        raise SystemExit("--num-envs and --num-gpus must be positive")
    if arguments.policy_seed < 0:
        raise SystemExit("--policy-seed must be non-negative")
    steps = (
        discover_checkpoint_steps(arguments.run_dir)
        if arguments.checkpoint_steps is None
        else arguments.checkpoint_steps
    )
    if arguments.print_command:
        for step in steps:
            command = build_evaluation_command(
                policy_path=checkpoint_policy_path(arguments.run_dir, step),
                dataset_repo_id=arguments.dataset_repo_id,
                dataset_root=_resolve_path(arguments.dataset_root),
                state_bank_root=_resolve_path(arguments.state_bank_root),
                output_dir=evaluation_dir(arguments.selection_dir, step),
                expected_episodes=arguments.expected_episodes,
                max_steps=arguments.max_steps,
                n_action_steps=arguments.n_action_steps,
                num_envs=arguments.num_envs,
                num_gpus=arguments.num_gpus,
                policy_seed=arguments.policy_seed,
            )
            print(shlex.join(command))
        return 0
    selection = evaluate_and_select(
        run_dir=arguments.run_dir,
        selection_dir=arguments.selection_dir,
        checkpoint_steps=steps,
        dataset_repo_id=arguments.dataset_repo_id,
        dataset_root=arguments.dataset_root,
        state_bank_root=arguments.state_bank_root,
        expected_episodes=arguments.expected_episodes,
        max_steps=arguments.max_steps,
        n_action_steps=arguments.n_action_steps,
        num_envs=arguments.num_envs,
        num_gpus=arguments.num_gpus,
        policy_seed=arguments.policy_seed,
        reuse_completed=arguments.reuse_completed,
        select_only=arguments.select_only,
    )
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
