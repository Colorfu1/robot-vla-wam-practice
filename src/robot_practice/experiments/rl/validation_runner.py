"""Three-stage validation orchestration, S1 training dispatch, and preflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import selectors
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .paired import read_jsonl, summarize_paired
from .residual_contract import compose_residual_action

REQUIRED_IMPLEMENTATION_PATHS = (
    "configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml",
    "scripts/experiments/rl/run_three_stage_validation.sh",
    "scripts/experiments/rl/run_policy_decorator_paper_multigpu.sh",
    "scripts/experiments/rl/run_policy_decorator_official_base_eval_multigpu.sh",
    "scripts/experiments/rl/run_policy_decorator_base_then_sac_multigpu.sh",
    "scripts/experiments/robotwin/run_lightwam_residual_rl.sh",
    "src/robot_practice/experiments/rl",
    "src/robot_practice/experiments/rl/policy_decorator_official_base_eval.py",
    "tests/experiments/rl",
    "tests/experiments/robotwin",
)
PACKAGE_NAMES = (
    "torch",
    "numpy",
    "deepspeed",
    "curobo",
    "gymnasium",
    "mani-skill2",
    "diffusers",
    "transformers",
)
S1_TRAINING_SCRIPT = "online/pi_dec_diffusion_maniskill2.py"
S1_FINAL_STEP = 4_000_000
S1_TRAINING_TOTAL = 4_000_000


def _repo_root() -> Path:
    configured = os.environ.get("PRACTICE_ROOT")
    return Path(configured).expanduser().resolve() if configured else Path.cwd().resolve()


def _path(value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else _repo_root() / candidate


def _relative(value: Path) -> str:
    try:
        return value.resolve().relative_to(_repo_root()).as_posix()
    except ValueError:
        return value.as_posix()


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - image and project env ship PyYAML
            raise RuntimeError("PyYAML is required to read the RL YAML config") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"RL config must be a mapping: {path}")
    if value.get("schema_version") != 1:
        raise ValueError(f"unsupported RL config schema: {value.get('schema_version')!r}")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:  # JSON is a valid YAML subset and keeps preflight usable in a minimal env.
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _status(output: Path, event: str, completed: int, total: int) -> None:
    line = f"{event} completed={completed} total={total}"
    print(line, flush=True)
    with (output / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
    _append_jsonl(
        output / "progress.jsonl",
        {"event": event, "completed": completed, "total": total},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> dict[str, str | None]:
    return {
        name: next(
            (
                importlib.metadata.version(candidate)
                for candidate in (name, name.replace("-", "_"))
                if _distribution_exists(candidate)
            ),
            None,
        )
        for name in PACKAGE_NAMES
    }


def _distribution_exists(name: str) -> bool:
    try:
        importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return {"torch_import_error": f"{type(exc).__name__}: {exc}", "count": 0}
    count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "count": count,
        "devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(count)
        ],
    }


def _split_audit(config: Mapping[str, Any]) -> dict[str, Any]:
    declared = config.get("splits", {})
    if not isinstance(declared, Mapping):
        raise ValueError("config.splits must be a mapping")
    normalized = {
        str(name): tuple(str(item) for item in values) for name, values in declared.items()
    }
    overlaps: dict[str, list[str]] = {}
    names = list(normalized)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            overlap = sorted(set(normalized[left_name]) & set(normalized[right_name]))
            if overlap:
                overlaps[f"{left_name}:{right_name}"] = overlap
    return {
        "splits": {name: list(values) for name, values in normalized.items()},
        "overlaps": overlaps,
        "disjoint": not overlaps,
    }


def _code_hashes(config_path: Path) -> dict[str, str | None]:
    candidates = [
        config_path,
        _repo_root() / "scripts/experiments/rl/run_three_stage_validation.sh",
        _repo_root() / "scripts/experiments/robotwin/run_lightwam_residual_rl.sh",
        _repo_root() / "src/robot_practice/experiments/rl/residual_contract.py",
        _repo_root() / "src/robot_practice/experiments/rl/validation_runner.py",
    ]
    return {_relative(path): _sha256(path) if path.is_file() else None for path in candidates}


def _revisions(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _repo_root()
    vendors = {
        "policy_decorator": root / "datasets/raw/policy_decorator/reference_repo",
        "lightwam": root / "third_party/lightwam/Light-WAM",
        "rlinf": root / "third_party/rlinf/RLinf",
        "rlinf_robotwin": root / "third_party/rlinf/RoboTwin",
    }
    return {
        "declared": dict(config.get("revisions", {})),
        "observed_git_heads": {name: _git_head(path) for name, path in vendors.items()},
        "vendor_paths_present": {name: path.is_dir() for name, path in vendors.items()},
    }


def _runtime_artifacts(
    output: Path,
    *,
    config_path: Path,
    config: dict[str, Any],
    dummy: bool,
    phase: str,
    required_stages: tuple[str, ...],
    formal_artifacts_checked: bool,
) -> dict[str, Any]:
    gpu = _gpu_info()
    formal_gpu_requirements = {
        "s1": int(config["stages"]["s1"].get("formal_gpu_count", 1)),
        "s2": int(config["stages"]["s2"].get("formal_gpu_count", 4)),
        "s3": int(config["stages"]["s3"].get("formal_gpu_count", 8)),
    }
    resource_gate = {
        stage: int(gpu["count"]) >= required
        for stage, required in formal_gpu_requirements.items()
    }
    resolved = dict(config)
    resolved["runtime"] = {
        "phase": phase,
        "dummy_smoke": dummy,
        "python": sys.version.split()[0],
        "repository_root": ".",
        "formal_resource_gate": resource_gate,
        "dummy_resource_gate": int(gpu["count"]) >= 1,
        "data_or_model_downloaded": False,
    }
    _write_yaml(output / "resolved_config.yaml", resolved)
    _write_json(output / "revisions.json", _revisions(config))
    versions = _package_versions()
    (output / "package_versions.txt").write_text(
        "\n".join(f"{name}={version or 'missing'}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )
    _write_json(output / "gpu.json", gpu)
    _write_json(output / "hashes.json", _code_hashes(config_path))
    split = _split_audit(config)
    _write_json(output / "split_audit.json", split)
    _write_json(
        output / "resource_gate.json",
        {
            "formal_required_gpu_count": formal_gpu_requirements,
            "observed_gpu_count": gpu["count"],
            "formal_gate": resource_gate,
            "required_stages": list(required_stages),
            "dummy_single_gpu_gate": int(gpu["count"]) >= 1,
        },
    )
    if not split["disjoint"]:
        raise RuntimeError(f"train/development/test split overlap: {split['overlaps']}")
    return {
        "gpu": gpu,
        "resource_gate": resource_gate,
        "required_stages": list(required_stages),
        "formal_artifacts_checked": formal_artifacts_checked,
        "split_audit": split,
    }


def _require_implementation_paths() -> None:
    missing = [path for path in REQUIRED_IMPLEMENTATION_PATHS if not _path(path).exists()]
    if missing:
        raise FileNotFoundError("missing required local implementation: " + ", ".join(missing))


def _require_formal_artifacts(arguments: argparse.Namespace, *, stage: str) -> None:
    if stage == "s1":
        checkpoint = _path(arguments.checkpoint) if arguments.checkpoint else None
        reference_value = arguments.reference_root or (
            "datasets/raw/policy_decorator/reference_repo"
        )
        reference = _path(reference_value)
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError("formal S1 audit requires --checkpoint")
        if reference is None or not (reference / "envs").is_dir():
            raise FileNotFoundError(
                "formal S1 audit requires a Policy Decorator checkout with envs/"
            )
        if not (reference / S1_TRAINING_SCRIPT).is_file():
            raise FileNotFoundError(
                "formal S1 training requires the pinned Policy Decorator online script: "
                f"{S1_TRAINING_SCRIPT}"
            )
    elif stage == "s2":
        if not (_repo_root() / "third_party/lightwam/Light-WAM").is_dir():
            raise FileNotFoundError("formal S2 audit requires third_party/lightwam/Light-WAM")
    elif stage == "s3":
        for relative in ("third_party/rlinf/RLinf", "third_party/rlinf/RoboTwin"):
            if not _path(relative).is_dir():
                raise FileNotFoundError(f"formal S3 audit requires {relative}")


def _s1_training_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the frozen S1 recipe without accepting experiment-time sweeps."""

    stage = config.get("stages", {}).get("s1", {})
    train = stage.get("train", {})
    residual = config.get("residual_contract", {})
    return {
        "env_id": str(stage.get("formal_env_id", "PegInsertionSide-v2")),
        "total_timesteps": int(train.get("total_timesteps", S1_TRAINING_TOTAL)),
        "num_envs": int(train.get("num_envs", 16)),
        "training_freq": 64,
        "gamma": float(residual.get("gamma", 0.97)),
        "utd": float(residual.get("utd_ratio", 0.25)),
        "batch_size": int(residual.get("batch_size", 1024)),
        "policy_lr": float(residual.get("actor_learning_rate", 1e-4)),
        "q_lr": float(residual.get("critic_learning_rate", 1e-4)),
        "tau": float(residual.get("tau", 0.01)),
        "res_scale": float(residual.get("scale", 0.1)),
        "prog_explore": int(residual.get("progressive_exploration_actions", 30_000)),
        # The shared config's actor_input describes the S2 physical-qpos
        # contract; S1 follows the pinned Policy Decorator recipe and uses obs.
        "actor_input": "obs",
        "critic_input": "sum",
        "learning_starts": 8_000,
        "eval_freq": 50_000,
        "num_eval_episodes": 50,
        "num_eval_envs": 5,
        "save_freq": 1_000_000,
        "ddim_steps": 4,
        "act_horizon": 4,
    }


def _s1_training_command(
    arguments: argparse.Namespace,
    contract: Mapping[str, Any],
    *,
    checkpoint: Path,
    output: Path,
) -> list[str]:
    """Build the pinned Policy Decorator command from the frozen contract."""

    return [
        sys.executable,
        S1_TRAINING_SCRIPT,
        "--exp-name",
        f"s1_residual_seed_{arguments.seed}",
        "--seed",
        str(arguments.seed),
        "--env-id",
        str(contract["env_id"]),
        "--base-policy-ckpt",
        str(checkpoint),
        "--total-timesteps",
        str(contract["total_timesteps"]),
        "--num-envs",
        str(contract["num_envs"]),
        "--training-freq",
        str(contract["training_freq"]),
        "--gamma",
        str(contract["gamma"]),
        "--utd",
        str(contract["utd"]),
        "--batch-size",
        str(contract["batch_size"]),
        "--learning-starts",
        str(contract["learning_starts"]),
        "--policy-lr",
        str(contract["policy_lr"]),
        "--q-lr",
        str(contract["q_lr"]),
        "--tau",
        str(contract["tau"]),
        "--res-scale",
        str(contract["res_scale"]),
        "--prog-explore",
        str(contract["prog_explore"]),
        "--critic-input",
        str(contract["critic_input"]),
        "--actor-input",
        str(contract["actor_input"]),
        "--ddim-steps",
        str(contract["ddim_steps"]),
        "--act-horizon",
        str(contract["act_horizon"]),
        "--eval-freq",
        str(contract["eval_freq"]),
        "--num-eval-episodes",
        str(contract["num_eval_episodes"]),
        "--num-eval-envs",
        str(contract["num_eval_envs"]),
        "--save-freq",
        str(contract["save_freq"]),
        "--policy-frequency",
        "1",
        "--target-network-frequency",
        "1",
        "--bootstrap-at-done",
        "truncated",
        "--max-grad-norm",
        "50",
        "--sync-venv=false",
        "--log-freq",
        "20000",
        "--torch-deterministic=true",
        "--track=false",
        "--capture-video=false",
        "--autotune=true",
        "--cuda=true",
        "--output-dir",
        str(output),
    ]


def _run_s1_formal_training(arguments: argparse.Namespace) -> int:
    """Run and audit one seed of the pinned Policy Decorator residual SAC."""

    _require_implementation_paths()
    config_path = _path(arguments.config)
    config = _load_config(config_path)
    checkpoint = _path(arguments.checkpoint) if arguments.checkpoint else None
    reference_value = arguments.reference_root or "datasets/raw/policy_decorator/reference_repo"
    reference = _path(reference_value)
    stage = config.get("stages", {}).get("s1", {})
    train = stage.get("train", {})
    if (
        str(stage.get("execution_mode", "smoke_only")) != "formal"
        or not bool(stage.get("formal_residual_training", False))
        or not bool(train.get("formal_training", False))
    ):
        raise RuntimeError(
            "formal S1 residual training is disabled by the experiment config; set "
            "stages.s1.execution_mode=formal, stages.s1.formal_residual_training=true, "
            "and stages.s1.train.formal_training=true before launching --phase=s1-train"
        )
    _require_formal_artifacts(arguments, stage="s1")
    if checkpoint is None:  # guarded by _require_formal_artifacts; keeps typing explicit
        raise FileNotFoundError("formal S1 training requires --checkpoint")

    output = _path(arguments.output_root)
    _prepare_output(output)
    contract = _s1_training_contract(config)
    total = int(contract["total_timesteps"])
    if total != S1_FINAL_STEP:
        raise ValueError(
            f"formal S1 requires total_timesteps={S1_FINAL_STEP}, got {total} from config"
        )
    if int(contract["num_envs"]) != 16 or int(contract["training_freq"]) != 64:
        raise ValueError("formal S1 requires num_envs=16 and training_freq=64")
    configured_seed_values = stage.get("seeds", train.get("seeds", (1, 2, 3)))
    configured_seeds = tuple(int(seed) for seed in configured_seed_values)
    if int(arguments.seed) not in configured_seeds:
        raise ValueError(
            f"formal S1 seed must be one of {configured_seeds}, got {arguments.seed}"
        )

    command = _s1_training_command(
        arguments,
        contract,
        checkpoint=checkpoint,
        output=output,
    )
    base_hash_before = _sha256(checkpoint)
    audit_command = [Path(command[0]).name]
    for value in command[1:]:
        if value == str(checkpoint):
            audit_command.append(_relative(checkpoint))
        elif value == str(output):
            audit_command.append(_relative(output))
        else:
            audit_command.append(value)
    _write_json(
        output / "launch.json",
        {
            "phase": "s1-train",
            "seed": int(arguments.seed),
            "reference_root": _relative(reference),
            "base_checkpoint": _relative(checkpoint),
            "base_checkpoint_sha256_before": base_hash_before,
            "training_script": S1_TRAINING_SCRIPT,
            "command": audit_command,
            "contract": dict(contract),
        },
    )
    _status(output, "STATUS", 0, total)

    child_env = os.environ.copy()
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env["PYTHONUNBUFFERED"] = "1"
    existing_pythonpath = child_env.get("PYTHONPATH")
    child_env["PYTHONPATH"] = (
        str(reference)
        + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    )
    process: subprocess.Popen[str] | None = None
    completed = 0
    last_status = time.monotonic()
    step_pattern = re.compile(r"global_step=(\d+)")
    run_log = output / "run.log"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(reference),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:  # pragma: no cover - subprocess always supplies PIPE
            raise RuntimeError("failed to capture S1 training output")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            with run_log.open("a", encoding="utf-8") as log:
                while True:
                    events = selector.select(timeout=5.0)
                    if events:
                        line = process.stdout.readline()
                        if line:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                            match = step_pattern.search(line)
                            if match:
                                completed = min(total, int(match.group(1)))
                                _status(output, "STATUS", completed, total)
                                last_status = time.monotonic()
                    if process.poll() is not None:
                        drained = False
                        while selector.select(timeout=0):
                            line = process.stdout.readline()
                            if not line:
                                break
                            drained = True
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        if not drained:
                            break
                    if time.monotonic() - last_status >= 60:
                        _status(output, "STATUS", completed, total)
                        last_status = time.monotonic()
        finally:
            selector.unregister(process.stdout)
            selector.close()
            process.stdout.close()
        return_code = process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
        raise

    if return_code != 0:
        _status(output, "FAILED", completed, total)
        raise subprocess.CalledProcessError(return_code, command)

    final_candidates = sorted(output.glob(f"**/checkpoints/{S1_FINAL_STEP}.pt"))
    if len(final_candidates) != 1:
        _status(output, "FAILED", completed, total)
        raise FileNotFoundError(
            "formal S1 training completed without exactly one 4M checkpoint under output root"
        )
    final_checkpoint = final_candidates[0]
    base_hash_after = _sha256(checkpoint)
    if base_hash_before != base_hash_after:
        _status(output, "FAILED", completed, total)
        raise RuntimeError("formal S1 training changed the frozen base checkpoint")
    args_path = final_checkpoint.parent.parent / "args.json"
    if not args_path.is_file():
        _status(output, "FAILED", completed, total)
        raise FileNotFoundError("formal S1 final checkpoint is missing its training args.json")
    _write_json(
        output / "training_artifact.json",
        {
            "phase": "s1-train",
            "seed": int(arguments.seed),
            "final_step": S1_FINAL_STEP,
            "final_checkpoint": _relative(final_checkpoint),
            "final_checkpoint_sha256": _sha256(final_checkpoint),
            "training_args": _relative(args_path),
            "base_checkpoint_sha256_before": base_hash_before,
            "base_checkpoint_sha256_after": base_hash_after,
            "base_parameters_frozen": True,
            "formal_result": True,
        },
    )
    _status(output, "DONE", total, total)
    return 0


def _run_preflight(arguments: argparse.Namespace, *, phase: str) -> int:
    _require_implementation_paths()
    dummy = bool(arguments.dummy)
    config_path = _path(arguments.config)
    config = _load_config(config_path)
    output = _path(arguments.output_root)
    _prepare_output(output)
    required_stages = {
        "preflight": ("s1", "s2", "s3"),
        "s1-audit": ("s1",),
        "s2-audit": ("s2",),
        "s3-audit": ("s3",),
    }.get(phase, ("s1", "s2", "s3"))
    formal_artifacts_checked = phase.endswith("-audit") and not dummy
    runtime = _runtime_artifacts(
        output,
        config_path=config_path,
        config=config,
        dummy=dummy,
        phase=phase,
        required_stages=required_stages,
        formal_artifacts_checked=formal_artifacts_checked,
    )
    limitations = []
    if dummy:
        limitations.extend(
            [
                "dummy smoke uses synthetic model/data and does not produce a formal result",
                "S1 formal PegInsertionSide-v2 registration remains supplied by Policy "
                "Decorator envs.maniskill_fixed",
                "S2/S3 formal GPU gates are recorded but not satisfied by a one-GPU smoke",
            ]
        )
    failed_stages = [
        stage for stage in required_stages if not runtime["resource_gate"].get(stage, False)
    ]
    gate_pass = dummy or not failed_stages
    if failed_stages:
        limitations.append(
            "formal GPU gate failed for: "
            + ", ".join(failed_stages)
            + "; use --dummy for a one-GPU interface smoke"
        )
    _write_json(
        output / "preflight.json",
        {
            "phase": phase,
            "gate_pass": gate_pass,
            "dummy": dummy,
            "formal_result": False if dummy else None,
            "formal_artifacts_checked": formal_artifacts_checked,
            "limitations": limitations,
            "runtime": runtime,
        },
    )
    _status(output, "STATUS", 1, 1)
    if not gate_pass:
        _status(output, "FAILED", 1, 1)
        raise RuntimeError("formal resource gate failed for " + ", ".join(failed_stages))
    _status(output, "DONE", 1, 1)
    return 0


def _dummy_s1_train(arguments: argparse.Namespace) -> int:
    import torch

    output = _path(arguments.output_root)
    _prepare_output(output)
    total = max(1, int(arguments.dummy_steps))
    base = torch.zeros((1, 4, 14), dtype=torch.float32)
    residual = torch.zeros((1, 4, 12), dtype=torch.float32)
    vectors = torch.ones(12), torch.full((12,), -1.0), torch.full((12,), 1.0)
    composed = compose_residual_action(base, residual, *vectors, res_scale=0.1)
    torch.testing.assert_close(composed, base)
    for completed in range(1, total + 1):
        _status(output, "STATUS", completed, total)
    _write_json(
        output / "metrics.json",
        {
            "schema_version": 1,
            "result_label": "Smoke",
            "dummy": True,
            "sac_updates": max(1, total - 1),
            "formal_result": False,
            "base_parameters_frozen": True,
            "zero_residual_action_equal": True,
        },
    )
    _status(output, "DONE", total, total)
    return 0


def _episode_file(value: str | None) -> Path:
    if not value:
        raise ValueError("an episode input path is required")
    path = _path(value)
    if path.is_file():
        return path
    candidate = path / "episodes.jsonl"
    if candidate.is_file():
        return candidate
    matches = sorted(path.glob("**/episodes.jsonl")) if path.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"could not resolve a unique episodes.jsonl under {value}")


def _summarize(arguments: argparse.Namespace, *, phase: str) -> int:
    output = _path(arguments.output_root)
    _prepare_output(output)
    result = summarize_paired(
        read_jsonl(_episode_file(arguments.base)),
        read_jsonl(_episode_file(arguments.treatment)),
    )
    formal = bool(getattr(arguments, "formal", False)) and not bool(arguments.dummy)
    result.update(
        {
            "phase": phase,
            "result_label": "Closed-loop" if formal else "Smoke",
            "formal_result": formal,
        }
    )
    _write_json(output / "metrics.json", result)
    _status(output, "STATUS", 1, 1)
    _status(output, "DONE", 1, 1)
    return 0


def _dummy_s2_eval(arguments: argparse.Namespace) -> int:
    from .robotwin_residual import run_dummy_rollout

    output = _path(arguments.output_root)
    _prepare_output(output)
    for replan_steps in (arguments.residual_replan_steps, arguments.official_replan_steps):
        if not replan_steps:
            continue
        target = output / ("replan4" if replan_steps == 4 else f"replan{replan_steps}")
        if replan_steps > 4:
            _prepare_output(target)
            _write_json(
                target / "metrics.json",
                {
                    "result_label": "Smoke",
                    "dummy": True,
                    "formal_result": False,
                    "executed": False,
                    "limitation": (
                        "dummy residual harness only models the 4-step residual decision horizon"
                    ),
                    "requested_replan_steps": replan_steps,
                },
            )
            continue
        child = SimpleNamespace(
            output_root=str(target),
            episodes=arguments.episodes,
            replan_steps=replan_steps,
            seed=arguments.seed,
            eval_seed=arguments.seed,
            task=arguments.task,
            task_config="dummy",
            zero_residual=True,
            res_scale=0.1,
            base_checkpoint=arguments.released_checkpoint,
            base_run=arguments.local_run,
            max_steps=max(8, replan_steps),
        )
        run_dummy_rollout(child)
    _write_json(output / "audit.json", {"dummy": True, "formal_result": False})
    return 0


def _s3_summary(arguments: argparse.Namespace) -> int:
    output = _path(arguments.output_root)
    _prepare_output(output)
    _write_json(
        output / "metrics.json",
        {
            "phase": "s3-summarize",
            "result_label": "Smoke",
            "dummy": bool(arguments.dummy),
            "formal_result": False,
            "released_eval_path_present": bool(
                arguments.released_eval and _path(arguments.released_eval).exists()
            ),
            "training_path_present": bool(
                arguments.training_runs and _path(arguments.training_runs).exists()
            ),
        },
    )
    _status(output, "STATUS", 1, 1)
    _status(output, "DONE", 1, 1)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument(
        "--config", default="configs/experiments/rl/three_stage_residual_lightwam_rlinf.yaml"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--dummy-steps", type=int, default=4)
    parser.add_argument(
        "--checkpoint",
        "--base-policy-ckpt",
        dest="checkpoint",
        help=(
            "checkpoint used by the selected validation phase; "
            "--base-policy-ckpt is a compatibility alias"
        ),
    )
    parser.add_argument("--reference-root")
    parser.add_argument("--base")
    parser.add_argument("--treatment")
    parser.add_argument("--released-checkpoint")
    parser.add_argument("--local-run")
    parser.add_argument("--settings")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--official-replan-steps", type=int, default=24)
    parser.add_argument("--residual-replan-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="move_stapler_pad")
    parser.add_argument("--protocol")
    parser.add_argument("--released-eval")
    parser.add_argument("--training-runs")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="mark a paired summary as a formal closed-loop result (requires non-dummy inputs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    arguments.dummy = bool(arguments.dummy or os.environ.get("DUMMY_SMOKE") == "1")
    phase = arguments.phase
    if phase == "preflight":
        return _run_preflight(arguments, phase=phase)
    if phase == "s1-audit":
        if not arguments.dummy:
            _require_formal_artifacts(arguments, stage="s1")
        result = _run_preflight(arguments, phase=phase)
        return result
    if phase == "s1-train":
        if arguments.dummy:
            return _dummy_s1_train(arguments)
        return _run_s1_formal_training(arguments)
    if phase in {"s1-summarize", "s2-summarize"}:
        return _summarize(arguments, phase=phase)
    if phase == "s2-audit":
        if not arguments.dummy:
            _require_formal_artifacts(arguments, stage="s2")
        return _run_preflight(arguments, phase=phase)
    if phase == "s2-eval-base":
        if not arguments.dummy:
            raise SystemExit(
                "formal Light-WAM evaluation is vendor-specific; use --dummy for wiring smoke"
            )
        return _dummy_s2_eval(arguments)
    if phase == "s3-audit":
        if not arguments.dummy:
            _require_formal_artifacts(arguments, stage="s3")
        return _run_preflight(arguments, phase=phase)
    if phase == "s3-summarize":
        return _s3_summary(arguments)
    raise SystemExit(f"unsupported validation phase: {phase}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
