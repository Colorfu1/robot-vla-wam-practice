"""S1 evaluator for the pinned Policy Decorator checkout and dummy smoke.

Formal inference imports the pinned checkout's ``BasePolicy`` and ``Actor``
classes instead of maintaining a second copy of the Diffusion Policy model.
``--dummy`` remains data-free and exercises the same ManiSkill2 reset/step/
render/action contract used by the formal evaluator.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .paired import read_jsonl, validate_manifest

FORMAL_ENV_ID = "PegInsertionSide-v2"
DUMMY_ENV_ID = "PegInsertionSide-v0"
ACTION_DIM = 7
STATE_DIM = 50
EPISODE_LIMIT = 200
OBS_HORIZON = 2
EXECUTION_ACTION_STEPS = 4
RESET_SEED_BASE = 1000
RESET_SEED_SCHEME = "contiguous_disjoint_v1"
SAMPLING_SEED_SCHEME = "policy_seed_reset_seed_chunk_v1"
_HEADLESS_DISPLAY_PROCESS: subprocess.Popen[bytes] | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


def _prepare_output(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"refusing to reuse output path that is not a directory: {path}")
        entries = list(path.iterdir())
        tee_placeholder = (
            len(entries) == 1
            and entries[0].name == "run.log"
            and entries[0].is_file()
            and entries[0].stat().st_size == 0
        )
        if entries and not tee_placeholder:
            raise FileExistsError(f"refusing to reuse non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _ensure_headless_display() -> None:
    """Start a private Xvfb when RGB rendering has no display endpoint.

    SAPIEN's Vulkan path can survive without ``DISPLAY`` on some drivers, but
    that behavior is not stable across repeated container processes.  The
    training image already contains Xvfb, so the dummy smoke uses it when
    available and keeps the formal protocol's RGB render requirement intact.
    """

    global _HEADLESS_DISPLAY_PROCESS
    if os.environ.get("DISPLAY") or _HEADLESS_DISPLAY_PROCESS is not None:
        return
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        return
    for display_number in range(90, 100):
        socket_path = Path(f"/tmp/.X11-unix/X{display_number}")
        if socket_path.exists():
            continue
        display = f":{display_number}"

        def kill_with_parent() -> None:
            # Let SAPIEN finish its interpreter cleanup before the child Xvfb
            # receives the parent-death signal.
            ctypes.CDLL(None).prctl(1, signal.SIGTERM)

        process = subprocess.Popen(
            [xvfb, display, "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=kill_with_parent,
        )
        deadline = time.monotonic() + 2.0
        while (
            process.poll() is None
            and not socket_path.exists()
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        if process.poll() is not None or not socket_path.exists():
            process.terminate()
            continue
        _HEADLESS_DISPLAY_PROCESS = process
        os.environ["DISPLAY"] = display
        return


def _state_id(observation: Any) -> str:
    values = np.ascontiguousarray(np.asarray(observation))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _success(env: Any) -> bool:
    result = env.unwrapped.evaluate()
    value = result.get("success", False)
    array = np.asarray(value).reshape(-1)
    return bool(array.size and array[0])


def _register_reference_env(reference_root: Path | None) -> tuple[bool, str | None]:
    """Add the pinned checkout and import its v2 registration when available."""

    if reference_root is None:
        return False, "no reference root supplied"
    root = reference_root.expanduser().resolve()
    if not root.is_dir():
        return False, f"reference root does not exist: {reference_root}"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        __import__("envs.maniskill_fixed")
    except Exception as exc:  # pragma: no cover - depends on external checkout
        return False, f"could not import envs.maniskill_fixed: {type(exc).__name__}: {exc}"
    return True, None


def _make_runtime_env(
    *, requested_env_id: str, reference_root: Path | None, dummy: bool
) -> tuple[Any, str, str | None]:
    import gymnasium as gym
    import mani_skill2.envs  # noqa: F401

    registered, registration_error = _register_reference_env(reference_root)
    runtime_env_id = requested_env_id
    fallback_reason: str | None = None
    try:
        gym.spec(runtime_env_id)
    except Exception as exc:
        if not dummy or requested_env_id != FORMAL_ENV_ID:
            raise RuntimeError(
                f"{requested_env_id} is not registered. The formal v2 route requires the pinned "
                "Policy Decorator checkout and envs.maniskill_fixed import; "
                f"registration detail: {registration_error or exc}"
            ) from exc
        runtime_env_id = DUMMY_ENV_ID
        fallback_reason = (
            f"formal {FORMAL_ENV_ID} unavailable in dummy process; exercised native "
            f"ManiSkill2 {DUMMY_ENV_ID}. reference_registered={registered}"
        )

    environment = gym.make(
        runtime_env_id,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        reward_mode="sparse",
        render_mode="rgb_array",
    )
    return environment, runtime_env_id, fallback_reason


def _reference_online_module(reference_root: Path):
    """Import the online Policy Decorator implementation from the pinned checkout."""

    root = reference_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Policy Decorator checkout does not exist: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        return importlib.import_module("online.pi_dec_diffusion_maniskill2")
    except Exception as exc:  # pragma: no cover - dependency-specific on the S1 image
        raise RuntimeError(
            "could not import online.pi_dec_diffusion_maniskill2 from the pinned "
            f"Policy Decorator checkout: {type(exc).__name__}: {exc}"
        ) from exc


def _load_base_args(
    checkpoint_path: Path,
    args_path: Path | None,
    *,
    ddim_steps: int,
    act_horizon: int,
) -> SimpleNamespace:
    if args_path is None:
        candidates = (
            checkpoint_path.parent.parent / "args.json",
            checkpoint_path.parent / "args.json",
        )
        args_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if args_path is None or not args_path.is_file():
        raise FileNotFoundError(
            "could not resolve S1 args.json; pass --base-policy-args explicitly"
        )
    payload = json.loads(args_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"S1 args.json must contain an object: {args_path}")
    payload["ddim_steps"] = int(ddim_steps)
    payload["act_horizon"] = int(act_horizon)
    required = (
        "obs_horizon",
        "pred_horizon",
        "diffusion_step_embed_dim",
        "unet_dims",
        "n_groups",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"S1 args.json is missing model fields: {missing}")
    if payload.get("env_id") != FORMAL_ENV_ID:
        raise ValueError(f"S1 args.json env_id must be {FORMAL_ENV_ID!r}")
    if payload.get("control_mode") != "pd_ee_delta_pose":
        raise ValueError("S1 args.json control_mode must be 'pd_ee_delta_pose'")
    return SimpleNamespace(**payload)


def _policy_vector_spec(gym: Any, environment: Any, obs_horizon: int) -> SimpleNamespace:
    observation_space = environment.observation_space
    action_space = environment.action_space
    if not isinstance(observation_space, gym.spaces.Box) or observation_space.shape is None:
        raise RuntimeError("formal S1 requires a flat Box state observation space")
    if len(observation_space.shape) != 1:
        raise RuntimeError(
            "formal S1 state observation must be one-dimensional before frame stacking; "
            f"got {observation_space.shape}"
        )
    if not isinstance(action_space, gym.spaces.Box) or action_space.shape != (ACTION_DIM,):
        raise RuntimeError(
            f"formal S1 action space must be a {ACTION_DIM}D Box, got {action_space}"
        )
    stacked_observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(int(obs_horizon), int(observation_space.shape[0])),
        dtype=np.float32,
    )
    return SimpleNamespace(
        single_observation_space=stacked_observation_space,
        single_action_space=action_space,
    )


def _torch_load(path: Path, *, device: Any) -> Any:
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older Torch builds
        return torch.load(path, map_location=device)


def _load_base_policy(
    reference_module: Any,
    environment: Any,
    checkpoint_path: Path,
    args_path: Path | None,
    *,
    device: Any,
    ddim_steps: int,
    act_horizon: int,
) -> tuple[Any, SimpleNamespace, int]:
    import gymnasium as gym

    args = _load_base_args(
        checkpoint_path,
        args_path,
        ddim_steps=ddim_steps,
        act_horizon=act_horizon,
    )
    vector_spec = _policy_vector_spec(gym, environment, int(args.obs_horizon))
    base_policy = reference_module.BasePolicy(vector_spec, args).to(device)
    checkpoint = _torch_load(checkpoint_path, device=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("S1 checkpoint must be a mapping")
    for key in ("ema_agent", "agent"):
        if key in checkpoint:
            base_policy.load_state_dict(checkpoint[key])
            print(f"Loaded S1 base policy from checkpoint[{key}]", flush=True)
            break
    else:
        raise RuntimeError("S1 checkpoint must contain ema_agent or agent")
    base_policy.eval()
    base_policy.requires_grad_(False)
    return base_policy, args, int(environment.observation_space.shape[0])


def _resolve_residual_checkpoint(root: Path, seed: int, step: int) -> Path:
    seed_root = root / f"seed_{seed}"
    candidates = sorted(seed_root.glob(f"**/checkpoints/{step}.pt")) if seed_root.is_dir() else []
    if not candidates:
        for candidate in sorted(root.glob(f"**/checkpoints/{step}.pt")):
            args_path = candidate.parent.parent / "args.json"
            if not args_path.is_file():
                continue
            payload = json.loads(args_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and int(payload.get("seed", -1)) == seed:
                candidates.append(candidate)
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) != 1:
        raise FileNotFoundError(
            f"expected exactly one residual checkpoint for seed {seed} at step {step}, "
            f"found {len(unique)} under {root}"
        )
    return unique[0]


def _load_residual_actor(
    reference_module: Any,
    environment: Any,
    checkpoint_path: Path,
    *,
    base_args: SimpleNamespace,
    device: Any,
    actor_input: str | None,
) -> tuple[Any, float, str]:
    import gymnasium as gym

    training_args_path = checkpoint_path.parent.parent / "args.json"
    training_payload: dict[str, Any] = {}
    if training_args_path.is_file():
        value = json.loads(training_args_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            training_payload = value
    resolved_actor_input = actor_input or str(training_payload.get("actor_input", "obs"))
    if resolved_actor_input not in {"obs", "obs_base_action"}:
        raise ValueError(f"unsupported residual actor_input: {resolved_actor_input}")
    horizon = int(base_args.act_horizon)
    action_low = np.tile(np.asarray(environment.action_space.low, dtype=np.float32), horizon)
    action_high = np.tile(np.asarray(environment.action_space.high, dtype=np.float32), horizon)
    actor_env = SimpleNamespace(
        single_observation_space=gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(int(environment.observation_space.shape[0]),),
            dtype=np.float32,
        ),
        single_action_space=gym.spaces.Box(
            low=action_low,
            high=action_high,
            shape=(ACTION_DIM * horizon,),
            dtype=np.float32,
        ),
    )
    actor_args = SimpleNamespace(**vars(base_args))
    actor_args.actor_input = resolved_actor_input
    actor = reference_module.Actor(actor_env, actor_args).to(device)
    checkpoint = _torch_load(checkpoint_path, device=device)
    if not isinstance(checkpoint, dict) or "res_actor" not in checkpoint:
        raise RuntimeError(f"residual checkpoint has no res_actor: {checkpoint_path}")
    actor.load_state_dict(checkpoint["res_actor"])
    actor.eval()
    actor.requires_grad_(False)
    res_scale = float(training_payload.get("res_scale", 0.1))
    return actor, res_scale, resolved_actor_input


def _formal_manifest(
    arguments: argparse.Namespace, seeds: list[int]
) -> list[dict[str, Any]]:
    total = len(seeds) * arguments.episodes_per_seed
    if arguments.manifest:
        rows = read_jsonl(arguments.manifest)
        validate_manifest(rows, source="input manifest")
        if len(rows) != total:
            raise ValueError(
                f"input manifest has {len(rows)} episodes, expected {total} from CLI seeds"
            )
        expected_contract = {
            "action_dim": ACTION_DIM,
            "n_obs_steps": OBS_HORIZON,
            "action_horizon": int(arguments.act_horizon),
            "execution_action_steps": EXECUTION_ACTION_STEPS,
        }
        seen_reset_seeds: set[int] = set()
        for row in rows:
            if row.get("requested_env_id") != arguments.env_id:
                raise ValueError("input manifest requested_env_id does not match --env-id")
            if "reset_seed" not in row:
                raise ValueError("input manifest must contain reset_seed")
            reset_seed = int(row["reset_seed"])
            if reset_seed in seen_reset_seeds:
                raise ValueError(
                    "input manifest must use a disjoint reset_seed for every episode; "
                    f"duplicate reset_seed={reset_seed}"
                )
            seen_reset_seeds.add(reset_seed)
            for field, expected in expected_contract.items():
                if row.get(field) != expected:
                    raise ValueError(
                        f"input manifest {field}={row.get(field)!r} does not match "
                        f"the S1 contract ({expected!r})"
                    )
            if row.get("policy_seed") is not None and int(row["policy_seed"]) not in seeds:
                raise ValueError(
                    f"input manifest policy_seed={row['policy_seed']!r} is not in --seeds"
                )
        return rows

    rows = []
    for seed_index, seed in enumerate(seeds):
        for episode_index in range(arguments.episodes_per_seed):
            rows.append(
                {
                    "episode_key": f"s1-{seed}-{episode_index}",
                    "policy_seed": seed,
                    # Keep each policy-seed block disjoint.  The first block
                    # starts at 1001 and later blocks cannot silently replay
                    # the same episodes.
                    "reset_seed": (
                        RESET_SEED_BASE
                        + 1
                        + seed_index * arguments.episodes_per_seed
                        + episode_index
                    ),
                    "requested_env_id": arguments.env_id,
                    "action_dim": ACTION_DIM,
                    "n_obs_steps": OBS_HORIZON,
                    "action_horizon": int(arguments.act_horizon),
                    "execution_action_steps": EXECUTION_ACTION_STEPS,
                }
            )
    return rows


def _manifest_policy_seed(row: dict[str, Any], index: int, seeds: list[int], episodes: int) -> int:
    value = row.get("policy_seed")
    if value is not None:
        return int(value)
    return seeds[index // episodes]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sampling_seed(policy_seed: int, reset_seed: int, chunk_index: int) -> int:
    """Derive a reproducible, paired diffusion-sampling seed.

    The previous evaluator keyed Torch sampling only by ``reset_seed``, so
    nominally different policy seeds replayed the same stochastic action
    sequence. Include all three episode coordinates while staying within the
    positive 32-bit range accepted by Torch and CUDA generators.
    """

    return int(
        (
            (int(policy_seed) * 1_000_003)
            + (int(reset_seed) * 97)
            + int(chunk_index)
        )
        % (2**31 - 1)
    )


def _parse_record_episode_ids(value: str | None) -> tuple[int, ...] | None:
    if value is None or not value.strip():
        return None
    try:
        episode_ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("--record-episode-ids must be a comma-separated integer list") from exc
    if not episode_ids:
        raise ValueError("--record-episode-ids must contain at least one integer")
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("--record-episode-ids must not contain duplicates")
    if any(episode_id < 0 for episode_id in episode_ids):
        raise ValueError("--record-episode-ids must contain non-negative episode IDs")
    return episode_ids


def _make_rollout_recorder(
    arguments: argparse.Namespace,
    *,
    metadata: dict[str, Any],
) -> Any | None:
    if not arguments.record_rollout_dir:
        return None
    if arguments.record_episodes <= 0:
        raise ValueError("--record-episodes must be positive")
    from .policy_decorator_peg_visualization import PolicyDecoratorPegRolloutRecorder

    return PolicyDecoratorPegRolloutRecorder(
        output_dir=arguments.record_rollout_dir,
        max_episodes=int(arguments.record_episodes),
        fps=float(arguments.record_fps),
        episode_ids=_parse_record_episode_ids(arguments.record_episode_ids),
        metadata=metadata,
    )


def _render_s1_frame(environment: Any) -> np.ndarray:
    frame = np.asarray(environment.render())
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f"S1 RGB render contract failed: shape={frame.shape}")
    if frame.dtype != np.uint8:
        if not np.issubdtype(frame.dtype, np.integer):
            raise RuntimeError(f"S1 RGB render contract requires integer pixels: {frame.dtype}")
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _reward_scalar(value: Any) -> float:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise RuntimeError(f"S1 reward must be one finite scalar, got {value!r}")
    return float(array[0])


def run_formal_evaluation(arguments: argparse.Namespace) -> dict[str, Any]:
    import torch

    checkpoint_path = Path(arguments.base_policy_ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"S1 base checkpoint does not exist: {checkpoint_path}")
    reference_root = Path(arguments.reference_root).expanduser().resolve()
    _ensure_headless_display()
    environment, runtime_env_id, fallback_reason = _make_runtime_env(
        requested_env_id=arguments.env_id,
        reference_root=reference_root,
        dummy=False,
    )
    output_dir = Path(arguments.output_dir)
    _prepare_output(output_dir)
    manifest_path = output_dir / "episode_manifest.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    progress_path = output_dir / "progress.jsonl"
    run_log_path = output_dir / "run.log"
    seeds = [int(value) for value in arguments.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if arguments.episodes_per_seed <= 0:
        raise ValueError("--episodes-per-seed must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference_module = _reference_online_module(reference_root)
    base_policy, base_args, state_dim = _load_base_policy(
        reference_module,
        environment,
        checkpoint_path,
        Path(arguments.base_policy_args).expanduser().resolve()
        if arguments.base_policy_args
        else None,
        device=device,
        ddim_steps=arguments.ddim_steps,
        act_horizon=arguments.act_horizon,
    )
    manifest_rows = _formal_manifest(arguments, seeds)
    residual_actors: dict[int, Any] = {}
    residual_actor_inputs: dict[int, str] = {}
    residual_scales: dict[int, float] = {}
    residual_checkpoint_hashes: dict[str, str] = {}
    if arguments.arm == "residual":
        if not arguments.residual_runs or arguments.checkpoint_step is None:
            raise ValueError("residual evaluation requires --residual-runs and --checkpoint-step")
        residual_root = Path(arguments.residual_runs).expanduser().resolve()
        for seed in seeds:
            residual_checkpoint = _resolve_residual_checkpoint(
                residual_root, seed, int(arguments.checkpoint_step)
            )
            residual_checkpoint_hashes[str(seed)] = _file_sha256(residual_checkpoint)
            actor, res_scale, resolved_actor_input = _load_residual_actor(
                reference_module,
                environment,
                residual_checkpoint,
                base_args=base_args,
                device=device,
                actor_input=arguments.actor_input,
            )
            residual_actors[seed] = actor
            residual_actor_inputs[seed] = resolved_actor_input
            residual_scales[seed] = (
                float(arguments.res_scale) if arguments.res_scale is not None else res_scale
            )

    recorder = _make_rollout_recorder(
        arguments,
        metadata={
            "requested_env_id": arguments.env_id,
            "runtime_env_id": runtime_env_id,
            "arm": arguments.arm,
            "ddim_steps": int(arguments.ddim_steps),
            "act_horizon": int(arguments.act_horizon),
            "base_checkpoint_sha256": _file_sha256(checkpoint_path),
            "residual_checkpoint_step": (
                int(arguments.checkpoint_step) if arguments.arm == "residual" else None
            ),
            "residual_checkpoint_sha256": residual_checkpoint_hashes,
            "residual_actor_input": residual_actor_inputs,
            "residual_scales": residual_scales,
        },
    )
    started = time.perf_counter()
    written_manifest_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    render_shape: list[int] | None = None
    try:
        for index, input_row in enumerate(manifest_rows):
            policy_seed = _manifest_policy_seed(
                input_row, index, seeds, arguments.episodes_per_seed
            )
            reset_seed = int(input_row["reset_seed"])
            environment.action_space.seed(policy_seed)
            environment.observation_space.seed(policy_seed)
            observation, _info = environment.reset(seed=reset_seed)
            observation_array = np.asarray(observation, dtype=np.float32)
            if observation_array.shape != (state_dim,):
                raise RuntimeError(
                    "S1 state contract expected shape "
                    f"{(state_dim,)}, got {observation_array.shape}"
                )
            capture_episode = recorder is not None and recorder.should_capture(index)
            if render_shape is None or capture_episode:
                frame = _render_s1_frame(environment)
            else:
                frame = None
            if render_shape is None:
                if frame is None:  # pragma: no cover - guarded above
                    raise RuntimeError("S1 render validation did not produce a frame")
                render_shape = list(frame.shape)
            actual_state_id = _state_id(observation_array)
            expected_state_id = input_row.get("reset_state_id")
            if expected_state_id is not None and expected_state_id != actual_state_id:
                raise RuntimeError(
                    f"reset_state_id mismatch for {input_row['episode_key']!r}: "
                    f"expected {expected_state_id}, got {actual_state_id}"
                )
            row = {
                **input_row,
                "policy_seed": policy_seed,
                "reset_state_id": actual_state_id,
                "requested_env_id": arguments.env_id,
                "runtime_env_id": runtime_env_id,
                "action_dim": ACTION_DIM,
                "n_obs_steps": int(base_args.obs_horizon),
                "action_horizon": int(arguments.act_horizon),
                "execution_action_steps": 4,
            }
            written_manifest_rows.append(row)
            _append_jsonl(manifest_path, row)
            if capture_episode:
                if frame is None:  # pragma: no cover - guarded above
                    raise RuntimeError("S1 rollout capture did not produce the reset frame")
                recorder.start_episode(
                    episode_id=index,
                    reset_seed=reset_seed,
                    state=observation_array,
                    frame=frame,
                    metadata=row,
                )
            history = [observation_array.copy() for _ in range(int(base_args.obs_horizon))]
            episode_started = time.perf_counter()
            success_once = False
            first_success_step: int | None = None
            executed_steps = 0
            terminated = False
            truncated = False
            episode_return = 0.0
            while executed_steps < EPISODE_LIMIT:
                obs_seq = np.stack(history, axis=0)[None, ...]
                decision_step = executed_steps // max(1, int(arguments.act_horizon))
                sampling_seed = _sampling_seed(
                    policy_seed,
                    reset_seed,
                    decision_step,
                )
                torch.manual_seed(sampling_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sampling_seed)
                obs_tensor = torch.from_numpy(obs_seq).to(device)
                with torch.no_grad():
                    base_action_tensor = base_policy.get_eval_action(obs_tensor)
                    residual_tensor = torch.zeros_like(base_action_tensor)
                    composed_action_tensor = base_action_tensor
                    if arguments.arm == "residual":
                        actor = residual_actors[policy_seed]
                        actor_input = obs_tensor[:, -1]
                        if residual_actor_inputs[policy_seed] == "obs_base_action":
                            actor_input = torch.cat(
                                [actor_input, base_action_tensor.flatten(start_dim=1)], dim=1
                            )
                        residual_tensor = actor.get_eval_action(actor_input).reshape(
                            -1, int(base_args.act_horizon), ACTION_DIM
                        )
                        composed_action_tensor = base_action_tensor + (
                            residual_scales[policy_seed] * residual_tensor
                        )
                base_action_seq = base_action_tensor.detach().cpu().numpy()[0]
                residual_unit_seq = residual_tensor.detach().cpu().numpy()[0]
                residual_delta_seq = (
                    residual_scales.get(policy_seed, 0.0) * residual_unit_seq
                )
                composed_action_seq = composed_action_tensor.detach().cpu().numpy()[0]
                action_seq = np.clip(
                    composed_action_seq,
                    np.asarray(environment.action_space.low),
                    np.asarray(environment.action_space.high),
                )
                for action_index, action in enumerate(action_seq):
                    observation, _reward, terminated, truncated, _info = environment.step(action)
                    observation_array = np.asarray(observation, dtype=np.float32)
                    history.append(observation_array)
                    history = history[-int(base_args.obs_horizon) :]
                    executed_steps += 1
                    reward = _reward_scalar(_reward)
                    episode_return += reward
                    success = _success(environment)
                    if success and not success_once:
                        success_once = True
                        first_success_step = executed_steps
                    if capture_episode:
                        recorder.record_step(
                            state=observation_array,
                            frame=_render_s1_frame(environment),
                            base_action=base_action_seq[action_index],
                            residual_unit=residual_unit_seq[action_index],
                            residual_delta=residual_delta_seq[action_index],
                            composed_action=action_seq[action_index],
                            reward=reward,
                            success=success,
                            success_once=success_once,
                            terminated=terminated,
                            truncated=truncated,
                            decision_step=decision_step,
                            horizon_index=action_index,
                        )
                    if terminated or truncated or executed_steps >= EPISODE_LIMIT:
                        break
                if terminated or truncated:
                    break
            episode = {
                **row,
                "success": bool(success_once),
                "first_success_step": first_success_step,
                "executed_action_count": executed_steps,
                "policy_chunk_count": math.ceil(
                    executed_steps / max(1, int(arguments.act_horizon))
                ),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "return": episode_return,
                "latency_s": time.perf_counter() - episode_started,
                "arm": arguments.arm,
                "model_kind": f"policy_decorator_{arguments.arm}",
            }
            episode_rows.append(episode)
            if capture_episode:
                recorder.finish_episode(
                    success=success_once,
                    first_success_step=first_success_step,
                    return_total=episode_return,
                    steps=executed_steps,
                    decision_steps=math.ceil(
                        executed_steps / max(1, int(arguments.act_horizon))
                    ),
                )
            _append_jsonl(episodes_path, episode_rows[-1])
            completed = len(episode_rows)
            line = f"STATUS completed={completed} total={len(manifest_rows)}"
            print(line, flush=True)
            with run_log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            _append_jsonl(
                progress_path,
                {"event": "STATUS", "completed": completed, "total": len(manifest_rows)},
            )
    finally:
        environment.close()

    rollout_manifest = None if recorder is None else recorder.finalize()
    validate_manifest(written_manifest_rows, source="formal manifest")
    validate_manifest(episode_rows, source="formal episodes")
    done_line = f"DONE completed={len(episode_rows)} total={len(manifest_rows)}"
    print(done_line, flush=True)
    with run_log_path.open("a", encoding="utf-8") as stream:
        stream.write(done_line + "\n")
    _append_jsonl(
        progress_path,
        {"event": "DONE", "completed": len(episode_rows), "total": len(manifest_rows)},
    )
    metrics = {
        "schema_version": 1,
        "result_label": "Closed-loop",
        "requested_env_id": arguments.env_id,
        "runtime_env_id": runtime_env_id,
        "runtime_fallback_reason": fallback_reason,
        "arm": arguments.arm,
        "dummy": False,
        "episodes": len(episode_rows),
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "render_shape": render_shape,
        "action_dim": ACTION_DIM,
        "elapsed_s": time.perf_counter() - started,
        "data_or_model_downloaded": False,
        "formal_result": True,
        "base_checkpoint_sha256": _file_sha256(checkpoint_path),
        "manifest_sha256": _file_sha256(manifest_path),
        "unique_reset_seeds": len({row["reset_seed"] for row in written_manifest_rows}),
        "reset_seed_scheme": RESET_SEED_SCHEME,
        "sampling_seed_scheme": SAMPLING_SEED_SCHEME,
        "input_manifest_sha256": (
            _file_sha256(Path(arguments.manifest).expanduser().resolve())
            if arguments.manifest
            else None
        ),
        "residual_checkpoint_step": (
            int(arguments.checkpoint_step) if arguments.arm == "residual" else None
        ),
        "residual_checkpoint_sha256": residual_checkpoint_hashes,
        "residual_actor_input": residual_actor_inputs,
        "residual_scales": residual_scales,
        "rollout_recording": (
            None
            if recorder is None
            else {
                "raw_dir": str(Path(arguments.record_rollout_dir).expanduser()),
                "episodes": len(rollout_manifest["episodes"]),
                "fps": float(arguments.record_fps),
            }
        ),
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "resolved_config.json",
        {
            "requested_env_id": arguments.env_id,
            "runtime_env_id": runtime_env_id,
            "arm": arguments.arm,
            "seeds": seeds,
            "episodes_per_seed": arguments.episodes_per_seed,
            "ddim_steps": arguments.ddim_steps,
            "act_horizon": arguments.act_horizon,
            "base_policy_args": "args.json",
            "input_manifest": bool(arguments.manifest),
            "reset_seed_scheme": RESET_SEED_SCHEME,
            "sampling_seed_scheme": SAMPLING_SEED_SCHEME,
            "residual_checkpoint_step": (
                int(arguments.checkpoint_step) if arguments.arm == "residual" else None
            ),
            "residual_checkpoint_sha256": residual_checkpoint_hashes,
            "residual_actor_input": residual_actor_inputs,
            "residual_scales": residual_scales,
            "record_rollout_dir": bool(arguments.record_rollout_dir),
            "record_episodes": int(arguments.record_episodes),
            "record_episode_ids": _parse_record_episode_ids(arguments.record_episode_ids),
            "record_fps": float(arguments.record_fps),
        },
    )
    return metrics


def run_dummy_evaluation(arguments: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(arguments.output_dir)
    _prepare_output(output_dir)
    seeds = [int(value) for value in arguments.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if arguments.episodes_per_seed <= 0:
        raise ValueError("--episodes-per-seed must be positive")
    if arguments.dummy_steps <= 0 or arguments.dummy_steps > EPISODE_LIMIT:
        raise ValueError(f"--dummy-steps must be in [1, {EPISODE_LIMIT}]")

    _ensure_headless_display()
    reference_root = Path(arguments.reference_root) if arguments.reference_root else None
    environment, runtime_env_id, fallback_reason = _make_runtime_env(
        requested_env_id=arguments.env_id,
        reference_root=reference_root,
        dummy=True,
    )
    manifest_path = output_dir / "episode_manifest.jsonl"
    episodes_path = output_dir / "episodes.jsonl"
    progress_path = output_dir / "progress.jsonl"
    run_log_path = output_dir / "run.log"
    manifest_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    total = len(seeds) * arguments.episodes_per_seed
    started = time.perf_counter()
    render_shape: list[int] | None = None
    recorder = _make_rollout_recorder(
        arguments,
        metadata={
            "requested_env_id": arguments.env_id,
            "runtime_env_id": runtime_env_id,
            "arm": arguments.arm,
            "ddim_steps": int(arguments.ddim_steps),
            "act_horizon": int(arguments.act_horizon),
            "model_kind": "dummy_zero_policy",
        },
    )
    try:
        for seed_index, seed in enumerate(seeds):
            for episode_index in range(arguments.episodes_per_seed):
                episode_id = len(episode_rows)
                ordinal = episode_id + 1
                reset_seed = (
                    RESET_SEED_BASE
                    + 1
                    + seed_index * arguments.episodes_per_seed
                    + episode_index
                )
                observation, _info = environment.reset(seed=reset_seed)
                observation_array = np.asarray(observation)
                if observation_array.shape != (STATE_DIM,):
                    raise RuntimeError(
                        "S1 state contract expected shape "
                        f"{(STATE_DIM,)}, got {observation_array.shape}"
                    )
                capture_episode = recorder is not None and recorder.should_capture(episode_id)
                if render_shape is None or capture_episode:
                    frame = _render_s1_frame(environment)
                else:
                    frame = None
                if render_shape is None:
                    if frame is None:  # pragma: no cover - guarded above
                        raise RuntimeError("S1 render validation did not produce a frame")
                    render_shape = list(frame.shape)
                episode_key = f"s1-{seed}-{episode_index}"
                manifest = {
                    "episode_key": episode_key,
                    "policy_seed": seed,
                    "reset_seed": reset_seed,
                    "reset_state_id": _state_id(observation_array),
                    "requested_env_id": arguments.env_id,
                    "runtime_env_id": runtime_env_id,
                    "action_dim": ACTION_DIM,
                    "n_obs_steps": OBS_HORIZON,
                    "action_horizon": int(arguments.act_horizon),
                    "execution_action_steps": EXECUTION_ACTION_STEPS,
                }
                manifest_rows.append(manifest)
                _append_jsonl(manifest_path, manifest)
                if capture_episode:
                    if frame is None:  # pragma: no cover - guarded above
                        raise RuntimeError("S1 rollout capture did not produce the reset frame")
                    recorder.start_episode(
                        episode_id=episode_id,
                        reset_seed=reset_seed,
                        state=observation_array,
                        frame=frame,
                        metadata=manifest,
                    )
                episode_started = time.perf_counter()
                success_once = False
                first_success_step: int | None = None
                executed_steps = 0
                terminated = False
                truncated = False
                episode_return = 0.0
                for step in range(arguments.dummy_steps):
                    # The dummy base is deliberately deterministic and neutral;
                    # the environment, controller, and observation path remain real.
                    action = np.zeros(ACTION_DIM, dtype=np.float32)
                    observation, _reward, terminated, truncated, _info = environment.step(action)
                    observation_array = np.asarray(observation, dtype=np.float32)
                    reward = _reward_scalar(_reward)
                    episode_return += reward
                    success = _success(environment)
                    if success and not success_once:
                        success_once = True
                        first_success_step = step + 1
                    executed_steps = step + 1
                    if capture_episode:
                        recorder.record_step(
                            state=observation_array,
                            frame=_render_s1_frame(environment),
                            base_action=action,
                            residual_unit=np.zeros(ACTION_DIM, dtype=np.float32),
                            residual_delta=np.zeros(ACTION_DIM, dtype=np.float32),
                            composed_action=action,
                            reward=reward,
                            success=success,
                            success_once=success_once,
                            terminated=terminated,
                            truncated=truncated,
                            decision_step=step // max(1, int(arguments.act_horizon)),
                            horizon_index=step % max(1, int(arguments.act_horizon)),
                        )
                    if terminated or truncated:
                        break
                episode = {
                    **manifest,
                    "success": bool(success_once),
                    "first_success_step": first_success_step,
                    "executed_action_count": executed_steps,
                    "policy_chunk_count": math.ceil(executed_steps / max(1, arguments.act_horizon)),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "return": episode_return,
                    "latency_s": time.perf_counter() - episode_started,
                    "arm": arguments.arm,
                    "model_kind": "dummy_zero_policy",
                }
                episode_rows.append(episode)
                if capture_episode:
                    recorder.finish_episode(
                        success=success_once,
                        first_success_step=first_success_step,
                        return_total=episode_return,
                        steps=executed_steps,
                        decision_steps=math.ceil(
                            executed_steps / max(1, int(arguments.act_horizon))
                        ),
                    )
                _append_jsonl(episodes_path, episode)
                line = f"STATUS completed={ordinal} total={total}"
                print(line, flush=True)
                with run_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                _append_jsonl(
                    progress_path,
                    {"event": "STATUS", "completed": ordinal, "total": total},
                )
    finally:
        environment.close()

    rollout_manifest = None if recorder is None else recorder.finalize()
    validate_manifest(manifest_rows)
    done_line = f"DONE completed={total} total={total}"
    print(done_line, flush=True)
    with run_log_path.open("a", encoding="utf-8") as stream:
        stream.write(done_line + "\n")
    _append_jsonl(progress_path, {"event": "DONE", "completed": total, "total": total})
    metrics = {
        "schema_version": 1,
        "result_label": "Smoke",
        "requested_env_id": arguments.env_id,
        "runtime_env_id": runtime_env_id,
        "runtime_fallback_reason": fallback_reason,
        "arm": arguments.arm,
        "dummy": True,
        "episodes": total,
        "success_rate": float(np.mean([row["success"] for row in episode_rows])),
        "render_shape": render_shape,
        "action_dim": ACTION_DIM,
        "elapsed_s": time.perf_counter() - started,
        "data_or_model_downloaded": False,
        "formal_result": False,
        "manifest_sha256": _file_sha256(manifest_path),
        "unique_reset_seeds": len({row["reset_seed"] for row in manifest_rows}),
        "reset_seed_scheme": RESET_SEED_SCHEME,
        "sampling_seed_scheme": SAMPLING_SEED_SCHEME,
        "rollout_recording": (
            None
            if recorder is None
            else {
                "raw_dir": str(Path(arguments.record_rollout_dir).expanduser()),
                "episodes": len(rollout_manifest["episodes"]),
                "fps": float(arguments.record_fps),
            }
        ),
    }
    _write_json(output_dir / "metrics.json", metrics)
    _write_json(
        output_dir / "resolved_config.json",
        {
            "requested_env_id": arguments.env_id,
            "runtime_env_id": runtime_env_id,
            "arm": arguments.arm,
            "seeds": seeds,
            "episodes_per_seed": arguments.episodes_per_seed,
            "dummy_steps": arguments.dummy_steps,
            "ddim_steps": arguments.ddim_steps,
            "act_horizon": arguments.act_horizon,
            "reset_seed_scheme": RESET_SEED_SCHEME,
            "sampling_seed_scheme": SAMPLING_SEED_SCHEME,
            "record_rollout_dir": bool(arguments.record_rollout_dir),
            "record_episodes": int(arguments.record_episodes),
            "record_episode_ids": _parse_record_episode_ids(arguments.record_episode_ids),
            "record_fps": float(arguments.record_fps),
        },
    )
    return metrics


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root")
    parser.add_argument("--env-id", default=FORMAL_ENV_ID)
    parser.add_argument("--base-policy-ckpt")
    parser.add_argument("--base-policy-args")
    parser.add_argument("--arm", choices=("base", "residual"), required=True)
    parser.add_argument("--residual-runs")
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--manifest")
    parser.add_argument("--res-scale", type=float)
    parser.add_argument("--actor-input", choices=("obs", "obs_base_action"))
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--episodes-per-seed", type=int, default=50)
    parser.add_argument("--ddim-steps", type=int, default=4)
    parser.add_argument("--act-horizon", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--dummy-steps", type=int, default=4)
    parser.add_argument(
        "--record-rollout-dir",
        help="write selected closed-loop episodes as dependency-free raw RGB/state bundles",
    )
    parser.add_argument(
        "--record-episodes",
        type=int,
        default=1,
        help="maximum number of episodes to record when --record-episode-ids is omitted",
    )
    parser.add_argument(
        "--record-episode-ids",
        help="comma-separated global 0-based episode IDs to record",
    )
    parser.add_argument("--record-fps", type=float, default=20.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.dummy:
        run_dummy_evaluation(arguments)
    else:
        if not arguments.base_policy_ckpt:
            raise SystemExit("formal S1 evaluation requires --base-policy-ckpt")
        run_formal_evaluation(arguments)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
