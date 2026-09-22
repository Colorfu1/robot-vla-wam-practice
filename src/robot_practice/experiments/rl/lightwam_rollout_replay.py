"""Replay recorded actions without model inference; no upstream or installed edits."""

import argparse
import os
from pathlib import Path

import numpy as np

from robot_practice.experiments.rl.lightwam_base import (
    _DPPOSquareEnvClient,
    _resolve_dppo_runtime,
)
from robot_practice.experiments.rl.rollout_forensics import memory_snapshot, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--start-method", choices=["fork", "spawn", "forkserver"], required=True)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument(
        "--serial-step",
        action="store_true",
        help="Keep all workers alive; dispatch step calls one at a time",
    )
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument(
        "--normalization", default="datasets/raw/dppo/robomimic/square-img/normalization.npz"
    )
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output_root)
    files = sorted(source.glob("chunk_*_input.npz"))
    if not files or [p.name for p in files] != [
        f"chunk_{i:04d}_input.npz" for i in range(len(files))
    ]:
        raise ValueError("Missing or non-contiguous recorded input chunks")
    with np.load(files[0], allow_pickle=False) as saved:
        seeds = saved["environment_seeds"].tolist()
    if args.worker_index is None:
        selection = slice(None)
    else:
        if not 0 <= args.worker_index < len(seeds):
            parser.error("worker-index out of bounds")
        selection = slice(args.worker_index, args.worker_index + 1)
        seeds = seeds[selection]
    if seeds != list(range(seeds[0], seeds[0] + len(seeds))):
        raise ValueError("Environment client requires contiguous seeds")
    output.mkdir(parents=True, exist_ok=False)
    evidence = output / "forensics"
    os.environ["ROBOT_PRACTICE_ROLLOUT_FORENSICS"] = str(evidence.resolve())
    os.environ["ROBOT_PRACTICE_ROLLOUT_START_METHOD"] = args.start_method
    os.environ["ROBOT_PRACTICE_ROLLOUT_SERIAL_STEP"] = "1" if args.serial_step else "0"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    python, root = _resolve_dppo_runtime(dppo_python_value=None, dppo_root_value=None)
    with (output / "run.log").open("a") as log:

        def emit(message):
            print(message, flush=True)
            print(message, file=log, flush=True)

        emit(
            f"START replay method={args.start_method} serial_step={args.serial_step} "
            f"seeds={seeds} total={len(files)}"
        )
        record(evidence, "replay_begin", method=args.start_method, memory=memory_snapshot())
        try:
            with _DPPOSquareEnvClient(
                dppo_python=python,
                dppo_root=root,
                normalization_path=Path(args.normalization).resolve(),
                num_envs=len(seeds),
                max_episode_steps=args.max_steps,
                action_horizon=4,
                policy_seed=seeds[0],
                worker_log=output / "env_worker.log",
            ) as env:
                obs = env.reset()
                for i, path in enumerate(files):
                    with np.load(path, allow_pickle=False) as saved:
                        if saved["environment_seeds"][selection].tolist() != env.environment_seeds:
                            raise ValueError("Recorded seeds differ from runtime seeds")
                        error = float(np.max(np.abs(obs["state"] - saved["state"][selection])))
                        emit(f"START chunk={i} state_max_abs_error={error}")
                        if not np.isfinite(error) or error > 1e-6:
                            raise ValueError("Replay state mismatch; not the same trajectory")
                        record(evidence, "replay_step_begin", chunk=i, state_error=error)
                        obs, *_ = env.step(saved["action_executed"][selection])
                    emit(f"STATUS completed={i + 1} total={len(files)}")
            emit(f"DONE replay completed={len(files)} total={len(files)}")
        except BaseException as exc:
            emit(f"FAILED replay error={type(exc).__name__}: {exc}")
            raise
        finally:
            record(evidence, "replay_end", memory=memory_snapshot())


if __name__ == "__main__":
    main()
