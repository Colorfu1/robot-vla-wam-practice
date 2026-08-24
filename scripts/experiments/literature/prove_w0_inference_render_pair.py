#!/usr/bin/env python3
"""Prove or falsify a same-physical-GPU CUDA/EGL failure in FastWAM eval.

This is a single-process, one-task diagnostic.  It uses the pinned FastWAM
model construction, processor, LIBERO environment and action inference path.
The only experimental variable is the physical GPU selected by EGL:

* no ``--render-physical``: CUDA and EGL target the same physical GPU;
* ``--render-physical N``: CUDA targets ``--compute-physical`` and EGL targets
  the physical GPU N, resolved through the host's PCI/EGL enumeration.

Every case runs in its own process because a native ``SIGABRT`` in
``mjr_readPixels`` cannot be recovered by Python.  The parent shell should run
the matched and separated cases independently and compare their logs.

The real case loads the checkpoint before creating the LIBERO environment,
matching ``experiments/libero/eval_libero_single.py``.  It explicitly calls
``torch.cuda.synchronize`` after every inference so a passing separated case
cannot be explained by unfinished model kernels.  The render-only case keeps a
CUDA context alive but skips model construction, separating CUDA/EGL context
interaction from the FastWAM model workload.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


REPO_ROOT = _repo_root()
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


class EventLog:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "run.log"
        self.stream = self.path.open("w", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> None:
        payload = {"event": event, **fields}
        line = (
            f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] "
            f"[W0-PROOF] {json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"
        )
        print(line, flush=True)
        self.stream.write(line + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def _resolve(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("real", "render-only"),
        default="real",
        help="real loads FastWAM and runs inference; render-only only initializes CUDA",
    )
    parser.add_argument("--case", required=True, help="stable label written to the log")
    parser.add_argument("--compute-physical", type=int, default=0)
    parser.add_argument(
        "--render-physical",
        type=int,
        default=None,
        help="physical CUDA index whose EGL index should render; omitted means same GPU",
    )
    parser.add_argument(
        "--egl-index",
        type=int,
        default=None,
        help="explicit EGL index; skips PCI/EGL derivation (for a known host map)",
    )
    parser.add_argument(
        "--include-render-cuda",
        action="store_true",
        help=(
            "also expose the physical CUDA ordinal corresponding to the selected EGL "
            "device; keeps the compute device first as cuda:0"
        ),
    )
    parser.add_argument(
        "--vendor-root",
        default=os.environ.get("W0_VENDOR_ROOT", "/opt/robot-practice/w0-vendor"),
    )
    parser.add_argument("--cache-root", default="pretrained/w0_fastwam")
    parser.add_argument(
        "--config",
        default="configs/experiments/literature/w0_fastwam_reproduction.json",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-stats", default=None)
    parser.add_argument("--action-dit", default=None)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=5)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _runtime_setup(args: argparse.Namespace, log: EventLog):
    """Set the same local model/artifact environment as the audit launcher."""

    from robot_practice.experiments.literature import w0_fastwam_audit as audit

    config = audit.load_config(args.config)
    vendor = _resolve(args.vendor_root)
    cache_root = _resolve(args.cache_root)
    paths = audit._artifact_paths(config, cache_root=cache_root)
    runtime_env = audit._model_runtime_env(paths, vendor_root=vendor)

    os.environ.update(runtime_env)
    os.environ["W0_VENDOR_ROOT"] = str(vendor)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    fastwam_root = vendor / "FastWAM"
    for path in (
        fastwam_root,
        fastwam_root / "src",
        fastwam_root / "experiments" / "libero",
        vendor / "LIBERO",
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # The formal launcher exposes W0_LIBERO_ASSETS before Python starts, so
    # sitecustomize runs automatically.  This diagnostic sets the runtime
    # environment inside the process in order to derive the EGL mapping first;
    # explicitly invoke the same hook here or LIBERO falls back to its
    # incomplete site-packages asset directory.
    from robot_practice.experiments.literature.w0_runtime import sitecustomize

    sitecustomize._configure_libero_assets()

    log.emit(
        "RUNTIME",
        repo=str(REPO_ROOT),
        vendor_root=str(vendor),
        fastwam_root=str(fastwam_root),
        cache_root=str(cache_root),
        python=sys.executable,
        case=args.case,
        mode=args.mode,
        libero_assets=os.environ.get("W0_LIBERO_ASSETS"),
    )
    return audit, config, paths, runtime_env, vendor


def _resolve_render_device(
    args: argparse.Namespace, audit, runtime_env, log: EventLog
) -> tuple[str, dict[str, Any]]:
    compute = str(args.compute_physical)
    render_physical = (
        args.compute_physical if args.render_physical is None else args.render_physical
    )

    if args.egl_index is not None:
        render_egl = str(args.egl_index)
        report: dict[str, Any] = {"source": "explicit", "render_physical": render_physical}
    else:
        physical_ids = sorted({compute, str(render_physical)}, key=int)
        same_gpu_egl, report = audit._pci_derived_egl_device_map(
            physical_ids, runtime_env=runtime_env
        )
        render_egl = same_gpu_egl[str(render_physical)]
        report = {
            "source": "pci-derived",
            "same_gpu_egl_by_cuda": same_gpu_egl,
            "render_physical": render_physical,
            "join": report,
        }

    # robosuite's binding_utils requires the EGL index string to occur in
    # CUDA_VISIBLE_DEVICES.  The compute device stays first, so cuda:0 remains
    # the model device.  The A/B mode additionally exposes the *physical CUDA
    # ordinal* that owns the selected EGL entry; the raw EGL index and CUDA
    # ordinal are different namespaces on this host.
    visible_devices = [compute]
    render_cuda: str | None = None
    if args.include_render_cuda:
        join_report = report.get("join", {})
        egl_to_cuda = join_report.get("egl_to_cuda", {})
        render_cuda = egl_to_cuda.get(render_egl)
        if render_cuda is None:
            raise RuntimeError(
                "cannot expose the physical CUDA ordinal for EGL index "
                f"{render_egl}; PCI/EGL report has no egl_to_cuda entry"
            )
        if render_cuda not in visible_devices:
            visible_devices.append(render_cuda)
    if render_egl not in visible_devices:
        visible_devices.append(render_egl)
    visible = ",".join(visible_devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    os.environ["MUJOCO_EGL_DEVICE_ID"] = render_egl
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    log.emit(
        "DEVICE_BINDING",
        compute_physical=args.compute_physical,
        render_physical=render_physical,
        render_cuda_physical=render_cuda,
        egl_index=int(render_egl),
        cuda_visible_devices=visible,
        visibility_mode=(
            "compute_plus_render_physical" if args.include_render_cuda else "legacy"
        ),
        local_model_device="cuda:0",
        same_physical_gpu=bool(args.compute_physical == render_physical),
        map_report=report,
    )
    return render_egl, report


def _cuda_snapshot(torch, log: EventLog, stage: str) -> None:
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    props = torch.cuda.get_device_properties(0)
    log.emit(
        "CUDA",
        stage=stage,
        current_device=torch.cuda.current_device(),
        device_name=props.name,
        total_mib=total_bytes >> 20,
        free_mib=free_bytes >> 20,
    )


def _load_real_pipeline(args, audit, config, paths, vendor, log: EventLog):
    """Construct the same model/processor objects as eval_libero_single.py."""

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    evaluator_path = vendor / "FastWAM" / "experiments" / "libero" / "eval_libero_single.py"
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"missing pinned evaluator: {evaluator_path}")

    from experiments.libero import eval_libero_single as evaluator

    checkpoint = _resolve(args.checkpoint)
    stats = _resolve(args.dataset_stats) if args.dataset_stats else paths["upstream_stats"]
    action_dit = _resolve(args.action_dit) if args.action_dit else paths["action_dit"]
    required_files = (
        (checkpoint, "checkpoint"),
        (stats, "dataset_stats"),
        (action_dit, "action_dit"),
    )
    for path, label in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"missing {label}: {path}")

    log.emit(
        "MODEL_INPUTS",
        checkpoint=str(checkpoint),
        dataset_stats=str(stats),
        action_dit=str(action_dit),
    )

    overrides = [
        f"ckpt={checkpoint}",
        f"model.action_dit_pretrained_path={action_dit}",
        f"EVALUATION.dataset_stats_path={stats}",
        f"EVALUATION.task_suite_name={args.suite}",
        f"EVALUATION.task_id={args.task_id}",
        "EVALUATION.device=cuda",
        f"EVALUATION.replan_steps={args.replan_steps}",
        f"EVALUATION.num_steps_wait={args.num_steps_wait}",
        f"EVALUATION.num_inference_steps={args.num_inference_steps}",
        "EVALUATION.visualize_future_video=false",
        "EVALUATION.use_action_ensembler=false",
        "EVALUATION.env_num=1",
        f"seed={args.seed}",
        "gpu_id=0",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=str(vendor / "FastWAM" / "configs")):
        cfg = compose(config_name="sim_libero", overrides=overrides)

    torch.cuda.set_device(0)
    _cuda_snapshot(torch, log, "before_model_construct")
    model_dtype = evaluator._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device="cuda")
    log.emit("MODEL_CONSTRUCTED", dtype=str(model_dtype))
    evaluator._load_model_checkpoint(model, str(checkpoint))
    model = model.to("cuda").eval()
    torch.cuda.synchronize(0)
    _cuda_snapshot(torch, log, "model_ready")

    dataset_stats = evaluator.load_dataset_stats_from_json(str(stats))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    action_horizon = int(
        cfg.EVALUATION.get("action_horizon") or (int(cfg.data.train.num_frames) - 1)
    )
    return evaluator, cfg, model, processor, action_horizon, input_w, input_h


def _make_env(args, evaluator, log: EventLog):
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    initial_states = list(suite.get_task_init_states(args.task_id))
    env, task_description = evaluator.get_libero_env(
        task, evaluator.LIBERO_ENV_RESOLUTION, args.seed
    )
    inner = getattr(env, "env", env)
    sim = getattr(inner, "sim", None)
    render_context = getattr(sim, "_render_context_offscreen", None)
    log.emit(
        "ENV_READY",
        suite=args.suite,
        task_id=args.task_id,
        task_description=task_description,
        initial_state_count=len(initial_states),
        hard_reset=getattr(inner, "hard_reset", "unknown"),
        render_context_id=id(render_context) if render_context is not None else None,
        gl_context_id=(
            id(getattr(render_context, "gl_ctx", None))
            if render_context is not None
            else None
        ),
    )
    return env, initial_states, task_description


def _action_summary(action) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(action, dtype=np.float64)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "min": float(values.min()),
        "max": float(values.max()),
        "norm": float(np.linalg.norm(values)),
    }


def _run_loop(
    args,
    evaluator,
    model,
    processor,
    cfg,
    action_horizon,
    input_w,
    input_h,
    env,
    initial_states,
    task_description,
    log,
):
    import torch

    obs = env.reset()
    log.emit("RESET_RETURNED", observation_keys=sorted(obs.keys()))
    obs = env.set_init_state(initial_states[0])
    log.emit("INIT_STATE_RETURNED", observation_keys=sorted(obs.keys()))

    pending_actions: list[list[float]] = []
    dummy_action = evaluator.get_libero_dummy_action()
    for step in range(args.steps):
        if step < args.num_steps_wait:
            action = dummy_action
            source = "warmup_dummy"
        else:
            if not pending_actions:
                if model is None:
                    action = dummy_action
                    source = "cuda_context_only_dummy"
                else:
                    log.emit(
                        "INFERENCE_BEGIN",
                        step=step,
                        device=str(next(model.parameters()).device),
                    )
                    action_chunk, _, _ = evaluator._predict_action_chunk(
                        obs=obs,
                        task_description=task_description,
                        model=model,
                        processor=processor,
                        cfg=cfg,
                        action_horizon=action_horizon,
                        input_w=input_w,
                        input_h=input_h,
                        model_device="cuda",
                    )
                    # Explicit barrier: this experiment is not allowed to
                    # rely on accidental CUDA stream completion.
                    torch.cuda.synchronize(0)
                    log.emit(
                        "INFERENCE_END",
                        step=step,
                        action=_action_summary(action_chunk),
                        explicit_cuda_synchronize=True,
                    )
                    pending_actions = action_chunk[: args.replan_steps].tolist()
                    action = pending_actions.pop(0)
                    source = "fastwam_inference"
            else:
                action = pending_actions.pop(0)
                source = "pending_action_chunk"

        log.emit(
            "ENV_STEP_BEGIN",
            step=step,
            source=source,
            action=_action_summary(action),
        )
        obs, reward, done, info = env.step(action)
        log.emit(
            "ENV_STEP_END",
            step=step,
            reward=float(reward),
            done=bool(done),
            observation_keys=sorted(obs.keys()),
        )
        if done:
            log.emit("EPISODE_DONE", step=step)
            break


def main() -> int:
    args = _parse_args()
    output_dir = _resolve(args.output_dir)
    log = EventLog(output_dir)
    try:
        # Remove inherited bindings before the absolute EGL list is inspected.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        audit, config, paths, runtime_env, vendor = _runtime_setup(args, log)
        _resolve_render_device(args, audit, runtime_env, log)

        import faulthandler

        faulthandler.enable(all_threads=True)
        import torch

        torch.cuda.set_device(0)
        _cuda_snapshot(torch, log, "cuda_context_initialized")

        model = None
        processor = None
        cfg = None
        evaluator = None
        action_horizon = input_w = input_h = None
        if args.mode == "real":
            if not args.checkpoint:
                raise ValueError("--checkpoint is required for --mode=real")
            (
                evaluator,
                cfg,
                model,
                processor,
                action_horizon,
                input_w,
                input_h,
            ) = _load_real_pipeline(args, audit, config, paths, vendor, log)
        else:
            log.emit("MODEL_SKIPPED", reason="CUDA context only; no checkpoint or policy inference")

        if evaluator is None:
            from experiments.libero import eval_libero_single as evaluator

        env, initial_states, task_description = _make_env(args, evaluator, log)
        log.emit("LOOP_BEGIN", steps=args.steps, trials=args.trials)
        for trial in range(args.trials):
            log.emit("TRIAL_BEGIN", trial=trial)
            _run_loop(
                args,
                evaluator,
                model,
                processor,
                cfg,
                action_horizon,
                input_w,
                input_h,
                env,
                initial_states,
                task_description,
                log,
            )
            log.emit("TRIAL_END", trial=trial)
        env.close()
        log.emit("ENV_CLOSED")
        log.emit("DONE", status="complete")
        return 0
    except BaseException as error:
        log.emit("PYTHON_ERROR", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
        return 1
    finally:
        log.close()
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
