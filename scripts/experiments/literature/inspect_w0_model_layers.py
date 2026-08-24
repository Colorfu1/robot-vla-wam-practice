#!/usr/bin/env python3
"""Inspect one FastWAM training forward pass in one Python process.

This is deliberately a forward-only tool.  It does not initialise torch.distributed,
DeepSpeed, an optimizer, or a DataLoader worker.  The call being traced is the same
``model(sample)`` call used by the upstream trainer, so the returned value is the
training loss tuple, but autograd is disabled.

FastWAM's MoT implementation does not call ``block.forward``.  It manually calls the
Q/K/V projections and the post-attention pieces inside ``MoT.forward``.  Consequently,
ordinary hooks on ``video_expert.blocks.N`` are not sufficient to observe a full layer.
This script adds a small runtime wrapper around MoT's post-layer function and records
``mot.video.layer_N`` / ``mot.action.layer_N`` explicitly, without editing the pinned
vendor checkout.

Typical host invocation (real video/action/state and cached T5 context):

    PYTHONNOUSERSITE=1 \
    conda run --no-capture-output -n fastwam-upstream python \
      scripts/experiments/literature/inspect_w0_model_layers.py \
      --module-regex '^(video_expert|action_expert)\\.blocks\\.\\d+\\.(self_attn|cross_attn|ffn|gate)' \
      --pdb-at mot.video.layer_03

For exact runtime parity, run the same command inside the pinned Docker image with
``--gpus all`` and the repository mounted read-only.  See ``--help`` for the complete
set of path and tracing options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

EXPECTED_FASTWAM_COMMIT = "45d8e1458921d83f8ad6cf9ce993d371208dabd0"
EXPECTED_LIBERO_COMMIT = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
EXPECTED_LEROBOT_COMMIT = "30da8e687a6dfc617fcd94afc367ac7071c376ce"
EXPECTED_PACKAGES = {
    "accelerate": "1.12.0",
    "deepspeed": "0.18.5",
    "einops": "0.8.1",
    "fastwam": "0.1.0",
    "libero": "0.1.0",
    "mujoco": "3.3.2",
    "omegaconf": "2.3.0",
    "torch": "2.7.1+cu128",
    "torchcodec": "0.5+cu128",
    "transformers": "4.49.0",
}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DEFAULT_TASK = "libero_uncond_2cam224_1e-4"
DEFAULT_MODULE_REGEX = (
    r"^(?:video_expert|action_expert)\."
    r"(?:blocks\.\d+\.(?:self_attn\.(?:q|k|v|norm_q|norm_k)|cross_attn|ffn|norm[123]|gate)"
    r"|head(?:\..*)?|action_encoder|text_embedding|time_embedding|time_projection)$"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _git_dirty(path: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={path}", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _dtype_from_name(name: str):
    import torch

    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


class RunLog:
    """Print status immediately and mirror it into a persistent log."""

    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "run.log"
        self.stream = self.path.open("a", encoding="utf-8")

    def emit(self, message: str) -> None:
        line = f"[{dt.datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.stream.write(line + "\n")
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


def _tensor_summary(value: Any) -> dict[str, Any] | None:
    import torch

    if not torch.is_tensor(value):
        return None
    detached = value.detach()
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "requires_grad": bool(value.requires_grad),
    }
    if detached.numel() == 0:
        return result
    # Reductions are intentionally part of this debug tool: each value is a compact
    # signal that can be compared across layers without copying the activation.
    if detached.is_complex():
        # RoPE frequencies are complex.  Report magnitude rather than silently
        # dropping the imaginary component during float casting.
        numeric = detached.abs().float()
        result["statistic"] = "abs"
    else:
        numeric = detached.float()
    result.update(
        {
            "min": float(numeric.amin().item()),
            "max": float(numeric.amax().item()),
            "mean": float(numeric.mean().item()),
            "std": float(numeric.std(unbiased=False).item()),
            "nan": int(torch.isnan(numeric).sum().item()),
            "inf": int(torch.isinf(numeric).sum().item()),
        }
    )
    return result


def _value_summary(value: Any, *, depth: int = 0, max_depth: int = 4) -> Any:
    tensor = _tensor_summary(value)
    if tensor is not None:
        return tensor
    if depth >= max_depth:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, Mapping):
        return {
            str(key): _value_summary(item, depth=depth + 1, max_depth=max_depth)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [
            _value_summary(item, depth=depth + 1, max_depth=max_depth)
            for item in value
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"type": type(value).__name__, "repr": repr(value)}


def _detach_tree(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    return value


class TraceRecorder:
    def __init__(
        self,
        output_dir: Path,
        log: RunLog,
        *,
        save_patterns: Sequence[str],
        pdb_pattern: str | None,
    ):
        self.output_dir = output_dir
        self.log = log
        self.save_patterns = [re.compile(pattern) for pattern in save_patterns]
        self.pdb_pattern = re.compile(pdb_pattern) if pdb_pattern else None
        self.pdb_fired = False
        self.events: list[dict[str, Any]] = []
        self.event_path = output_dir / "events.jsonl"
        self.event_stream = self.event_path.open("w", encoding="utf-8")
        self.save_dir = output_dir / "tensors"

    def _matches_save(self, name: str) -> bool:
        return any(pattern.search(name) for pattern in self.save_patterns)

    def _maybe_save(self, name: str, call_index: int, value: Any) -> str | None:
        if not self._matches_save(name):
            return None
        self.save_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "event"
        path = self.save_dir / f"{call_index:04d}_{safe_name}.pt"
        import torch

        torch.save(_detach_tree(value), path)
        return str(path.relative_to(self.output_dir))

    def record(
        self,
        name: str,
        output: Any,
        *,
        kind: str,
        inputs: Any = None,
        module_type: str | None = None,
    ) -> dict[str, Any]:
        call_index = len(self.events)
        event = {
            "call_index": call_index,
            "name": name,
            "kind": kind,
            "module_type": module_type,
            "inputs": _value_summary(inputs),
            "output": _value_summary(output),
            "saved": self._maybe_save(name, call_index, output),
        }
        self.events.append(event)
        self.event_stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.event_stream.flush()
        self.log.emit(
            f"[TRACE] {name} kind={kind} output={json.dumps(event['output'], ensure_ascii=False, separators=(',', ':'))}"
        )
        if self.pdb_pattern and not self.pdb_fired and self.pdb_pattern.search(name):
            self.pdb_fired = True
            self.log.emit(f"[PDB] stopped at {name}; continue with `p`/`n`/`c` in pdb")
            import pdb  # noqa: T100

            pdb.set_trace()  # noqa: T100
        return event

    def close(self) -> None:
        self.event_stream.close()
        (self.output_dir / "trace.json").write_text(
            json.dumps(self.events, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


class HookRegistry:
    """Install module hooks and runtime wrappers, then restore everything."""

    def __init__(self, model: Any, recorder: TraceRecorder, *, module_pattern: str | None, all_modules: bool):
        self.model = model
        self.recorder = recorder
        self.module_pattern = re.compile(module_pattern) if module_pattern else None
        self.all_modules = all_modules
        self.handles: list[Any] = []
        self.restore_callbacks: list[Callable[[], None]] = []

    def install_module_hooks(self) -> int:
        count = 0
        for name, module in self.model.named_modules():
            if not name:
                continue
            if not self.all_modules and self.module_pattern is None:
                continue
            if not self.all_modules and not self.module_pattern.search(name):
                continue

            def hook(mod, inputs, output, event_name=name):
                self.recorder.record(
                    event_name,
                    output,
                    kind="module",
                    inputs=inputs,
                    module_type=type(mod).__name__,
                )

            self.handles.append(module.register_forward_hook(hook))
            count += 1
        return count

    def wrap_method(
        self,
        owner: Any,
        method_name: str,
        event_name: str | Callable[[Any, tuple[Any, ...], dict[str, Any]], str],
    ) -> None:
        original = getattr(owner, method_name)

        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            name = event_name(result, args, kwargs) if callable(event_name) else event_name
            self.recorder.record(name, result, kind="method", inputs={"args": args, "kwargs": kwargs})
            return result

        setattr(owner, method_name, wrapped)

        def restore() -> None:
            setattr(owner, method_name, original)

        self.restore_callbacks.append(restore)

    def install_fastwam_boundaries(self) -> None:
        self.wrap_method(self.model, "build_inputs", "model.build_inputs")
        self.wrap_method(self.model, "_encode_video_latents", "video.vae.encode")
        self.wrap_method(self.model, "_build_mot_attention_mask", "mot.attention_mask")
        self.wrap_method(self.model.video_expert, "pre_dit", "video.pre_dit")
        self.wrap_method(self.model.action_expert, "pre_dit", "action.pre_dit")
        self.wrap_method(self.model.video_expert, "post_dit", "video.post_dit")
        self.wrap_method(self.model.action_expert, "post_dit", "action.post_dit")

        def layer_name(result, args, kwargs):
            del result, args
            block = kwargs.get("block")
            if block is None:
                return "mot.layer.unknown"
            for expert_name, expert in self.model.mot.mixtures.items():
                for layer_index, candidate in enumerate(expert.blocks):
                    if candidate is block:
                        return f"mot.{expert_name}.layer_{layer_index:02d}"
            return "mot.layer.unknown"

        original = self.model.mot._apply_post_with_optional_checkpoint

        def wrapped_layer(*args, **kwargs):
            result = original(*args, **kwargs)
            name = layer_name(result, args, kwargs)
            selected_inputs = {
                key: kwargs.get(key)
                for key in ("residual_x", "mixed_slice", "gate_msa", "shift_mlp", "scale_mlp", "gate_mlp")
                if key in kwargs
            }
            self.recorder.record(name, result, kind="mot_layer", inputs=selected_inputs)
            return result

        setattr(self.model.mot, "_apply_post_with_optional_checkpoint", wrapped_layer)  # noqa: B010

        def restore_layer() -> None:
            setattr(  # noqa: B010
                self.model.mot, "_apply_post_with_optional_checkpoint", original
            )

        self.restore_callbacks.append(restore_layer)

        self.wrap_method(self.model.mot, "forward", "mot")

        def record_loss(result, args, kwargs):
            del args, kwargs
            if isinstance(result, tuple) and len(result) == 2:
                self.recorder.record("loss.total", result[0], kind="loss")
                self.recorder.record("loss.components", result[1], kind="loss")
            else:
                self.recorder.record("loss", result, kind="loss")
            return "loss.return"

        self.wrap_method(self.model, "training_loss", record_loss)

    def install_scheduler_boundaries(self) -> None:
        for prefix, scheduler in (
            ("video", self.model.train_video_scheduler),
            ("action", self.model.train_action_scheduler),
        ):
            self.wrap_method(scheduler, "sample_training_t", f"{prefix}.scheduler.timestep")

            def add_noise_event(result, args, kwargs, event_prefix=prefix):
                del args, kwargs
                return f"{event_prefix}.scheduler.noisy"

            def target_event(result, args, kwargs, event_prefix=prefix):
                del args, kwargs
                return f"{event_prefix}.scheduler.target"

            self.wrap_method(scheduler, "add_noise", add_noise_event)
            self.wrap_method(scheduler, "training_target", target_event)

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        for restore in reversed(self.restore_callbacks):
            restore()


def _parse_args() -> argparse.Namespace:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vendor-root", default=os.environ.get("W0_VENDOR_ROOT"))
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--task-config", default=None, help="FastWAM configs/task YAML; defaults to --task")
    parser.add_argument("--cache-root", default=str(repo / "pretrained/w0_fastwam"))
    parser.add_argument("--dataset-root", default=None, help="v2.1 LIBERO dataset root; auto-detected below cache-root")
    parser.add_argument("--dataset-stats", default=None)
    parser.add_argument("--text-cache-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--action-dit", default=None)
    parser.add_argument("--index", type=int, default=0, help="Global dataset frame index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--module-regex", default=DEFAULT_MODULE_REGEX)
    parser.add_argument("--all-modules", action="store_true", help="Hook every named module; can be very verbose")
    parser.add_argument(
        "--save",
        action="append",
        default=[],
        metavar="REGEX",
        help="Save matching outputs under output-dir/tensors; repeatable",
    )
    parser.add_argument("--pdb-at", default=None, metavar="REGEX", help="Enter pdb once when an event name matches")
    parser.add_argument(
        "--dummy-context",
        action="store_true",
        help="EXPLICIT structural smoke only: replace the real T5 context with zeros",
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="Fail if TorchCodec cannot load or PYTHONNOUSERSITE is not 1",
    )
    parser.add_argument(
        "--keep-going-on-missing-checkpoint",
        action="store_true",
        help="Build random/base weights when the fine-tune checkpoint is absent (debug only)",
    )
    return parser.parse_args()


def _prepare_environment(args: argparse.Namespace) -> dict[str, Path]:
    repo = _repo_root()
    vendor_raw = args.vendor_root or str(repo / "third_party/w0_fastwam")
    vendor = _resolve_path(vendor_raw, repo)
    fastwam_root = vendor / "FastWAM"
    if not fastwam_root.is_dir():
        raise FileNotFoundError(f"FastWAM checkout not found: {fastwam_root}")

    # Set these before importing any FastWAM module.  In Docker the values point at
    # the image's pinned vendor tree; on the host they point at the repository checkout.
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ["W0_VENDOR_ROOT"] = str(vendor)
    cache_root = _resolve_path(args.cache_root, repo)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(cache_root / "models")
    os.environ["W0_FASTWAM_MODEL_ROOT"] = str(cache_root / "models")
    # The image's base environment advertises HOME=/root even when the container
    # is run as an ordinary user.  HuggingFace datasets otherwise tries to create
    # /root/.cache while it materialises the local Parquet index.  Keep all such
    # transient state beside the repository-local W0 artifacts in both runtimes.
    hf_home = cache_root / "dataset/.cache/huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_DATASETS_CACHE"] = str(hf_home / "datasets")
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_ASSETS_CACHE"] = str(hf_home / "assets")
    os.environ["HF_LEROBOT_HOME"] = str(hf_home / "lerobot")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "1"

    # TorchCodec's FFmpeg builds use CUDA NPP for image conversion.  Docker has
    # this runtime library in /usr/local/cuda; the host fastwam-upstream env gets
    # the same CUDA 12.8 NPP payload from nvidia-npp-cu12, whose lib directory is
    # not automatically added to the ELF loader search path.
    npp_candidates = [
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/npp/lib",
        *sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/npp/lib")),
    ]
    for npp_lib in npp_candidates:
        if npp_lib.is_dir():
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            entries = [entry for entry in current_ld.split(":") if entry]
            if str(npp_lib) not in entries:
                os.environ["LD_LIBRARY_PATH"] = ":".join([str(npp_lib), *entries])
            break

    runtime_support = repo / "src/robot_practice/experiments/literature/w0_runtime"
    source_root = fastwam_root / "src"
    for path in (source_root, runtime_support, repo / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    return {"repo": repo, "vendor": vendor, "fastwam_root": fastwam_root, "cache_root": cache_root}


def _find_npp_lib_dir() -> Path | None:
    candidates = [
        Path(sys.prefix)
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/nvidia/npp/lib",
        *sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/npp/lib")),
        Path("/usr/local/cuda/targets/x86_64-linux/lib"),
        Path("/usr/local/cuda-12.8/targets/x86_64-linux/lib"),
    ]
    for candidate in candidates:
        if (candidate / "libnppc.so.12").is_file() and (candidate / "libnppicc.so.12").is_file():
            return candidate.resolve()
    return None


def _preload_npp_runtime() -> dict[str, Any]:
    """Make NPP available to TorchCodec even when LD_LIBRARY_PATH was set too late."""
    npp_dir = _find_npp_lib_dir()
    if npp_dir is None:
        return {"available": False, "path": None, "error": "NPP runtime directory not found"}
    try:
        import ctypes

        # Load the dependency first and keep both handles globally visible.  The
        # dynamic loader reads LD_LIBRARY_PATH only at process startup, while this
        # function runs after Python has started.
        ctypes.CDLL(str(npp_dir / "libnppc.so.12"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(str(npp_dir / "libnppicc.so.12"), mode=ctypes.RTLD_GLOBAL)
    except OSError as error:
        return {"available": False, "path": str(npp_dir), "error": str(error)}
    return {"available": True, "path": str(npp_dir), "error": None}


def _environment_report(paths: Mapping[str, Path], args: argparse.Namespace, log: RunLog) -> dict[str, Any]:
    import torch

    npp_status = _preload_npp_runtime()
    fastwam_spec = importlib.util.find_spec("fastwam")
    fastwam_path = None if fastwam_spec is None else fastwam_spec.origin
    package_versions = {name: _package_version(name) for name in EXPECTED_PACKAGES}
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": str(Path.cwd()),
        "torch": {
            "version": torch.__version__,
            "path": str(torch.__file__),
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "packages": package_versions,
        "fastwam_import": fastwam_path,
        "vendor": {
            "root": str(paths["vendor"]),
            "fastwam_commit": _git_head(paths["vendor"] / "FastWAM"),
            "libero_commit": _git_head(paths["vendor"] / "LIBERO"),
            "lerobot_commit": _git_head(paths["vendor"] / "lerobot"),
            "fastwam_dirty": _git_dirty(paths["vendor"] / "FastWAM"),
        },
        "npp": npp_status,
        "env": {
            key: os.environ.get(key)
            for key in (
                "PYTHONNOUSERSITE",
                "W0_VENDOR_ROOT",
                "DIFFSYNTH_MODEL_BASE_PATH",
                "W0_FASTWAM_MODEL_ROOT",
                "PYTHONPATH",
                "LD_LIBRARY_PATH",
            )
        },
    }

    torchcodec_status: dict[str, Any]
    if importlib.util.find_spec("torchcodec") is None:
        torchcodec_status = {"available": False, "error": "package not importable"}
    else:
        try:
            from torchcodec.decoders import VideoDecoder  # noqa: F401

            torchcodec_status = {"available": True, "error": None}
        except Exception as error:  # noqa: BLE001
            torchcodec_status = {
                "available": False,
                "error": f"{type(error).__name__}: {error}",
            }
    report["torchcodec"] = torchcodec_status

    log.emit(f"[ENV] python={report['python']} executable={report['executable']}")
    log.emit(
        f"[ENV] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
        f"device={report['torch']['device']}"
    )
    log.emit(f"[ENV] fastwam_import={fastwam_path}")
    log.emit(f"[ENV] commits={json.dumps(report['vendor'], ensure_ascii=False, separators=(',', ':'))}")
    log.emit(f"[ENV] npp={json.dumps(npp_status, ensure_ascii=False)}")
    log.emit(f"[ENV] torchcodec={json.dumps(torchcodec_status, ensure_ascii=False)}")

    expected_commits = {
        "fastwam_commit": EXPECTED_FASTWAM_COMMIT,
        "libero_commit": EXPECTED_LIBERO_COMMIT,
        "lerobot_commit": EXPECTED_LEROBOT_COMMIT,
    }
    for key, expected in expected_commits.items():
        if report["vendor"].get(key) != expected:
            raise RuntimeError(
                f"{key} mismatch: got {report['vendor'].get(key)!r}, expected {expected!r}"
            )
    if Path(fastwam_path or "").resolve() != (paths["vendor"] / "FastWAM/src/fastwam/__init__.py").resolve():
        raise RuntimeError(
            "fastwam imported from an unexpected checkout: "
            f"{fastwam_path}; expected {paths['vendor'] / 'FastWAM/src/fastwam'}"
        )

    problems = []
    for name, expected in EXPECTED_PACKAGES.items():
        if report["packages"].get(name) != expected:
            problems.append(f"{name}={report['packages'].get(name)!r} (expected {expected!r})")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        problems.append("PYTHONNOUSERSITE is not 1")
    if not torchcodec_status["available"]:
        problems.append(f"TorchCodec unavailable: {torchcodec_status['error']}")
    if problems:
        message = "runtime contract warnings: " + "; ".join(problems)
        if args.strict_environment:
            raise RuntimeError(message)
        log.emit(f"[ENV][WARN] {message}")

    (paths["output_dir"] / "environment.json").write_text(
        json.dumps(_json_safe(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _find_dataset_root(cache_root: Path, explicit: str | None, repo: Path) -> Path:
    if explicit:
        root = _resolve_path(explicit, repo)
        candidates = [root]
    else:
        candidates = [
            cache_root / "dataset/data/libero_mujoco3.3.2",
            cache_root / "dataset/lerobot_v30",
        ]
    for root in candidates:
        if all((root / f"{suite}_no_noops_lerobot/meta/info.json").is_file() for suite in SUITES):
            # The pinned FastWAM dataset reader is v2.1.  v3.0 metadata is kept as a
            # fallback candidate only so the error can name it explicitly; do not use it.
            info_path = root / "libero_spatial_no_noops_lerobot/meta/info.json"
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                info = {}
            if info.get("codebase_version") == "v2.1":
                return root
            if explicit:
                raise RuntimeError(
                    f"dataset root {root} has codebase_version={info.get('codebase_version')!r}; "
                    "this pinned FastWAM reader requires v2.1 metadata"
                )
    raise FileNotFoundError(
        "No v2.1 LIBERO dataset found. Expected "
        f"{cache_root / 'dataset/data/libero_mujoco3.3.2'} or pass --dataset-root."
    )


def _default_file(value: str | None, default: Path, repo: Path) -> Path:
    return _resolve_path(value, repo) if value else default.resolve()


def _compose_config(paths: Mapping[str, Path], args: argparse.Namespace):
    from hydra import compose, initialize_config_dir

    task_config = args.task_config
    if task_config:
        task_path = _resolve_path(task_config, paths["repo"])
        task_name = task_path.stem
        config_dir = task_path.parent.parent
    else:
        task_name = args.task
        config_dir = paths["fastwam_root"] / "configs"
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")

    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(config_name="train", overrides=[f"task={task_name}"])
    return cfg


def _configure_runtime_config(cfg: Any, args: argparse.Namespace, paths: Mapping[str, Path], files: Mapping[str, Path]):
    from omegaconf import OmegaConf

    dataset_dirs = [
        str(files["dataset_root"] / f"{suite}_no_noops_lerobot") for suite in SUITES
    ]
    OmegaConf.update(cfg, "data.train.dataset_dirs", dataset_dirs, merge=False)
    OmegaConf.update(cfg, "data.train.pretrained_norm_stats", str(files["dataset_stats"]), force_add=True)
    OmegaConf.update(cfg, "data.train.text_embedding_cache_dir", str(files["text_cache_dir"]), merge=False)
    OmegaConf.update(cfg, "model.action_dit_pretrained_path", str(files["action_dit"]), merge=False)
    OmegaConf.update(cfg, "model.mot_checkpoint_mixed_attn", False, merge=False)
    OmegaConf.update(cfg, "model.load_text_encoder", False, merge=False)
    OmegaConf.update(cfg, "mixed_precision", args.dtype, merge=False)
    OmegaConf.update(cfg, "seed", int(args.seed), merge=False)
    return cfg


def _validate_files(files: Mapping[str, Path], *, dummy_context: bool) -> None:
    required_files = ("checkpoint", "action_dit", "dataset_stats")
    for key in required_files:
        if not files[key].is_file():
            raise FileNotFoundError(f"missing {key}: {files[key]}")
    for suite in SUITES:
        path = files["dataset_root"] / f"{suite}_no_noops_lerobot"
        if not path.is_dir():
            raise FileNotFoundError(f"missing dataset suite: {path}")
    model_root = files["cache_root"] / "models"
    required_model_files = (
        model_root / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors",
        model_root / "DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors",
        model_root / "Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json",
    )
    for path in required_model_files:
        if not path.is_file():
            raise FileNotFoundError(f"missing offline model artifact: {path}")
    if not dummy_context and not files["text_cache_dir"].is_dir():
        raise FileNotFoundError(
            f"text cache directory does not exist: {files['text_cache_dir']}; "
            "pass --dummy-context for a structural smoke or prepare the pinned T5 cache"
        )


def _batchify(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.unsqueeze(0)
    if isinstance(value, Mapping):
        return {key: _batchify(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_batchify(item) for item in value)
    if isinstance(value, list):
        return [_batchify(item) for item in value]
    return value


def _load_sample(cfg: Any, args: argparse.Namespace, files: Mapping[str, Path], log: RunLog):
    import torch
    from hydra.utils import instantiate

    dataset = instantiate(cfg.data.train)
    if args.dummy_context:
        context_len = int(cfg.data.train.context_len)

        def dummy_context(prompt: str):
            del prompt
            return torch.zeros((context_len, 4096), dtype=torch.float32), torch.ones(
                (context_len,), dtype=torch.bool
            )

        dataset._get_cached_text_context = dummy_context
        log.emit("[DATA] using --dummy-context: real video/action/state, zero T5 context")
    log.emit(f"[DATA] dataset_root={files['dataset_root']} length={len(dataset)} index={args.index}")
    if args.index < 0 or args.index >= len(dataset):
        raise IndexError(f"--index={args.index} outside dataset length {len(dataset)}")
    sample = dataset[args.index]
    log.emit(f"[DATA] task={sample.get('task')!r} prompt={sample.get('prompt')!r}")
    return _batchify(sample)


def _load_model(cfg: Any, args: argparse.Namespace, files: Mapping[str, Path], log: RunLog):
    import torch
    from hydra.utils import instantiate

    dtype = _dtype_from_name(args.dtype)
    log.emit(f"[MODEL] constructing dtype={args.dtype} device={args.device}")
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    if files["checkpoint"].is_file():
        log.emit(f"[MODEL] loading checkpoint={files['checkpoint']}")
        model.load_checkpoint(str(files["checkpoint"]))
    elif not args.keep_going_on_missing_checkpoint:
        raise FileNotFoundError(f"missing checkpoint: {files['checkpoint']}")
    else:
        log.emit("[MODEL][WARN] checkpoint missing; continuing with base/random trainable weights")
    model = model.to(args.device)
    # Match Wan22Trainer._apply_dit_only_train_mode: the VAE/text side stays in
    # eval mode while the trainable DiT/MoT side remains in train mode.  We still
    # run under no_grad below, so this does not allocate a backward graph.
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    proprio_encoder = getattr(model, "proprio_encoder", None)
    if proprio_encoder is not None:
        proprio_encoder.train()
        proprio_encoder.requires_grad_(True)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    log.emit(
        f"[MODEL] ready; training forward modes: model.training={model.training} "
        f"dit.training={model.dit.training} forward-only=no_grad"
    )
    return model


def _memory_snapshot(device: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return {}
    target = torch.device(device)
    return {
        "allocated_gib": round(torch.cuda.memory_allocated(target) / 2**30, 3),
        "reserved_gib": round(torch.cuda.memory_reserved(target) / 2**30, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(target) / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(target) / 2**30, 3),
    }


def _autocast_context(args: argparse.Namespace):
    import torch

    if args.dtype == "fp32":
        return nullcontext()
    device_type = torch.device(args.device).type
    if device_type == "cuda" and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=_dtype_from_name(args.dtype))
    if device_type == "cpu" and args.dtype == "bf16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _run_forward(model: Any, sample: Mapping[str, Any], recorder: TraceRecorder, args: argparse.Namespace, log: RunLog):
    import torch

    recorder.record("input.sample", sample, kind="input")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    log.emit("[STATUS] forward_start step=1/1")
    started = time.perf_counter()
    with torch.no_grad(), _autocast_context(args):
        result = model(sample)
    elapsed = time.perf_counter() - started
    if isinstance(result, tuple) and len(result) == 2:
        total_loss, loss_dict = result
        log.emit(f"[DONE] forward elapsed_s={elapsed:.3f} loss={float(total_loss.item()):.8f} components={loss_dict}")
    else:
        log.emit(f"[DONE] forward elapsed_s={elapsed:.3f} result={_value_summary(result)}")
    return result, elapsed


def main() -> int:
    args = _parse_args()
    paths = _prepare_environment(args)
    timestamp = dt.datetime.now(dt.timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    output_arg = args.output_dir or str(paths["repo"] / "outputs/literature/w0_fastwam/layer_debug" / timestamp)
    paths = dict(paths)
    paths["output_dir"] = _resolve_path(output_arg, paths["repo"])
    log = RunLog(paths["output_dir"])
    # RobotVideoDataset mirrors the upstream trainer and writes a copy of the
    # normalization stats through FastWAM's work-dir registry.  Point that registry
    # at this debug run so dataset construction never falls back to ./runs.
    from fastwam.utils.misc import register_work_dir

    register_work_dir(paths["output_dir"])
    recorder: TraceRecorder | None = None
    registry: HookRegistry | None = None
    try:
        report = _environment_report(paths, args, log)
        del report
        dataset_root = _find_dataset_root(paths["cache_root"], args.dataset_root, paths["repo"])
        files = {
            **paths,
            "dataset_root": dataset_root,
            "dataset_stats": _default_file(
                args.dataset_stats,
                paths["cache_root"] / "upstream/libero_uncond_2cam224_dataset_stats.json",
                paths["repo"],
            ),
            "text_cache_dir": _default_file(
                args.text_cache_dir,
                paths["cache_root"] / "dataset/text_embeds_cache/libero",
                paths["repo"],
            ),
            "checkpoint": _default_file(
                args.checkpoint,
                paths["cache_root"] / "upstream/libero_uncond_2cam224.pt",
                paths["repo"],
            ),
            "action_dit": _default_file(
                args.action_dit,
                paths["cache_root"] / "ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
                paths["repo"],
            ),
        }
        _validate_files(files, dummy_context=args.dummy_context)
        (paths["output_dir"] / "inputs.json").write_text(
            json.dumps(_json_safe({"args": vars(args), "files": files}), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        cfg = _compose_config(paths, args)
        cfg = _configure_runtime_config(cfg, args, paths, files)
        sample = _load_sample(cfg, args, files, log)
        model = _load_model(cfg, args, files, log)
        recorder = TraceRecorder(
            paths["output_dir"],
            log,
            save_patterns=args.save,
            pdb_pattern=args.pdb_at,
        )
        registry = HookRegistry(
            model,
            recorder,
            module_pattern=args.module_regex,
            all_modules=args.all_modules,
        )
        module_count = registry.install_module_hooks()
        registry.install_fastwam_boundaries()
        registry.install_scheduler_boundaries()
        log.emit(
            f"[TRACE] installed module_hooks={module_count} custom_boundaries=mot_layers+loss+pre_post+vae+schedulers"
        )
        result, elapsed = _run_forward(model, sample, recorder, args, log)
        del result
        summary = {
            "status": "complete",
            "elapsed_s": elapsed,
            "event_count": len(recorder.events),
            "memory": _memory_snapshot(args.device),
            "output_dir": str(paths["output_dir"]),
            "note": "forward-only; no backward/optimizer/distributed process group",
        }
        (paths["output_dir"] / "summary.json").write_text(
            json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        log.emit(f"[DONE] events={len(recorder.events)} memory={summary['memory']} output={paths['output_dir']}")
        return 0
    except Exception as error:
        log.emit(f"[FAIL] {type(error).__name__}: {error}")
        raise
    finally:
        if registry is not None:
            registry.close()
        if recorder is not None:
            recorder.close()
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
