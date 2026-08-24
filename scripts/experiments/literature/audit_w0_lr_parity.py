#!/usr/bin/env python
"""Audit the LR-REL observation pipeline against the official UP-REL contract.

LR-REL runs to completion with 0/20 success while UP-REL reaches 96.45% with the
same nominal policy.  Every stage that could silently change the model input is
deterministic, so it can be compared numerically without a rollout, without the
12 GiB DiT and without an EGL context.

Static reading already showed that several suspected differences do **not**
exist: the 180-degree flip and the 8D state assembly are applied by
``lerobot.processor.env_processor.LiberoProcessorStep`` (invoked at
``lerobot_eval.py:278``), camera concatenation is ordered by sorted keys, and the
gripper post-processing is mathematically identical.  What is not yet verified is
the **image value range**, because the two stacks split the ``[0,1] -> [-1,1]``
mapping differently:

* official: ``x * (2/255) - 1`` in ``_obs_to_model_input``; the FastWAM core does
  not rescale (``src/fastwam/models/wan22/fastwam.py:261``).
* LeRobot: ``preprocess_observation`` yields ``[0,1]``, and the LeRobot FastWAM
  core rescales with ``input_image * 2.0 - 1.0``
  (``policies/fastwam/wan/modular.py:1086``).

The LeRobot processor source states VISUAL must therefore stay ``IDENTITY``, but
the released checkpoint's ``config.json`` may set another mode.  If VISUAL is
normalised with ``mean=0.5, std=0.5`` the mapping is applied twice
(``4x - 3`` instead of ``2x - 1``), which distorts every frame while leaving the
whole pipeline apparently healthy.

This script reproduces the exact eval-time pipeline on a synthetic observation
and reports, per stage, the tensor range plus the deviation from the official
target.  It also verifies the flip empirically with an asymmetric marker image.

Run inside the LeRobot environment.  Nothing is written outside ``--output``.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-path",
        required=True,
        help="local LeRobot FastWAM checkpoint directory",
    )
    parser.add_argument(
        "--upstream-stats",
        default="pretrained/w0_fastwam/upstream/libero_uncond_2cam224_dataset_stats.json",
        help="official dataset stats JSON used by the UP-REL evaluator",
    )
    parser.add_argument(
        "--visual-normalization",
        choices=("checkpoint", "identity"),
        default="checkpoint",
        help=(
            "'checkpoint' (default) audits the pipeline exactly as shipped. "
            "'identity' forces norm_map.VISUAL=IDENTITY on the loaded normalizer "
            "step, which is what the launcher's symlink overlay does, so the fix can "
            "be verified without building an overlay or touching the artifact."
        ),
    )
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument(
        "--build-model",
        action="store_true",
        help="also construct the policy to report checkpoint tensors dropped by "
        "the shape-aware loader (slow: loads the Wan VAE)",
    )
    parser.add_argument("--output", default="", help="optional JSON report path")
    return parser.parse_args(argv)


def _read_safetensors_floats(path: Path) -> dict[str, np.ndarray]:
    """Read a small safetensors file without importing torch."""

    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len])
    base = 8 + header_len
    out: dict[str, np.ndarray] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        dtype = {"F32": np.float32, "F64": np.float64, "F16": np.float16}.get(meta["dtype"])
        if dtype is None:
            continue
        out[key] = np.frombuffer(raw[base + start : base + end], dtype=dtype).reshape(
            meta["shape"]
        )
    return out


def _marker_images(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Two uint8 frames with an asymmetric marker so a flip is detectable."""

    front = np.zeros((height, width, 3), dtype=np.uint8)
    front[:, :] = 40
    front[: height // 8, : width // 8] = (255, 0, 0)  # top-left red
    front[-height // 8 :, -width // 8 :] = (0, 255, 0)  # bottom-right green
    wrist = np.full((height, width, 3), 200, dtype=np.uint8)
    wrist[: height // 8, : width // 8] = (0, 0, 255)  # top-left blue
    return front, wrist


def _describe(name: str, tensor: Any) -> dict[str, Any]:
    array = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    record = {
        "stage": name,
        "shape": list(array.shape),
        "min": round(float(array.min()), 6),
        "max": round(float(array.max()), 6),
        "mean": round(float(array.mean()), 6),
    }
    print(
        f"[LR-AUDIT] {name:38s} shape={record['shape']} "
        f"min={record['min']:+.4f} max={record['max']:+.4f} mean={record['mean']:+.4f}",
        flush=True,
    )
    return record


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy_dir = Path(args.policy_path).expanduser().resolve()
    if not policy_dir.is_dir():
        raise NotADirectoryError(f"policy path is not a directory: {policy_dir}")

    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import NormalizationMode
    from lerobot.envs.utils import preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.fastwam.configuration_fastwam import FastWAMConfig
    from lerobot.policies.fastwam.modeling_fastwam import _input_image_from_batch
    from lerobot.processor.env_processor import LiberoProcessorStep

    report: dict[str, Any] = {"policy_path": str(policy_dir), "stages": []}

    # ---------------------------------------------------------------- config
    # `config.json` carries a `type` discriminator, so it must be decoded through
    # the ChoiceRegistry base class; decoding straight into FastWAMConfig makes
    # draccus reject `type` as an unknown field.
    config = PreTrainedConfig.from_pretrained(str(policy_dir))
    if not isinstance(config, FastWAMConfig):
        raise TypeError(
            f"expected a FastWAM checkpoint, got {type(config).__name__} "
            f"from {policy_dir / 'config.json'}"
        )
    config.device = "cpu"
    try:
        default_mapping = {
            key: str(value) for key, value in FastWAMConfig().normalization_mapping.items()
        }
    except Exception as error:  # pragma: no cover - depends on dataclass defaults
        default_mapping = {"error": f"{type(error).__name__}: {error}"}
    mapping = {key: str(value) for key, value in config.normalization_mapping.items()}
    report["normalization_mapping"] = mapping
    report["code_default_mapping"] = default_mapping
    print(f"[LR-AUDIT] checkpoint normalization_mapping = {mapping}", flush=True)
    print(f"[LR-AUDIT] code default            mapping = {default_mapping}", flush=True)
    visual_mode = config.normalization_mapping.get("VISUAL")
    visual_is_identity = visual_mode == NormalizationMode.IDENTITY
    report["visual_is_identity"] = bool(visual_is_identity)
    if not visual_is_identity:
        print(
            f"[LR-AUDIT] WARNING VISUAL={visual_mode} but the LeRobot FastWAM core "
            "already maps [0,1] -> [-1,1] (wan/modular.py:1086). Any VISUAL "
            "normalisation is applied on top of that mapping.",
            flush=True,
        )

    # ------------------------------------------------------- baked-in stats
    normalizer_files = sorted(policy_dir.glob("policy_preprocessor_step_*normalizer*.safetensors"))
    baked: dict[str, np.ndarray] = {}
    for path in normalizer_files:
        baked.update(_read_safetensors_floats(path))
    report["baked_stat_keys"] = sorted(baked)
    for key in sorted(baked):
        if key.startswith("observation.images."):
            print(f"[LR-AUDIT] baked {key:44s} = {np.round(baked[key].reshape(-1), 6)}", flush=True)

    upstream_stats_path = Path(args.upstream_stats)
    if upstream_stats_path.is_file():
        upstream = json.loads(upstream_stats_path.read_text(encoding="utf-8"))
        report["upstream_stats_keys"] = sorted(upstream)
        for group in ("state", "action"):
            block = upstream.get(group, {}).get("default", {})
            for stat in ("min", "max", "mean", "std"):
                if stat not in block:
                    continue
                lerobot_key = ("observation.state" if group == "state" else "action") + f".{stat}"
                if lerobot_key not in baked:
                    continue
                official = np.asarray(block[stat], dtype=np.float32).reshape(-1)
                mine = baked[lerobot_key].reshape(-1)
                if official.shape != mine.shape:
                    print(
                        f"[LR-AUDIT] {lerobot_key}: shape mismatch "
                        f"official={official.shape} checkpoint={mine.shape}",
                        flush=True,
                    )
                    continue
                delta = float(np.max(np.abs(official - mine)))
                report[f"stats_delta.{lerobot_key}"] = delta
                print(
                    f"[LR-AUDIT] stats {lerobot_key:26s} max|official-checkpoint| = {delta:.3e}",
                    flush=True,
                )
    else:
        print(f"[LR-AUDIT] upstream stats not found: {upstream_stats_path}", flush=True)

    # ---------------------------------------------- replay the eval pipeline
    front, wrist = _marker_images(args.height, args.width)
    state_8d = np.array([0.1, -0.2, 1.05, 0.3, -0.4, 0.5, 0.02, -0.02], dtype=np.float32)
    raw_observation = {
        # LiberoEnv._format_raw_obs hands over the *unflipped* camera arrays.
        "pixels": {"image": front[None], "image2": wrist[None]},
        "robot_state": {
            "eef": {
                "pos": state_8d[None, :3],
                "quat": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                "mat": np.eye(3, dtype=np.float32)[None],
            },
            "gripper": {
                "qpos": state_8d[None, 6:8],
                "qvel": np.zeros((1, 2), dtype=np.float32),
            },
            "joints": {
                "pos": np.zeros((1, 7), dtype=np.float32),
                "vel": np.zeros((1, 7), dtype=np.float32),
            },
        },
    }

    observation = preprocess_observation(raw_observation)
    report["stages"].append(_describe("preprocess_observation image", observation["observation.images.image"]))

    libero_step = LiberoProcessorStep()
    if hasattr(libero_step, "_process_observation"):
        observation = libero_step._process_observation(observation)
    else:  # pragma: no cover - depends on the pinned LeRobot revision
        raise AttributeError(
            "LiberoProcessorStep has no _process_observation; inspect "
            "lerobot/processor/env_processor.py for the current entry point"
        )
    report["stages"].append(_describe("LiberoProcessorStep image", observation["observation.images.image"]))
    if "observation.state" in observation:
        report["stages"].append(_describe("LiberoProcessorStep state", observation["observation.state"]))
        report["state_8d"] = [round(float(v), 6) for v in observation["observation.state"][0]]
        print(f"[LR-AUDIT] assembled 8D state = {report['state_8d']}", flush=True)

    # Empirical flip check: the marker must move from top-left to bottom-right.
    flipped = observation["observation.images.image"][0].permute(1, 2, 0).numpy()
    corner_top_left = flipped[: args.height // 8, : args.width // 8].mean(axis=(0, 1))
    corner_bottom_right = flipped[-args.height // 8 :, -args.width // 8 :].mean(axis=(0, 1))
    rot180_applied = bool(corner_bottom_right[0] > corner_top_left[0])
    report["rot180_applied"] = rot180_applied
    print(
        f"[LR-AUDIT] rot180 applied by env processor = {rot180_applied} "
        f"(top-left RGB={np.round(corner_top_left, 3)}, "
        f"bottom-right RGB={np.round(corner_bottom_right, 3)})",
        flush=True,
    )

    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    if args.visual_normalization == "identity":
        patched_steps = []
        for step in getattr(preprocessor, "steps", []):
            norm_map = getattr(step, "norm_map", None)
            if not isinstance(norm_map, dict) or "VISUAL" not in norm_map:
                continue
            previous = norm_map["VISUAL"]
            norm_map["VISUAL"] = NormalizationMode.IDENTITY
            patched_steps.append(f"{type(step).__name__}: {previous} -> IDENTITY")
        report["visual_norm_forced_identity"] = patched_steps
        if not patched_steps:
            print(
                "[LR-AUDIT] WARNING no loaded step exposes a norm_map with VISUAL; "
                "the forced-identity request had no effect",
                flush=True,
            )
        for line in patched_steps:
            print(f"[LR-AUDIT] forced {line}", flush=True)

    batch = preprocessor(observation)
    report["stages"].append(_describe("policy preprocessor image", batch["observation.images.image"]))
    if "observation.state" in batch:
        report["stages"].append(_describe("policy preprocessor state", batch["observation.state"]))

    model_input = _input_image_from_batch(batch, config)
    report["stages"].append(_describe("model input (before core)", model_input))
    core_input = model_input.float() * 2.0 - 1.0  # wan/modular.py:1086
    report["stages"].append(_describe("core input after *2-1", core_input))

    # ------------------------------------------------- official reference
    official_front = np.ascontiguousarray(front[::-1, ::-1])
    official_wrist = np.ascontiguousarray(wrist[::-1, ::-1])
    official_rgb = np.concatenate([official_front, official_wrist], axis=1)
    official = torch.tensor(official_rgb).permute(2, 0, 1).unsqueeze(0).float()
    official = official * (2.0 / 255.0) - 1.0
    report["stages"].append(_describe("official VAE input", official))

    if tuple(official.shape) == tuple(core_input.shape):
        delta = float((core_input - official).abs().max())
        report["vae_input_max_abs_delta"] = delta
        print(f"[LR-AUDIT] max|LeRobot_core_input - official_VAE_input| = {delta:.6f}", flush=True)
        if delta > 1e-3:
            ratio = float(
                (core_input.abs().mean() / official.abs().mean()).item()
            )
            print(
                f"[LR-AUDIT] MISMATCH the two stacks feed different pixels to the VAE "
                f"(mean-abs ratio LeRobot/official = {ratio:.4f})",
                flush=True,
            )
        else:
            print("[LR-AUDIT] OK VAE inputs agree; look elsewhere for the LR-REL gap", flush=True)
    else:
        report["vae_input_shape_mismatch"] = [
            list(core_input.shape),
            list(official.shape),
        ]
        print(
            f"[LR-AUDIT] MISMATCH shape LeRobot={tuple(core_input.shape)} "
            f"official={tuple(official.shape)}",
            flush=True,
        )

    # -------------------------------------------- optional loader integrity
    if args.build_model:
        from safetensors import safe_open

        from lerobot.policies.fastwam.modeling_fastwam import FastWAMPolicy

        print("[LR-AUDIT] constructing policy to check for dropped tensors...", flush=True)
        config.load_text_encoder = False
        policy = FastWAMPolicy(config)
        model_state = policy.state_dict()
        weights = policy_dir / "model.safetensors"
        mismatched: list[str] = []
        missing: list[str] = []
        with safe_open(str(weights), framework="pt") as handle:
            checkpoint_keys = set(handle.keys())
            for key in checkpoint_keys:
                if key in model_state and tuple(model_state[key].shape) != tuple(
                    handle.get_slice(key).get_shape()
                ):
                    mismatched.append(key)
            missing = sorted(k for k in model_state if k not in checkpoint_keys)
        report["checkpoint_shape_mismatched"] = sorted(mismatched)
        report["checkpoint_missing_in_file"] = missing[:50]
        report["checkpoint_missing_count"] = len(missing)
        print(
            f"[LR-AUDIT] shape-mismatched tensors (silently dropped, left random) = "
            f"{len(mismatched)}",
            flush=True,
        )
        for key in sorted(mismatched)[:20]:
            print(f"[LR-AUDIT]   mismatched: {key}", flush=True)
        print(f"[LR-AUDIT] model tensors absent from the file = {len(missing)}", flush=True)
        for key in missing[:20]:
            print(f"[LR-AUDIT]   missing: {key}", flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[LR-AUDIT] report written to {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
