"""Small, explicit adapter for the robomimic/DPPO pixel residual contract.

The upstream DPPO checkout remains the source of the diffusion policy and the
robomimic wrappers.  This module owns only the downstream contract: a frozen
base, a bounded residual actor, and a critic.  ``--dummy`` deliberately uses a
deterministic toy environment so the wiring can be checked without downloading
the robomimic archive or a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


def _sha256_module(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        # Avoid the NumPy ABI at hash time: the Docker image may intentionally
        # use a NumPy version different from the host running the unit tests.
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).tolist().__repr__().encode())
    return h.hexdigest()


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def compose_bounded_residual(
    base_action: torch.Tensor,
    residual: torch.Tensor,
    bound: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compose and clip a residual without ever changing the base tensor in-place."""

    bound = bound.to(device=base_action.device, dtype=base_action.dtype)
    bounded = torch.clamp(residual, -bound, bound)
    return torch.clamp(base_action + float(scale) * bounded, -1.0, 1.0)


class FrozenDPPOBase(nn.Module):
    """Call an upstream DPPO-like policy while keeping it outside autograd.

    DPPO's diffusion models return a named tuple with ``trajectories``.  Tests
    and integrations may instead provide ``get_action`` or a tensor-returning
    callable; all three forms are accepted here.
    """

    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy
        _freeze(self.policy)
        self._initial_hash = _sha256_module(self.policy)

    @property
    def parameter_hash(self) -> str:
        return _sha256_module(self.policy)

    def assert_unchanged(self) -> None:
        current = self.parameter_hash
        if current != self._initial_hash:
            raise RuntimeError(
                f"frozen DPPO base changed: before={self._initial_hash} after={current}"
            )

    @torch.no_grad()
    def forward(self, cond: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.policy, "get_action"):
            action = self.policy.get_action(cond)
        else:
            try:
                sample = self.policy(cond=cond, deterministic=True)
            except TypeError:
                # Useful for the tiny deterministic smoke base and for simple
                # adapters supplied by downstream callers.
                features = pixel_state_features(cond) if "rgb" in cond else cond["state"].flatten(1)
                sample = self.policy(features)
            action = getattr(sample, "trajectories", sample)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action)
        return action.detach()


class ResidualActor(nn.Module):
    def __init__(self, feature_dim: int, horizon: int, action_dim: int):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, horizon * action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1, self.horizon, self.action_dim)


class ResidualCritic(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def pixel_state_features(obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Build a compact feature vector without admitting object observations."""

    unexpected = sorted(set(obs) - {"rgb", "state"})
    if unexpected:
        raise ValueError(f"object or unknown observations leaked into residual: {unexpected}")
    state = obs["state"].float().flatten(1)
    rgb = obs["rgb"].float()
    # Mean/std preserve a cheap pixel signal for the adapter smoke while the
    # real DPPO policy still consumes the full [3,96,96] image tensor.
    image_stats = torch.cat([rgb.mean(dim=(2, 3)), rgb.std(dim=(2, 3))], dim=1)
    return torch.cat([state, image_stats], dim=1)


class DummySquareEnv:
    """Deterministic pixel-shaped env used only for contract/smoke validation."""

    def __init__(self, seed: int, episodes: int = 1):
        self.generator = torch.Generator().manual_seed(seed)
        self.episodes = episodes
        self.t = 0
        self.state = torch.zeros(9)

    def reset(self) -> dict[str, torch.Tensor]:
        self.t = 0
        self.state = torch.randn(9, generator=self.generator) * 0.1
        return self._obs()

    def _obs(self) -> dict[str, torch.Tensor]:
        rgb = torch.full((1, 3, 96, 96), self.t / 10.0)
        return {"rgb": rgb, "state": self.state[None]}

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], float, bool, bool]:
        self.t += 1
        flat_action = action.detach().cpu().reshape(-1, action.shape[-1])[0]
        self.state[: min(7, len(self.state))] += flat_action[:7] * 0.01
        reward = float(torch.clamp(1.0 - torch.linalg.vector_norm(self.state[:2]), 0.0, 1.0))
        terminated = self.t >= 8
        return self._obs(), reward, terminated, False


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - Docker provides PyYAML
        raise RuntimeError("PyYAML is required by the DPPO adapter") from exc
    raw = yaml.safe_load(path.read_text()) or {}

    env_pattern = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*))?\}")

    def expand(value: Any) -> Any:
        if isinstance(value, str):
            def replace(match: re.Match[str]) -> str:
                name, _, default = match.groups()
                return os.environ.get(name, default or match.group(0))

            return env_pattern.sub(replace, value)
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    expanded = expand(raw)
    # Resolve the few local references used by the config without requiring
    # OmegaConf in the lightweight smoke path.
    for _ in range(4):
        changed = False

        def resolve(value: Any) -> Any:
            nonlocal changed
            if isinstance(value, str):
                def replace_ref(match: re.Match[str]) -> str:
                    nonlocal changed
                    cursor: Any = expanded
                    try:
                        for part in match.group(1).split("."):
                            cursor = cursor[part]
                    except (KeyError, TypeError):
                        return match.group(0)
                    changed = True
                    return str(cursor)

                return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_.]*)\}", replace_ref, value)
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            if isinstance(value, list):
                return [resolve(v) for v in value]
            return value

        expanded = resolve(expanded)
        if not changed:
            break
    return expanded


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {message}\n"
    with path.open("a") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def _check_contract(cfg: Mapping[str, Any]) -> None:
    obs = cfg["observation"]
    if obs.get("allow_object_obs"):
        raise ValueError("pixel contract must keep allow_object_obs=false")
    if any("object" in str(key).lower() for key in obs.get("low_dim_keys", [])):
        raise ValueError("object observation key found in low_dim_keys")
    if obs.get("image_shape") != [3, 96, 96] or obs.get("state_dim") != 9:
        raise ValueError("unexpected DPPO Square pixel observation shape")
    if cfg.get("action_dim") != 7 or cfg.get("horizon_steps") != cfg.get("act_steps"):
        raise ValueError("unexpected DPPO Square action/horizon contract")


def _run_dummy(phase: str, cfg: Mapping[str, Any], output: Path, smoke: bool, max_iterations: int | None) -> None:
    _check_contract(cfg)
    seed = int(cfg["train"]["seeds"][0])
    random.seed(seed)
    try:
        np.random.seed(seed)
    except Exception:  # host may have an incompatible optional NumPy ABI
        pass
    torch.manual_seed(seed)
    feature_dim = int(cfg["observation"]["state_dim"]) + 6
    horizon = int(cfg["horizon_steps"])
    action_dim = int(cfg["action_dim"])
    base_policy = nn.Linear(feature_dim, horizon * action_dim, bias=False)
    base = FrozenDPPOBase(base_policy)
    actor = ResidualActor(feature_dim, horizon, action_dim)
    critic = ResidualCritic(feature_dim)
    bound = torch.tensor(cfg["train"]["residual_bound"], dtype=torch.float32).view(1, 1, -1)
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-3)
    log_path = output / "run.log"
    iterations = 1 if smoke else int(max_iterations or cfg["train"]["n_train_itr"])
    total = iterations
    completed = 0
    _log(log_path, f"CONTRACT phase={phase} dummy=true base_parameter_hash={base.parameter_hash}")
    env = DummySquareEnv(seed)
    metrics: dict[str, Any] = {
        "phase": phase,
        "dummy": True,
        "observation_keys": ["rgb", "state"],
        "action_dim": action_dim,
        "horizon_steps": horizon,
        "base_parameter_hash_before": base.parameter_hash,
        "zero_residual_action_equal": True,
        "iterations": iterations,
    }
    for iteration in range(iterations):
        obs = env.reset()
        feat = pixel_state_features(obs)
        base_action = base({"state": obs["state"], "rgb": obs["rgb"]}).view(-1, horizon, action_dim)
        residual = actor(feat)
        action = compose_bounded_residual(base_action, residual, bound, cfg["train"]["residual_scale"])
        zero_action = compose_bounded_residual(base_action, torch.zeros_like(residual), bound, 1.0)
        metrics["zero_residual_action_equal"] &= bool(torch.equal(zero_action, base_action.clamp(-1.0, 1.0)))
        _, reward, _, _ = env.step(action[:, :1])
        value = critic(feat)
        loss = (value - torch.tensor([reward])).square().mean() + residual.square().mean() * 1e-3
        if phase == "train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 1.0)
            optimizer.step()
        completed = iteration + 1
        _log(log_path, f"STATUS completed={completed} total={total} loss={loss.item():.6f}")
        with (output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps({"completed": completed, "total": total, "loss": loss.item()}) + "\n")
    base.assert_unchanged()
    metrics["base_parameter_hash_after"] = base.parameter_hash
    metrics["base_hash_unchanged"] = metrics["base_parameter_hash_before"] == metrics["base_parameter_hash_after"]
    if phase == "train":
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "metrics": metrics}, output / "checkpoint.pt")
    _write_json(output / "metrics.json", metrics)
    _log(log_path, f"DONE completed={completed} total={total}")


def summarize(root: Path) -> None:
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    _write_json(root / "summary.json", {"runs": rows, "count": len(rows)})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["preflight", "base-eval", "train", "eval", "summarize"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--time-stages", action="store_true")
    parser.add_argument("--residual-mode", choices=["learned", "zero", "random-bounded"], default="learned")
    args = parser.parse_args(argv)
    if args.phase == "summarize":
        summarize(args.output_root)
        return 0
    cfg = _load_yaml(args.config)
    _check_contract(cfg)
    if args.phase == "preflight":
        paths = cfg["paths"]
        missing = [key for key in ("train_dataset", "normalization", "base_checkpoint") if not Path(paths[key]).exists()]
        output = args.output_root / "gates.json"
        _write_json(output, {"contract_ok": True, "missing_assets": missing, "dummy": args.dummy})
        if missing and not args.dummy:
            print("Missing assets (download commands are in the guide): " + ", ".join(missing))
            return 3
        return 0
    if not args.dummy:
        raise RuntimeError("real DPPO execution is intentionally gated: run preflight after mounting downloaded assets")
    phase = "base-eval" if args.phase == "base-eval" else args.phase
    _run_dummy(phase, cfg, args.output_root, args.smoke, args.max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
