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
import math
import os
import pickle
import random
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F


def _sha256_module(module: nn.Module) -> str:
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        # Prefer a compact byte view in Docker; retain a no-NumPy fallback for
        # the host smoke environment whose optional NumPy ABI may differ.
        compact = tensor.detach().cpu().contiguous()
        try:
            h.update(compact.numpy().tobytes())
        except RuntimeError:
            h.update(compact.view(torch.uint8).tolist().__repr__().encode())
    return h.hexdigest()


def _freeze(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def compose_bounded_residual(
    base_action: torch.Tensor,
    residual: torch.Tensor,
    bound: torch.Tensor | None = None,
    scale: float = 1.0,
) -> torch.Tensor:
    """Compose an optional residual and apply the upstream [-1, 1] action contract.

    Bounds are deliberately optional: this repair plan must not pre-register a
    residual scale in physical controller units. A later residual experiment can
    pass an explicitly chosen normalized-space bound.
    """

    if bound is not None:
        bound = bound.to(device=base_action.device, dtype=base_action.dtype)
        residual = torch.clamp(residual, -bound, bound)
    bounded = residual
    return torch.clamp(base_action + float(scale) * bounded, -1.0, 1.0)


def unnormalize_action(action: torch.Tensor, action_min: torch.Tensor, action_max: torch.Tensor) -> torch.Tensor:
    """Mirror DPPO's robomimic_image wrapper action mapping exactly."""

    action_min = action_min.to(device=action.device, dtype=action.dtype)
    action_max = action_max.to(device=action.device, dtype=action.dtype)
    return (action + 1.0) / 2.0 * (action_max - action_min) + action_min


def create_optimizers(actor: nn.Module, critic: nn.Module, cfg: Mapping[str, Any]):
    """Create independent actor/critic optimizers, matching upstream's split."""

    return (
        torch.optim.Adam(actor.parameters(), lr=float(cfg["train"]["actor_lr"])),
        torch.optim.Adam(critic.parameters(), lr=float(cfg["train"]["critic_lr"])),
    )


def resolve_reward_shaping(cfg: Mapping[str, Any], *, anchor: bool = False) -> bool:
    """Return the reward mode actually passed to an environment constructor."""

    return False if anchor else bool(cfg["env"]["reward_shaping"])


def path_usage(cfg: Mapping[str, Any]) -> dict[str, str]:
    """Record the only two assets consumed by the frozen-base experiment."""

    anchor = cfg["paths"]["anchor"]
    return {
        "anchor_normalization": str(Path(anchor["normalization"]).resolve()),
        "anchor_base_checkpoint": str(Path(anchor["base_checkpoint"]).resolve()),
    }


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
    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        action_dim: int,
        residual_bound: torch.Tensor | Sequence[float] | None = None,
        initial_log_std: float = -1.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, horizon * action_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_std = nn.Parameter(torch.full((horizon, action_dim), float(initial_log_std)))
        if residual_bound is None:
            residual_bound = torch.ones(action_dim)
        bound = torch.as_tensor(residual_bound, dtype=torch.float32).view(1, 1, action_dim)
        if torch.any(bound <= 0):
            raise ValueError("residual_bound entries must be positive")
        self.register_buffer("residual_bound", bound)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).view(-1, self.horizon, self.action_dim)

    def distribution(self, features: torch.Tensor) -> Normal:
        mean = self(features)
        std = self.log_std.clamp(-5.0, 1.0).exp().view(1, self.horizon, self.action_dim)
        return Normal(mean, std)

    def residual_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(latent) * self.residual_bound

    def sample(
        self, features: torch.Tensor, deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.distribution(features)
        latent = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(latent).sum(dim=(1, 2))
        return self.residual_from_latent(latent), log_prob, latent

    def log_prob(self, features: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(features)
        return dist.log_prob(latent).sum(dim=(1, 2)), dist.entropy().sum(dim=(1, 2))


class ResidualCritic(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(), nn.Linear(256, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def pixel_state_features(obs: Mapping[str, torch.Tensor], image_feature_hw: int = 12) -> torch.Tensor:
    """Build spatial pixel features without admitting privileged observations."""

    unexpected = sorted(set(obs) - {"rgb", "state"})
    if unexpected:
        raise ValueError(f"object or unknown observations leaked into residual: {unexpected}")
    state = obs["state"].float()
    if state.ndim == 3:
        state = state[:, -1]
    state = state.flatten(1)
    rgb = obs["rgb"].float()
    if rgb.ndim == 5:
        rgb = rgb[:, -1]
    if rgb.ndim != 4:
        raise ValueError(f"rgb must be [B,C,H,W] or [B,T,C,H,W], got {tuple(rgb.shape)}")
    # DPPO's wrapper emits float RGB in [0, 255]. A fixed spatial grid keeps
    # object location information while making the on-policy buffer small.
    spatial = F.adaptive_avg_pool2d(rgb / 255.0, (image_feature_hw, image_feature_hw))
    return torch.cat([state, spatial.flatten(1)], dim=1)


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

        def resolve(value: Any, expanded_snapshot: Any = expanded) -> Any:
            nonlocal changed
            if isinstance(value, str):
                def replace_ref(match: re.Match[str]) -> str:
                    nonlocal changed
                    cursor: Any = expanded_snapshot
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
    if int(cfg.get("ft_denoising_steps", -1)) != 0:
        raise ValueError("the supervised state_4000.pt base requires ft_denoising_steps=0")


def _check_path_groups(cfg: Mapping[str, Any]) -> None:
    paths = cfg.get("paths", {})
    anchor = paths.get("anchor", {})
    required_anchor = {"normalization", "base_checkpoint"}
    if not required_anchor <= set(anchor):
        raise ValueError("paths.anchor must contain normalization and base_checkpoint")
    if "pretrain" in paths:
        raise ValueError("the frozen-base residual experiment must not require pretrain data")


def _validate_anchor_assets(cfg: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    anchor = cfg["paths"]["anchor"]
    files = {
        "anchor_normalization": Path(anchor["normalization"]),
        "anchor_base_checkpoint": Path(anchor["base_checkpoint"]),
    }
    missing = [name for name, path in files.items() if not path.is_file() or path.stat().st_size == 0]
    errors: list[str] = []
    if "anchor_normalization" not in missing:
        try:
            with np.load(files["anchor_normalization"]) as normalization:
                required = {"obs_min", "obs_max", "action_min", "action_max"}
                absent = sorted(required - set(normalization.files))
                if absent:
                    errors.append("anchor_normalization missing keys: " + ", ".join(absent))
        except (EOFError, OSError, ValueError) as exc:
            errors.append(f"anchor_normalization unreadable: {type(exc).__name__}: {exc}")
    if "anchor_base_checkpoint" not in missing:
        try:
            checkpoint = torch.load(
                files["anchor_base_checkpoint"], map_location="cpu", weights_only=True
            )
            model = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
            if not isinstance(model, Mapping) or not any(
                str(key).startswith("network.") for key in model
            ):
                errors.append("anchor_base_checkpoint has no supervised network.* weights")
        except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
            errors.append(f"anchor_base_checkpoint unreadable: {type(exc).__name__}: {exc}")
    return missing, errors


def build_episode_manifest(
    seeds: Sequence[int], episodes_per_seed: int, reset_seed_base: int = 1001,
) -> list[dict[str, int | str]]:
    """Return a deterministic, disjoint reset schedule for paired evaluation."""

    manifest: list[dict[str, int | str]] = []
    seen: set[int] = set()
    for seed_index, _policy_seed in enumerate(seeds):
        for episode_index in range(int(episodes_per_seed)):
            reset_seed = int(reset_seed_base) + 1 + seed_index * int(episodes_per_seed) + episode_index
            if reset_seed in seen:
                raise ValueError(f"duplicate reset_seed generated: {reset_seed}")
            seen.add(reset_seed)
            manifest.append({
                "episode_key": f"seed_{seed_index}/episode_{episode_index}",
                "seed_index": seed_index,
                "episode_index": episode_index,
                "reset_seed": reset_seed,
            })
    return manifest


def write_episode_manifest(path: Path, manifest: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def load_episode_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keys = [row.get("episode_key") for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate episode_key in manifest: {path}")
    seeds = [row.get("reset_seed") for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate reset_seed in manifest: {path}")
    return rows


def assert_paired_manifests(base_path: Path, residual_path: Path) -> None:
    base = load_episode_manifest(base_path)
    residual = load_episode_manifest(residual_path)
    base_contract = [(row["episode_key"], row["seed_index"], row["episode_index"], row["reset_seed"]) for row in base]
    residual_contract = [(row["episode_key"], row["seed_index"], row["episode_index"], row["reset_seed"]) for row in residual]
    if base_contract != residual_contract:
        raise ValueError("base and residual episode manifests are not paired")


def _manifest_seed_index(cfg: Mapping[str, Any], seed: int) -> int:
    seeds = [int(value) for value in cfg["train"]["seeds"]]
    try:
        return seeds.index(int(seed))
    except ValueError as exc:
        raise ValueError(f"seed={seed} is not listed in train.seeds={seeds}") from exc


def _run_dummy(phase: str, cfg: Mapping[str, Any], output: Path, smoke: bool, max_iterations: int | None) -> None:
    _check_contract(cfg)
    _check_path_groups(cfg)
    seed = int(cfg["train"]["seeds"][0])
    random.seed(seed)
    try:
        np.random.seed(seed)
    except (ImportError, RuntimeError):  # host may have an incompatible optional NumPy ABI
        print("WARNING NumPy RNG seeding unavailable in host-only dummy mode", flush=True)
    torch.manual_seed(seed)
    image_feature_hw = int(cfg["train"]["image_feature_hw"])
    feature_dim = int(cfg["observation"]["state_dim"]) + 3 * image_feature_hw**2
    horizon = int(cfg["horizon_steps"])
    action_dim = int(cfg["action_dim"])
    base_policy = nn.Linear(feature_dim, horizon * action_dim, bias=False)
    base = FrozenDPPOBase(base_policy)
    actor = ResidualActor(
        feature_dim, horizon, action_dim, cfg["train"]["residual_bound"],
        cfg["train"]["initial_log_std"],
    )
    critic = ResidualCritic(feature_dim)
    actor_optimizer, critic_optimizer = create_optimizers(actor, critic, cfg)
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
        "reward_shaping": resolve_reward_shaping(cfg),
        "actor_lr": actor_optimizer.param_groups[0]["lr"],
        "critic_lr": critic_optimizer.param_groups[0]["lr"],
        "base_parameter_hash_before": base.parameter_hash,
        "zero_residual_action_equal": True,
        "iterations": iterations,
        "path_usage": path_usage(cfg),
        "episode_manifest": str(output / "episode_manifest.jsonl"),
    }
    manifest = build_episode_manifest(cfg["train"]["seeds"], int(cfg["evaluation"]["episodes"]), int(cfg["evaluation"]["random_seed_base"]))
    write_episode_manifest(output / "episode_manifest.jsonl", manifest)
    for iteration in range(iterations):
        obs = env.reset()
        feat = pixel_state_features(obs, image_feature_hw)
        base_action = base({"state": obs["state"], "rgb": obs["rgb"]}).view(-1, horizon, action_dim)
        residual, _log_prob, _latent = actor.sample(feat)
        action = compose_bounded_residual(base_action, residual)
        zero_action = compose_bounded_residual(base_action, torch.zeros_like(residual))
        metrics["zero_residual_action_equal"] &= bool(torch.equal(zero_action, base_action.clamp(-1.0, 1.0)))
        _, reward, _, _ = env.step(action[:, :1])
        value = critic(feat)
        loss = (value - torch.tensor([reward])).square().mean() + residual.square().mean() * 1e-3
        if phase == "train":
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(cfg["train"]["grad_clip"]))
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(cfg["train"]["grad_clip"]))
            actor_optimizer.step()
            critic_optimizer.step()
        completed = iteration + 1
        _log(log_path, f"STATUS completed={completed} total={total} loss={loss.item():.6f}")
        with (output / "progress.jsonl").open("a") as handle:
            handle.write(json.dumps({"completed": completed, "total": total, "loss": loss.item()}) + "\n")
    base.assert_unchanged()
    if hasattr(env, "close"):
        env.close()
    metrics["base_parameter_hash_after"] = base.parameter_hash
    metrics["base_hash_unchanged"] = metrics["base_parameter_hash_before"] == metrics["base_parameter_hash_after"]
    if phase == "train":
        torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "metrics": metrics}, output / "checkpoint.pt")
    _write_json(output / "metrics.json", metrics)
    _log(log_path, f"DONE completed={completed} total={total}")


def summarize(root: Path, cfg: Mapping[str, Any]) -> None:
    base_manifest = root / "base" / "episode_manifest.jsonl"
    base_metrics_path = root / "base" / "metrics.json"
    if not base_manifest.exists() or not base_metrics_path.exists():
        raise FileNotFoundError("summary requires base/metrics.json and base/episode_manifest.jsonl")
    base_metrics = json.loads(base_metrics_path.read_text())
    expected_seeds = [int(row["seed"]) for row in base_metrics["per_seed"]]

    def load_group(name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for seed in expected_seeds:
            run_root = root / name / f"seed_{seed}"
            metrics_path = run_root / "metrics.json"
            manifest_path = run_root / "episode_manifest.jsonl"
            if not metrics_path.exists() or not manifest_path.exists():
                raise FileNotFoundError(f"summary requires {metrics_path} and {manifest_path}")
            assert_paired_manifests(base_manifest, manifest_path)
            row = json.loads(metrics_path.read_text())
            if int(row["seed"]) != seed:
                raise ValueError(f"seed mismatch in {metrics_path}")
            rows.append(row)
        return rows

    zero_rows = load_group("zero_eval")
    learned_rows = load_group("residual_eval")
    random_rows = load_group("random_residual")
    train_rows: list[dict[str, Any]] = []
    for seed in expected_seeds:
        run_root = root / "residual" / f"seed_{seed}"
        metrics_path = run_root / "metrics.json"
        checkpoint_path = run_root / "checkpoint.pt"
        if not metrics_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"summary requires {metrics_path} and {checkpoint_path}")
        row = json.loads(metrics_path.read_text())
        if int(row["seed"]) != seed:
            raise ValueError(f"seed mismatch in {metrics_path}")
        train_rows.append(row)
    base_by_seed = {int(row["seed"]): float(row["success_rate"]) for row in base_metrics["per_seed"]}
    base_rng_by_seed = {
        int(row["seed"]): int(row["policy_rng_seed"]) for row in base_metrics["per_seed"]
    }

    def comparison_row(label: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rates = [float(row["success_rate"]) for row in rows]
        return {
            "label": label,
            "success_rate_mean": float(np.mean(rates)),
            "success_rate_by_seed": {str(row["seed"]): float(row["success_rate"]) for row in rows},
            "episodes": int(sum(int(row["episodes"]) for row in rows)),
            "saturation_fraction_mean": float(np.mean([
                float(row.get("saturation_fraction", 0.0)) for row in rows
            ])),
        }

    comparison = [
        {
            "label": "frozen_base",
            "success_rate_mean": float(base_metrics["success_rate"]),
            "success_rate_by_seed": {str(k): v for k, v in base_by_seed.items()},
            "episodes": int(base_metrics["episodes"]),
            "saturation_fraction_mean": 0.0,
        },
        comparison_row("zero_residual", zero_rows),
        comparison_row("learned_bounded_residual", learned_rows),
        comparison_row("random_bounded_residual", random_rows),
    ]
    base_rate = comparison[0]["success_rate_mean"]
    learned_gain = comparison[2]["success_rate_mean"] - base_rate
    random_gain = comparison[3]["success_rate_mean"] - base_rate
    judgement_cfg = cfg["judgement"]
    contract_gates = {
        "manifests_paired": True,
        "base_diffusion_rng_paired": all(
            int(row["policy_rng_seed"]) == base_rng_by_seed[int(row["seed"])]
            for row in [*zero_rows, *learned_rows, *random_rows]
        ),
        "formal_episode_counts_complete": (
            int(base_metrics["episodes"]) == len(expected_seeds) * int(cfg["evaluation"]["episodes"])
            and all(
                int(row["episodes"]) == int(cfg["evaluation"]["episodes"])
                for row in [*zero_rows, *learned_rows, *random_rows]
            )
        ),
        "formal_training_budget_complete": all(
            int(row["iterations"]) == int(cfg["train"]["n_train_itr"])
            for row in train_rows
        ),
        "zero_matches_base": all(
            math.isclose(float(row["success_rate"]), base_by_seed[int(row["seed"])], abs_tol=1e-12)
            for row in zero_rows
        ),
        "base_hash_unchanged": all(
            bool(row.get("base_hash_unchanged"))
            for row in [*train_rows, *zero_rows, *learned_rows, *random_rows]
        ),
        "base_has_headroom": (
            float(judgement_cfg["base_success_min"]) <= base_rate
            <= float(judgement_cfg["base_success_max"])
        ),
    }
    hypothesis_gates = {
        "gain_at_least_threshold": learned_gain >= float(judgement_cfg["residual_gain_min"]),
        "all_seeds_improve": all(
            float(row["success_rate"]) > base_by_seed[int(row["seed"])] for row in learned_rows
        ),
        "random_gain_below_fraction": (
            learned_gain > 0
            and random_gain < learned_gain * float(judgement_cfg["random_gain_fraction_max"])
        ),
        "saturation_below_threshold": (
            comparison[2]["saturation_fraction_mean"]
            < float(judgement_cfg["saturation_fraction_max"])
        ),
    }
    summary: dict[str, Any] = {
        "result_label": "Formal Closed-loop",
        "comparison": comparison,
        "learned_gain": learned_gain,
        "random_gain": random_gain,
        "contract_gates": contract_gates,
        "hypothesis_gates": hypothesis_gates,
        "experiment_valid": all(contract_gates.values()),
        "hypothesis_supported": all(contract_gates.values()) and all(hypothesis_gates.values()),
        "manifest_state_count": len({row["reset_seed"] for row in load_episode_manifest(base_manifest)}),
    }
    _write_json(root / "summary.json", summary)


def _write_base_metrics(
    cfg: Mapping[str, Any], output: Path, result_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write the same base metrics for a partial seed run or the merged run."""

    if not result_rows:
        raise ValueError("base eval produced no seed results")
    anchor = cfg["paths"]["anchor"]
    eval_steps = int(cfg["env"]["max_episode_steps"]) // int(cfg["act_steps"])
    total_episodes = sum(int(row["episodes"]) for row in result_rows)
    success_rate = sum(
        int(row["episodes"]) * float(row["success_rate"]) for row in result_rows
    ) / total_episodes
    avg_best_reward = sum(
        int(row["episodes"]) * float(row["avg_best_reward"]) for row in result_rows
    ) / total_episodes
    elapsed_seconds = sum(float(row["elapsed_seconds"]) for row in result_rows)
    metrics = {
        "result_label": "Formal Closed-loop",
        "phase": "base-eval",
        "official_upstream": True,
        "task_suite": "robomimic",
        "task": cfg["task"],
        "dataset_size": 0,
        "requested_env_id": cfg["env_id"],
        "observation_keys": ["rgb", "state"],
        "action_space": "normalized Box[-1,1]^7",
        "action_dim": cfg["action_dim"],
        "horizon_steps": cfg["horizon_steps"],
        "control_frequency_hz": int(cfg["control_freq"]),
        "episodes": total_episodes,
        "success_rate": success_rate,
        "closed_loop_success_rate": success_rate,
        "failure_count": round(total_episodes * (1.0 - success_rate)),
        "avg_best_reward": avg_best_reward,
        "reward_shaping": False,
        "train_loss": None,
        "open_loop_action_error": None,
        "elapsed_seconds": elapsed_seconds,
        "latency_ms_per_vector_step": 1000.0 * elapsed_seconds / (len(result_rows) * eval_steps),
        "base_parameter_hash": hashlib.sha256(Path(anchor["base_checkpoint"]).read_bytes()).hexdigest(),
        "base_hash_unchanged": True,
        "per_seed": list(result_rows),
        "result_paths": [str(row["result_path"]) for row in result_rows],
        "path_usage": path_usage(cfg),
        "episode_manifest": str(output / "episode_manifest.jsonl"),
    }
    _write_json(output / "metrics.json", metrics)


def _run_official_base_eval(
    cfg: Mapping[str, Any], output: Path, selected_seed: int | None = None,
) -> None:
    """Run all base seeds serially, or one selected seed for multi-GPU orchestration."""

    dppo_root = Path(os.environ.get("DPPO_ROOT", cfg["dppo_root"])).resolve()
    paths = cfg["paths"]
    anchor = paths["anchor"]
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    full_manifest = build_episode_manifest(
        cfg["train"]["seeds"], int(cfg["evaluation"]["episodes"]),
        int(cfg["evaluation"]["random_seed_base"]),
    )
    configured_seeds = [int(value) for value in cfg["train"]["seeds"]]
    if selected_seed is None:
        selected = list(enumerate(configured_seeds))
        manifest = full_manifest
    else:
        selected_index = _manifest_seed_index(cfg, selected_seed)
        selected = [(selected_index, configured_seeds[selected_index])]
        manifest = [
            row for row in full_manifest if int(row["seed_index"]) == selected_index
        ]
    write_episode_manifest(output / "episode_manifest.jsonl", manifest)
    env = os.environ.copy()
    env["DPPO_DATA_DIR"] = str(Path(paths["data_dir"]).resolve())
    env["DPPO_LOG_DIR"] = str(Path(paths["log_dir"]).resolve())
    env["PYTHONNOUSERSITE"] = "1"
    _log(log_path, "START official DPPO frozen-base pixel eval (paired manifest)")
    result_rows: list[dict[str, Any]] = []
    episodes_per_seed = int(cfg["evaluation"]["episodes"])
    eval_steps = int(cfg["env"]["max_episode_steps"]) // int(cfg["act_steps"])
    for completed, (seed_index, policy_seed) in enumerate(selected, start=1):
        reset_seed = int(manifest[0]["reset_seed"]) if len(selected) == 1 else int(
            cfg["evaluation"]["random_seed_base"]
        ) + 1 + seed_index * episodes_per_seed
        seed_logdir = (
            Path(paths["log_dir"]) / "robomimic-eval-paired" / f"seed_{seed_index}"
        )
        command = [
            "python", "script/run.py", "--config-name=eval_diffusion_mlp_img",
            "--config-dir=cfg/robomimic/eval/square",
            f"seed={reset_seed}", f"env.n_envs={episodes_per_seed}",
            f"n_steps={eval_steps}",
            f"logdir={seed_logdir.resolve()}",
            f"base_policy_path={Path(anchor['base_checkpoint']).resolve()}",
            f"normalization_path={Path(anchor['normalization']).resolve()}",
        ]
        _log(log_path, "COMMAND " + " ".join(command))
        with log_path.open("a") as handle:
            process = subprocess.Popen(
                command, cwd=dppo_root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                handle.write(line)
                print(line, end="", flush=True)
            return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"official DPPO eval seed_index={seed_index} failed with status {return_code}"
            )
        result_path = seed_logdir / "result.npz"
        if not result_path.exists():
            raise FileNotFoundError(f"upstream eval completed without {result_path}")
        with np.load(result_path) as result:
            num_episode = int(result["num_episode"])
            if num_episode != episodes_per_seed:
                raise RuntimeError(
                    f"official eval produced {num_episode} episodes; expected {episodes_per_seed}"
                )
            result_rows.append({
                "seed": int(policy_seed),
                "policy_rng_seed": reset_seed,
                "reset_seed_start": reset_seed,
                "episodes": num_episode,
                "success_rate": float(result["eval_success_rate"]),
                "avg_best_reward": float(result["eval_best_reward"]),
                "elapsed_seconds": float(result["time"]),
                "result_path": str(result_path),
            })
        _log(log_path, f"STATUS completed={completed} total={len(selected)}")
    _write_base_metrics(cfg, output, result_rows)
    episodes = sum(int(row["episodes"]) for row in result_rows)
    _log(log_path, f"DONE completed={episodes} total={episodes}")


def _merge_official_base_eval(cfg: Mapping[str, Any], output: Path) -> None:
    """Merge independently evaluated base seeds and validate their manifests."""

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    configured_seeds = [int(value) for value in cfg["train"]["seeds"]]
    full_manifest = build_episode_manifest(
        configured_seeds, int(cfg["evaluation"]["episodes"]),
        int(cfg["evaluation"]["random_seed_base"]),
    )
    result_rows: list[dict[str, Any]] = []
    _log(log_path, "START merge multi-GPU frozen-base eval")
    for completed, seed in enumerate(configured_seeds, start=1):
        seed_index = _manifest_seed_index(cfg, seed)
        run_root = output / f"seed_{seed}"
        metrics_path = run_root / "metrics.json"
        manifest_path = run_root / "episode_manifest.jsonl"
        if not metrics_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"missing base seed output: {run_root}")
        expected_manifest = [
            row for row in full_manifest if int(row["seed_index"]) == seed_index
        ]
        if load_episode_manifest(manifest_path) != expected_manifest:
            raise ValueError(f"base seed manifest mismatch: {manifest_path}")
        metrics = json.loads(metrics_path.read_text())
        rows = metrics.get("per_seed", [])
        if len(rows) != 1 or int(rows[0].get("seed", -1)) != seed:
            raise ValueError(f"invalid base seed metrics: {metrics_path}")
        row = rows[0]
        if int(row.get("policy_rng_seed", -1)) != int(expected_manifest[0]["reset_seed"]):
            raise ValueError(f"base RNG seed mismatch: {metrics_path}")
        if int(row.get("episodes", -1)) != int(cfg["evaluation"]["episodes"]):
            raise ValueError(f"base episode count mismatch: {metrics_path}")
        result_rows.append(row)
        _log(log_path, f"STATUS completed={completed} total={len(configured_seeds)}")
    write_episode_manifest(output / "episode_manifest.jsonl", full_manifest)
    _write_base_metrics(cfg, output, result_rows)
    episodes = sum(int(row["episodes"]) for row in result_rows)
    _log(log_path, f"DONE completed={episodes} total={episodes}")


def _make_real_env(cfg: Mapping[str, Any], dppo_root: Path, n_envs: int, *, anchor: bool = False):
    """Construct the exact DPPO robomimic image+state wrapper stack."""

    import sys

    sys.path.insert(0, str(dppo_root))
    from env.gym_utils import make_async
    from omegaconf import OmegaConf

    shape_meta = {
        "obs": {"rgb": {"shape": [3, 96, 96]}, "state": {"shape": [9]}},
        "action": {"shape": [7]},
    }
    wrappers = {
        "robomimic_image": {
            "normalization_path": str(Path(cfg["paths"]["anchor"]["normalization"]).resolve()),
            "low_dim_keys": list(cfg["observation"]["low_dim_keys"]),
            "image_keys": [cfg["observation"]["image_key"]],
            "shape_meta": shape_meta,
        },
        "multi_step": {
            "n_obs_steps": 1,
            "n_action_steps": int(cfg["act_steps"]),
            "max_episode_steps": int(cfg["env"]["max_episode_steps"]),
            "reset_within_step": True,
        },
    }
    return make_async(
        id="square", num_envs=n_envs, asynchronous=True, wrappers=OmegaConf.create(wrappers),
        robomimic_env_cfg_path=str(dppo_root / "cfg/robomimic/env_meta/square-img.json"),
        use_image_obs=True, render_offscreen=True,
        reward_shaping=resolve_reward_shaping(cfg, anchor=anchor), shape_meta=shape_meta,
        max_episode_steps=int(cfg["env"]["max_episode_steps"]), obs_steps=1,
        act_steps=int(cfg["act_steps"]), action_dim=int(cfg["action_dim"]),
    )


def _load_real_dppo_base(cfg: Mapping[str, Any], dppo_root: Path, device: str):
    """Hydra-compose the upstream eval model without invoking its downloader."""

    import hydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with hydra.initialize_config_dir(version_base=None, config_dir=str(dppo_root / "cfg/robomimic/eval/square")):
        dppo_cfg = hydra.compose(
            config_name="eval_diffusion_mlp_img",
            overrides=[
                f"device={device}",
                f"base_policy_path={Path(cfg['paths']['anchor']['base_checkpoint']).resolve()}",
                f"normalization_path={Path(cfg['paths']['anchor']['normalization']).resolve()}",
            ],
        )
    OmegaConf.resolve(dppo_cfg)
    model = instantiate(dppo_cfg.model)
    model.eval()
    return model


def _numpy_obs_to_torch(obs: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in obs.items()}


def _run_real_residual(cfg: Mapping[str, Any], output: Path, phase: str, residual_mode: str,
                       max_iterations: int | None, time_stages: bool, seed: int,
                       residual_root: Path | None = None,
                       manifest_path: Path | None = None) -> None:
    """Run a frozen DPPO base plus a tanh-bounded Gaussian residual with PPO."""

    if phase == "train" and residual_mode != "learned":
        raise ValueError("training only supports residual_mode=learned")
    dppo_root = Path(os.environ.get("DPPO_ROOT", cfg["dppo_root"])).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_envs = int(cfg["env"]["n_envs"])
    iterations = int(max_iterations or cfg["train"]["n_train_itr"])
    image_feature_hw = int(cfg["train"]["image_feature_hw"])
    feature_dim = int(cfg["observation"]["state_dim"]) + 3 * image_feature_hw**2
    horizon, action_dim = int(cfg["horizon_steps"]), int(cfg["action_dim"])
    bound = torch.as_tensor(cfg["train"]["residual_bound"], dtype=torch.float32, device=device)
    if bound.numel() != action_dim:
        raise ValueError("train.residual_bound must have one normalized bound per action dimension")

    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "run.log"
    _write_json(output / "run_config.json", cfg)

    reset_seeds: list[int] | None = None
    if phase == "eval":
        if manifest_path is None or not manifest_path.exists():
            raise FileNotFoundError(
                "eval requires --manifest pointing to base/episode_manifest.jsonl"
            )
        manifest = load_episode_manifest(manifest_path)
        write_episode_manifest(output / "episode_manifest.jsonl", manifest)
        selected_index = _manifest_seed_index(cfg, seed)
        reset_seeds = [
            int(row["reset_seed"])
            for row in manifest
            if int(row["seed_index"]) == selected_index
        ]
        if len(reset_seeds) != n_envs:
            raise ValueError(f"evaluation manifest has {len(reset_seeds)} states; expected {n_envs}")

    # Upstream EvalAgent uses the first environment reset seed for Python,
    # NumPy, Torch, and the contiguous vector-env reset schedule. Reproduce
    # that coupling so Frozen Base and Zero Residual consume identical initial
    # diffusion noise on the paired manifest.
    policy_rng_seed = reset_seeds[0] if reset_seeds is not None else seed
    random.seed(policy_rng_seed)
    np.random.seed(policy_rng_seed)
    torch.manual_seed(policy_rng_seed)
    base = FrozenDPPOBase(_load_real_dppo_base(cfg, dppo_root, str(device)))
    base_cpu_rng_state = torch.get_rng_state()
    base_cuda_rng_states = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    actor = ResidualActor(
        feature_dim, horizon, action_dim, bound, cfg["train"]["initial_log_std"],
    ).to(device)
    critic = ResidualCritic(feature_dim).to(device)
    actor_optimizer, critic_optimizer = create_optimizers(actor, critic, cfg)
    residual_root = residual_root or output
    checkpoint = residual_root / "checkpoint.pt"
    start_iteration = 0
    if phase == "eval" and residual_mode == "learned":
        if not checkpoint.exists():
            raise FileNotFoundError(f"learned residual checkpoint is missing: {checkpoint}")
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
    elif phase == "train" and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        start_iteration = int(state["iteration"]) + 1
        _log(log_path, f"RESUME iteration={start_iteration} checkpoint={checkpoint}")
        if start_iteration >= iterations and (output / "metrics.json").exists():
            _log(log_path, f"ALREADY_DONE iterations={iterations}")
            return
    actor.eval() if phase == "eval" else actor.train()
    critic.eval() if phase == "eval" else critic.train()

    env = _make_real_env(cfg, dppo_root, n_envs)
    if hasattr(env, "seed"):
        env.seed(reset_seeds if reset_seeds is not None else [seed + i for i in range(n_envs)])
    obs = env.reset()
    if phase == "eval":
        torch.set_rng_state(base_cpu_rng_state)
        if base_cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(base_cuda_rng_states)
    residual_generator = torch.Generator(device=device)
    residual_generator.manual_seed(policy_rng_seed + 10_000_000)
    train_steps = int(cfg["train"]["n_steps"])
    eval_steps = int(cfg["env"]["max_episode_steps"]) // int(cfg["act_steps"])
    steps_per_iteration = eval_steps if phase == "eval" else train_steps
    total_work = steps_per_iteration if phase == "eval" else iterations * steps_per_iteration
    metrics: dict[str, Any] = {
        "result_label": "Formal Closed-loop" if phase == "eval" else "Training",
        "phase": phase,
        "residual_mode": residual_mode,
        "seed": seed,
        "policy_rng_seed": policy_rng_seed,
        "official_upstream": False,
        "device": str(device),
        "task_suite": "robomimic",
        "task": cfg["task"],
        "dataset_size": 0,
        "n_envs": n_envs,
        "iterations": iterations,
        "resumed_from_iteration": start_iteration,
        "steps_per_iteration": steps_per_iteration,
        "observation_keys": ["rgb", "state"],
        "action_space": "normalized Box[-1,1]^7",
        "action_dim": action_dim,
        "horizon_steps": horizon,
        "control_frequency_hz": int(cfg["control_freq"]),
        "reward_shaping": resolve_reward_shaping(cfg),
        "residual_bound_normalized": [float(value) for value in bound.cpu().tolist()],
        "actor_lr": actor_optimizer.param_groups[0]["lr"],
        "critic_lr": critic_optimizer.param_groups[0]["lr"],
        "base_parameter_hash_before": base.parameter_hash,
        "path_usage": path_usage(cfg),
        "episode_manifest": str(output / "episode_manifest.jsonl") if phase == "eval" else None,
        "open_loop_action_error": None,
        "timing_probe": bool(time_stages),
    }
    eval_rewards: list[np.ndarray] = []
    stage_seconds = {"base_inference": 0.0, "residual_inference": 0.0, "env_step": 0.0, "update": 0.0}
    saturated, residual_values = 0, 0
    zero_residual_action_equal = True
    last_losses: dict[str, float] = {}
    _log(
        log_path,
        f"START real residual phase={phase} mode={residual_mode} seed={seed} "
        f"n_envs={n_envs} device={device}",
    )
    started = time.perf_counter()
    try:
        for iteration in range(start_iteration, iterations if phase == "train" else 1):
            features_buf: list[torch.Tensor] = []
            latent_buf: list[torch.Tensor] = []
            logprob_buf: list[torch.Tensor] = []
            reward_buf: list[torch.Tensor] = []
            done_buf: list[torch.Tensor] = []
            value_buf: list[torch.Tensor] = []
            for step in range(steps_per_iteration):
                cond = _numpy_obs_to_torch(obs, device)
                features = pixel_state_features(cond, image_feature_hw)
                stage_start = time.perf_counter()
                with torch.no_grad():
                    base_action = base(cond).view(-1, horizon, action_dim)
                stage_seconds["base_inference"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter()
                with torch.no_grad():
                    if residual_mode == "zero":
                        residual = torch.zeros_like(base_action)
                        logprob = torch.zeros(n_envs, device=device)
                        latent = torch.zeros_like(base_action)
                    elif residual_mode == "random-bounded":
                        residual = torch.empty_like(base_action).uniform_(
                            -1.0, 1.0, generator=residual_generator
                        ) * bound.view(1, 1, -1)
                        logprob = torch.zeros(n_envs, device=device)
                        latent = torch.zeros_like(base_action)
                    else:
                        residual, logprob, latent = actor.sample(
                            features, deterministic=(phase == "eval")
                        )
                    values = critic(features) if phase == "train" else None
                stage_seconds["residual_inference"] += time.perf_counter() - stage_start
                saturated += int((residual.abs() >= 0.95 * bound.view(1, 1, -1)).sum().item())
                residual_values += residual.numel()
                action = compose_bounded_residual(base_action, residual)
                if residual_mode == "zero":
                    zero_residual_action_equal &= bool(torch.equal(action, base_action.clamp(-1.0, 1.0)))

                stage_start = time.perf_counter()
                obs, reward, terminated, truncated, _ = env.step(
                    action[:, : int(cfg["act_steps"])].detach().cpu().numpy()
                )
                stage_seconds["env_step"] += time.perf_counter() - stage_start
                done = np.asarray(terminated) | np.asarray(truncated)
                if phase == "eval":
                    eval_rewards.append(np.asarray(reward, dtype=np.float32).reshape(-1))
                else:
                    features_buf.append(features.detach().cpu())
                    latent_buf.append(latent.detach().cpu())
                    logprob_buf.append(logprob.detach().cpu())
                    reward_buf.append(torch.as_tensor(reward, dtype=torch.float32).cpu())
                    done_buf.append(torch.as_tensor(done, dtype=torch.float32).cpu())
                    assert values is not None
                    value_buf.append(values.detach().cpu())
                if step % 10 == 0:
                    completed = iteration * steps_per_iteration + step
                    _log(log_path, f"STATUS completed={completed} total={total_work}")

            if phase == "train":
                update_started = time.perf_counter()
                with torch.no_grad():
                    next_cond = _numpy_obs_to_torch(obs, device)
                    next_features = pixel_state_features(next_cond, image_feature_hw)
                    next_value = critic(next_features).detach().cpu()
                features_t = torch.stack(features_buf)
                latents_t = torch.stack(latent_buf)
                old_logprobs_t = torch.stack(logprob_buf)
                rewards_t = torch.stack(reward_buf)
                dones_t = torch.stack(done_buf)
                old_values_t = torch.stack(value_buf)
                advantages_t = torch.zeros_like(rewards_t)
                last_gae = torch.zeros(n_envs)
                gamma = float(cfg["train"]["gamma"])
                gae_lambda = float(cfg["train"]["gae_lambda"])
                for step in reversed(range(steps_per_iteration)):
                    next_values = next_value if step == steps_per_iteration - 1 else old_values_t[step + 1]
                    nonterminal = 1.0 - dones_t[step]
                    delta = rewards_t[step] + gamma * next_values * nonterminal - old_values_t[step]
                    last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
                    advantages_t[step] = last_gae
                returns_t = advantages_t + old_values_t
                flat_features = features_t.flatten(0, 1)
                flat_latents = latents_t.flatten(0, 1)
                flat_old_logprobs = old_logprobs_t.flatten()
                flat_old_values = old_values_t.flatten()
                flat_returns = returns_t.flatten()
                flat_advantages = advantages_t.flatten()
                flat_advantages = (
                    (flat_advantages - flat_advantages.mean())
                    / (flat_advantages.std() + 1e-8)
                )
                batch_size = int(cfg["train"]["batch_size"])
                clip_coef = float(cfg["train"]["clip_coef"])
                value_clip_coef = float(cfg["train"]["value_clip_coef"])
                approx_kl = 0.0
                stop_update = False
                for _epoch in range(int(cfg["train"]["update_epochs"])):
                    permutation = torch.randperm(len(flat_features))
                    for start in range(0, len(permutation), batch_size):
                        indices = permutation[start : start + batch_size]
                        features_b = flat_features[indices].to(device)
                        latents_b = flat_latents[indices].to(device)
                        old_logprobs_b = flat_old_logprobs[indices].to(device)
                        advantages_b = flat_advantages[indices].to(device)
                        returns_b = flat_returns[indices].to(device)
                        old_values_b = flat_old_values[indices].to(device)

                        new_logprobs, entropy = actor.log_prob(features_b, latents_b)
                        log_ratio = new_logprobs - old_logprobs_b
                        ratio = log_ratio.exp()
                        policy_loss = torch.maximum(
                            -advantages_b * ratio,
                            -advantages_b * ratio.clamp(1.0 - clip_coef, 1.0 + clip_coef),
                        ).mean()
                        actor_loss = policy_loss - float(cfg["train"]["ent_coef"]) * entropy.mean()
                        if iteration >= int(cfg["train"]["critic_warmup_iterations"]):
                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                actor.parameters(), float(cfg["train"]["grad_clip"])
                            )
                            actor_optimizer.step()

                        new_values = critic(features_b)
                        value_unclipped = (new_values - returns_b).square()
                        value_clipped = old_values_b + (new_values - old_values_b).clamp(
                            -value_clip_coef, value_clip_coef
                        )
                        value_loss = 0.5 * torch.maximum(
                            value_unclipped, (value_clipped - returns_b).square()
                        ).mean()
                        critic_optimizer.zero_grad()
                        (float(cfg["train"]["vf_coef"]) * value_loss).backward()
                        torch.nn.utils.clip_grad_norm_(
                            critic.parameters(), float(cfg["train"]["grad_clip"])
                        )
                        critic_optimizer.step()
                        with torch.no_grad():
                            approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                        last_losses = {
                            "policy_loss": float(policy_loss.detach().item()),
                            "value_loss": float(value_loss.detach().item()),
                            "entropy": float(entropy.detach().mean().item()),
                            "approx_kl": approx_kl,
                        }
                        if (
                            iteration >= int(cfg["train"]["critic_warmup_iterations"])
                            and approx_kl > float(cfg["train"]["target_kl"])
                        ):
                            stop_update = True
                            break
                    if stop_update:
                        break
                stage_seconds["update"] += time.perf_counter() - update_started
                checkpoint_payload = {
                    "iteration": iteration,
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                    "seed": seed,
                }
                checkpoint_tmp = checkpoint.with_suffix(".tmp")
                torch.save(checkpoint_payload, checkpoint_tmp)
                checkpoint_tmp.replace(checkpoint)
                progress = {
                    "completed": (iteration + 1) * steps_per_iteration,
                    "total": total_work,
                    "iteration": iteration + 1,
                    **last_losses,
                }
                with (output / "progress.jsonl").open("a") as handle:
                    handle.write(json.dumps(progress, sort_keys=True) + "\n")
                _log(
                    log_path,
                    f"STATUS completed={progress['completed']} total={total_work} "
                    f"policy_loss={last_losses.get('policy_loss', float('nan')):.6f} "
                    f"value_loss={last_losses.get('value_loss', float('nan')):.6f}",
                )
    finally:
        if hasattr(env, "close"):
            env.close()

    base.assert_unchanged()
    metrics["base_parameter_hash_after"] = base.parameter_hash
    metrics["base_hash_unchanged"] = metrics["base_parameter_hash_before"] == metrics["base_parameter_hash_after"]
    metrics["zero_residual_action_equal"] = zero_residual_action_equal if residual_mode == "zero" else None
    metrics["elapsed_seconds"] = time.perf_counter() - started
    metrics["stage_seconds"] = stage_seconds
    decisions = max(1, (1 if phase == "eval" else max(0, iterations - start_iteration)) * steps_per_iteration)
    metrics["latency_ms_per_vector_step"] = 1000.0 * (
        stage_seconds["base_inference"] + stage_seconds["residual_inference"] + stage_seconds["env_step"]
    ) / decisions
    metrics["saturation_fraction"] = saturated / max(1, residual_values)
    metrics["train_loss"] = last_losses or None
    if eval_rewards:
        reward_matrix = np.stack(eval_rewards, axis=0)
        best = reward_matrix.max(axis=0) / int(cfg["act_steps"])
        success = best >= float(cfg["evaluation"]["success_threshold"])
        metrics["episodes"] = int(best.size)
        metrics["success_rate"] = float(np.mean(success))
        metrics["closed_loop_success_rate"] = metrics["success_rate"]
        metrics["avg_best_reward"] = float(np.mean(best))
        metrics["failure_count"] = int((~success).sum())
    _write_json(output / "metrics.json", metrics)
    completed = total_work
    _log(log_path, f"DONE completed={completed} total={total_work}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", required=True,
        choices=["preflight", "base-eval", "base-eval-merge", "train", "eval", "summarize"],
    )
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/rl/residual_robomimic_dppo_square.yaml"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--time-stages", action="store_true")
    parser.add_argument("--residual-mode", choices=["learned", "zero", "random-bounded"], default="learned")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--residual-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="paired episode_manifest.jsonl used by residual eval")
    args = parser.parse_args(argv)
    cfg = _load_yaml(args.config)
    _check_contract(cfg)
    _check_path_groups(cfg)
    if args.phase == "summarize":
        summarize(args.output_root, cfg)
        return 0
    if args.phase == "preflight":
        assets = {
            "anchor_normalization": cfg["paths"]["anchor"]["normalization"],
            "anchor_base_checkpoint": cfg["paths"]["anchor"]["base_checkpoint"],
        }
        missing, asset_errors = _validate_anchor_assets(cfg)
        output = args.output_root / "gates.json"
        _write_json(output, {
            "contract_ok": not asset_errors,
            "missing_assets": missing,
            "asset_errors": asset_errors,
            "assets": assets,
            "dummy": args.dummy,
        })
        if (missing or asset_errors) and not args.dummy:
            details = [*missing, *asset_errors]
            print("Invalid assets (download commands are in the guide): " + "; ".join(details))
            return 3
        return 0
    if args.phase == "base-eval" and not args.dummy:
        paths = cfg["paths"]
        missing = [key for key in ("normalization", "base_checkpoint")
                   if not Path(paths["anchor"][key]).exists()]
        if missing:
            raise FileNotFoundError("missing assets: " + ", ".join(missing))
        _run_official_base_eval(cfg, args.output_root, args.seed)
        return 0
    if args.phase == "base-eval-merge" and not args.dummy:
        missing, asset_errors = _validate_anchor_assets(cfg)
        if missing or asset_errors:
            raise RuntimeError("invalid base assets: " + "; ".join([*missing, *asset_errors]))
        _merge_official_base_eval(cfg, args.output_root)
        return 0
    if not args.dummy and args.phase in {"train", "eval"}:
        paths = cfg["paths"]
        missing = [key for key in ("normalization", "base_checkpoint")
                   if not Path(paths["anchor"][key]).exists()]
        if missing:
            raise FileNotFoundError("missing assets: " + ", ".join(missing))
        seed = args.seed if args.seed is not None else int(cfg["train"]["seeds"][0])
        random.seed(seed)
        torch.manual_seed(seed)
        _run_real_residual(cfg, args.output_root, args.phase, args.residual_mode, args.max_iterations, args.time_stages, seed, args.residual_root, args.manifest)
        return 0
    phase = "base-eval" if args.phase == "base-eval" else args.phase
    _run_dummy(phase, cfg, args.output_root, args.smoke, args.max_iterations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
