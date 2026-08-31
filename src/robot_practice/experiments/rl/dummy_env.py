"""Deterministic vector environment used only by the residual smoke path."""

from __future__ import annotations

import numpy as np


class DummyRobotWinEnv:
    """A tiny qpos-target environment with the same 14D action shape as S2."""

    def __init__(self, num_envs: int, *, obs_dim: int = 32, episode_length: int = 64) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if obs_dim < 14:
            raise ValueError("obs_dim must leave room for the 14D proprioception prefix")
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.episode_length = episode_length
        self._rng = np.random.default_rng(0)
        self._qpos = np.zeros((num_envs, 14), dtype=np.float32)
        self._target = np.zeros((num_envs, 14), dtype=np.float32)
        self._steps = np.zeros(num_envs, dtype=np.int64)

    def reset(self, seed: int = 0) -> np.ndarray:
        self._rng = np.random.default_rng(seed)
        self._qpos = self._rng.normal(0.0, 0.05, size=(self.num_envs, 14)).astype(np.float32)
        self._target = np.zeros((self.num_envs, 14), dtype=np.float32)
        self._steps.fill(0)
        return self.observation()

    def observation(self) -> np.ndarray:
        obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        obs[:, :14] = self._qpos
        obs[:, 14:28] = self._target
        obs[:, 28] = self._steps.astype(np.float32) / max(1, self.episode_length)
        return obs

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (self.num_envs, 14):
            raise ValueError(f"expected action shape {(self.num_envs, 14)}, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("dummy action is not finite")
        self._qpos += 0.2 * np.clip(values - self._qpos, -1.0, 1.0)
        self._steps += 1
        distance = np.linalg.norm(self._qpos - self._target, axis=1)
        reward = (-distance).astype(np.float32)
        success = distance < 0.01
        truncated = self._steps >= self.episode_length
        return self.observation(), reward, success, truncated

    @property
    def qpos(self) -> np.ndarray:
        return self._qpos.copy()

