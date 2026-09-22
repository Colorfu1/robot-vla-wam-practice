"""Per-control-step scene evidence; reconstruction is gated by full prefix replay."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np


class SceneCapture:
    def __init__(self, env, directory, obs, diagnostic_views=False):
        import mujoco

        self.env = env
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mj = mujoco
        self.sim = env.env.sim
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.states, self.rgb, self.proprio, self.integration = [], [], [], []
        self.actions, self.rewards, self.successes = [], [], []
        self.views = None
        if diagnostic_views:
            from robot_practice.experiments.rl.scene_views import SceneViews

            self.views = SceneViews(env, self.directory)
        # Actual runtime XML, not merely the source XML before path processing.
        (self.directory / "runtime_model.xml").write_text(self.sim.model.get_xml())
        self.append(obs)

    def append(self, obs):
        m, d = self.sim.model._model, self.sim.data._data
        state = np.empty(self.mj.mj_stateSize(m, self.spec), dtype=np.float64)
        self.mj.mj_getState(m, d, state, self.spec)
        self.integration.append(state)
        self.states.append(self.env.get_state()["states"].copy())
        self.rgb.append(np.asarray(obs["rgb"]).copy())
        self.proprio.append(np.asarray(obs["state"]).copy())
        if self.views is not None:
            self.views.append(obs)

    def step(self, action, obs, reward, success):
        self.actions.append(np.asarray(action).copy())
        self.rewards.append(reward)
        self.successes.append(success)
        self.append(obs)

    def finish(self):
        np.savez_compressed(
            self.directory / "control_trace.npz", sim_states=self.states,
            integration_states=self.integration, rgb=self.rgb, proprio=self.proprio,
            actions=self.actions, rewards=self.rewards, successes=self.successes,
        )
        metadata = {
            "state_spec": int(self.spec), "mujoco_version": self.mj.__version__,
            "package_versions": {p: version(p) for p in ("robosuite", "robomimic", "numpy")},
            "control_steps": len(self.actions), "observations": len(self.rgb),
            "indexing": "obs[t] precedes actions[t]; obs[t+1] follows it",
            "reconstruction": "source XML + initial state + executed action prefix",
            "direct_snapshot_resume_verified": False,
            "limitation": "Integration state excludes Python controller and policy state. "
            "Do not use integration_states alone as an exact branch checkpoint. "
            "XML mesh/texture dependencies require the matching simulator installation; "
            "this is not a standalone simulator asset bundle.",
        }
        if self.views is not None:
            metadata["diagnostic_views"] = self.views.finish()
        (self.directory / "capture.json").write_text(json.dumps(metadata, indent=2))
        return metadata


def compare_traces(reference, replay, tolerance=1e-8):
    """No success-only gate: compare every recorded state and observation."""
    errors = {}
    for key in ("sim_states", "integration_states", "proprio", "rgb", "rewards", "successes"):
        a, b = np.asarray(reference[key]), np.asarray(replay[key])
        if a.shape != b.shape:
            return {"passed": False, "shape_mismatch": key}
        difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
        errors[key] = float(difference.max(initial=0))
    return {"passed": all(np.isfinite(x) and x <= tolerance for x in errors.values()),
            "max_abs_errors": errors, "tolerance": tolerance,
            "scope": "same-runtime initial reset and full executed-action-prefix replay; "
            "not direct mid-episode snapshot restore"}
