"""Square dual-view protocol and disk-backed, temporally aligned training windows."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROTOCOL = {"action_horizon": 32, "execute_steps": 4, "video_frames": 9,
            "video_stride": 4, "control_hz": 20, "image_size": [224, 448],
            "cameras": ["agentview", "robot0_eye_in_hand"],
            "layout": "agentview left, wrist right", "native_camera_size": [224, 224]}


def window_starts(lengths):
    starts, offset = [], 0
    for length in lengths:
        starts.extend(range(offset, offset + max(0, int(length) - 32)))
        offset += int(length)
    return np.asarray(starts, dtype=np.int64)


def executed_actions(prediction):
    if tuple(prediction.shape[-2:]) != (32, 7):
        raise ValueError("Dual-view policy must predict 32x7 actions")
    return prediction[..., :4, :]


def render_dual_rgb(env):
    from robosuite import macros
    from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING

    sim = env.env.sim
    if any(c not in sim.model.camera_names for c in PROTOCOL["cameras"]):
        raise ValueError("Required native agentview or robot0_eye_in_hand camera missing")
    flip = -IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    views = [sim.render(width=224, height=224, camera_name=c, depth=False)[::flip].copy()
             for c in PROTOCOL["cameras"]]
    return np.concatenate(views, axis=1).transpose(2, 0, 1).copy()


class DualViewDataset(Dataset):
    def __init__(self, root, context_path=None):
        self.root = Path(root)
        manifest = json.loads((self.root / "manifest.json").read_text())
        if manifest["status"] != "complete" or manifest["protocol"] != PROTOCOL:
            raise ValueError("Incomplete or incompatible dual-view dataset")
        self._arrays = None
        self.traj_lengths = np.load(self.root / "traj_lengths.npy")
        self.windows = window_starts(self.traj_lengths)
        self.context = self.context_mask = None
        if context_path is not None:
            from robot_practice.experiments.rl.lightwam_base import load_context_cache

            self.context, self.context_mask = load_context_cache(context_path)
        arrays = self.arrays
        n = int(sum(self.traj_lengths))
        if (arrays["images"].shape != (n, 3, 224, 448) or arrays["images"].dtype != np.uint8
                or arrays["states"].shape != (n, 9) or arrays["actions"].shape != (n, 7)):
            raise ValueError("Invalid dual-view array shapes or image dtype")
        if not len(self.windows):
            raise ValueError("No complete 32-action/33-state windows")

    @property
    def arrays(self):
        if self._arrays is None:
            self._arrays = {k: np.load(self.root / f"{k}.npy", mmap_mode="r")
                            for k in ("images", "states", "actions")}
        return self._arrays

    @property
    def states(self):
        return self.arrays["states"]

    @property
    def actions(self):
        return self.arrays["actions"]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None  # spawn workers reopen mmap, never pickle the image corpus
        return state

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        start = int(self.windows[index])
        frames = start + np.arange(9) * 4
        video = torch.from_numpy(self.arrays["images"][frames].copy()).float() / 127.5 - 1
        sample = {"video": video.permute(1, 0, 2, 3).contiguous(),
                  "proprio": torch.from_numpy(self.states[frames].copy()).float(),
                  "action": torch.from_numpy(self.actions[start:start + 32].copy()).float(),
                  "prompt": "insert the square nut into the square peg",
                  "image_is_pad": torch.zeros(9, dtype=torch.bool),
                  "action_is_pad": torch.zeros(32, dtype=torch.bool)}
        if self.context is not None:
            sample.update(context=self.context, context_mask=self.context_mask)
        return sample
