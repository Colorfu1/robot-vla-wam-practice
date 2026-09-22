import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robot_practice.experiments.rl.fastwam_rollout_video import (
    VideoWriter,
    case_name,
    ffmpeg_binary,
    locate_batch,
    panel,
    read_observation,
    run_case,
    select_cases,
    write_index,
)


def test_selection_is_outcome_only_and_sorted():
    data = {"status": "complete", "evaluated": 80, "arguments": {"policy": "fastwam"},
            "per_episode": [{"environment_seed": s, "success_600": s % 4 == 0}
                            for s in range(10080, 10000, -1)]}
    chosen = select_cases(data)
    assert [r["environment_seed"] for r in chosen] == [10004, 10008, 10012, 10001, 10002, 10003]
    assert case_name(chosen[0]) == "success_10004"
    assert case_name(chosen[-1]) == "failure_10003"
    with pytest.raises(ValueError):
        select_cases({**data, "status": "running"})


def test_recording_requires_all_150_inputs(tmp_path):
    directory = tmp_path / "gpu_0/batch_000/forensics"
    directory.mkdir(parents=True)
    path = directory / "chunk_0000_input.npz"
    np.savez(path, environment_seeds=[10001, 10002],
             rgb=np.zeros((2, 1, 3, 96, 96)), state=np.zeros((2, 1, 9)),
             action_raw=np.ones((2, 4, 7)))
    obs, action = read_observation(directory, 0, 10002)
    assert obs["rgb"].shape == (3, 96, 96) and obs["state"].shape == (9,)
    assert action.shape == (4, 7)
    with pytest.raises(FileNotFoundError):
        locate_batch(tmp_path, 10002)
    for chunk in range(1, 150):
        (directory / f"chunk_{chunk:04d}_input.npz").touch()
    assert locate_batch(tmp_path, 10002) == directory
    with pytest.raises(ValueError):
        read_observation(directory, 0, 10003)


def test_real_mp4_writer(tmp_path):
    binary = ffmpeg_binary()
    frame = Image.new("RGB", (224, 224), "red")
    view = panel([frame, frame, None], ["Current", "Pred +4", "Actual +4"], "Test")
    assert view.size == (672, 272)
    output = tmp_path / "test.mp4"
    writer = VideoWriter(binary, output, view.size, 5)
    for _ in range(3):
        writer.append(view)
    writer.close()
    assert output.stat().st_size > 0
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(output),
    ], capture_output=True, text=True, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (672, 272)
    assert stream["nb_frames"] == "3" and stream["r_frame_rate"] == "5/1"


def test_full_150_chunk_contract_without_gpu(monkeypatch, tmp_path):
    import torch
    from torch import nn

    from robot_practice.experiments.rl import fastwam_rollout_video as module
    from robot_practice.experiments.rl import fastwam_square_eval as loader
    from robot_practice.experiments.rl.lightwam_base import LightWAMBaseAdapter

    frame = Image.new("RGB", (224, 224), "black")
    calls = []

    class Model(nn.Module):
        device = "cpu"

        def infer_joint(self, **kwargs):
            assert kwargs["action"] is None and kwargs["num_video_frames"] == 5
            assert kwargs["seed"] == 10004 and kwargs["num_inference_steps"] == 20
            calls.append(kwargs)
            return {"video": [frame] * 5, "action": torch.zeros(4, 7)}

    model = Model()
    adapter = LightWAMBaseAdapter(model, context=torch.zeros(2, 4096),
                                  context_mask=torch.ones(2))
    monkeypatch.setattr(loader, "load_fastwam", lambda *a: (model, adapter, {}))
    monkeypatch.setattr(module, "ffmpeg_binary", lambda: "unused")
    monkeypatch.setattr(module, "locate_batch", lambda *a: tmp_path)
    reads = []

    def observation(directory, chunk, seed):
        assert 0 <= chunk < 150
        reads.append(chunk)
        return {"rgb": np.zeros((3, 96, 96)), "state": np.zeros(9)}, np.zeros((4, 7))

    monkeypatch.setattr(module, "read_observation", observation)
    monkeypatch.setattr(module.np, "savez_compressed", lambda *a, **kw: None)
    videos = {}

    class Writer:
        def __init__(self, binary, path, size, fps):
            self.count = 0
            self.size = size
            self.closed = False
            videos[path.name] = self

        def append(self, image):
            assert image.size == self.size
            self.count += 1

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "VideoWriter", Writer)
    cfg = tmp_path / "config.yaml"
    cfg.write_text("{}")
    out = tmp_path / "case"
    row = {"environment_seed": 10004, "success_600": True}
    run_case(SimpleNamespace(output=str(out), eval_root="unused", checkpoint="unused",
                             config=str(cfg)), row, "hash")
    data = json.loads((out / "metrics.json").read_text())
    assert len(calls) == 150 and data["status"] == "complete"
    assert data["endpoint_summary"]["scored_chunks"] == 149
    assert data["chunks"][-1]["prediction_endpoint_mae"] is None
    assert max(reads) == 149
    assert [videos[k].count for k in (
        "actual_5hz.mp4", "predicted_20hz.mp4", "comparison_5hz.mp4", "forecast_strip_5hz.mp4"
    )] == [150, 600, 150, 150]
    assert all(v.closed for v in videos.values())
    write_index(tmp_path, [row])
    assert "success_10004/comparison_5hz.mp4" in (tmp_path / "index.html").read_text()
