from types import SimpleNamespace

import numpy as np

from robot_practice.experiments.rl.readpixels_probe import input_metadata


def test_offsamples_override_precedes_buffer_creation(tmp_path):
    from robot_practice.experiments.rl.readpixels_probe import install_offsamples_override

    observed = []

    class Context:
        def __init__(self):
            self.model = SimpleNamespace(vis=SimpleNamespace(quality=SimpleNamespace(offsamples=4)))

        def _set_mujoco_context_and_buffers(self):
            observed.append(self.model.vis.quality.offsamples)
            self.con = SimpleNamespace(offSamples=self.model.vis.quality.offsamples)

    install_offsamples_override(Context, 0, tmp_path, 0)
    context = Context()
    context._set_mujoco_context_and_buffers()
    context._set_mujoco_context_and_buffers()
    assert observed == [0, 0]


def test_cuda_probe_does_not_load_driver(monkeypatch):
    from robot_practice.experiments.rl import readpixels_probe as probe

    def unavailable(name, mode):
        assert name == "libcuda.so.1"
        assert mode & probe.os.RTLD_NOLOAD
        raise OSError("not loaded")

    monkeypatch.setattr(probe.C, "CDLL", unavailable)
    assert probe.cuda_snapshot()["driver_query"] == "library_not_loaded"


def test_buffer_and_viewport_validation():
    viewport = SimpleNamespace(left=0, bottom=0, width=96, height=96)
    con = SimpleNamespace(currentBuffer=1, offWidth=96, offHeight=96)
    rgb = np.empty((96, 96, 3), dtype=np.uint8)
    assert input_metadata(rgb, None, viewport, con)["issues"] == []
    assert "rgb_buffer_invalid" in input_metadata(rgb[:, ::-1], None, viewport, con)["issues"]
    viewport.width = 97
    assert "viewport_exceeds_offscreen" in input_metadata(rgb, None, viewport, con)["issues"]
