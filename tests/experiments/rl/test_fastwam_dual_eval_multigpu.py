import json
from types import SimpleNamespace

import pytest

from robot_practice.experiments.rl.fastwam_dual_data import PROTOCOL
from robot_practice.experiments.rl.fastwam_dual_eval_multigpu import (
    assignments,
    build_shard_command,
    collect,
    parse_mapping,
    shard_environment,
)


def test_plan_assigns_each_global_episode_once():
    plan = assignments([str(gpu) for gpu in range(8)], 17)
    assert plan[0] == ("0", [0, 8, 16])
    assert plan[7] == ("7", [7, 15])
    assert sorted(index for _, indices in plan for index in indices) == list(range(17))


@pytest.mark.parametrize("gpus,episodes", [([], 8), (["0", "0"], 8), (["-1"], 8),
                                            (["x"], 8), (["0"], 0)])
def test_invalid_plan(gpus, episodes):
    with pytest.raises(ValueError):
        assignments(gpus, episodes)


def test_parse_mapping_uses_physical_gpu_not_egl_or_cuda_ordinal():
    table = ("EGL_ID\tCUDA_ORDINAL_UNMASKED\tNVIDIA_SMI_INDEX\tPCI_BUS\tGPU_UUID\n"
             "6\t4\t0\t0000:65:02.0\tGPU-A\n"
             "7\t1\t1\t0000:65:03.0\tGPU-B\n"
             "DONE enumerated=8 unresolved=0\n")
    mapping = parse_mapping(table)
    assert mapping["0"]["egl_id"] == 6
    assert mapping["1"]["egl_id"] == 7
    assert mapping["0"]["cuda_ordinal_unmasked"] == 4


def test_shard_pins_policy_gpu_but_passes_unmasked_egl_id(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    env = shard_environment("4", 2)
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert "MUJOCO_EGL_DEVICE_ID" not in env
    assert env["OMP_NUM_THREADS"] == "2"
    args = SimpleNamespace(checkpoint="weights.pt", episodes=8, seed=10001, max_steps=600,
                           render_backend="egl", hdf5="demo.hdf5", normalization="norm.npz")
    cmd = build_shard_command(args, tmp_path, "0", [0], 6)
    assert cmd[cmd.index("--episode-indices") + 1] == "0"
    assert cmd[cmd.index("--egl-device-id") + 1] == "6"
    assert cmd[cmd.index("--render-backend") + 1] == "egl"


@pytest.mark.parametrize("table", [
    "6\t0\t0\t0000:65:02.0\tGPU-A\n",
    "6\t0\t0\t0000:65:02.0\tGPU-A\n7\t1\t0\t0000:65:03.0\tGPU-B\n"
    + "DONE enumerated=8 unresolved=0\n",
    "6\t0\t0\t0000:65:02.0\tUNRESOLVED\nDONE enumerated=8 unresolved=1\n",
])
def test_mapping_requires_complete_unique_identity(table):
    with pytest.raises(ValueError):
        parse_mapping(table)


def make_shards(tmp_path):
    args = SimpleNamespace(episodes=10, seed=10001, max_steps=600, render_backend="egl")
    plan = assignments(["0", "1"], args.episodes)
    mapping = {"0": {"egl_id": 6}, "1": {"egl_id": 7}}
    for gpu, indices in plan:
        directory = tmp_path / f"gpu_{gpu}"
        directory.mkdir()
        rows = [{"episode_index": i, "seed": args.seed + i,
                 "task_success_any_step": i % 2 == 0, "official_chunk_success": i % 2 == 0,
                 "within_400": {"task_success_any_step": False,
                                "official_chunk_success": False}} for i in indices]
        (directory / "metrics.json").write_text(json.dumps({
            "status": "complete", "checkpoint_sha256": "weights", "protocol": PROTOCOL,
            "arguments": {"episodes": args.episodes, "seed": args.seed,
                          "max_steps": args.max_steps, "render_backend": "egl",
                          "egl_device_id": mapping[gpu]["egl_id"], "episode_indices": indices},
            "rendering": {"backend": "egl", "context_class": "EGLGLContext",
                          "egl_device_id": str(mapping[gpu]["egl_id"]),
                          "cuda_visible_devices": None},
            "episodes": rows, "evaluated": len(rows), "requested": len(indices),
        }))
    return args, plan, mapping


def test_collect_requires_exact_coverage_and_renderer(tmp_path):
    args, plan, mapping = make_shards(tmp_path)
    result = collect(tmp_path, plan, mapping, args, "weights", final=True)
    assert result["evaluated"] == result["requested"] == 10
    assert result["success_rate_600"] == 0.5
    assert result["success_rate_400"] == 0


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "checkpoint", "backend",
                                       "context", "seed", "status"])
def test_collect_rejects_corrupt_or_incomplete_shard(tmp_path, mutation):
    args, plan, mapping = make_shards(tmp_path)
    path = tmp_path / "gpu_0/metrics.json"
    data = json.loads(path.read_text())
    if mutation == "duplicate":
        data["episodes"].append(data["episodes"][0])
    elif mutation == "missing":
        data["episodes"].pop()
    elif mutation == "checkpoint":
        data["checkpoint_sha256"] = "wrong"
    elif mutation == "backend":
        data["arguments"]["render_backend"] = "osmesa"
    elif mutation == "context":
        data["rendering"]["context_class"] = "OSMesaGLContext"
    elif mutation == "seed":
        data["episodes"][0]["seed"] = 999
    else:
        data["status"] = "running"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        collect(tmp_path, plan, mapping, args, "weights", final=True)
