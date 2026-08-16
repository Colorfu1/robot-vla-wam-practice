from robot_practice.experiments.maniskill.ddp_device import (
    rank_render_backend,
    rank_sim_backend,
)


def test_gpu_backends_are_bound_to_ddp_local_rank() -> None:
    assert rank_sim_backend("physx_cuda", 2) == "physx_cuda:2"
    assert rank_sim_backend("gpu", 3) == "physx_cuda:3"
    assert rank_render_backend("physx_cuda", 2) == "sapien_cuda:2"


def test_cpu_backend_is_not_reassigned() -> None:
    assert rank_sim_backend("physx_cpu", 3) == "physx_cpu"
    assert rank_render_backend("physx_cpu", 3) == "sapien_cpu"
