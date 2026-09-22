import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from robot_practice.experiments.rl.fastwam_square_distributed import (
    Batches,
    learning_rate,
    optimizer_update,
)


class TinyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, target):
        return (self.weight - target).square().mean(), {}


def ddp_worker(rank, init_file, result_file):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        model = TinyLoss()
        wrapped = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        targets = iter([torch.tensor(0.1 + rank * 0.2), torch.tensor(0.2 + rank * 0.2)])
        optimizer_update(wrapped, optimizer, list(model.parameters()), lambda: next(targets),
                         2, torch.device("cpu"))
        if rank == 0:
            with open(result_file, "w") as stream:
                json.dump({"weight": model.weight.item()}, stream)
    finally:
        dist.destroy_process_group()


def test_real_two_process_ddp_matches_global_batch(tmp_path):
    result_file = tmp_path / "result.json"
    mp.spawn(ddp_worker, args=(str(tmp_path / "init"), str(result_file)), nprocs=2, join=True)
    reference = TinyLoss()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    optimizer_update(reference, optimizer, list(reference.parameters()),
                     lambda: torch.tensor([0.1, 0.2, 0.3, 0.4]), 1, torch.device("cpu"))
    assert json.loads(result_file.read_text())["weight"] == pytest.approx(reference.weight.item())


def test_lr_warmup_and_cosine():
    def lr(seconds):
        return learning_rate(seconds, 1e-4, 1e-6, 43200, 0.01)

    assert lr(0) == pytest.approx(1e-6)
    assert lr(432) == pytest.approx(1e-4)
    assert lr((432 + 43200) / 2) == pytest.approx(5.05e-5)
    assert lr(43200) == pytest.approx(1e-6)
    assert lr(50000) == pytest.approx(1e-6)
    assert all(lr(t) >= lr(t + 1) for t in range(432, 43200, 100))


def test_resume_uses_cumulative_time_and_zero_warmup():
    previous, since_resume = 7200, 1000
    assert learning_rate(previous + since_resume, 1e-4, 1e-6, 43200, 0.01) == (
        learning_rate(8200, 1e-4, 1e-6, 43200, 0.01)
    )
    assert learning_rate(0, 1e-4, 1e-6, 43200, 0) == pytest.approx(1e-4)


def test_batch_cursor_resume():
    class Sampler:
        def set_epoch(self, epoch):
            self.epoch = epoch

    sampler = Sampler()
    batches = Batches(["a", "b", "c"], sampler, epoch=5, offset=2)
    assert batches() == "c"
    assert (batches.epoch, batches.offset) == (5, 3)
    assert batches() == "a"
    assert (batches.epoch, batches.offset, sampler.epoch) == (6, 1, 6)


def test_nonfinite_loss_stops_update():
    model = TinyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError):
        optimizer_update(model, optimizer, list(model.parameters()),
                         lambda: torch.tensor(float("nan")), 1, torch.device("cpu"))
    assert model.weight.item() == 0
