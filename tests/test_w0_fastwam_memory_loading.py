from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _import_w0_module(name: str):
    source_root = Path("third_party/w0_fastwam/FastWAM/src")
    if not source_root.is_dir():
        pytest.skip("pinned FastWAM checkout is not present")
    source = str(source_root.resolve())
    if source not in sys.path:
        sys.path.insert(0, source)
    return importlib.import_module(name)


def test_target_device_construction_avoids_a_cpu_fp32_staging_model() -> None:
    torch = pytest.importorskip("torch")
    io = _import_w0_module("fastwam.models.wan22.helpers.io")

    class Probe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4, 4))
            self.plain_tensor = torch.ones(2)

    original_dtype = torch.get_default_dtype()
    model = io.construct_module_on_device(
        Probe,
        {},
        device="meta",
        torch_dtype=torch.bfloat16,
    )

    assert model.weight.device.type == "meta"
    assert model.weight.dtype == torch.bfloat16
    assert model.plain_tensor.device.type == "meta"
    assert model.plain_tensor.dtype == torch.bfloat16
    assert torch.get_default_dtype() == original_dtype


def test_large_w0_components_use_target_device_construction() -> None:
    loader = _import_w0_module("fastwam.models.wan22.helpers.loader")
    action_dit = _import_w0_module("fastwam.models.wan22.action_dit")

    loader_source = Path(loader.__file__).read_text(encoding="utf-8")
    action_source = Path(action_dit.__file__).read_text(encoding="utf-8")

    assert "construct_module_on_device(" in loader_source
    assert "WanVideoDiT(**validated_dit_config).to(" not in loader_source
    assert "construct_module_on_device(" in action_source


def test_safetensors_are_copied_into_the_model_one_tensor_at_a_time(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    io = _import_w0_module("fastwam.models.wan22.helpers.io")

    checkpoint = tmp_path / "probe.safetensors"
    safetensors_torch.save_file(
        {
            "source_bias": torch.tensor([3.0, 4.0]),
            "source_weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        },
        checkpoint,
    )
    model = torch.nn.Linear(3, 2)

    copied = io.load_safetensors_into_model(
        model,
        checkpoint,
        device="cpu",
        torch_dtype=torch.float32,
        state_dict_converter=lambda item: {
            key.removeprefix("source_"): value for key, value in item.items()
        },
    )

    assert copied == 2
    assert torch.equal(model.weight, torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert torch.equal(model.bias, torch.tensor([3.0, 4.0]))


def test_patch_helper_handles_a_host_owned_vendor_mount_without_global_git_config() -> None:
    body = Path("scripts/experiments/literature/apply_w0_fastwam_memory_patch.sh").read_text(
        encoding="utf-8"
    )

    assert 'git_command=(git -c "safe.directory=${fastwam_root}" -C "${fastwam_root}")' in body
    assert 'git config --global' not in body


def test_docker_launcher_pins_nvidia_devices_without_privileged_mode() -> None:
    body = Path("run_w0_fastwam_debug_docker.sh").read_text(encoding="utf-8")

    assert "/dev/nvidiactl" in body
    assert "/dev/nvidia-uvm" in body
    assert "/dev/nvidia[0-9]*" in body
    assert '--device "${nvidia_device}:${nvidia_device}:rwm"' in body
    assert "--pull=never" in body
    assert "--privileged" not in body
    assert "docker pull" not in body
    assert "curl " not in body
    assert "wget " not in body
