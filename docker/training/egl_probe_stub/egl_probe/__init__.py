"""Compatibility shim for robomimic's optional legacy EGL probe."""

from .get_available_devices import get_available_devices, get_cuda_device

__all__ = ["get_available_devices", "get_cuda_device"]
