"""Minimal fallback for the obsolete ``egl_probe`` dependency.

The W0 image uses ``hf-egl-probe`` for EGL probing.  This module only exists
to satisfy robomimic 0.2.0's legacy package requirement without compiling its
unmaintained CMake helper.
"""


def get_available_devices():
    return [0]


def get_cuda_device(minor_idx):
    return int(minor_idx)
