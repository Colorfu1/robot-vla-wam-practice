#!/usr/bin/env python
"""Report the EGL device list robosuite selects from, and map it back to CUDA.

``robosuite.renderers.context.egl_context.create_initialized_egl_device_display``
indexes ``eglQueryDevicesEXT()`` directly with ``MUJOCO_EGL_DEVICE_ID``.  That
list is **not** filtered by ``CUDA_VISIBLE_DEVICES``, its order is not
guaranteed to match CUDA's enumeration, and its length need not equal the GPU
count.  On an 8-GPU H20 node this list has 10 entries, so
``MUJOCO_EGL_DEVICE_ID=7`` can select something other than CUDA device 7 --
and rendering then aborts natively inside ``mjr_readPixels``.

This probe answers three questions without touching the evaluator:

1. How many EGL devices exist, and what is each one?
2. For each EGL index, which CUDA device does the driver say it is
   (``EGL_CUDA_DEVICE_NV``)?
3. With ``--probe-context``, can that index initialise a display, create a
   context, make it current, and report GL vendor/renderer/version strings?

Run enumeration with ``CUDA_VISIBLE_DEVICES`` unset so the mapping is absolute.
Run ``--probe-context --device N`` in its own process per index; a bad device
can abort natively and would otherwise take the whole sweep down.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
from typing import Any, Callable

# EGL/GL enums that the bundled bindings do not always expose by name.
EGL_CUDA_DEVICE_NV = 0x323A
EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377
EGL_EXTENSIONS = 0x3055
EGL_VENDOR = 0x3053
EGL_VERSION = 0x3054
GL_VENDOR = 0x1F00
GL_RENDERER = 0x1F01
GL_VERSION_STR = 0x1F02


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-context",
        action="store_true",
        help="also initialise a display/context on --device and report GL strings",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="EGL device index to probe (defaults to MUJOCO_EGL_DEVICE_ID, else 0)",
    )
    return parser.parse_args(argv)


def _load_device_query() -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    """Resolve eglQueryDeviceAttribEXT/eglQueryDeviceStringEXT across bindings.

    ``mujoco.egl.egl_ext`` only re-exports the enumeration entry point, so the
    query entry points have to come from PyOpenGL's extension modules.
    """

    attrib = None
    string = None
    for module_name in (
        "OpenGL.EGL.EXT.device_query",
        "OpenGL.EGL.EXT.device_base",
        "OpenGL.EGL.EXT.device_enumeration",
        "mujoco.egl.egl_ext",
    ):
        try:
            module = __import__(module_name, fromlist=["*"])
        except ImportError:
            continue
        attrib = attrib or getattr(module, "eglQueryDeviceAttribEXT", None)
        string = string or getattr(module, "eglQueryDeviceStringEXT", None)
        if attrib is not None and string is not None:
            break
    return attrib, string


def _query_attrib(attrib: Callable[..., Any] | None, device: Any, name: int) -> str:
    if attrib is None:
        return "<no binding>"
    # PyOpenGL exposes both an out-parameter form and a value-returning form.
    try:
        value = ctypes.c_ssize_t(-1)
        ok = attrib(device, name, ctypes.byref(value))
        if ok:
            return str(value.value)
        return "<unsupported>"
    except (TypeError, ctypes.ArgumentError):
        pass
    except Exception as error:  # pragma: no cover - driver dependent
        return f"<error {type(error).__name__}>"
    try:
        return str(attrib(device, name))
    except Exception as error:  # pragma: no cover - driver dependent
        return f"<error {type(error).__name__}>"


def _query_string(string: Callable[..., Any] | None, device: Any, name: int) -> str:
    if string is None:
        return "<no binding>"
    try:
        value = string(device, name)
    except Exception as error:  # pragma: no cover - driver dependent
        return f"<error {type(error).__name__}>"
    if value is None:
        return "<none>"
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def _gl_strings() -> str:
    """Read GL vendor/renderer/version through libGL to identify the device."""

    for library in ("libGL.so.1", "libOpenGL.so.0", "libGLESv2.so.2"):
        try:
            handle = ctypes.CDLL(library)
        except OSError:
            continue
        try:
            handle.glGetString.restype = ctypes.c_char_p
            handle.glGetString.argtypes = [ctypes.c_uint]
            values = [
                handle.glGetString(name)
                for name in (GL_VENDOR, GL_RENDERER, GL_VERSION_STR)
            ]
        except Exception as error:  # pragma: no cover - driver dependent
            return f"<{library}: error {type(error).__name__}>"
        if any(value for value in values):
            decoded = [
                value.decode("utf-8", "replace") if value else "<none>" for value in values
            ]
            return f"via={library} vendor={decoded[0]} renderer={decoded[1]} version={decoded[2]}"
    return "<no GL library returned strings>"


def _report_cuda_view() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("[EGL] torch not importable; skipping CUDA view", flush=True)
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    order = os.environ.get("CUDA_DEVICE_ORDER", "<unset>")
    print(
        f"[EGL] CUDA_VISIBLE_DEVICES={visible} CUDA_DEVICE_ORDER={order} "
        f"torch_device_count={torch.cuda.device_count()}",
        flush=True,
    )
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"[EGL] CUDA index={index} name={properties.name} "
            f"total_mib={properties.total_memory >> 20} "
            f"uuid={getattr(properties, 'uuid', None)}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    from mujoco.egl import egl_ext as EGL

    attrib, string = _load_device_query()
    devices = EGL.eglQueryDevicesEXT()
    print(
        f"[EGL] MUJOCO_EGL_DEVICE_ID={os.environ.get('MUJOCO_EGL_DEVICE_ID', '<unset>')} "
        f"egl_device_count={len(devices)} "
        f"attrib_binding={attrib is not None} string_binding={string is not None}",
        flush=True,
    )

    for index, device in enumerate(devices):
        print(
            f"[EGL] index={index} "
            f"cuda_device={_query_attrib(attrib, device, EGL_CUDA_DEVICE_NV)} "
            f"drm={_query_string(string, device, EGL_DRM_DEVICE_FILE_EXT)} "
            f"render_node={_query_string(string, device, EGL_DRM_RENDER_NODE_FILE_EXT)} "
            f"extensions={_query_string(string, device, EGL_EXTENSIONS)[:200]}",
            flush=True,
        )

    _report_cuda_view()

    if not args.probe_context:
        print("[EGL] enumeration only; pass --probe-context to test a device", flush=True)
        return 0

    requested = args.device
    if requested is None:
        requested = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))
    if not 0 <= requested < len(devices):
        print(
            f"[EGL] FAIL requested index={requested} is out of range for "
            f"egl_device_count={len(devices)}",
            flush=True,
        )
        return 2

    print(f"[EGL] probing context on egl index={requested}", flush=True)
    display = EGL.eglGetPlatformDisplayEXT(
        EGL.EGL_PLATFORM_DEVICE_EXT, devices[requested], None
    )
    if display == EGL.EGL_NO_DISPLAY:
        print("[EGL] FAIL eglGetPlatformDisplayEXT returned EGL_NO_DISPLAY", flush=True)
        return 3
    if not EGL.eglInitialize(display, None, None):
        print(f"[EGL] FAIL eglInitialize error=0x{EGL.eglGetError():x}", flush=True)
        return 4
    print(
        f"[EGL] display vendor={EGL.eglQueryString(display, EGL_VENDOR)} "
        f"version={EGL.eglQueryString(display, EGL_VERSION)}",
        flush=True,
    )

    # Same config/context request robosuite makes.
    attributes = (
        EGL.EGL_RED_SIZE, 8, EGL.EGL_GREEN_SIZE, 8, EGL.EGL_BLUE_SIZE, 8,
        EGL.EGL_ALPHA_SIZE, 8, EGL.EGL_DEPTH_SIZE, 24, EGL.EGL_STENCIL_SIZE, 8,
        EGL.EGL_COLOR_BUFFER_TYPE, EGL.EGL_RGB_BUFFER,
        EGL.EGL_SURFACE_TYPE, EGL.EGL_PBUFFER_BIT,
        EGL.EGL_RENDERABLE_TYPE, EGL.EGL_OPENGL_BIT,
        EGL.EGL_NONE,
    )
    num_configs = ctypes.c_long()
    config = EGL.EGLConfig()
    EGL.eglChooseConfig(display, attributes, ctypes.byref(config), 1, num_configs)
    print(f"[EGL] matching_configs={num_configs.value}", flush=True)
    if num_configs.value < 1:
        print("[EGL] FAIL no framebuffer config matches robosuite's attributes", flush=True)
        return 5

    EGL.eglBindAPI(EGL.EGL_OPENGL_API)
    context = EGL.eglCreateContext(display, config, EGL.EGL_NO_CONTEXT, None)
    if not context:
        print(f"[EGL] FAIL eglCreateContext error=0x{EGL.eglGetError():x}", flush=True)
        return 6
    if not EGL.eglMakeCurrent(display, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, context):
        print(f"[EGL] FAIL eglMakeCurrent error=0x{EGL.eglGetError():x}", flush=True)
        return 7

    print(f"[EGL] OK index={requested} {_gl_strings()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
