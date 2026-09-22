"""Opt-in call-boundary evidence; never changes readPixels arguments or GL bindings."""

import ctypes as C
import os
import sys
from pathlib import Path

from robot_practice.experiments.rl.rollout_forensics import record


def cuda_snapshot():
    """Observe only: never import torch, load a new CUDA library, or call cuInit.

    cuCtxGetCurrent covers only this CPU thread, not every context in the process.
    """
    torch = sys.modules.get("torch")
    result = {
        "torch_imported": torch is not None,
        "torch_cuda_initialized": None,
        "scope": "calling_thread",
    }
    if torch is not None and hasattr(torch, "cuda"):
        result["torch_cuda_initialized"] = bool(torch.cuda.is_initialized())
    try:
        driver = C.CDLL("libcuda.so.1", mode=os.RTLD_NOLOAD | os.RTLD_LOCAL)
    except OSError:
        result["driver_query"] = "library_not_loaded"
        return result
    get_current = driver.cuCtxGetCurrent
    get_current.argtypes = [C.POINTER(C.c_void_p)]
    get_current.restype = C.c_int
    context = C.c_void_p()
    status = int(get_current(C.byref(context)))
    result.update(
        driver_query="queried_without_cuInit",
        status=status,
        current_context=context.value if status == 0 else None,
    )
    return result


def array_metadata(value):
    if value is None:
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "strides": list(value.strides),
        "nbytes": int(value.nbytes),
        "contiguous": bool(value.flags.c_contiguous),
        "writeable": bool(value.flags.writeable),
        "aligned": bool(value.flags.aligned),
    }


def input_metadata(rgb, depth, viewport, con):
    rect = {key: int(getattr(viewport, key)) for key in ("left", "bottom", "width", "height")}
    context = {
        key: int(getattr(con, key))
        for key in (
            "currentBuffer",
            "offWidth",
            "offHeight",
            "offSamples",
            "offFBO",
            "offFBO_r",
        )
        if hasattr(con, key)
    }
    issues = []
    w, h = rect["width"], rect["height"]
    if w <= 0 or h <= 0 or rect["left"] < 0 or rect["bottom"] < 0:
        issues.append("invalid_viewport")
    for name, array, shape, dtype in (
        ("rgb", rgb, (h, w, 3), "uint8"),
        ("depth", depth, (h, w), "float32"),
    ):
        if array is not None and (
            array.shape != shape
            or str(array.dtype) != dtype
            or not array.flags.c_contiguous
            or not array.flags.writeable
        ):
            issues.append(f"{name}_buffer_invalid")
    if context.get("currentBuffer") == 1 and (
        rect["left"] + w > context["offWidth"] or rect["bottom"] + h > context["offHeight"]
    ):
        issues.append("viewport_exceeds_offscreen")
    return {
        "rgb": array_metadata(rgb),
        "depth": array_metadata(depth),
        "viewport": rect,
        "context": context,
        "issues": issues,
    }


def install_offsamples_override(context_class, samples, directory, worker_index):
    """Diagnostic only: change model visual settings before MjrContext allocates buffers."""
    if samples not in (0, 4):
        raise ValueError("Diagnostic offsamples must be 0 or 4")
    original = context_class._set_mujoco_context_and_buffers

    def create(context):
        quality = context.model.vis.quality
        previous = int(quality.offsamples)
        quality.offsamples = samples
        result = original(context)
        actual = int(context.con.offSamples)
        record(
            directory,
            "offsamples_created",
            worker_index=worker_index,
            previous=previous,
            requested=samples,
            actual=actual,
        )
        print(
            f"[DEBUG-readpixels] worker={worker_index} offSamples={actual} "
            f"requested={samples} previous={previous}",
            flush=True,
        )
        if actual != samples:
            raise RuntimeError(f"offSamples override did not take effect: {actual} != {samples}")
        return result

    context_class._set_mujoco_context_and_buffers = create


def install(directory, worker_index):
    import mujoco

    samples = os.environ.get("ROBOT_PRACTICE_DIAGNOSTIC_OFFSAMPLES")
    if samples is not None:
        from robosuite.utils.binding_utils import MjRenderContext

        install_offsamples_override(MjRenderContext, int(samples), directory, worker_index)

    original = mujoco.mjr_readPixels
    backend = os.environ.get("ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND", "egl")
    if backend == "osmesa":
        gl = C.CDLL("libOSMesa.so.8")
        gl.OSMesaGetCurrentContext.restype = C.c_void_p
        gl.OSMesaGetCurrentContext.argtypes = []
    else:
        egl = C.CDLL("libEGL.so.1")
        for name in ("eglGetCurrentContext", "eglGetCurrentDisplay"):
            getattr(egl, name).restype = C.c_void_p
            getattr(egl, name).argtypes = []
        gl = C.CDLL("libGL.so.1")
    gl.glGetString.argtypes = [C.c_uint]
    gl.glGetString.restype = C.c_char_p
    gl.glGetIntegerv.argtypes = [C.c_uint, C.POINTER(C.c_int)]
    gl.glGetIntegerv.restype = None
    gl.glCheckFramebufferStatus.argtypes = [C.c_uint]
    gl.glCheckFramebufferStatus.restype = C.c_uint
    sequence = 0
    record(
        directory,
        "readpixels_probe_installed",
        worker_index=worker_index,
        process_status=Path("/proc/self/status").read_text(),
        selection={
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "MUJOCO_EGL_DEVICE_ID",
                "__EGL_VENDOR_LIBRARY_FILENAMES",
                "LD_LIBRARY_PATH",
            )
        },
    )

    def wrapped(rgb, depth, viewport, con):
        nonlocal sequence
        sequence += 1
        fields = input_metadata(rgb, depth, viewport, con)
        fields.update(call=sequence, worker_index=worker_index)
        # Flush before querying the driver: distinguish a probe failure from readPixels.
        record(directory, "readpixels_arguments", **fields)
        if backend == "osmesa":
            graphics = {"backend": backend, "osmesa_context": gl.OSMesaGetCurrentContext()}
            current = graphics["osmesa_context"]
            if not current:
                raise RuntimeError("OSMesa requested but no current OSMesa context")
        else:
            graphics = {
                "backend": backend,
                "egl_context": egl.eglGetCurrentContext(),
                "egl_display": egl.eglGetCurrentDisplay(),
            }
            current = graphics["egl_context"]
        if current:
            if sequence == 1:
                renderer = (gl.glGetString(0x1F01) or b"").decode(errors="replace")
                record(
                    directory,
                    "renderer_verified",
                    backend=backend,
                    renderer=renderer,
                    worker_index=worker_index,
                )
                print(
                    f"[DEBUG-readpixels] worker={worker_index} backend={backend} "
                    f"renderer={renderer}",
                    flush=True,
                )
            for name, token in (
                ("read_fbo", 0x8CAA),
                ("draw_fbo", 0x8CA6),
                ("pack_alignment", 0x0D05),
                ("pack_row_length", 0x0D02),
                ("pack_skip_rows", 0x0D03),
                ("pack_skip_pixels", 0x0D04),
                ("pixel_pack_buffer", 0x88ED),
            ):
                value = C.c_int()
                gl.glGetIntegerv(token, C.byref(value))
                graphics[name] = value.value
            graphics["read_fbo_status"] = int(gl.glCheckFramebufferStatus(0x8CA8))
        # Deliberately no glGetError: it would consume the original error state.
        if os.environ.get("ROBOT_PRACTICE_CUDA_CONTEXT_PROBE") == "1":
            graphics["cuda"] = cuda_snapshot()
        record(directory, "readpixels_before", graphics=graphics, **fields)
        result = original(rgb=rgb, depth=depth, viewport=viewport, con=con)
        record(directory, "readpixels_after", call=sequence, worker_index=worker_index)
        return result

    mujoco.mjr_readPixels = wrapped


def probed_shared_worker(*args):
    from env.gym_utils import async_vector_env as module

    if os.environ.get("ROBOT_PRACTICE_DIAGNOSTIC_RENDER_BACKEND") == "osmesa":
        from robomimic.utils import env_utils

        # DPPO sets MUJOCO_GL=egl immediately before calling this factory.
        # Override only this diagnostic child, at the factory boundary.
        original_factory = env_utils.create_env_from_metadata

        def software_factory(*factory_args, **factory_kwargs):
            os.environ["MUJOCO_GL"] = "osmesa"
            os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            result = original_factory(*factory_args, **factory_kwargs)
            from robosuite.utils.binding_utils import GLContext

            if "osmesa" not in GLContext.__module__:
                raise RuntimeError(f"Wrong renderer class: {GLContext}")
            return result

        env_utils.create_env_from_metadata = software_factory

    install(os.environ["ROBOT_PRACTICE_ROLLOUT_FORENSICS"], args[0])
    if os.environ.get("ROBOT_PRACTICE_GRASP_PROBE") == "1":
        from robot_practice.experiments.rl.grasp_probe import install as install_grasp

        install_grasp(os.environ["ROBOT_PRACTICE_ROLLOUT_FORENSICS"], args[0])
    target = getattr(module, "_probe_original_shared_worker", module._worker_shared_memory)
    return target(*args)
