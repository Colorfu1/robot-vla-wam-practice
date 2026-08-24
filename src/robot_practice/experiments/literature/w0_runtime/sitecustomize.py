"""Inject the repository-local LIBERO simulator asset path into W0 runtimes.

The pinned LIBERO release only looks for assets beside the installed package or
under ``~/.cache/libero/assets``.  W0 keeps all inputs under the repository, so
the launcher sets ``W0_LIBERO_ASSETS`` and this startup hook fills LIBERO's
existing asset-path cache before any environment class is imported.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _report(message: str) -> None:
    """Write a diagnostic to stderr, never stdout.

    ``conda run`` builds a temporary activation script and ``eval``s the output of a helper
    ``python`` invocation. Because PYTHONPATH already contains this directory, that helper
    imports this module, so anything printed to stdout is swallowed by the eval. A message
    containing parentheses -- e.g. "(kill -USR1 <pid> ...)" -- then aborts the activation
    with "syntax error near unexpected token '('", conda falls back to the wrong
    environment, and training dies with "DeepSpeed is not installed" because it launched
    from robot-practice instead of fastwam-upstream.

    stderr keeps the message visible in train.log (which captures both streams), so the
    positive evidence that a safeguard is active is preserved without polluting any
    command substitution.
    """

    sys.stderr.write(f"{message}\n")
    sys.stderr.flush()


def _fatal(message: str) -> None:
    """Stop the interpreter with a visible reason.

    ``site.execsitecustomize`` catches ``Exception`` and lets the program run on
    with only a one-line ``Error in sitecustomize`` note, so raising here would
    silently disable these patches instead of stopping a misconfigured run.
    ``SystemExit`` reaches the user only as ``Fatal Python error:
    init_import_site``, so write the real reason and exit directly.
    """

    sys.stderr.write(f"[W0-RUNTIME] fatal: {message}\n")
    sys.stderr.flush()
    os._exit(2)


def _configure_libero_assets() -> None:
    raw_path = os.environ.get("W0_LIBERO_ASSETS")
    if not raw_path:
        return

    assets_path = Path(raw_path).expanduser().resolve()
    required = assets_path / "scenes/libero_tabletop_base_style.xml"
    if not required.is_file():
        # A silently ignored asset root would let LIBERO fall back to the
        # packaged scenes and quietly change the evaluated environment.
        _fatal(f"W0_LIBERO_ASSETS is missing the required LIBERO scene: {required}")

    try:
        import libero.libero as libero_module
    except ModuleNotFoundError as error:
        # ``conda run`` can start a helper Python process before it enters
        # fastwam-upstream.  That interpreter inherits PYTHONPATH and
        # W0_LIBERO_ASSETS but does not have LIBERO installed.  Python reports
        # sitecustomize import failures even though the real child process can
        # continue, so make this hook a no-op for that outer interpreter.  Do
        # not hide missing dependencies of an installed LIBERO package.
        if error.name == "libero" or error.name.startswith("libero."):
            return
        raise

    libero_module._assets_path_cache = str(assets_path)
    libero_module.get_assets_path = lambda: str(assets_path)


def _patch_egl_cleanup() -> None:
    """Make robosuite's EGL destructor tolerant of late process teardown.

    robosuite registers ``eglTerminate`` with ``atexit`` when the first EGL
    context is created.  Python may finalize the context object afterwards;
    its destructor then calls ``eglMakeCurrent`` against an already-terminated
    display and emits an ``EGL_NOT_INITIALIZED`` traceback.  This is a cleanup
    ordering issue, not a rendering failure.  Patch only that specific error;
    all initialization and other EGL errors remain visible.
    """

    try:
        from OpenGL import EGL
        from OpenGL.raw.EGL._errors import EGLError
        from robosuite.renderers.context import egl_context
    except ModuleNotFoundError as error:
        # The outer interpreter used by ``conda run`` can inherit this
        # PYTHONPATH without having the FastWAM-only graphics packages.  The
        # actual fastwam-upstream interpreter imports both packages and gets
        # the patch below.
        if error.name.startswith("OpenGL") or error.name.startswith("robosuite"):
            return
        raise

    original_free = egl_context.EGLGLContext.free
    if getattr(original_free, "_robot_practice_egl_safe", False):
        return

    def safe_free(context: object) -> None:
        if getattr(context, "_context", None) is None:
            return
        try:
            original_free(context)
        except EGLError as error:
            error_text = repr(error)
            if getattr(error, "err", None) == EGL.EGL_NOT_INITIALIZED or "EGL_NOT_INITIALIZED" in error_text:
                setattr(context, "_context", None)
                return
            raise

    safe_free._robot_practice_egl_safe = True
    egl_context.EGLGLContext.free = safe_free


def _patch_render_context_current() -> None:
    """Bind every MuJoCo render call to its own GL context.

    robosuite's ``MjRenderContext.render`` and ``read_pixels`` never re-make
    their GL context current; they rely on the thread's current context still
    being the one installed at construction time.  ``EGLGLContext.__init__``
    calls ``eglReleaseThread()`` and then ``make_current()``, so creating any
    additional render context silently un-currents the previous one.  With
    robosuite's default ``hard_reset=True`` every ``env.reset()`` destroys the
    sim and builds a fresh render context, and ``MjRenderContext.__del__``
    frees MuJoCo's GL objects without making its own context current -- which
    deletes object ids that now belong to the live context.  Both paths end in
    a native abort inside ``mjr_readPixels`` that Python cannot catch.

    This is opt-in via ``W0_GL_CONTEXT_GUARD=1`` so the unpatched behaviour
    stays available as the control arm of an A/B.
    """

    if os.environ.get("W0_GL_CONTEXT_GUARD") != "1":
        return

    try:
        from robosuite.utils import binding_utils
    except ModuleNotFoundError as error:
        # ``conda run`` may start a helper interpreter that inherits this
        # PYTHONPATH without the FastWAM-only simulator packages.
        if error.name.startswith("robosuite"):
            return
        raise

    context_cls = binding_utils.MjRenderContext
    if getattr(context_cls, "_robot_practice_context_guard", False):
        return

    def make_current(context: object) -> None:
        gl_ctx = getattr(context, "gl_ctx", None)
        if gl_ctx is not None:
            gl_ctx.make_current()

    for method_name in ("render", "read_pixels", "upload_texture"):
        original = getattr(context_cls, method_name, None)
        if original is None:
            continue

        def guarded(self, *args, _original=original, **kwargs):
            make_current(self)
            return _original(self, *args, **kwargs)

        guarded.__name__ = method_name
        guarded.__doc__ = getattr(original, "__doc__", None)
        setattr(context_cls, method_name, guarded)

    original_del = getattr(context_cls, "__del__", None)
    if original_del is not None:

        def guarded_del(self, _original=original_del) -> None:
            # A destructor must never raise; an already-destroyed context here
            # is expected during interpreter teardown.
            try:
                make_current(self)
            except Exception:  # pragma: no cover - teardown ordering
                pass
            _original(self)

        context_cls.__del__ = guarded_del

    context_cls._robot_practice_context_guard = True
    _report("[W0-RUNTIME] MjRenderContext GL-context guard enabled")


def _resolve_hard_reset_override() -> bool | None:
    raw = os.environ.get("W0_HARD_RESET")
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes"}:
        _fatal(f"W0_HARD_RESET must be one of 0/1/false/true/no/yes, got {raw!r}")
    return normalized in {"1", "true", "yes"}


def _egl_device_count() -> str:
    """Report how many entries eglQueryDevicesEXT() exposes, or why it is unknown.

    ``MUJOCO_EGL_DEVICE_ID`` is a position in this list, not a CUDA device id, so
    the list length is the only way to tell whether a requested index can exist.
    """

    try:
        from mujoco.egl import egl_ext

        return str(len(egl_ext.eglQueryDevicesEXT()))
    except Exception:  # pragma: no cover - driver/binding dependent
        return "unknown"


def _patch_libero_render_truth() -> None:
    """Make W0_HARD_RESET authoritative and report the state that actually applies.

    The pinned FastWAM evaluator prints ``hard_reset`` by echoing the
    environment variable, while the value that really decides whether robosuite
    rebuilds the sim and its offscreen render context on every ``reset()`` is the
    ``hard_reset`` keyword of ``MujocoEnv.__init__``.  A vendored checkout whose
    ``libero_utils.get_libero_env`` never forwards that keyword therefore logs
    ``hard_reset=0`` while running with robosuite's default ``hard_reset=True``
    -- a diagnostic that confirms a setting it never applied.

    Patching the constructor fixes both halves at once: the environment variable
    becomes authoritative regardless of which vendored copy is deployed, and one
    line records the effective render configuration, including the EGL device
    index and list length that ``MUJOCO_EGL_DEVICE_ID`` indexes into.
    """

    override = _resolve_hard_reset_override()

    try:
        from robosuite.environments.base import MujocoEnv
    except ModuleNotFoundError as error:
        # ``conda run`` can start a helper interpreter that inherits this
        # PYTHONPATH without the simulator packages installed.
        if error.name.startswith("robosuite"):
            return
        raise

    original_init = MujocoEnv.__init__
    if getattr(original_init, "_robot_practice_render_truth", False):
        return

    def patched_init(self: object, *args: object, **kwargs: object) -> None:
        if override is not None:
            kwargs["hard_reset"] = override
        original_init(self, *args, **kwargs)
        if getattr(MujocoEnv, "_robot_practice_render_reported", False):
            return
        MujocoEnv._robot_practice_render_reported = True
        _report(
            "[W0-RUNTIME] render config "
            f"requested_hard_reset={os.environ.get('W0_HARD_RESET', 'unset')} "
            f"effective_hard_reset={getattr(self, 'hard_reset', 'unknown')} "
            f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
            f"mujoco_egl_device_id={os.environ.get('MUJOCO_EGL_DEVICE_ID', 'unset')} "
            f"egl_device_count={_egl_device_count()} "
            f"mujoco_gl={os.environ.get('MUJOCO_GL', 'unset')}"
        )

    patched_init._robot_practice_render_truth = True
    MujocoEnv.__init__ = patched_init


def _patch_libero_num_steps_wait() -> None:
    """Let W0 align LeRobot's post-reset settling with the official evaluator.

    ``lerobot.envs.libero.LiberoEnv`` defaults to ``num_steps_wait=10`` while the
    official FastWAM evaluator uses ``5``
    (``experiments/libero/eval_libero_single.py``).  Both step a dummy action that
    many times after ``set_init_state`` before handing control to the policy, so
    the value decides how long the scene settles.  robosuite notes that objects
    can be floating or intersecting immediately after a state restore, which makes
    the duration matter for precariously stacked configurations such as
    ``libero_spatial`` task 5, "pick up the black bowl on the ramekin".

    ``LiberoEnv``'s config exposes only four keys through its ``gym_kwargs``
    property, so the value is unreachable from the CLI.  Those kwargs are
    forwarded verbatim into the constructor, so injecting the key there is enough.
    Opt in with ``W0_LIBERO_NUM_STEPS_WAIT`` so the default behaviour is untouched
    and the comparison stays an explicit A/B.
    """

    raw = os.environ.get("W0_LIBERO_NUM_STEPS_WAIT")
    if raw is None:
        return
    try:
        steps = int(raw)
    except ValueError:
        _fatal(f"W0_LIBERO_NUM_STEPS_WAIT must be an integer, got {raw!r}")
    if steps < 0:
        _fatal(f"W0_LIBERO_NUM_STEPS_WAIT must be non-negative, got {steps}")

    try:
        from lerobot.envs.configs import LiberoEnv
    except ModuleNotFoundError as error:
        # The upstream FastWAM interpreter shares this PYTHONPATH but has no LeRobot.
        if error.name.startswith("lerobot"):
            return
        raise

    descriptor = LiberoEnv.__dict__.get("gym_kwargs")
    if descriptor is None or not isinstance(descriptor, property):
        _fatal(
            "LiberoEnv.gym_kwargs is not a property; inspect "
            "lerobot/envs/configs.py before forcing num_steps_wait"
        )
    original = descriptor.fget
    if getattr(original, "_robot_practice_num_steps_wait", False):
        return

    def patched(self: object) -> dict:
        kwargs = dict(original(self))
        kwargs["num_steps_wait"] = steps
        return kwargs

    patched._robot_practice_num_steps_wait = True
    LiberoEnv.gym_kwargs = property(patched)
    _report(f"[W0-RUNTIME] LiberoEnv num_steps_wait forced to {steps}")


def _configure_cuda_home() -> None:
    """Point CUDA_HOME at a prefix that actually contains ``bin/nvcc``.

    ``import deepspeed`` runs ``git_version_info.py``, which calls
    ``is_compatible()`` on every op builder; ``FpQuantizerBuilder`` then calls
    ``installed_cuda_version()``, which shells out to ``$CUDA_HOME/bin/nvcc -V``
    purely to read a version number.  ``installed_cuda_version`` only raises a
    catchable ``MissingCUDAException`` when ``CUDA_HOME`` is ``None``; when the
    directory exists but has no compiler the ``subprocess`` call escapes as
    ``FileNotFoundError`` and kills the launcher before training starts.
    ``DS_SKIP_CUDA_CHECK`` does not help -- it guards
    ``assert_no_cuda_mismatch``, a different function.

    The W0 image ships a CUDA runtime layout at ``/usr/local/cuda`` with no
    ``bin/``, so torch resolves ``CUDA_HOME`` to a prefix that cannot answer the
    probe.  Nothing is ever compiled here (ZeRO-1 with offload disabled needs no
    custom op), so satisfying the probe is enough: pick the first candidate
    prefix that has ``bin/nvcc``.  A no-op when the current value already works.

    Opt in with ``W0_CUDA_HOME_FALLBACK=1``, which the training launcher sets.
    Only processes that import DeepSpeed need this, and leaving evaluation
    processes untouched keeps their stdout identical to the runs that produced the
    recorded UP-REL/LR-REL numbers.
    """

    if os.environ.get("W0_CUDA_HOME_FALLBACK") != "1":
        return

    current = os.environ.get("CUDA_HOME")
    if current and (Path(current) / "bin/nvcc").is_file():
        return

    candidates = [
        Path(sys.prefix),  # conda env, e.g. after `conda install cuda-nvcc`
        Path("/usr/local/cuda-12.8"),
        Path("/usr/local/cuda-12"),
        Path("/usr/local/cuda"),
        # Last resort for hosts with no package index: a repository-tracked shim
        # that answers `-V` and refuses anything else. Real toolchains above win,
        # so it goes unused once the image ships cuda-nvcc.
        Path(__file__).resolve().parents[5] / "docker/training/cuda_nvcc_probe_shim",
    ]
    for prefix in candidates:
        if (prefix / "bin/nvcc").is_file():
            os.environ["CUDA_HOME"] = str(prefix)
            kind = "probe shim" if prefix.name == "cuda_nvcc_probe_shim" else "toolchain"
            _report(
                f"[W0-RUNTIME] CUDA_HOME={prefix} ({kind}; previous={current or 'unset'}; "
                "needed only for DeepSpeed's import-time nvcc version probe)"
            )
            return


def _enable_stall_tracebacks() -> None:
    """Make a hung rank print its own Python stack instead of dying silently.

    Two 8-GPU training runs died to a 600 s NCCL ALLREDUCE timeout. Both times exactly
    one rank was absent from the watchdog reports -- step 260/rank 3, then step 590/rank 1
    -- meaning that rank never reached the collective and was stuck in host-side code while
    the other seven waited. NCCL FlightRecorder does not help here: it records collectives,
    and the stuck rank had not enqueued one. Nothing in the logs said where it was.

    ``faulthandler`` is stdlib, needs no network, and changes no numerics. Registering
    SIGTERM means torch elastic's own teardown of the stuck rank dumps that rank's stack
    into train.log. SIGUSR1 allows dumping on demand during the 600 s window
    (``kill -USR1 <pid>``). ``chain=True`` keeps the default signal behaviour so teardown
    still proceeds.

    Disable with W0_STALL_TRACEBACKS=0 if a handler ever interferes.
    """

    if os.environ.get("W0_STALL_TRACEBACKS", "1") == "0":
        return
    try:
        import faulthandler
        import signal
    except Exception as exc:  # pragma: no cover - stdlib import guard
        _report(f"[W0-RUNTIME] faulthandler unavailable: {exc}")
        return

    try:
        # Fatal signals (SIGABRT/SIGSEGV): the ranks that abort at the NCCL timeout.
        faulthandler.enable(all_threads=True)
    except Exception as exc:  # pragma: no cover
        _report(f"[W0-RUNTIME] faulthandler.enable failed: {exc}")
        return

    # chain matters: it decides whether the *default* action still runs after the dump.
    #
    # SIGTERM -> chain=True. torch elastic sends it to tear a rank down, and the teardown
    # must still happen; we only want the stack on the way out.
    #
    # SIGUSR1/SIGUSR2 -> chain=False. These are for dumping a *live* hung rank on demand.
    # Their default action is Term (signal(7)), so chaining kills the process. That is not
    # theoretical: an on-demand dump chained to SIG_DFL killed 57 dataloader workers, whose
    # deaths raised DataLoader's SIGCHLD handler in the ranks and destroyed the run 15 s
    # after the dump -- taking the evidence with it.
    handlers = (("SIGTERM", True), ("SIGUSR1", False), ("SIGUSR2", False))
    registered = []
    for name, chain in handlers:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            faulthandler.register(sig, all_threads=True, chain=chain)
        except Exception:  # pragma: no cover - platform dependent
            continue
        registered.append(f"{name}{'' if chain else '(no-chain)'}")
    _report(
        "[W0-RUNTIME] stall tracebacks enabled: fatal signals + "
        f"{'/'.join(registered) or 'none'} (kill -USR1 <pid> to dump a hung rank)"
    )


def _enable_remote_debugger() -> None:
    """Let a VS Code debugger attach to one training rank, if asked.

    Single-process debugging of the causal training is not possible: ZeRO-1 does not shard at
    ``world_size=1``, so one rank needs 92.4 GiB and OOMs on a 95.6 GiB card. Attaching to a
    live rank is the only way to step through the training path.

    Gated on ``LOCAL_RANK`` so exactly one process binds the port. The elastic agent and the
    conda activation helpers have no ``LOCAL_RANK``, and eight ranks binding one port would
    fail for seven of them.

    ``W0_DEBUGPY_WAIT=1`` blocks the rank until a client attaches. Use it only when the
    breakpoint is before the first collective: while one rank waits, the other seven sit in
    the gradient allreduce and the 600 s NCCL watchdog will kill the job.
    """

    port_spec = os.environ.get("W0_DEBUGPY", "").strip()
    if not port_spec:
        return
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    if local_rank != os.environ.get("W0_DEBUGPY_RANK", "0").strip():
        return
    try:
        port = int(port_spec)
    except ValueError:
        _report(f"[W0-RUNTIME] W0_DEBUGPY={port_spec!r} is not a port; debugger not started")
        return
    try:
        import debugpy

        debugpy.listen(("0.0.0.0", port))
    except Exception as exc:  # pragma: no cover - depends on the runtime image
        _report(f"[W0-RUNTIME] debugpy.listen({port}) failed: {type(exc).__name__}: {exc}")
        return
    _report(f"[W0-RUNTIME] debugpy listening on 0.0.0.0:{port} for LOCAL_RANK={local_rank}")
    if os.environ.get("W0_DEBUGPY_WAIT", "").strip() not in ("", "0"):
        _report("[W0-RUNTIME] blocking until a debugger attaches (W0_DEBUGPY_WAIT set)")
        debugpy.wait_for_client()
        _report("[W0-RUNTIME] debugger attached")


_configure_libero_assets()
_configure_cuda_home()
_patch_egl_cleanup()
_patch_render_context_current()
_patch_libero_render_truth()
_patch_libero_num_steps_wait()
_enable_stall_tracebacks()
_enable_remote_debugger()
