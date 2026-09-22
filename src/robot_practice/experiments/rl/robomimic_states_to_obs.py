"""Run the installed robomimic converter with deterministic environment cleanup.

Accepts the installed dataset_states_to_obs CLI unchanged. Use the repository
launcher run_square_states_to_obs.sh for a persistent log and the Square smoke.
Only the environment factory is temporarily wrapped; package files are untouched.
This entrypoint is single-threaded and intended for robosuite environments.
"""

import runpy
import sys
from contextlib import contextmanager


@contextmanager
def managed_environments(env_utils):
    """Close owned environments before Python executes EGL's atexit callbacks."""
    factory = env_utils.create_env_for_data_processing
    environments = []

    def create(*args, **kwargs):
        env = factory(*args, **kwargs)
        environments.append(env)
        return env

    env_utils.create_env_for_data_processing = create
    try:
        yield
    finally:
        conversion_failed = sys.exc_info()[0] is not None
        env_utils.create_env_for_data_processing = factory
        errors = []
        original_hook = sys.unraisablehook

        def capture_unraisable(event):
            # MjSim.free() invokes destructors: those errors otherwise leave exit code 0.
            errors.append(event.exc_value)
            original_hook(event)

        sys.unraisablehook = capture_unraisable
        try:
            total = len(environments)
            for completed, env in enumerate(reversed(environments), 1):
                try:
                    native = getattr(env, "env", env)
                    native.close()
                except Exception as exc:  # noqa: BLE001 - finish cleanup, then fail the entrypoint
                    errors.append(exc)
                print(
                    f"STATUS phase=cleanup completed={completed} total={total} "
                    f"errors={len(errors)}",
                    flush=True,
                )
        finally:
            sys.unraisablehook = original_hook
        if errors:
            print(f"FAILED phase=cleanup errors={len(errors)}", file=sys.stderr, flush=True)
            if not conversion_failed:
                raise RuntimeError("Environment cleanup failed; see cleanup log") from errors[0]


def main():
    from robomimic.utils import env_utils

    with managed_environments(env_utils):
        runpy.run_module("robomimic.scripts.dataset_states_to_obs", run_name="__main__")
    print("DONE phase=conversion_and_cleanup", flush=True)


if __name__ == "__main__":
    main()
