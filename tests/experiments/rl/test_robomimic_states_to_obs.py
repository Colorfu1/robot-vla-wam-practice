"""Lifecycle regression tests; no renderer or GPU required."""

import sys
from types import SimpleNamespace

import pytest

from robot_practice.experiments.rl.robomimic_states_to_obs import managed_environments


@pytest.mark.parametrize("fail_conversion", [False, True])
def test_closes_cyclic_environment_before_exit_and_restores_factory(fail_conversion):
    events = []
    native = SimpleNamespace(close=lambda: events.append("close"))
    env = SimpleNamespace(env=native)
    native.owner = env  # Real renderers and simulators also have reference cycles.
    factory = lambda **kwargs: env
    utils = SimpleNamespace(create_env_for_data_processing=factory)

    def convert():
        with managed_environments(utils):
            assert utils.create_env_for_data_processing(camera_height=224) is env
            events.append("convert")
            if fail_conversion:
                raise ValueError("conversion failed")

    if fail_conversion:
        with pytest.raises(ValueError, match="conversion failed"):
            convert()
    else:
        convert()
    events.append("terminate display")
    assert events == ["convert", "close", "terminate display"]
    assert utils.create_env_for_data_processing is factory


@pytest.mark.parametrize("unraisable", [False, True])
def test_cleanup_failure_is_not_reported_as_success(unraisable, monkeypatch):
    def close():
        error = RuntimeError("EGL cleanup failed")
        if unraisable:
            sys.unraisablehook(SimpleNamespace(exc_value=error))
        else:
            raise error

    monkeypatch.setattr(sys, "unraisablehook", lambda event: None)
    hook = sys.unraisablehook
    utils = SimpleNamespace(
        create_env_for_data_processing=lambda: SimpleNamespace(env=SimpleNamespace(close=close))
    )
    with pytest.raises(RuntimeError, match="cleanup failed"), managed_environments(utils):
        utils.create_env_for_data_processing()
    assert sys.unraisablehook is hook


def test_original_conversion_failure_survives_cleanup_failure():
    def close():
        raise RuntimeError("cleanup failed")

    utils = SimpleNamespace(
        create_env_for_data_processing=lambda: SimpleNamespace(env=SimpleNamespace(close=close))
    )
    with pytest.raises(ValueError, match="conversion failed"), managed_environments(utils):
        utils.create_env_for_data_processing()
        raise ValueError("conversion failed")


def test_remaining_environments_close_after_one_cleanup_fails():
    events = []

    def fail():
        events.append("failed close")
        raise RuntimeError("failed close")

    envs = iter([
        SimpleNamespace(close=lambda: events.append("successful close")),
        SimpleNamespace(close=fail),
    ])
    utils = SimpleNamespace(create_env_for_data_processing=lambda: next(envs))
    with pytest.raises(RuntimeError, match="cleanup failed"), managed_environments(utils):
        utils.create_env_for_data_processing()
        utils.create_env_for_data_processing()
    assert events == ["failed close", "successful close"]


def test_entrypoint_preserves_official_cli_and_closes_environment(monkeypatch, capsys):
    from robot_practice.experiments.rl import robomimic_states_to_obs as entry

    events = []
    utils = SimpleNamespace(
        create_env_for_data_processing=lambda: SimpleNamespace(close=lambda: events.append("close"))
    )
    monkeypatch.setitem(sys.modules, "robomimic", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "robomimic.utils", SimpleNamespace(env_utils=utils))
    argv = ["entry", "--n", "2", "--camera_height", "224"]
    monkeypatch.setattr(sys, "argv", argv)

    def run_module(module, run_name):
        assert module == "robomimic.scripts.dataset_states_to_obs"
        assert run_name == "__main__"
        assert sys.argv is argv
        utils.create_env_for_data_processing()
        events.append("convert")

    monkeypatch.setattr(entry.runpy, "run_module", run_module)
    entry.main()
    assert events == ["convert", "close"]
    assert "DONE phase=conversion_and_cleanup" in capsys.readouterr().out
