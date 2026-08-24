from __future__ import annotations

import importlib
import json
import os
import subprocess
from typing import Sequence
import sys
from pathlib import Path
from typing import Any

import pytest

from robot_practice.experiments.literature.w0_fastwam_audit import (
    DEFAULT_CONFIG,
    _artifact_paths,
    _collect_lerobot_metrics,
    _egl_device_map,
    eval_causal_pair,
    _lerobot_worker_command,
    _model_runtime_env,
    _parse_egl_device_map,
    _assert_sibling_budget_matches,
    _dataset_fingerprint,
    _training_debug_env,
    _pair_contract,
    _prepare_lerobot_policy_overlay,
    _training_allocator_env,
    _prepare_output_directory,
    _portable,
    _upstream_worker_specs,
    _visible_gpu_ids,
    _worker_environment,
    _worker_eval_command,
    build_parser,
    load_config,
    main,
    train_causal_variant,
    validate_artifacts,
    validate_config,
    validate_environments,
)


def test_w0_config_matches_frozen_protocol() -> None:
    config = load_config(DEFAULT_CONFIG)
    result = validate_config(config)
    assert result["valid"] is True
    assert result["fixed_protocol"] == {
        "tasks": 40,
        "demos": 2000,
        "action": "7D EEF pose + gripper",
        "state": "8D proprio",
        "horizon": 32,
        "execute": 10,
    }
    assert result["causal_variants"] == {"with_video": 1.0, "no_video": 0.0}


def test_w0_config_rejects_a_changed_causal_budget() -> None:
    config = load_config(DEFAULT_CONFIG)
    config["training"]["global_batch_size"] = 127
    with pytest.raises(ValueError, match="global batch"):
        validate_config(config)


def test_model_tree_uses_the_upstream_diffusion_synth_layout() -> None:
    config = load_config(DEFAULT_CONFIG)
    paths = _artifact_paths(config)
    assert paths["model_base"].as_posix().endswith("pretrained/w0_fastwam/models")
    assert paths["wan"].as_posix().endswith("Wan-AI/Wan2.2-TI2V-5B")
    assert paths["tokenizer"].as_posix().endswith("Wan-AI/Wan2.1-T2V-1.3B")
    assert paths["common"].as_posix().endswith(
        "DiffSynth-Studio/Wan-Series-Converted-Safetensors"
    )
    runtime_env = _model_runtime_env(paths)
    assert runtime_env["DIFFSYNTH_MODEL_BASE_PATH"].endswith("pretrained/w0_fastwam/models")
    assert runtime_env["W0_LIBERO_ASSETS"].endswith("pretrained/w0_fastwam/libero_assets")
    assert "w0_runtime" in runtime_env["PYTHONPATH"]
    assert runtime_env["HF_HUB_OFFLINE"] == "1"
    assert runtime_env["TRANSFORMERS_OFFLINE"] == "1"


def test_cli_help_renders_without_percent_formatting_errors() -> None:
    """--help must not crash.

    argparse runs every help string through % formatting, so a literal "5% warmup" raises
    ValueError: unsupported format character 'w'. That shipped unnoticed because --help was
    never exercised, and it makes the tool undiscoverable at exactly the moment someone needs
    to look up an option.
    """

    import argparse

    from robot_practice.experiments.literature import w0_fastwam_audit as audit

    original = argparse.ArgumentParser.parse_args
    captured: dict[str, argparse.ArgumentParser] = {}

    def capture(self: argparse.ArgumentParser, *args: object, **kwargs: object):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture  # type: ignore[method-assign]
    try:
        with pytest.raises(SystemExit):
            audit.main([])
    finally:
        argparse.ArgumentParser.parse_args = original  # type: ignore[method-assign]

    parser = captured.get("parser")
    assert parser is not None, "main() did not construct an ArgumentParser"
    rendered = parser.format_help()
    assert "--train-num-workers" in rendered
    assert "--train-epochs" in rendered


def test_train_num_workers_is_forwarded_and_recorded(monkeypatch, tmp_path: Path) -> None:
    """The host-RAM lever must reach the training command and the manifest.

    Four runs died to a cgroup OOM: a DataLoader worker was SIGKILLed, its rank blocked in
    queue.get, and the other seven spun in DeepSpeed's gradient allreduce until the 600 s
    NCCL watchdog killed the job. num_workers is the lever, and it is numerically inert
    because the sampler order comes from the seed, so batches stay bit-identical.
    """

    from robot_practice.experiments.literature import w0_fastwam_audit as audit

    seen: dict[str, object] = {}

    def fake_train(*args: object, **kwargs: object) -> dict[str, object]:
        seen.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(audit, "train_causal_variant", fake_train)
    source = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )

    # The override actually handed to Hydra, and the manifest field that records it.
    override = "num_workers={train_num_workers if train_num_workers is not None else"
    assert override in source
    assert '"num_workers": (' in source
    # And the argparse value must reach the function; a missing dispatch line silently
    # discards the option, which is exactly how --train-epochs was lost for a whole run.
    assert "train_num_workers=args.train_num_workers," in source


def test_stall_watcher_attributes_anonymous_memory_by_process_role() -> None:
    """Growth must be attributable to ranks or to workers, and ps RSS cannot do it.

    Workers are forked from their rank and share its pages copy-on-write, so ps counts the
    same memory once per process: 64 workers each reported 34 GiB while the cgroup held
    1368 GiB in total. Pss_Anon divides every shared page by its number of sharers, so
    summing across processes gives the real total and the rank/worker split is meaningful.
    Without that split there is no way to tell whether lowering num_workers can help.
    """

    body = Path("scripts/experiments/literature/watch_w0_stall.sh").read_text(encoding="utf-8")
    assert "anon_kb()" in body
    assert "report_attribution()" in body
    # Share-adjusted metric, not raw RSS.
    assert "Pss_Anon:" in body
    assert "smaps_rollup" in body
    # Older kernels lack smaps_rollup; the fallback must still show the trend.
    assert "RssAnon:" in body
    assert "share-adjusted" in body
    # The split relies on the same depth rule that picks ranks for the stack dump.
    assert "mapfile -t ranks < <(gpu_pids)" in body
    assert "mapfile -t all < <(candidate_pids)" in body
    assert "report_attribution" in body


def test_trial_shard_layout_cannot_silently_drop_trials() -> None:
    """Sharding a task's trials requires evaluator support the image does not ship.

    ``+EVALUATION.trial_start=N`` is a Hydra addition. The pinned upstream evaluator never
    reads the key, so it is silently ignored: every shard would evaluate
    ``init_states[0:count]`` and the later trials would never run. Two earlier tests asserted
    that the vendored file contained the patch, and passed only because a local scratch
    checkout was dirty -- the image is clean at the pinned commit and has neither the
    ``trial_start`` support nor the render diagnostics.

    The recorded UP-REL result is unaffected: it used
    ``trials_per_worker == trials_per_task == 50``, so every task ran in one child with
    ``trial_start = 0``, and its summary shows one source per task, 50 unique trial indices
    and ``missing_trial_count = 0``.

    The aggregate check would eventually catch a bad layout, but only after hours of
    evaluation, so the launcher must refuse it up front.
    """

    from robot_practice.experiments.literature.w0_fastwam_audit import (
        _assert_vendor_supports_trial_start,
    )

    vendor = Path("third_party/w0_fastwam")
    evaluator = vendor / "FastWAM/experiments/libero/eval_libero_single.py"
    if not evaluator.is_file():
        pytest.skip("pinned FastWAM checkout is not present")

    # One child per task keeps trial_start at 0, so the missing support cannot matter.
    _assert_vendor_supports_trial_start(vendor, trials=50, per_worker=50)
    _assert_vendor_supports_trial_start(vendor, trials=50, per_worker=99)

    supported = 'cfg.EVALUATION.get("trial_start"' in evaluator.read_text(encoding="utf-8")
    if supported:
        # The patch is applied in this checkout, so sharding is legitimate.
        _assert_vendor_supports_trial_start(vendor, trials=50, per_worker=1)
    else:
        with pytest.raises(RuntimeError) as excinfo:
            _assert_vendor_supports_trial_start(vendor, trials=50, per_worker=1)
        message = str(excinfo.value)
        assert "EVALUATION.trial_start" in message
        # The message must name both ways out, or the operator has to guess.
        assert "--trials-per-worker=50" in message
        assert "vendor-patches" in message

    source = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )
    # The guard has to run before any child is spawned.
    assert "_assert_vendor_supports_trial_start(\n        vendor_root," in source

    # The patches themselves must be tracked, so a future image build can apply them instead
    # of the content living only in an untracked checkout.
    patches = Path("docker/training/vendor-patches")
    assert (patches / "fastwam-eval-trial-start-and-render-diagnostics.patch").is_file()
    assert (patches / "lerobot-local-wan-loader.patch").is_file()


def test_dataset_memory_probe_isolates_and_attributes_the_leak(tmp_path: Path) -> None:
    """A single-process reproducer, because every cheap hypothesis is already dead.

    Ruled out by measurement: page cache (memory.stat put 1368 GiB in rss against 35 GiB of
    cache), the shared-memory tensor transport (/dev/shm held 4.8 GB), the worker count
    (8 -> 2 left the slope unchanged while per-rank stayed flat), and glibc arena
    fragmentation (MALLOC_ARENA_MAX with a pinned low mmap threshold also left it unchanged).
    Ruled out by reading: per-sample state on the dataset, a retained view of the decoded
    tensor in the returned dict, and a missing container close.

    The probe therefore has to do two things reading cannot: exercise one layer at a time,
    and separate Python-visible retention from retention below Python. A live-tensor census
    that grows means a Python reference and names the shape; a flat census under a climbing
    RssAnon means no amount of dataset-code reading will find it.
    """

    script = Path("scripts/experiments/literature/probe_w0_dataset_memory.py")
    assert script.is_file()
    assert os.access(script, os.X_OK)
    body = script.read_text(encoding="utf-8")

    # Anonymous RSS is the figure the cgroup blamed, so it must be the one reported.
    assert "RssAnon" in body
    assert "def _rss_kb()" in body
    # The Python-vs-below-Python discriminator.
    assert "def _tensor_census()" in body
    assert "gc.get_objects()" in body
    assert "untyped_storage().nbytes()" in body
    # gc.collect() before the census, so "not collected yet" cannot be misread as retention.
    assert "collected = gc.collect()" in body
    assert "gc freed" in body
    # Naming the holder is the point: the retained group includes intermediate tensors, which
    # only a live frame explains, so referrers must be resolved to a code name.
    assert "def _name_referrers(" in body
    assert 'f"frame:{holder.f_code.co_name}' in body
    assert "def _largest_retained_tensor()" in body
    # And the retry rate should be comparable to the leak rate if __getitem__ is the source.
    assert "_get raised on" in body

    # With gc.collect() the growth vanishes and _get raises on 0 of 120 samples, so it is
    # uncollected cycle garbage rather than a leak or a retry. Naming the cycle needs
    # DEBUG_SAVEALL, and it must not be preceded by a collect: that would free exactly the
    # cycles being named, which is how the first version reported zero.
    assert "--cycles" in body
    assert "gc.set_debug(gc.DEBUG_SAVEALL)" in body
    assert "No pre-collect" in body
    saveall = body.index("gc.set_debug(gc.DEBUG_SAVEALL)")
    window = body[max(0, saveall - 400) : saveall]
    assert "gc.collect()" not in window, "a collect precedes DEBUG_SAVEALL again"
    assert "gc.garbage" in body
    assert "cycle members" in body

    # The named cycle contains decode_video_frames_torchcodec's frame plus exactly two
    # tracebacks per sample -- one per camera -- because find_spec("torchcodec") succeeds
    # while its shared library cannot load. Forcing the backend must therefore be testable,
    # and it has to be applied after construction since BaseLerobotDataset builds
    # MultiLeRobotDataset without forwarding video_backend.
    assert '"--video-backend"' in body
    assert 'choices=(None, "pyav", "torchcodec")' in body
    assert 'getattr(multi, "_datasets", [])' in body
    assert 'target.video_backend = args.video_backend' in body
    # Layer attribution.
    assert 'choices=("full", "lerobot", "text")' in body
    # run_training registers a work dir before building datasets, and the dataset writes
    # dataset_stats.json there after a full 1712-episode normalization pass. Without the
    # registration the default ./runs does not exist and instantiation dies with
    # FileNotFoundError only after that pass has already been paid for.
    assert "misc.register_work_dir(work_dir)" in body
    assert "work_dir.mkdir(parents=True, exist_ok=True)" in body
    assert '"--work-dir"' in body

    # It must instantiate only the dataset, not the model or DeepSpeed.
    assert "instantiate(cfg.data.train)" in body
    # No model, no DeepSpeed, no NCCL: mentioning them in the docstring is fine, importing
    # them would reintroduce the eight-GPU cost the probe exists to avoid.
    imports = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith(("import ", "from ")) or "importlib" in line
    ]
    for forbidden in ("deepspeed", "accelerate", "torch.distributed"):
        assert not any(forbidden in line for line in imports), forbidden
    assert "instantiate(cfg.model" not in body

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script), *args], capture_output=True, text=True, check=False
        )

    # Bad paths must fail fast with a named cause rather than part-way through a long run.
    missing_vendor = run("--vendor-root", str(tmp_path))
    assert missing_vendor.returncode == 2
    assert "no FastWAM/configs" in missing_vendor.stderr

    assert "--samples" in run("--help").stdout
    assert "--stage" in run("--help").stdout


def test_stall_watcher_reports_host_memory() -> None:
    """The watcher must sample cgroup memory, not only Python stacks.

    Host RAM was never measured across four crashes; GPU memory was, and it was fine at
    64.2 of 95.6 GiB per rank. The growth shape is what picks the fix: a plateau means the
    prefetch buffers are oversized for 64 workers, a straight line means the pyav fallback
    decoder leaks and lowering num_workers would only postpone the OOM.
    """

    body = Path("scripts/experiments/literature/watch_w0_stall.sh").read_text(encoding="utf-8")
    assert "report_memory()" in body
    # Both cgroup generations, because v2 paths are absent on the training host.
    assert "/sys/fs/cgroup/memory.current" in body
    assert "/sys/fs/cgroup/memory/memory.usage_in_bytes" in body
    assert "/sys/fs/cgroup/memory/memory.limit_in_bytes" in body
    # failcnt/oom_kill is the counter that proved the OOM; it must be surfaced.
    assert "memory.failcnt" in body
    assert "oom/failcnt" in body
    # Sampled on every step change so the curve survives the crash.
    assert body.count('report_memory "${step}"') >= 2


def test_stall_watcher_dumps_without_killing_the_run() -> None:
    """The watcher must observe a stall, not end it.

    Its first version did both, and the combination destroyed a live run: SIGUSR1's default
    action is Term (signal(7)), faulthandler was registered with chain=True, and
    ``pgrep -f scripts/train.py`` matched 65 processes because DataLoader workers are forked
    from the ranks and share their cmdline. Signalling all of them killed 57 workers, whose
    deaths raised DataLoader's SIGCHLD handler in every rank, and the run was gone 15 s after
    the dump -- so the stall evidence went with it.
    """

    script = Path("scripts/experiments/literature/watch_w0_stall.sh")
    assert script.is_file()
    assert os.access(script, os.X_OK)
    body = script.read_text(encoding="utf-8")

    # DataLoader workers must be excluded by depth in the candidate process tree. Three
    # earlier discriminators all failed:
    #   * cmdline alone matched 65 processes (workers are forked and share the cmdline),
    #     and signalling them killed a run;
    #   * "parent is also a candidate" dropped the ranks along with the workers and kept
    #     only the agent, whose stack is a monitoring loop;
    #   * nvidia-smi --query-compute-apps returns nothing (or host-namespace pids) in this
    #     container, so a CUDA-context requirement found zero ranks and exited at once.
    assert "gpu_pids()" in body
    assert "depth=$(( depth + 1 ))" in body
    assert "if (( depth <= 1 )); then" in body
    assert "workers excluded" in body
    # The rank lookup must not depend on nvidia-smi at all.
    executable = "\n".join(
        line for line in body.splitlines() if not line.lstrip().startswith("#")
    )
    assert "nvidia-smi" not in executable, "nvidia-smi reintroduced into executable code"

    # Liveness and rank identity are separate questions: during the ~10 minute model load
    # the ranks exist but hold no CUDA context and emit no step line. Conflating them made
    # the watcher exit immediately when started right after launch.
    assert "candidate_pids()" in body
    assert 'if [[ -z "${step}" ]]; then' in body
    assert "do not treat as finished" in body

    # It must refuse to run at all against a chaining SIGUSR1 handler.
    assert "grep -q '(\"SIGUSR1\", False)'" in body
    assert "exit 4" in body

    # Concurrent dumps to one stderr are unreadable; ranks are signalled one at a time.
    assert "sleep 2" in body
    assert "Serialize" in body

    # Kernel-side state discriminates causes that look identical in Python: D means blocked
    # in a syscall (storage/network IO), S means waiting on a lock, R means spinning.
    assert "report_kernel_state()" in body
    assert '/proc/${pid}/status' in body
    assert '/proc/${pid}/wchan' in body
    assert 'in D state wchan=' in body

    # Two passes: one snapshot cannot separate "wedged at a frame" from "far too slow".
    assert "for pass in 1 2; do" in body
    assert "waiting 60s before the second pass" in body

    sitecustomize = Path(
        "src/robot_practice/experiments/literature/w0_runtime/sitecustomize.py"
    ).read_text(encoding="utf-8")
    # SIGTERM keeps chain=True so torch elastic's teardown still proceeds; the on-demand
    # signals must not chain, because their default action would kill the rank.
    assert '("SIGTERM", True), ("SIGUSR1", False), ("SIGUSR2", False)' in sitecustomize
    assert "faulthandler.register(sig, all_threads=True, chain=chain)" in sitecustomize

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *args], capture_output=True, text=True, check=False
        )

    assert run().returncode == 2
    assert "usage:" in run().stderr
    for bad in (("/tmp/x.log", "0"), ("/tmp/x.log", "150", "abc"), ("/tmp/x.log", "-5")):
        result = run(*bad)
        assert result.returncode == 2, bad
        assert "positive integers" in result.stderr, bad


def test_stall_watcher_refuses_a_chaining_sigusr1_handler(tmp_path: Path) -> None:
    """Guard the guard: flipping SIGUSR1 back to chain=True must block the watcher.

    Without this, a future edit could silently restore the configuration that killed a run,
    and the watcher would once again terminate the ranks it was meant to inspect.
    """

    sitecustomize = Path(
        "src/robot_practice/experiments/literature/w0_runtime/sitecustomize.py"
    )
    original = sitecustomize.read_text(encoding="utf-8")
    assert '("SIGUSR1", False)' in original
    try:
        sitecustomize.write_text(
            original.replace('("SIGUSR1", False)', '("SIGUSR1", True)', 1),
            encoding="utf-8",
        )
        blocked = subprocess.run(
            ["bash", "scripts/experiments/literature/watch_w0_stall.sh", str(tmp_path / "t.log")],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        sitecustomize.write_text(original, encoding="utf-8")

    assert blocked.returncode == 4, blocked
    assert "default action is" in blocked.stderr
    assert "Sync the no-chain fix first" in blocked.stderr


def test_w0_sitecustomize_keeps_stdout_clean_for_command_substitution() -> None:
    """sitecustomize diagnostics must never reach stdout.

    ``conda run`` builds a temporary activation script and ``eval``s the output of a helper
    ``python`` invocation. PYTHONPATH already contains this directory, so that helper
    imports this module; anything printed to stdout is swallowed by the eval. A message
    containing parentheses -- "(kill -USR1 <pid> ...)" -- aborted the activation with
    "syntax error near unexpected token '('", conda fell back to the wrong environment, and
    an 8-GPU training run died with "DeepSpeed is not installed" because accelerate launched
    from robot-practice instead of fastwam-upstream.

    This reproduces that shape directly: capture only stdout, as command substitution does,
    and require it to contain the payload and nothing else.
    """

    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(runtime)
    env["PYTHONNOUSERSITE"] = "1"
    # Force the noisiest code path: the CUDA_HOME fallback announces itself.
    env["W0_CUDA_HOME_FALLBACK"] = "1"
    env.pop("CUDA_HOME", None)

    result = subprocess.run(
        [sys.executable, "-c", "print('2026-08-20_07-57-10')"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    # Exactly what `RUN_ID="$(python - <<PY ... PY)"` would capture.
    captured = result.stdout.strip()
    assert captured == "2026-08-20_07-57-10", f"stdout polluted: {captured!r}"
    assert "[W0-RUNTIME]" not in result.stdout
    assert "(" not in result.stdout

    # The diagnostics must still be visible -- they land in train.log via stderr.
    assert "[W0-RUNTIME]" in result.stderr
    assert "stall tracebacks enabled" in result.stderr

    source = (runtime / "sitecustomize.py").read_text(encoding="utf-8")
    # Every diagnostic must route through the stderr helper; no bare print() allowed.
    assert "def _report(message: str) -> None:" in source
    assert 'sys.stderr.write(f"{message}\\n")' in source
    bare_prints = [
        line
        for line in source.splitlines()
        if line.strip().startswith("print(")
    ]
    assert not bare_prints, f"stdout writes reintroduced: {bare_prints}"


def test_w0_sitecustomize_registers_stall_tracebacks(tmp_path: Path) -> None:
    """A hung rank must print its own Python stack instead of dying silently.

    Two 8-GPU runs died to a 600 s NCCL ALLREDUCE timeout with exactly one rank absent from
    the watchdog reports -- step 260/rank 3, then step 590/rank 1. That rank never enqueued
    the collective, so it was stuck in host-side code, and NCCL FlightRecorder cannot help
    because it only records collectives. faulthandler is stdlib, needs no network, and
    installs a C-level handler that walks interpreter state without the GIL, which is
    precisely what makes it usable on a hang.
    """

    source = Path(
        "src/robot_practice/experiments/literature/w0_runtime/sitecustomize.py"
    ).read_text(encoding="utf-8")

    # Fatal signals cover the ranks that SIGABRT at the timeout; SIGTERM covers the stuck
    # rank that torch elastic tears down; SIGUSR1 allows an on-demand dump in the window.
    assert "faulthandler.enable(all_threads=True)" in source
    assert 'handlers = (("SIGTERM", True), ("SIGUSR1", False), ("SIGUSR2", False))' in source
    assert "faulthandler.register(sig, all_threads=True, chain=chain)" in source
    # SIGTERM chains so torch elastic's teardown still proceeds; SIGUSR1/SIGUSR2 must not,
    # because their default action is Term and chaining would kill the rank being inspected.
    assert '("SIGUSR1", False)' in source and '("SIGUSR2", False)' in source
    # It must announce itself: the FlightRecorder gap was the absence of positive evidence.
    assert "[W0-RUNTIME] stall tracebacks enabled" in source
    # And it must be switchable off without editing the file.
    assert 'os.environ.get("W0_STALL_TRACEBACKS", "1") == "0"' in source
    assert "_enable_stall_tracebacks()" in source.split("_patch_libero_num_steps_wait()")[-1]

    probe = tmp_path / "probe.py"
    probe.write_text("print('probe-ok')\n", encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(
        Path("src/robot_practice/experiments/literature/w0_runtime").resolve()
    )
    env["PYTHONNOUSERSITE"] = "1"
    env["W0_CUDA_HOME_FALLBACK"] = "0"
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, env=env, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "probe-ok", "diagnostics must not pollute stdout"
    assert "[W0-RUNTIME] stall tracebacks enabled" in result.stderr
    assert "SIGTERM" in result.stderr and "SIGUSR1" in result.stderr

    # Opting out must be silent and must not break startup.
    env["W0_STALL_TRACEBACKS"] = "0"
    disabled = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, env=env, check=False
    )
    assert disabled.returncode == 0, disabled.stderr
    assert "stall tracebacks enabled" not in disabled.stderr


def _payload_lines(stdout: str) -> str:
    """Drop sitecustomize's own [W0-RUNTIME] diagnostics from a probe's stdout.

    sitecustomize deliberately announces what it activated (CUDA_HOME fallback, stall
    tracebacks, the effective hard-reset value) so a run's log carries positive evidence
    that a numerically-inert safeguard is on -- a gap that previously cost a debugging
    cycle. These probes assert on the payload their own snippet printed, so the
    diagnostics are filtered rather than suppressed at the source.
    """

    return "\n".join(
        line
        for line in stdout.strip().splitlines()
        if not line.startswith("[W0-RUNTIME]")
    ).strip()


def test_w0_sitecustomize_is_nonfatal_before_libero_environment(tmp_path: Path) -> None:
    assets = tmp_path / "libero_assets"
    scene = assets / "scenes" / "libero_tabletop_base_style.xml"
    scene.parent.mkdir(parents=True)
    scene.write_text("<mujoco/>", encoding="utf-8")
    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runtime)
    environment["W0_LIBERO_ASSETS"] = str(assets)
    result = subprocess.run(
        [sys.executable, "-c", "print('sitecustomize-startup-ok')"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert _payload_lines(result.stdout) == "sitecustomize-startup-ok"
    assert _payload_lines(result.stderr) == ""


def test_w0_sitecustomize_injects_assets_when_libero_is_available(tmp_path: Path) -> None:
    assets = tmp_path / "libero_assets"
    scene = assets / "scenes" / "libero_tabletop_base_style.xml"
    scene.parent.mkdir(parents=True)
    scene.write_text("<mujoco/>", encoding="utf-8")

    fake_packages = tmp_path / "fake_packages"
    libero_package = fake_packages / "libero"
    libero_package.mkdir(parents=True)
    (libero_package / "__init__.py").write_text("", encoding="utf-8")
    (libero_package / "libero.py").write_text(
        "_assets_path_cache = None\n"
        "def get_assets_path():\n"
        "    return _assets_path_cache\n",
        encoding="utf-8",
    )

    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(runtime), str(fake_packages)))
    environment["W0_LIBERO_ASSETS"] = str(assets)
    result = subprocess.run(
        [sys.executable, "-c", "import libero.libero as m; print(m.get_assets_path())"],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert _payload_lines(result.stdout) == str(assets.resolve())
    assert _payload_lines(result.stderr) == ""


def test_w0_sitecustomize_suppresses_only_late_egl_teardown_error(tmp_path: Path) -> None:
    fake_packages = tmp_path / "fake_packages"
    for package in (
        fake_packages / "OpenGL" / "raw" / "EGL",
        fake_packages / "robosuite" / "renderers" / "context",
    ):
        package.mkdir(parents=True)
        for parent in [package, *package.parents]:
            if parent == tmp_path:
                break
            init = parent / "__init__.py"
            if parent.is_relative_to(fake_packages) and not init.exists():
                init.write_text("", encoding="utf-8")
    (fake_packages / "OpenGL" / "EGL.py").write_text(
        "EGL_NOT_INITIALIZED = object()\n", encoding="utf-8"
    )
    (fake_packages / "OpenGL" / "raw" / "EGL" / "_errors.py").write_text(
        "class EGLError(Exception):\n"
        "    def __init__(self, err):\n"
        "        self.err = err\n"
        "        super().__init__('EGL_NOT_INITIALIZED')\n",
        encoding="utf-8",
    )
    (fake_packages / "robosuite" / "renderers" / "context" / "egl_context.py").write_text(
        "from OpenGL.raw.EGL._errors import EGLError\n"
        "from OpenGL import EGL\n"
        "class EGLGLContext:\n"
        "    def __init__(self):\n"
        "        self._context = object()\n"
        "    def free(self):\n"
        "        raise EGLError(EGL.EGL_NOT_INITIALIZED)\n",
        encoding="utf-8",
    )

    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()
    environment = os.environ.copy()
    environment.pop("W0_LIBERO_ASSETS", None)
    environment["PYTHONPATH"] = os.pathsep.join((str(runtime), str(fake_packages)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from robosuite.renderers.context.egl_context import EGLGLContext; "
            "ctx = EGLGLContext(); ctx.free(); print(ctx._context is None)",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert _payload_lines(result.stdout) == "True"
    assert _payload_lines(result.stderr) == ""


def test_w0_sitecustomize_refuses_a_wrong_libero_asset_root(tmp_path: Path) -> None:
    """A missing asset root must stop the run, not fall back to packaged scenes.

    ``site.execsitecustomize()`` swallows exceptions, so the previous
    ``RuntimeError`` let the interpreter continue with LIBERO's own assets and
    silently changed the evaluated environment.
    """

    empty = tmp_path / "no_assets"
    empty.mkdir()
    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(runtime)
    environment["W0_LIBERO_ASSETS"] = str(empty)
    result = subprocess.run(
        [sys.executable, "-c", "print('body-should-not-run')"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result
    assert "body-should-not-run" not in result.stdout
    assert "missing the required LIBERO scene" in result.stderr


def test_w0_sitecustomize_makes_hard_reset_authoritative_and_report_truthful(
    tmp_path: Path,
) -> None:
    """The evaluator's own log echoed W0_HARD_RESET instead of the applied value.

    A vendored ``libero_utils.get_libero_env`` that never forwards ``hard_reset``
    leaves robosuite on its ``hard_reset=True`` default while the log still shows
    ``hard_reset=0``.  sitecustomize must apply the variable itself and report the
    value that actually took effect, plus the EGL device index in force.
    """

    fake_packages = tmp_path / "fake_packages"
    base = fake_packages / "robosuite" / "environments"
    base.mkdir(parents=True)
    (fake_packages / "robosuite" / "__init__.py").write_text("", encoding="utf-8")
    (base / "__init__.py").write_text("", encoding="utf-8")
    (base / "base.py").write_text(
        "class MujocoEnv:\n"
        "    def __init__(self, *, hard_reset=True, **kwargs):\n"
        "        self.hard_reset = hard_reset\n",
        encoding="utf-8",
    )

    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()

    def run(hard_reset_value: str | None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("W0_LIBERO_ASSETS", None)
        environment.pop("W0_GL_CONTEXT_GUARD", None)
        environment["PYTHONPATH"] = os.pathsep.join((str(runtime), str(fake_packages)))
        environment["CUDA_VISIBLE_DEVICES"] = "7,0"
        environment["MUJOCO_EGL_DEVICE_ID"] = "0"
        if hard_reset_value is None:
            environment.pop("W0_HARD_RESET", None)
        else:
            environment["W0_HARD_RESET"] = hard_reset_value
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from robosuite.environments.base import MujocoEnv;"
                "print('applied', MujocoEnv(hard_reset=True).hard_reset)",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )

    disabled = run("0")
    assert disabled.returncode == 0, disabled.stderr
    assert "applied False" in disabled.stdout
    assert "requested_hard_reset=0" in disabled.stderr
    assert "effective_hard_reset=False" in disabled.stderr
    assert "mujoco_egl_device_id=0" in disabled.stderr
    assert "cuda_visible=7,0" in disabled.stderr
    assert "egl_device_count=" in disabled.stderr

    default = run(None)
    assert default.returncode == 0, default.stderr
    assert "applied True" in default.stdout
    assert "requested_hard_reset=unset" in default.stderr
    assert "effective_hard_reset=True" in default.stderr

    # site.execsitecustomize() swallows exceptions, so a misconfigured value
    # must stop the interpreter explicitly instead of disabling the patch.
    malformed = run("maybe")
    assert malformed.returncode == 2, malformed
    assert "W0_HARD_RESET must be one of" in malformed.stderr
    assert "applied" not in malformed.stdout


def test_lerobot_policy_overlay_disables_only_visual_normalization(tmp_path: Path) -> None:
    """The released checkpoint must stay byte-identical while VISUAL is corrected.

    ``policy_preprocessor.json`` ships ``norm_map.VISUAL=MEAN_STD`` with
    ``mean=std=0.5``, which already maps images to ``[-1,1]``; the pinned LeRobot
    FastWAM core then applies ``*2-1`` again, feeding the VAE ``4x-3`` instead of
    the official ``2x-1``. The overlay restores the contract the pinned code
    documents without touching the 12 GiB weights.
    """

    policy_dir = tmp_path / "released"
    policy_dir.mkdir()
    (policy_dir / "model.safetensors").write_bytes(b"weights")
    (policy_dir / "config.json").write_text('{"type": "fastwam"}', encoding="utf-8")
    nested = policy_dir / "google"
    nested.mkdir()
    (nested / "tokenizer.json").write_text("{}", encoding="utf-8")
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "norm_map": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
                },
                "state_file": "stats.safetensors",
            },
        ],
    }
    (policy_dir / "policy_preprocessor.json").write_text(
        json.dumps(preprocessor), encoding="utf-8"
    )
    original_bytes = (policy_dir / "policy_preprocessor.json").read_bytes()

    overlay, record = _prepare_lerobot_policy_overlay(policy_dir, tmp_path / "overlay")

    # Weights and every other entry are symlinks, so nothing is duplicated.
    assert (overlay / "model.safetensors").is_symlink()
    assert (overlay / "model.safetensors").read_bytes() == b"weights"
    assert (overlay / "google").is_symlink()
    assert (overlay / "google" / "tokenizer.json").read_text(encoding="utf-8") == "{}"

    # Only the preprocessor config is materialised, and only VISUAL changes.
    assert not (overlay / "policy_preprocessor.json").is_symlink()
    patched = json.loads((overlay / "policy_preprocessor.json").read_text(encoding="utf-8"))
    norm_map = patched["steps"][1]["config"]["norm_map"]
    assert norm_map == {"VISUAL": "IDENTITY", "STATE": "MIN_MAX", "ACTION": "MIN_MAX"}
    assert patched["steps"][1]["state_file"] == "stats.safetensors"
    assert record["visual_norm_changes"] == [
        {"step_index": 1, "from": "MEAN_STD", "to": "IDENTITY"}
    ]

    # The pinned artifact is untouched.
    assert (policy_dir / "policy_preprocessor.json").read_bytes() == original_bytes

    # Rebuilding is idempotent and a checkpoint already set to IDENTITY records no change.
    (policy_dir / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "name": "policy_preprocessor",
                "steps": [
                    {
                        "registry_name": "normalizer_processor",
                        "config": {"norm_map": {"VISUAL": "IDENTITY"}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _, second = _prepare_lerobot_policy_overlay(policy_dir, tmp_path / "overlay")
    assert second["visual_norm_changes"] == []


def test_lerobot_visual_normalization_choice_is_explicit() -> None:
    parser = build_parser()
    default = parser.parse_args(["--phase=eval-lerobot-released", "--policy-path=x"])
    assert default.visual_normalization == "identity"
    kept = parser.parse_args(
        ["--phase=eval-lerobot-released", "--policy-path=x", "--visual-normalization=checkpoint"]
    )
    assert kept.visual_normalization == "checkpoint"
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--phase=eval-lerobot-released", "--policy-path=x", "--visual-normalization=bogus"]
        )


def test_lerobot_shard_command_pins_one_task_and_aggregates_by_episode_count(
    tmp_path: Path,
) -> None:
    """LR-REL must shard like UP-REL instead of serialising four suites on one GPU.

    ``--eval.n_episodes`` counts episodes per task, and ``--env.task_ids=[N]``
    restricts a run to a single task, so the 40 protocol tasks are independently
    schedulable.
    """

    command = _lerobot_worker_command(
        policy=Path("/policy"),
        suite="libero_10",
        task_id=7,
        trials=50,
        episode_length=600,
        output=tmp_path / "shard",
        seed=42,
    )
    assert "--env.task=libero_10" in command
    assert "--env.task_ids=[7]" in command
    assert "--eval.n_episodes=50" in command
    assert "--env.episode_length=600" in command
    assert "--seed=42" in command

    # Sharded reports must be weighted by episode count, not averaged as percentages.
    shards = tmp_path / "shards"
    for task_id, (percent, episodes) in enumerate([(100.0, 10), (0.0, 30)]):
        directory = shards / f"libero_goal_task{task_id:02d}"
        directory.mkdir(parents=True)
        (directory / "eval_info.json").write_text(
            json.dumps({"overall": {"pc_success": percent, "n_episodes": episodes}}),
            encoding="utf-8",
        )
    metrics = _collect_lerobot_metrics(tmp_path, ["libero_goal", "libero_object"])
    assert metrics["libero_goal"]["n_episodes"] == 40
    assert metrics["libero_goal"]["successes"] == 10
    assert metrics["libero_goal"]["pc_success"] == 25.0  # not the 50.0 of a naive mean
    assert metrics["libero_goal"]["reports"] == 2
    assert metrics["libero_object"] == {"status": "missing"}
    assert metrics["aggregate"] == {"successes": 10, "n_episodes": 40, "pc_success": 25.0}

    # The sequential layout (<output>/<suite>/eval_info.json) still works.
    sequential = tmp_path / "seq"
    (sequential / "libero_spatial").mkdir(parents=True)
    (sequential / "libero_spatial" / "eval_info.json").write_text(
        json.dumps({"overall": {"pc_success": 90.0, "n_episodes": 20}}), encoding="utf-8"
    )
    fallback = _collect_lerobot_metrics(sequential, ["libero_spatial"])
    assert fallback["libero_spatial"]["pc_success"] == 90.0
    assert fallback["libero_spatial"]["n_episodes"] == 20


def test_w0_sitecustomize_can_force_libero_settling_steps(tmp_path: Path) -> None:
    """``num_steps_wait`` differs between the two stacks and is not CLI-reachable.

    LeRobot's ``LiberoEnv`` waits 10 dummy steps after ``set_init_state`` while the
    official evaluator waits 5, and the value is absent from ``LiberoEnv``'s
    ``gym_kwargs`` property.  Those kwargs reach the constructor verbatim, so
    injecting the key there aligns the settling duration.
    """

    fake_packages = tmp_path / "fake_packages"
    envs = fake_packages / "lerobot" / "envs"
    envs.mkdir(parents=True)
    (fake_packages / "lerobot" / "__init__.py").write_text("", encoding="utf-8")
    (envs / "__init__.py").write_text("", encoding="utf-8")
    (envs / "configs.py").write_text(
        "class LiberoEnv:\n"
        "    observation_height = 224\n"
        "    @property\n"
        "    def gym_kwargs(self):\n"
        "        return {'obs_type': 'pixels_agent_pos',\n"
        "                'observation_height': self.observation_height}\n",
        encoding="utf-8",
    )

    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()

    def run(value: str | None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("W0_LIBERO_ASSETS", None)
        environment.pop("W0_GL_CONTEXT_GUARD", None)
        environment.pop("W0_HARD_RESET", None)
        environment["PYTHONPATH"] = os.pathsep.join((str(runtime), str(fake_packages)))
        if value is None:
            environment.pop("W0_LIBERO_NUM_STEPS_WAIT", None)
        else:
            environment["W0_LIBERO_NUM_STEPS_WAIT"] = value
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from lerobot.envs.configs import LiberoEnv;"
                "print('kwargs', sorted(LiberoEnv().gym_kwargs.items()))",
            ],
            env=environment,
            capture_output=True,
            text=True,
        )

    forced = run("5")
    assert forced.returncode == 0, forced.stderr
    assert "('num_steps_wait', 5)" in forced.stdout
    # The four original keys must survive the injection.
    assert "('obs_type', 'pixels_agent_pos')" in forced.stdout
    assert "('observation_height', 224)" in forced.stdout
    assert "num_steps_wait forced to 5" in forced.stderr

    untouched = run(None)
    assert untouched.returncode == 0, untouched.stderr
    assert "num_steps_wait" not in untouched.stdout

    malformed = run("later")
    assert malformed.returncode == 2, malformed
    assert "W0_LIBERO_NUM_STEPS_WAIT must be an integer" in malformed.stderr


def test_w0_sitecustomize_repoints_cuda_home_at_a_prefix_with_nvcc(tmp_path: Path) -> None:
    """DeepSpeed's import-time probe needs ``$CUDA_HOME/bin/nvcc`` to exist.

    ``installed_cuda_version()`` only raises a catchable exception when
    ``CUDA_HOME`` is ``None``; a prefix that exists but has no ``bin/`` makes the
    ``nvcc -V`` subprocess escape as ``FileNotFoundError`` and kills the launcher.
    Nothing is compiled, so the fix is to satisfy the probe, not to install a
    toolchain.
    """

    runtime = Path("src/robot_practice/experiments/literature/w0_runtime").resolve()
    runtime_only = tmp_path / "cuda_runtime_only"
    (runtime_only / "targets").mkdir(parents=True)
    with_nvcc = tmp_path / "cuda_with_nvcc"
    (with_nvcc / "bin").mkdir(parents=True)
    (with_nvcc / "bin" / "nvcc").write_text("#!/bin/sh\necho 'release 12.8'\n", encoding="utf-8")

    def run(cuda_home: str | None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.pop("W0_LIBERO_ASSETS", None)
        environment.pop("W0_GL_CONTEXT_GUARD", None)
        environment.pop("W0_HARD_RESET", None)
        environment.pop("W0_LIBERO_NUM_STEPS_WAIT", None)
        environment["PYTHONPATH"] = str(runtime)
        environment["W0_CUDA_HOME_FALLBACK"] = "1"
        if cuda_home is None:
            environment.pop("CUDA_HOME", None)
        else:
            environment["CUDA_HOME"] = cuda_home
        return subprocess.run(
            [sys.executable, "-c", "import os;print('CUDA_HOME', os.environ.get('CUDA_HOME'))"],
            env=environment,
            capture_output=True,
            text=True,
        )

    # A runtime-only prefix must be replaced by one that can answer `nvcc -V`.
    # sys.prefix is probed first, so point CUDA_HOME at the broken prefix and rely
    # on the /usr/local candidates only when they exist on this host; assert on the
    # observable contract instead: a valid CUDA_HOME is never overwritten.
    valid = run(str(with_nvcc))
    assert valid.returncode == 0, valid.stderr
    assert f"CUDA_HOME {with_nvcc}" in valid.stdout
    assert "W0-RUNTIME] CUDA_HOME" not in valid.stderr

    broken = run(str(runtime_only))
    assert broken.returncode == 0, broken.stderr
    # Either a candidate prefix on this host provided nvcc (then CUDA_HOME moved and
    # was logged), or none did (then the value is left untouched for the original
    # error message to stay readable).
    if "W0-RUNTIME] CUDA_HOME" in broken.stderr:
        assert f"previous={runtime_only}" in broken.stderr
        assert f"CUDA_HOME {runtime_only}" not in broken.stdout
    else:
        assert f"CUDA_HOME {runtime_only}" in broken.stdout

    # Evaluation processes must stay untouched so their stdout matches the runs
    # that produced the recorded UP-REL/LR-REL numbers.
    ungated = run(str(runtime_only))
    del ungated
    environment_free = os.environ.copy()
    environment_free.pop("W0_LIBERO_ASSETS", None)
    environment_free.pop("W0_CUDA_HOME_FALLBACK", None)
    environment_free["PYTHONPATH"] = str(runtime)
    environment_free["CUDA_HOME"] = str(runtime_only)
    quiet = subprocess.run(
        [sys.executable, "-c", "import os;print('CUDA_HOME', os.environ.get('CUDA_HOME'))"],
        env=environment_free,
        capture_output=True,
        text=True,
    )
    assert quiet.returncode == 0, quiet.stderr
    assert f"CUDA_HOME {runtime_only}" in quiet.stdout
    assert "W0-RUNTIME] CUDA_HOME" not in quiet.stderr

    source = (runtime / "sitecustomize.py").read_text(encoding="utf-8")
    assert 'Path(sys.prefix),  # conda env' in source
    assert "DS_SKIP_CUDA_CHECK" in source  # the misleading alternative is documented
    assert 'os.environ.get("W0_CUDA_HOME_FALLBACK") != "1"' in source

    # The training launcher must be the thing that opts in.
    audit = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py")
    assert '"W0_CUDA_HOME_FALLBACK": "1"' in audit.read_text(encoding="utf-8")


def test_cuda_nvcc_probe_shim_answers_version_and_refuses_to_compile() -> None:
    """The shim exists so `import deepspeed` survives on a host with no nvcc.

    DeepSpeed's import-time probe only reads a version number, and the W0 training
    path compiles nothing, so answering `-V` is sufficient. Any other invocation
    must fail loudly instead of pretending to be a compiler.
    """

    shim = Path("docker/training/cuda_nvcc_probe_shim/bin/nvcc")
    assert shim.is_file()
    assert os.access(shim, os.X_OK), "shim must be executable for CUDA_HOME probing"

    version = subprocess.run([str(shim), "-V"], capture_output=True, text=True)
    assert version.returncode == 0, version.stderr
    # Mirror deepspeed.ops.op_builder.builder.installed_cuda_version parsing.
    fields = version.stdout.split()
    release = fields[fields.index("release") + 1].replace(",", "").split(".")
    assert (int(release[0]), int(release[1])) == (12, 8), version.stdout

    compile_attempt = subprocess.run(
        [str(shim), "-c", "kernel.cu"], capture_output=True, text=True
    )
    assert compile_attempt.returncode != 0
    assert "not a CUDA compiler" in compile_attempt.stderr

    # sitecustomize must prefer a real toolchain and only fall back to the shim.
    runtime = Path("src/robot_practice/experiments/literature/w0_runtime/sitecustomize.py")
    source = runtime.read_text(encoding="utf-8")
    shim_index = source.index("cuda_nvcc_probe_shim")
    for real in ("Path(sys.prefix)", '"/usr/local/cuda-12.8"'):
        assert source.index(real) < shim_index, f"{real} must be probed before the shim"

    # The image-level fix must be recorded, not only the stopgap.
    dockerfile = Path("docker/training/Dockerfile.w0-fastwam").read_text(encoding="utf-8")
    assert "cuda-nvcc=12.8" in dockerfile
    assert "--name fastwam-upstream" in dockerfile


def test_causal_pair_lambda_overrides_match_the_pinned_hydra_config() -> None:
    """The one declared difference must actually reach the model.

    ``configs/model/fastwam.yaml`` defines ``loss.lambda_action`` but not
    ``loss.lambda_video``; the pinned code defaults the latter to 1.0 via
    ``runtime.py``'s ``loss.get("lambda_video", 1.0)`` and uses it in
    ``loss_total = lambda_video * loss_video + lambda_action * loss_action``.
    Hydra's struct mode therefore needs ``+`` for the absent key and rejects ``+``
    for the present one.  Getting this wrong either crashes the launcher or, if
    the key were silently ignored, would make FW-V and FW-NV identical runs.
    """

    config = Path("third_party/w0_fastwam/FastWAM/configs/model/fastwam.yaml")
    if not config.is_file():
        pytest.skip("pinned FastWAM checkout is not present")
    loss_block = config.read_text(encoding="utf-8").split("loss:", 1)[1]
    has_video = "lambda_video" in loss_block.split("\n\n", 1)[0]
    has_action = "lambda_action" in loss_block.split("\n\n", 1)[0]
    assert has_action, "lambda_action is expected in the pinned model config"

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")

    if has_video:
        # Upstream added the key: the `+` must be dropped or Hydra raises
        # "An item is already at 'model.loss.lambda_video'".
        assert 'f"model.loss.lambda_video={lambda_video}"' in source
        assert 'f"+model.loss.lambda_video={lambda_video}"' not in source
    else:
        assert 'f"+model.loss.lambda_video={lambda_video}"' in source

    # lambda_action exists in the config, so it must never be appended.
    assert '"model.loss.lambda_action=1.0"' in source
    assert '"+model.loss.lambda_action' not in source

    # The consuming code path must still be the one this reasoning relies on.
    runtime = Path("third_party/w0_fastwam/FastWAM/src/fastwam/runtime.py")
    if runtime.is_file():
        assert 'loss.get("lambda_video", 1.0)' in runtime.read_text(encoding="utf-8")


def test_causal_smoke_logs_every_step_so_loss_is_observable() -> None:
    """A 5-step smoke must emit a loss, not finish silently.

    ``configs/task/libero_uncond_2cam224_1e-4.yaml`` sets ``log_every: 10``, so a
    5-step run never reaches a logging step: the first 8-GPU smoke completed and
    wrote checkpoints while ``grep -ci loss`` over its 253 KB log returned 0.
    Without a loss the run cannot fill the required train-loss field, and a
    checkpoint alone does not show that optimisation happened.
    """

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    # Epoch mode is always formal-scale, so it never needs per-step logging.
    assert (
        'f"log_every={1 if (train_epochs is None and steps <= 10) else 10}"' in source
    )
    # `log_every` exists in the pinned task config, so it must be a plain
    # override; `+log_every` would raise "An item is already at 'log_every'".
    assert '"+log_every' not in source

    task_config = Path(
        "third_party/w0_fastwam/FastWAM/configs/task/libero_uncond_2cam224_1e-4.yaml"
    )
    if task_config.is_file():
        assert "log_every:" in task_config.read_text(encoding="utf-8")


def test_causal_eval_smoke_preserves_gpu_count_and_uses_one_suite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The causal smoke must exercise concurrent inference/render workers.

    A previous implementation silently changed every smoke to one GPU and one task, which
    could not expose the multi-process EGL failures seen in formal evaluation.  The smoke keeps
    the requested GPU count and uses all ten tasks in one suite, while reducing each task to two
    trials.
    """

    calls: list[dict[str, Any]] = []

    def fake_eval(_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr(
        "robot_practice.experiments.literature.w0_fastwam_audit._require_resource_phase",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "robot_practice.experiments.literature.w0_fastwam_audit.eval_upstream_released",
        fake_eval,
    )

    result = eval_causal_pair(
        load_config(DEFAULT_CONFIG),
        with_video=tmp_path / "with_video.pt",
        no_video=tmp_path / "no_video.pt",
        seed_manifest=None,
        pair_contract=None,
        output_root=tmp_path / "causal_smoke",
        vendor_root=tmp_path / "vendor",
        trials=50,
        num_gpus=8,
        smoke=True,
        deployment_evidence=None,
        dry_run=True,
        egl_device_map_spec="0:2,1:3,2:0,3:1",
    )

    assert result == {"status": "dry_run"}
    assert len(calls) == 2
    for call in calls:
        assert call["trials"] == 2
        assert call["num_gpus"] == 8
        assert call["smoke"] is False
        assert call["suite"] == "libero_spatial"
        assert call["task_id"] is None
        assert call["egl_device_map_spec"] == "0:2,1:3,2:0,3:1"


def test_training_defaults_to_expandable_segments_but_yields_to_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fragmentation control must be on by default, not left to a remembered flag.

    Measured: the single-GPU attempt reported ``29.81 GiB is reserved by PyTorch but
    unallocated`` alongside a 22.43 GiB allocation failure, and the 8-GPU run holds
    ~74.8 GiB of 95.2 GiB per rank.  The setting is numerically inert, so the cost
    of enabling it is zero while forgetting it costs an OOM hours into a 20k-step
    run.  A deliberate operator value must still survive, because ``_run`` merges
    this mapping over ``os.environ``.
    """

    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert _training_allocator_env() == {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
    }

    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512")
    assert _training_allocator_env() == {}

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    # Applied to the training run only; evaluation must stay byte-identical to the
    # environment that produced the recorded UP-REL/LR-REL numbers.
    assert "**_training_allocator_env()," in source
    assert source.count("_training_allocator_env()") == 2  # definition + one call site
    for evaluator in ("def eval_lerobot_released(", "def _run_upstream_workers("):
        body = source.split(evaluator, 1)[1].split("\ndef ", 1)[0]
        assert "_training_allocator_env" not in body, evaluator


def test_dataset_fingerprint_matches_released_stats_and_rejects_drift(tmp_path: Path) -> None:
    """The frozen protocol must pin the dataset by content, not by "directory exists".

    The released checkpoint ships num_episodes=1712 / num_transition=277713. If the local
    dataset differs, the causal pair has no valid reference and a 17-hour run is wasted,
    so the check raises instead of warning.
    """

    config = load_config(DEFAULT_CONFIG)
    protocol = config["protocol"]
    # total_demos stays the raw LIBERO count; the training set is the _no_noops subset.
    assert protocol["total_demos"] == 2000
    assert protocol["train_episodes"] == 1712
    assert protocol["train_frames"] == 277713
    assert sum(protocol["train_episodes_per_suite"].values()) == 1712
    # Upstream defines no validation or test split at all.
    assert protocol["val_episodes"] == 0
    assert protocol["test_episodes"] == 0

    counts = {
        "libero_spatial": (434, 53229),
        "libero_object": (457, 67309),
        "libero_goal": (433, 52895),
        "libero_10": (388, 104280),
    }

    def write(values: dict[str, tuple[int, int]]) -> None:
        for suite, (episodes, frames) in values.items():
            meta = tmp_path / f"{suite}_no_noops_lerobot/meta"
            meta.mkdir(parents=True, exist_ok=True)
            (meta / "info.json").write_text(
                json.dumps({"total_episodes": episodes, "total_frames": frames}),
                encoding="utf-8",
            )

    write(counts)
    fingerprint = _dataset_fingerprint(config, tmp_path)
    assert fingerprint["total_episodes"] == 1712
    assert fingerprint["total_frames"] == 277713
    assert fingerprint["matches_released_stats"] is True
    assert all(entry["matches"] for entry in fingerprint["per_suite"].values())

    for broken in ({"libero_goal": (500, 52895)}, {"libero_10": (388, 104281)}):
        write({**counts, **broken})
        with pytest.raises(RuntimeError, match="does not match the frozen protocol"):
            _dataset_fingerprint(config, tmp_path)
    write(counts)

    # A missing info.json must fail loudly rather than be treated as zero episodes.
    (tmp_path / "libero_object_no_noops_lerobot/meta/info.json").unlink()
    with pytest.raises(RuntimeError, match="missing"):
        _dataset_fingerprint(config, tmp_path)


def test_training_env_announcements_stay_off_stdout(capsys) -> None:
    """Env announcements must go to stderr.

    The launcher evaluates ``conda run`` output. A line containing parentheses previously
    broke activation with ``syntax error near unexpected token '('``, which silently fell
    back to the wrong environment and then failed on a missing DeepSpeed, costing a run.
    """

    from robot_practice.experiments.literature import w0_fastwam_audit as audit

    audit._report_training_env("keeping operator EXAMPLE=1")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "keeping operator EXAMPLE=1" in captured.err

    source = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )
    assert 'print(f"[W0] keeping operator' not in source


def test_training_enables_nccl_flight_recorder_but_yields_to_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hung collective must be diagnosable without spending another run to reproduce it.

    An 8-GPU run died to a 600 s ALLREDUCE timeout where seven ranks reported
    ``last completed work: 11911`` and rank 3 reported nothing -- one rank never reached the
    collective. The log said the stack trace was unavailable because FlightRecorder was
    disabled. These variables only record collective metadata and dump it on timeout, so
    they are numerically inert and default on.
    """

    for key in (
        "TORCH_NCCL_TRACE_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_NCCL_TRACE_CPP_STACK",
        "TORCH_NCCL_DESYNC_DEBUG",
    ):
        monkeypatch.delenv(key, raising=False)

    env = _training_debug_env()
    assert env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "2000"
    assert env["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert env["TORCH_NCCL_TRACE_CPP_STACK"] == "1"
    # DESYNC_DEBUG names the culprit rank but has per-collective overhead, and this host is
    # already below its measured throughput baseline. Opt-in only.
    assert "TORCH_NCCL_DESYNC_DEBUG" not in env

    # A deliberate operator setting must survive.
    monkeypatch.setenv("TORCH_NCCL_TRACE_BUFFER_SIZE", "0")
    overridden = _training_debug_env()
    assert "TORCH_NCCL_TRACE_BUFFER_SIZE" not in overridden
    assert overridden["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    # Scoped to training only: evaluation workers must not inherit these.
    assert source.count("**_training_debug_env(),") == 1
    # The host-allocator defaults sit between them; all three must be applied together, and
    # only on the training path so evaluation stays identical to the recorded UP/LR runs.
    # glibc tuning was tried here and removed: MALLOC_ARENA_MAX with a pinned low
    # MALLOC_MMAP_THRESHOLD_ left the growth slope unchanged, because the cause was a
    # reference cycle from torchcodec's failed import rather than arena fragmentation.
    # Forcing every >=128 KiB allocation through mmap costs syscalls for no benefit.
    assert "MALLOC_ARENA_MAX" not in source
    assert "**_training_allocator_env(),\n            **_training_debug_env()," in source


def test_cli_dispatch_forwards_every_training_option(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard the argparse-to-function seam, not just each side of it.

    A real 21-minute 8-GPU run was wasted because ``--train-epochs=10`` was accepted by
    argparse and then dropped at the dispatch site, so the job silently used
    ``max_steps=20000``. ``--train-seed`` had the same defect. Earlier tests missed it
    because they exercised argparse alone and called ``train_causal_variant`` directly,
    which bypasses exactly the line that was broken. This test drives ``main`` with argv.
    """

    captured: dict[str, Any] = {}

    def fake_train(config: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(
        "robot_practice.experiments.literature.w0_fastwam_audit.train_causal_variant",
        fake_train,
    )

    argv = [
        "--phase=train-causal-pair",
        "--variant=no_video",
        "--train-epochs=10",
        "--train-seed=43",
        "--steps=20000",
        "--num-gpus=8",
        "--output-root=/tmp/w0-dispatch-test",
        "--dry-run",
    ]
    monkeypatch.setattr(sys, "argv", ["w0_fastwam_audit", *argv])
    main()

    assert captured["variant"] == "no_video"
    assert captured["train_epochs"] == 10, "--train-epochs must reach train_causal_variant"
    assert captured["train_seed"] == 43, "--train-seed must reach train_causal_variant"
    assert captured["steps"] == 20000
    assert captured["num_gpus"] == 8
    assert str(captured["output_root"]) == "/tmp/w0-dispatch-test"

    # Defaults must arrive as None so the config/protocol value stays authoritative.
    captured.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        ["w0_fastwam_audit", "--phase=train-causal-pair", "--variant=with_video", "--dry-run"],
    )
    main()
    assert captured["train_epochs"] is None
    assert captured["train_seed"] is None

    # Every option the parser defines for this phase must be forwarded by name.
    forwarded = set(captured)
    for option in ("variant", "steps", "num_gpus", "output_root", "vendor_root", "train_seed", "train_epochs"):
        assert option in forwarded, option


def test_egl_device_map_is_derived_from_pci_not_guessed(monkeypatch, tmp_path: Path) -> None:
    """MUJOCO_EGL_DEVICE_ID is a position in the EGL list, not a CUDA device id.

    Measured on an 8x H20 node: CUDA enumerates PCI domains 65,67,69,6b ascending while
    card1..card8 enumerate them descending, so the correct map is
    {0:6, 1:7, 2:4, 3:5, 4:2, 5:3, 6:0, 7:1} -- identity is wrong for **every** device.

    The old render-probe calibration could not find this. Cross-GPU rendering mostly works, so
    the probe accepted identity and a 2,000-episode evaluation completed; a later evaluation
    aborted natively inside mjr_readPixels at about the 141st cumulative render, just past the
    120-step probe window. EGL_CUDA_DEVICE_NV would answer directly but returns EGLError for
    every index on driver 535.261.03, so joining on PCI bus id is the deterministic route.
    """

    from robot_practice.experiments.literature import w0_fastwam_audit as audit

    # Use the exact strings each side reports. They disagree on the domain field --
    # nvidia-smi gives eight hex digits, sysfs gives four -- and a naive prefix strip left one
    # side with a domain and the other without, so the join silently produced nothing and the
    # run fell back to render probing, which then accepted the wrong identity map.
    cuda = [f"00000000:{b}" for b in
            ("65:02.0", "65:03.0", "67:02.0", "67:03.0", "69:02.0", "69:03.0", "6b:02.0", "6b:03.0")]
    drm = ["6b:02.0", "6b:03.0", "69:02.0", "69:03.0", "67:02.0", "67:03.0", "65:02.0", "65:03.0"]
    monkeypatch.setattr(audit, "_cuda_pci_by_index", lambda: {str(i): p for i, p in enumerate(cuda)})
    monkeypatch.setattr(
        audit,
        "_drm_pci_by_card",
        # card0 is a non-NVIDIA display device and must not break the join.
        lambda: {"card0": "00:1e.0", **{f"card{i}": p for i, p in zip(range(1, 9), drm)}},
    )
    assert audit._normalize_pci("00000000:65:02.0") == audit._normalize_pci("0000:65:02.0")
    # Entries 8 and 9 duplicate earlier cards and lack EGL_NV_device_cuda; the first occurrence
    # must win so a duplicate never becomes the chosen index.
    monkeypatch.setattr(
        audit,
        "_egl_drm_by_index",
        lambda runtime_env: {**{str(i): f"card{i + 1}" for i in range(8)}, "8": "card8", "9": "card7"},
    )

    same_gpu, report = audit._pci_derived_egl_device_map(
        [str(i) for i in range(8)], runtime_env={}
    )
    # This is the index that names the CUDA device's own physical GPU -- the one to avoid. A
    # measured 8x8 matrix on this node failed exactly one cell per row, and those cells matched
    # this join cell for cell: rendering on the same physical GPU as the CUDA context aborts
    # immediately, every other index survives.
    assert same_gpu == {"0": "6", "1": "7", "2": "4", "3": "5", "4": "2", "5": "3", "6": "0", "7": "1"}

    chosen = audit._select_egl_render_indices(
        [str(i) for i in range(8)], forbidden=same_gpu
    )
    # The permutation here is four transpositions, so it has no fixed point and identity is
    # already safe. That is why identity "worked" and why the probe kept accepting it.
    assert chosen == {str(i): str(i) for i in range(8)}
    assert all(chosen[gpu] != same_gpu[gpu] for gpu in chosen)
    assert len(set(chosen.values())) == 8, "two workers must not share a render device"

    # A host whose DRM order matches CUDA order would forbid identity everywhere. Selection
    # must then produce a derangement, and still a bijection. (The forbidden map is always a
    # bijection on a real host: distinct physical GPUs get distinct EGL indices.)
    all_forbidden = {str(i): str(i) for i in range(8)}
    deranged = audit._select_egl_render_indices(
        [str(i) for i in range(8)], forbidden=all_forbidden
    )
    assert all(deranged[gpu] != gpu for gpu in deranged)
    assert len(set(deranged.values())) == 8

    # The derived pair must actually be verified, not merely documented: a derivation that
    # disagrees with sysfs or the EGL list has to fail loudly before any worker starts.
    monkeypatch.setattr(audit, "_render_fingerprint", lambda *_a, **_k: {"gpu_count": 8})
    # The derivation names the forbidden index; selection must then pick a different one and
    # the probe must verify that selection, not the forbidden pair.
    monkeypatch.setattr(
        audit, "_pci_derived_egl_device_map", lambda gpu_ids, *, runtime_env: (
            {"0": "6", "1": "0"}, {"identity_is_correct": False}
        )
    )
    probed: list[tuple[str, str]] = []

    def refuse(*, cuda_device: str, egl_index: str, runtime_env, steps=0) -> bool:
        probed.append((cuda_device, egl_index))
        return False

    monkeypatch.setattr(audit, "_probe_render_pair", refuse)
    monkeypatch.setattr(audit, "DEFAULT_EGL_CALIBRATION", str(tmp_path / "cal.json"))
    with pytest.raises(RuntimeError) as excinfo:
        audit._egl_device_map(["0", "1"], runtime_env={"W0_TEST": "1"}, allow_calibration=True)
    assert "failed the render probe" in str(excinfo.value)
    # CUDA 0 must avoid EGL 6, so it takes 0; the probe verifies that pair.
    assert probed and probed[0] == ("0", "0")

    source = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )
    # PCI derivation must run before the render probe, and the probe is only a verification.
    assert "_pci_derived_egl_device_map(gpu_ids, runtime_env=runtime_env)" in source
    # The map must be re-derived every run and never read back from the cache. Caching is what
    # let a wrong map survive across runs: the record's fingerprint still matched this host.
    assert "never read from cache" in source
    assert '"source": "pci-derived",' in source
    # The cache is kept only as a record and to name drift.
    assert "map changed since" in source


def test_stats_lookup_reaches_the_run_root_and_tolerates_near_zero_fields() -> None:
    """Two measured bugs in the stats resolver, both silent.

    1. The trainer writes ``dataset_stats.json`` at the run root while checkpoints land under
       ``checkpoints/weights/``. Looking only at ``checkpoint.parent`` missed it, so the FW-V
       eval normalized with the upstream stats instead of its own training stats. Harmless that
       time -- the two files agreed to float32 epsilon -- but it was luck, not design.
    2. The difference test was purely relative, so a 7.6e-10 absolute gap on a ~3e-05
       ``stepwise_mean`` field reported a 2.4e-05 relative difference and would have warned
       "differ beyond float32 noise" about pure rounding.
    """

    import json as _json
    import tempfile

    from robot_practice.experiments.literature.w0_fastwam_audit import (
        _resolve_eval_dataset_stats,
    )

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        run_root = root / "with_video"
        weights = run_root / "checkpoints" / "weights"
        weights.mkdir(parents=True)
        checkpoint = weights / "step_021700.pt"
        checkpoint.write_bytes(b"")

        # Near-zero field: 7.6e-10 apart in absolute terms, 2.4e-05 apart relatively.
        local = {"action": {"default": {"stepwise_mean": [-3.128913158e-05], "std": [1.0]}}}
        upstream_payload = {
            "action": {"default": {"stepwise_mean": [-3.128989556e-05], "std": [1.0]}}
        }
        (run_root / "dataset_stats.json").write_text(_json.dumps(local), encoding="utf-8")
        upstream = root / "upstream_stats.json"
        upstream.write_text(_json.dumps(upstream_payload), encoding="utf-8")

        chosen, report = _resolve_eval_dataset_stats(checkpoint, upstream)

        # (1) the run-root file two levels up must win over the upstream file
        assert chosen == run_root / "dataset_stats.json"
        assert report["chosen"] == "checkpoint_sibling"
        assert report["checkpoint_sibling"] is not None
        assert len(report["searched_for_sibling"]) >= 3
        # (2) float32 rounding on a near-zero field must not be flagged as a data difference
        assert report["fields_beyond_tolerance"] == []
        assert report["max_relative_difference"] > 1e-5
        assert report["max_absolute_difference"] < 1e-8

        # A genuinely different scale must still be caught.
        (run_root / "dataset_stats.json").write_text(
            _json.dumps({"action": {"default": {"stepwise_mean": [-3.1e-05], "std": [2.0]}}}),
            encoding="utf-8",
        )
        _, changed = _resolve_eval_dataset_stats(checkpoint, upstream)
        assert changed["fields_beyond_tolerance"], changed


def test_eval_uses_the_stats_the_checkpoint_was_trained_with(tmp_path: Path) -> None:
    """A locally trained checkpoint must be denormalized with its own stats.

    ``eval_libero_single.py`` prefers an explicit ``EVALUATION.dataset_stats_path`` and only
    falls back to ``<checkpoint parent>/dataset_stats.json``, so passing the upstream file
    unconditionally is silently wrong whenever training data differ from the released run --
    actions come out on the wrong scale with no error anywhere.

    The check must be on values rather than paths. On the W0 pair the two files differ in 6
    fields at the 8th significant digit, about 2e-7 relative, which is float32 machine epsilon
    and comes from a different reduction order over identical data; refusing that would block a
    correct workflow.
    """

    import json as _json

    from robot_practice.experiments.literature.w0_fastwam_audit import (
        _resolve_eval_dataset_stats,
    )

    upstream = tmp_path / "upstream_stats.json"
    payload = {"action": {"default": {"global_mean": [0.125, -0.25], "global_std": [1.5, 2.5]}}}
    upstream.write_text(_json.dumps(payload), encoding="utf-8")
    checkpoint = tmp_path / "run" / "checkpoint_step_00021700.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"x")

    # No sibling: the upstream file is the only candidate.
    chosen, report = _resolve_eval_dataset_stats(checkpoint, upstream)
    assert chosen == upstream
    assert report["chosen"] == "upstream"

    def scaled(factor: float) -> dict[str, object]:
        return {
            "action": {
                "default": {
                    "global_mean": [v * factor for v in payload["action"]["default"]["global_mean"]],
                    "global_std": [v * factor for v in payload["action"]["default"]["global_std"]],
                }
            }
        }

    sibling = checkpoint.parent / "dataset_stats.json"

    # float32 reduction noise must be accepted, and the sibling still wins because it is what
    # training normalized with.
    sibling.write_text(_json.dumps(scaled(1 + 2e-7)), encoding="utf-8")
    chosen, report = _resolve_eval_dataset_stats(checkpoint, upstream)
    assert chosen == sibling
    assert report["chosen"] == "checkpoint_sibling"
    assert report["max_relative_difference"] < 1e-5

    # A genuine data difference must be surfaced, not silently absorbed.
    sibling.write_text(_json.dumps(scaled(1.01)), encoding="utf-8")
    chosen, report = _resolve_eval_dataset_stats(checkpoint, upstream)
    assert chosen == sibling
    assert report["max_relative_difference"] > 1e-3
    assert report["max_relative_difference_field"] is not None

    source = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )
    # The choice has to land in the manifest, or the run cannot be audited afterwards.
    assert '"dataset_stats": stats_report,' in source


def test_freeze_switch_refuses_to_run_unwired(tmp_path: Path) -> None:
    """W0_FREEZE_VIDEO_EXPERT without the patch is a silent no-op, which must not be allowed.

    The switch lives in a vendored source patch. If it is missing, training runs as FW-NV while
    the manifest records ``freeze_video_expert: true``, making two different experiments
    indistinguishable. Containers get recreated routinely here -- three times in one day -- and
    an edit applied inside a live container disappears with it.
    """

    from robot_practice.experiments.literature.w0_fastwam_audit import (
        _assert_freeze_switch_is_wired,
    )

    vendor = tmp_path / "vendor"
    trainer = vendor / "FastWAM/src/fastwam/trainer.py"
    trainer.parent.mkdir(parents=True)
    trainer.write_text("trainable_params = list(self.model.dit.parameters())\n", encoding="utf-8")

    for value in ("", "0"):
        os.environ["W0_FREEZE_VIDEO_EXPERT"] = value
        try:
            _assert_freeze_switch_is_wired(vendor)  # unset/0 must be a no-op
        finally:
            os.environ.pop("W0_FREEZE_VIDEO_EXPERT", None)

    os.environ["W0_FREEZE_VIDEO_EXPERT"] = "1"
    try:
        with pytest.raises(RuntimeError) as excinfo:
            _assert_freeze_switch_is_wired(vendor)
        message = str(excinfo.value)
        assert "freeze patch is not applied" in message
        # The message must name both ways out.
        assert "w0-fastwam-20260821-fwfz" in message
        assert "apply docker/training/vendor-patches/" in message

        # With both markers present it must pass.
        trainer.write_text(
            'if os.environ.get("W0_FREEZE_VIDEO_EXPERT", "").strip() not in ("", "0"):\n'
            "trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]\n",
            encoding="utf-8",
        )
        _assert_freeze_switch_is_wired(vendor)
    finally:
        os.environ.pop("W0_FREEZE_VIDEO_EXPERT", None)


def test_frozen_arm_gets_its_own_output_parent_and_a_tracked_patch() -> None:
    """FW-FZ must not be able to overwrite FW-NV, and its code change must be tracked.

    FW-FZ is the attribution arm: video expert frozen, lambda_video=0. It is also a
    ``no_video`` run, so if its output parent were passed separately it could be forgotten --
    the run would overwrite FW-NV's artifacts and the sibling budget check would compare it
    against the wrong group. The parent is therefore derived from W0_FREEZE_VIDEO_EXPERT.

    FW-V vs FW-NV keeps the frozen, pre-registered gate; FW-FZ only feeds the attribution
    discussion, because it was added after FW-V had already started.
    """

    body = Path("scripts/experiments/literature/launch_w0_causal.sh").read_text(encoding="utf-8")
    assert 'W0_FREEZE_VIDEO_EXPERT:-' in body
    assert 'pair_suffix="_frozen"' in body
    assert 'causal_${epochs}ep${pair_suffix}' in body
    # The launcher must say which arm it started.
    assert "printf 'freeze" in body

    # The manifest must record the arm, or two no_video runs become indistinguishable.
    audit = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )
    assert '"freeze_video_expert": os.environ.get("W0_FREEZE_VIDEO_EXPERT"' in audit

    patch = Path(
        "docker/training/vendor-patches/fastwam-trainer-freeze-video-expert.patch"
    )
    assert patch.is_file()
    text = patch.read_text(encoding="utf-8")
    # Both edits are required. Freezing alone would leave the 5B frozen params in the
    # optimizer, because _apply_dit_only_train_mode runs before a list(...) that ignores
    # requires_grad, so ZeRO-1 would still allocate Adam state for them.
    assert "W0_FREEZE_VIDEO_EXPERT" in text
    assert 'mixtures["video"]' in text
    assert "if p.requires_grad]" in text
    # Unset must reproduce upstream behaviour exactly so one image serves all three arms.
    assert 'not in ("", "0")' in text

    plan = Path("plans/2026-08-17-W0-Fast-WAM-LIBERO复现与机制锚点.md").read_text(encoding="utf-8")
    assert "FW-FZ" in plan
    # The plan no longer sets pass/fail thresholds: the operator reads the numbers and judges.
    # What it must still pin down is what to measure and when a number is meaningful at all,
    # because "the run finished" and "the number means something" are different claims.
    assert "不设 pass/fail 阈值" in plan
    assert "completeness" in plan
    assert "equal budget" in plan
    # FW-V vs FW-FZ changes two things at once, so the report has to say so rather than let it
    # read as a single-variable ablation.
    assert "架构对比而非单变量消融" in plan
    # A wide CI straddling zero is not a licence to pick a direction.
    assert "不得挑方向叙述" in plan
    # The Dockerfile layer that applies the patch, and its self-reporting check.
    dockerfile = Path("docker/training/Dockerfile.w0-fastwam-torchcodec-fix").read_text(
        encoding="utf-8"
    )
    assert "vendor-patches/fastwam-trainer-freeze-video-expert.patch" in dockerfile
    assert "git apply --check" in dockerfile
    # The image should report its own deviation through git rather than hide it.
    assert 'test "$(git status --porcelain | wc -l)" -eq 1' in dockerfile


def test_causal_launcher_forwards_the_host_ram_lever() -> None:
    """The launcher must be able to pass --train-num-workers, and must omit it by default.

    Without this the only way to change workers is the ~450-character direct invocation that
    soft-wrapping keeps mangling, which is why the launcher exists at all. Omitting the flag
    must produce no argument at all rather than an empty one, so the protocol default of 8
    still comes from the config.
    """

    script = Path("scripts/experiments/literature/launch_w0_causal.sh")
    body = script.read_text(encoding="utf-8")
    assert "workers=\"${3:-}\"" in body
    assert "worker_args=(--train-num-workers=\"${workers}\")" in body
    # An empty bash array expands to zero words, so the flag disappears entirely.
    assert 'worker_args=()' in body
    assert '"${worker_args[@]}"' in body
    assert "workers must be a non-negative integer" in body

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *args], capture_output=True, text=True, check=False
        )

    bad = run("with_video", "10", "abc")
    assert bad.returncode == 2
    assert "workers must be a non-negative integer" in bad.stderr
    assert "[workers]" in run("bogus").stderr


def test_causal_launcher_script_survives_a_broken_paste(tmp_path: Path) -> None:
    """The remote command must stay short, and must not silently drop arguments.

    The direct invocation is a ~450-character single line. Soft-wrapping during a paste has
    repeatedly split it, turning the tail into a second command: ``&`` and the redirect
    detach from ``nohup`` so the run goes to the foreground, and
    --num-gpus/--output-root/--overwrite are dropped, sending the run to the default output
    root. This wrapper keeps the remote command to two tokens.
    """

    script = Path("scripts/experiments/literature/launch_w0_causal.sh")
    assert script.is_file()
    assert os.access(script, os.X_OK), "launcher must be executable"
    body = script.read_text(encoding="utf-8")

    # Every argument that a broken paste used to drop must be pinned inside the script.
    for required in (
        "--phase=train-causal-pair",
        "--train-epochs=",
        "--num-gpus=8",
        "--output-root=",
        "--overwrite",
    ):
        assert required in body, required
    # Backgrounding and redirection must live here, not in the pasted line.
    assert "nohup bash" in body and ">\"${log}\" 2>&1 &" in body
    # Both groups must land under one parent so the pair contract and the sibling
    # budget check both apply.
    assert 'pair_root="${repo}/outputs/literature/w0_fastwam/causal_${epochs}ep${pair_suffix}"' in body
    assert 'output_root="${pair_root}/${variant}"' in body
    # A stale render pin would defeat the autocalibrated EGL map.
    assert "unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID" in body
    # Two concurrent 8-GPU trainings would OOM; refuse instead.
    assert 'pgrep -f "scripts/train.py"' in body
    assert "set -euo pipefail" in body

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(script), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    assert run().returncode == 2
    assert "usage:" in run().stderr
    assert run("bogus").returncode == 2
    for bad_epochs in ("0", "abc", "-1"):
        result = run("with_video", bad_epochs)
        assert result.returncode == 2, bad_epochs
        assert "positive integer" in result.stderr, bad_epochs


def test_causal_pair_groups_must_share_one_step_budget(tmp_path: Path) -> None:
    """The two groups are launched ~19h apart, so the budget must be checked, not trusted.

    ``--steps`` is always required, so omitting ``--train-epochs`` on the second command
    silently yields one 10-epoch run and one 20k-step run: different cosine schedule
    length, different warmup, comparison void, and no symptom until evaluation.
    ``pair_contract.json`` cannot catch it because it pins initialization/data/noise rather
    than the budget.
    """

    with_video = tmp_path / "with_video"
    no_video = tmp_path / "no_video"
    for directory in (with_video, no_video):
        directory.mkdir(parents=True)

    def write_sibling(payload: dict[str, object]) -> None:
        (no_video / "run_manifest.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    # No sibling yet: the first group of the pair must not be blocked.
    assert (
        _assert_sibling_budget_matches(
            with_video, "with_video", train_epochs=10, steps=20000
        )
        is None
    )

    write_sibling({"step_budget_mode": "epochs", "train_epochs": 10, "steps": None})
    matched = _assert_sibling_budget_matches(
        with_video, "with_video", train_epochs=10, steps=20000
    )
    assert matched is not None and matched["budget"] == 10

    # Mixed modes and mismatched epoch counts must both be rejected.
    for kwargs in (
        {"train_epochs": None, "steps": 20000},
        {"train_epochs": 5, "steps": 20000},
    ):
        with pytest.raises(ValueError, match="must share one step budget"):
            _assert_sibling_budget_matches(with_video, "with_video", **kwargs)

    write_sibling({"step_budget_mode": "max_steps", "train_epochs": None, "steps": 20000})
    assert (
        _assert_sibling_budget_matches(
            with_video, "with_video", train_epochs=None, steps=20000
        )["budget"]
        == 20000
    )
    with pytest.raises(ValueError, match="must share one step budget"):
        _assert_sibling_budget_matches(
            with_video, "with_video", train_epochs=10, steps=20000
        )

    # Manifests written before step_budget_mode existed must still be interpreted.
    write_sibling({"train_epochs": 10})
    assert (
        _assert_sibling_budget_matches(
            with_video, "with_video", train_epochs=10, steps=20000
        )["budget_mode"]
        == "epochs"
    )

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    # The check must run before the training command is built, and be recorded.
    assert "budget_match = _assert_sibling_budget_matches(" in source
    assert '"sibling_budget_check": budget_match,' in source


def test_train_epochs_mode_reproduces_the_upstream_step_budget(tmp_path: Path) -> None:
    """Epoch budgeting is not interchangeable with a pinned max_steps.

    The upstream task config ships num_epochs=10 / max_steps=null, and the trainer derives
    the total from the dataset. That total also sets the cosine schedule length and the 5%
    warmup, so pinning max_steps=20000 changes the LR curve rather than only the run
    length. Epoch mode must therefore pass num_epochs AND max_steps=null explicitly, and
    must count as a formal run for the resource gate.
    """

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    assert 'f"num_epochs={train_epochs}", "max_steps=null"' in source
    # A formal-scale epoch run must not bypass the 8x>=80GiB preflight gate.
    assert (
        'formal = train_epochs is not None or steps >= int(config["training"]["steps"])'
        in source
    )
    # save_every must stay at the protocol value, not be clamped by the unused --steps.
    assert (
        "int(config['training']['save_every']) if train_epochs is not None"
        in source
    )
    # The manifest must record which budgeting mode produced the run.
    assert '"step_budget_mode": "epochs" if train_epochs is not None else "max_steps",' in source
    assert '"steps": None if train_epochs is not None else steps,' in source

    parser = build_parser()
    default = parser.parse_args(["--phase=train-causal-pair", "--variant=with_video"])
    assert default.train_epochs is None
    epochs = parser.parse_args(
        ["--phase=train-causal-pair", "--variant=with_video", "--train-epochs=10"]
    )
    assert epochs.train_epochs == 10

    config = load_config(DEFAULT_CONFIG)
    with pytest.raises(ValueError, match="train_epochs must be positive"):
        train_causal_variant(
            config,
            variant="with_video",
            steps=20000,
            num_gpus=8,
            output_root=tmp_path / "out",
            vendor_root=tmp_path / "vendor",
            dry_run=True,
            train_epochs=0,
        )


def test_train_seed_override_keys_the_pair_contract_per_seed(tmp_path: Path) -> None:
    """Multi-seed runs need per-seed isolation, not a shared contract.

    A single FW-V/FW-NV pair cannot separate a mechanism effect from run-to-run
    variance, so the gate may require a second seed per arm. The two arms of one
    pair must share a seed, while different seeds must not share a
    ``pair_contract.json`` -- otherwise the contract check would pass across seeds
    and the comparison would silently mix them.
    """

    config = load_config(DEFAULT_CONFIG)
    dataset = tmp_path / "dataset"
    action_dit = tmp_path / "action_dit.pt"

    default = _pair_contract(config, dataset, action_dit)
    explicit = _pair_contract(config, dataset, action_dit, train_seed=42)
    other = _pair_contract(config, dataset, action_dit, train_seed=43)

    assert config["protocol"]["train_seed"] == 42
    assert default["init_hash"] == explicit["init_hash"]
    assert default["init_hash"] != other["init_hash"]
    for key in ("seed", "noise_seed", "data_order_seed", "update_seed"):
        assert other["contract"][key] == 43, key
    # Every derived hash must move with the seed, not just init_hash.
    for key in ("data_hash", "noise_hash", "update_hash"):
        assert default[key] != other[key], key

    parser = build_parser()
    assert parser.parse_args(["--phase=train-causal-pair", "--variant=with_video"]).train_seed is None
    forced = parser.parse_args(
        ["--phase=train-causal-pair", "--variant=with_video", "--train-seed=43"]
    )
    assert forced.train_seed == 43

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    # The Hydra override and the manifest must both report the effective seed.
    assert 'f"seed={effective_seed}"' in source
    assert '"train_seed": effective_seed,' in source
    # A seed mismatch must name the seeds rather than failing opaquely.
    assert "Give each seed its own --output-root parent." in source


def test_pair_contract_is_variant_independent_and_portable(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    dataset = tmp_path / "dataset"
    action_dit = tmp_path / "action.pt"
    left = _pair_contract(config, dataset, action_dit)
    right = _pair_contract(config, dataset, action_dit)
    assert left == right
    assert left["global_batch_size"] == 128
    assert _portable(Path("outputs/literature/w0_fastwam")) == "outputs/literature/w0_fastwam"
    json.dumps(left)


def test_script_forwards_arbitrary_argument_order() -> None:
    script = Path("scripts/experiments/literature/run_w0_fastwam.sh").read_text(encoding="utf-8")
    assert '"$@"' in script
    assert '"${@:2}"' not in script


def test_upstream_eval_uses_direct_workers_without_tmux() -> None:
    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    assert "run_libero_manager.py" not in source
    assert "subprocess.Popen" in source
    assert "tmux" in source  # documentation explains the removed upstream dependency

    command = _worker_eval_command(
        checkpoint=Path("checkpoint.pt"),
        stats=Path("stats.json"),
        action_dit=Path("action_dit.pt"),
        output=Path("outputs/eval"),
        trials=50,
        suite="libero_spatial",
        task_id=0,
        gpu_slot=3,
        seed=42,
    )
    assert any("eval_libero_single.py" in item for item in command)
    assert "gpu_id=3" in command
    assert "MULTIRUN.num_gpus=8" not in command


def test_upstream_worker_command_forwards_trial_start() -> None:
    command = _worker_eval_command(
        checkpoint=Path("checkpoint.pt"),
        stats=Path("stats.json"),
        action_dit=Path("action_dit.pt"),
        output=Path("outputs/eval"),
        trials=1,
        trial_start=7,
        suite="libero_spatial",
        task_id=0,
        gpu_slot=3,
        seed=42,
    )
    assert "EVALUATION.num_trials=1" in command
    assert "+EVALUATION.trial_start=7" in command


def test_upstream_worker_specs_default_to_one_trial_per_process() -> None:
    specs = _upstream_worker_specs(
        suite="libero_spatial",
        task_id=0,
        trials=3,
        trials_per_worker=1,
    )
    assert specs == [
        ("libero_spatial", 0, 0, 1),
        ("libero_spatial", 0, 1, 1),
        ("libero_spatial", 0, 2, 1),
    ]


def test_trials_per_worker_defaults_to_one_process_per_task() -> None:
    """The default must not reload the 12 GiB checkpoint once per trial.

    Per-trial process isolation was a workaround for the EGL render-device
    mismatch; with the resolved device map a task's trials share one process, so
    a 40-task x 50-trial run loads the checkpoint 40 times instead of 2000.
    """

    parser = build_parser()
    args = parser.parse_args(["--phase=eval-upstream-released", "--checkpoint=x"])
    assert args.trials_per_worker == 0

    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    assert "trials_per_worker = trials" in source

    # 0 must expand to a single shard covering every trial of the task.
    specs = _upstream_worker_specs(
        suite="libero_spatial",
        task_id=0,
        trials=50,
        trials_per_worker=50,
    )
    assert specs == [("libero_spatial", 0, 0, 50)]


def test_upstream_default_is_one_worker_per_gpu() -> None:
    source = Path(
        "src/robot_practice/experiments/literature/w0_fastwam_audit.py"
    ).read_text(encoding="utf-8")
    assert "DEFAULT_MAX_UPSTREAM_TASKS_PER_GPU = 1" in source


def test_upstream_gpu_slots_respect_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert _visible_gpu_ids(4) == ["4", "5", "6", "7"]


def test_internal_call_sites_match_function_signatures() -> None:
    """Guard against keyword arguments that no longer exist on the callee.

    ``_run_upstream_workers`` and ``_upstream_worker_specs`` take the same
    ``trials_per_worker`` keyword, so a mis-targeted edit can add a parameter to
    the wrong function and only fail at launch time, after a checkpoint load.
    """

    import ast
    import inspect

    module_path = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    module = importlib.import_module(
        "robot_practice.experiments.literature.w0_fastwam_audit"
    )

    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        target = getattr(module, node.func.id, None)
        if not callable(target) or not node.func.id.startswith("_"):
            continue
        try:
            parameters = inspect.signature(target).parameters
        except (TypeError, ValueError):
            continue
        if any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        ):
            continue
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            assert keyword.arg in parameters, (
                f"{node.func.id}(...) is called with unknown keyword "
                f"{keyword.arg!r} at line {node.lineno}; "
                f"accepted: {sorted(parameters)}"
            )
            checked += 1

    assert checked > 20, f"signature guard inspected too few keywords: {checked}"


def test_upstream_worker_environment_preserves_conda_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONDA_EXE", "/opt/conda/bin/conda")
    monkeypatch.setenv("PATH", "/opt/conda/condabin:/opt/conda/bin")
    environment = _worker_environment({"W0_TEST": "1"}, "4")
    assert environment["CONDA_EXE"] == "/opt/conda/bin/conda"
    assert environment["PATH"].startswith("/opt/conda")
    assert environment["W0_TEST"] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == "4"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "4"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["PYOPENGL_PLATFORM"] == "egl"
    assert environment["PYTHONFAULTHANDLER"] == "1"
    assert environment["TORCH_SHOW_CPP_STACKTRACES"] == "1"


def test_worker_environment_exposes_both_devices_when_render_device_differs() -> None:
    """robosuite asserts MUJOCO_EGL_DEVICE_ID appears in CUDA_VISIBLE_DEVICES.

    The compute device must stay first so ``EVALUATION.device=cuda`` resolves to
    the intended GPU as ``cuda:0``.
    """

    environment = _worker_environment({}, "7", "3")
    assert environment["CUDA_VISIBLE_DEVICES"] == "7,3"
    assert environment["MUJOCO_EGL_DEVICE_ID"] == "3"
    assert environment["MUJOCO_EGL_DEVICE_ID"] in environment["CUDA_VISIBLE_DEVICES"]

    same_device = _worker_environment({}, "7", "7")
    assert same_device["CUDA_VISIBLE_DEVICES"] == "7"
    assert same_device["MUJOCO_EGL_DEVICE_ID"] == "7"


def test_parse_egl_device_map_rejects_malformed_entries() -> None:
    assert _parse_egl_device_map("0:0, 7:3") == {"0": "0", "7": "3"}
    for bad in ("0", "0:", ":3", "a:1", "0:b"):
        with pytest.raises(ValueError):
            _parse_egl_device_map(bad)


def test_egl_device_map_prefers_explicit_then_always_rederives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit wins; otherwise the map is re-derived, never read back from the cache.

    A cached map is exactly how a wrong one survived across runs: its render fingerprint still
    matched the host, so it was reused without question, and identity was wrong for all eight
    devices. Deriving costs one nvidia-smi call, one sysfs walk and one short child process, so
    there is nothing to save by caching it.
    """

    module = "robot_practice.experiments.literature.w0_fastwam_audit"
    monkeypatch.delenv("W0_EGL_DEVICE_MAP", raising=False)
    host = {"gpus": ["uuid-a|H20"], "gpu_count": 8, "driver_version": "580", "egl_device_count": 10}
    monkeypatch.setattr(f"{module}._render_fingerprint", lambda *_args, **_kwargs: host)

    explicit, source = _egl_device_map(["0", "7"], explicit="0:0,7:3")
    assert explicit == {"0": "0", "7": "3"}
    assert source == "explicit"

    monkeypatch.setenv("W0_EGL_DEVICE_MAP", "0:1,7:2")
    from_env, env_source = _egl_device_map(["0", "7"])
    assert from_env == {"0": "1", "7": "2"}
    assert env_source == "explicit"
    monkeypatch.delenv("W0_EGL_DEVICE_MAP")

    # An explicit map that omits a requested CUDA device must fail loudly rather than silently
    # falling back to the unsafe identity assumption.
    with pytest.raises(ValueError):
        _egl_device_map(["0", "7"], explicit="0:0")

    report = tmp_path / "egl_calibration.json"
    monkeypatch.setattr(f"{module}.DEFAULT_EGL_CALIBRATION", str(report))
    report.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "source": "pci-derived",
                "render_fingerprint": host,
                "device_map": {"0": "0", "7": "7"},
                "safe_egl_indices_by_cuda_device": {"0": [0], "7": [7]},
            }
        ),
        encoding="utf-8",
    )

    # Even a matching, pci-derived record must not decide the map: derivation runs and wins.
    derived_calls: list[Sequence[str]] = []

    def fake_derive(gpu_ids, *, runtime_env):
        derived_calls.append(list(gpu_ids))
        # Forbidden indices: CUDA 0's own GPU is EGL 6, CUDA 7's is EGL 1.
        return {"0": "6", "7": "1"}, {"identity_is_correct": False}

    monkeypatch.setattr(f"{module}._pci_derived_egl_device_map", fake_derive)
    monkeypatch.setattr(f"{module}._probe_render_pair", lambda **_kwargs: True)

    mapping, map_source = _egl_device_map(
        ["0", "7"], runtime_env={"W0_TEST": "1"}, allow_calibration=True
    )
    # Each device avoids its own GPU and the assignment stays a bijection over {1, 6}.
    assert mapping == {"0": "1", "7": "6"}
    assert map_source.startswith("pci-derived:")
    assert derived_calls == [["0", "7"]]
    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["device_map"] == {"0": "1", "7": "6"}

    # With no subprocess allowed, identity is returned but labelled so it cannot be mistaken
    # for a measured map. Nothing renders on that path.
    identity, identity_source = _egl_device_map(["0", "7"], allow_calibration=False)
    assert identity == {"0": "0", "7": "7"}
    assert identity_source == "identity-unverified"


def test_eval_manifests_record_the_render_stack() -> None:
    """Every eval manifest must carry the render fingerprint, not just the chosen map.

    A node change silently altered rendering behaviour: the same identity map ran 2,000
    episodes across 8 workers on one 8x H20 host and aborted under 8-way concurrency on
    another. The difference could not be explained, because no eval manifest recorded the
    fingerprint and the calibration file -- the only other place it lived -- had been
    overwritten by the next run. Both hosts' driver versions and GPU UUIDs are now
    unrecoverable.
    """

    source = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )
    # Both eval paths: upstream/local-checkpoint and lerobot.
    assert source.count('"render_fingerprint": _render_fingerprint(') == 2
    assert '"render_fingerprint": _render_fingerprint(upstream_runtime_env),' in source
    assert '"render_fingerprint": _render_fingerprint(runtime_env),' in source
    # The fingerprint has to include what actually varies between hosts.
    fingerprint = source[source.index("def _render_fingerprint(") :]
    fingerprint = fingerprint[: fingerprint.index("\ndef ")]
    for field in ("uuid", "driver_version", "egl_device_count"):
        assert field in fingerprint, field


def test_egl_pair_sweep_tool_exists_and_states_its_own_limits() -> None:
    """Inference about the EGL pairing has failed twice, so measurement needs a tool.

    EGL_CUDA_DEVICE_NV returns EGLError for every index on driver 535.261.03, and joining
    nvidia-smi PCI ids with /sys/class/drm and the EGL device list produced
    {0:6, 1:7, 2:4, 3:5, 4:2, 5:3, 6:0, 7:1} whose first pair aborts immediately. The sweep
    measures every cell in an isolated child, because a bad cell aborts natively.

    It must also state what a PASS does not mean: an evaluation aborted at about the 141st
    cumulative render after a 120-step probe had accepted the pair, so a short PASS is weak
    evidence rather than proof.
    """

    script = Path("scripts/experiments/literature/sweep_w0_egl_pairs.py")
    assert script.is_file()
    assert os.access(script, os.X_OK)
    body = script.read_text(encoding="utf-8")

    assert "_probe_render_pair(" in body
    assert "--steps" in body and "--cuda" in body and "--egl" in body
    # A PASS is not proof; the tool has to say so where the reader will see it.
    assert "which is not proof the pair is correct" in body
    # The three shapes worth distinguishing, so nobody reads a formula into the matrix.
    assert "identity is the only working map" in body
    assert "raise --steps" in body
    assert "no systematic index offset explains" in body


def test_render_probe_reports_why_it_rejected_a_pair() -> None:
    """A rejected pair must say why, or the two failure modes are indistinguishable.

    The probe used to return a bare boolean, so a pair rejected for an unrelated reason -- a
    missing asset, a bad env, an import error -- looked exactly like a genuine native render
    abort. The caller then either scanned for another index or refused to launch, with nothing
    in the log to tell which had happened.
    """

    source = Path("src/robot_practice/experiments/literature/w0_fastwam_audit.py").read_text(
        encoding="utf-8"
    )
    probe = source[source.index("def _probe_render_pair(") : source.index("def _autocalibrate_egl")]
    assert "probe failed cuda=" in probe
    assert "returncode=" in probe
    # The two env vars that define the pair, so the log is self-contained.
    assert "CUDA_VISIBLE_DEVICES=" in probe
    assert "MUJOCO_EGL_DEVICE_ID=" in probe
    assert "stdout tail:" in probe and "stderr tail:" in probe
    # Timeout and spawn failure must be distinguishable from a rejection too.
    assert "probe timed out" in probe
    assert "probe could not start" in probe


def test_probe_only_fallback_is_refused_unless_requested(monkeypatch, tmp_path: Path) -> None:
    """A failed PCI derivation must stop the run, not quietly downgrade to probing.

    That downgrade already happened once: the join produced nothing because nvidia-smi reports
    an eight-digit PCI domain while sysfs reports four, the render probe then accepted identity
    -- wrong for all eight devices -- and the evaluation aborted natively inside mjr_readPixels
    after about 141 cumulative renders. The probe can only fail to reject a pair inside its
    window; it can never prove one is right.
    """

    from robot_practice.experiments.literature import w0_fastwam_audit as audit

    monkeypatch.delenv("W0_EGL_DEVICE_MAP", raising=False)
    monkeypatch.delenv("W0_ALLOW_EGL_PROBE_FALLBACK", raising=False)
    monkeypatch.setattr(audit, "DEFAULT_EGL_CALIBRATION", str(tmp_path / "cal.json"))
    monkeypatch.setattr(audit, "_render_fingerprint", lambda *_a, **_k: {"gpu_count": 8})

    def no_pci(*_args, **_kwargs):
        raise RuntimeError("no drm= fields")

    monkeypatch.setattr(audit, "_pci_derived_egl_device_map", no_pci)
    probed: list[str] = []
    monkeypatch.setattr(
        audit,
        "_probe_render_pair",
        lambda **kwargs: probed.append(kwargs["egl_index"]) or True,
    )

    with pytest.raises(RuntimeError) as excinfo:
        audit._egl_device_map(["0"], runtime_env={"W0_TEST": "1"}, allow_calibration=True)
    message = str(excinfo.value)
    assert "PCI derivation failed" in message
    # The message must name every way out, or the operator has to read the source.
    assert "W0_EGL_DEVICE_MAP" in message
    assert "W0_ALLOW_EGL_PROBE_FALLBACK=1" in message
    assert probed == [], "no probing may happen before the operator accepts the downgrade"

    monkeypatch.setenv("W0_ALLOW_EGL_PROBE_FALLBACK", "1")
    mapping, source = audit._egl_device_map(
        ["0"], runtime_env={"W0_TEST": "1"}, allow_calibration=True
    )
    assert mapping == {"0": "0"}
    assert source.startswith("autocalibrated:")


def test_egl_autocalibration_prefers_identity_then_scans_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The render-probe fallback, for hosts where the PCI join is unavailable.

    PCI derivation is the primary route and runs first; this covers what happens when it
    cannot run at all (no sysfs DRM entries, no drm= fields in the EGL enumeration). Identity
    is probed first so the common case costs one probe per GPU, and only a device whose
    identity pair aborts falls back to scanning other indices.

    Note the limitation this fallback carries, and why it is no longer the primary route:
    cross-GPU rendering mostly works, so a probe of any finite length can accept a wrong pair.
    One evaluation aborted natively inside mjr_readPixels at about the 141st cumulative render
    after the 120-step probe had passed.
    """

    module = "robot_practice.experiments.literature.w0_fastwam_audit"
    monkeypatch.delenv("W0_EGL_DEVICE_MAP", raising=False)
    report = tmp_path / "egl_calibration.json"
    host = {"gpus": ["uuid-a|H20"], "gpu_count": 8, "driver_version": "580", "egl_device_count": 10}
    monkeypatch.setattr(f"{module}.DEFAULT_EGL_CALIBRATION", str(report))
    monkeypatch.setattr(f"{module}._render_fingerprint", lambda *_args, **_kwargs: host)

    def no_pci(*_args, **_kwargs):
        raise RuntimeError("no drm= fields in this environment")

    monkeypatch.setattr(f"{module}._pci_derived_egl_device_map", no_pci)
    # The fallback is opt-in: a probe-only map reached a real run once and cost an aborted
    # evaluation, so it must be requested rather than inherited.
    monkeypatch.setenv("W0_ALLOW_EGL_PROBE_FALLBACK", "1")

    attempted: list[tuple[str, str]] = []

    def fake_probe(*, cuda_device: str, egl_index: str, runtime_env, steps=0) -> bool:
        attempted.append((cuda_device, egl_index))
        # Mirror the measured 8x H20 behaviour: only (cuda 7, egl 7) aborts.
        return not (cuda_device == "7" and egl_index == "7")

    monkeypatch.setattr(f"{module}._probe_render_pair", fake_probe)

    mapping, source = _egl_device_map(
        ["0", "7"], runtime_env={"W0_TEST": "1"}, allow_calibration=True
    )
    assert mapping == {"0": "0", "7": "0"}
    assert source.startswith("autocalibrated:")
    # One probe for CUDA 0, then identity-then-fallback for CUDA 7.
    assert attempted == [("0", "0"), ("7", "7"), ("7", "0")]

    cached = json.loads(report.read_text(encoding="utf-8"))
    assert cached["render_fingerprint"] == host
    assert cached["selected_map"] == {"0": "0", "7": "0"}

    # A render-probe cache is deliberately NOT reused, so the next launch measures again.
    # Reusing it is what let a wrong identity map survive across runs: the probe cannot prove a
    # pair is good, only that it did not abort within the window. Re-probing costs one short
    # child per GPU on this fallback path; trusting a weak measurement costs an aborted
    # evaluation hours later.
    attempted.clear()
    reused, reused_source = _egl_device_map(
        ["0", "7"], runtime_env={"W0_TEST": "1"}, allow_calibration=True
    )
    assert reused == {"0": "0", "7": "0"}
    assert reused_source.startswith("autocalibrated:")
    assert attempted == [("0", "0"), ("7", "7"), ("7", "0")]

    # If no index works the launcher must refuse to start rather than abort later.
    report.unlink()
    monkeypatch.setattr(
        f"{module}._probe_render_pair", lambda **_kwargs: False
    )
    with pytest.raises(RuntimeError, match="no EGL render device index works"):
        _egl_device_map(["3"], runtime_env={"W0_TEST": "1"}, allow_calibration=True)


def test_upstream_overwrite_archives_previous_output(tmp_path: Path) -> None:
    output = tmp_path / "upstream_released"
    output.mkdir()
    (output / "old_result.json").write_text("old\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        _prepare_output_directory(output, overwrite=False)

    backup = _prepare_output_directory(output, overwrite=True)
    assert backup is not None
    assert backup.name.startswith("upstream_released.backup-")
    assert (backup / "old_result.json").read_text(encoding="utf-8") == "old\n"
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_upstream_overwrite_allows_stale_running_manifest(tmp_path: Path) -> None:
    output = tmp_path / "upstream_released"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"status": "running", "launcher_pid": 999999999}), encoding="utf-8"
    )

    backup = _prepare_output_directory(output, overwrite=True)
    assert backup is not None
    assert (backup / "run_manifest.json").is_file()
    assert output.is_dir()


def test_validation_phases_do_not_create_or_download_inputs(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    vendor = tmp_path / "third_party" / "w0_fastwam"
    cache = tmp_path / "pretrained" / "w0_fastwam"

    environment = validate_environments(
        config,
        vendor_root=vendor,
        output_root=tmp_path / "environment",
        dry_run=True,
    )
    artifacts = validate_artifacts(
        config,
        cache_root=cache,
        vendor_root=vendor,
        output_root=tmp_path / "artifacts",
        dry_run=True,
    )

    assert environment["validation_only"] is True
    assert environment["installs_performed"] is False
    assert environment["downloads_performed"] is False
    assert artifacts["validation_only"] is True
    assert artifacts["installs_performed"] is False
    assert artifacts["downloads_performed"] is False
    assert not vendor.exists()
    assert not cache.exists()
    assert artifacts["commands"] == []
    forbidden = {"create", "install", "clone", "fetch", "download", "extract"}
    for command in environment["commands"]:
        assert not forbidden.intersection(command)
