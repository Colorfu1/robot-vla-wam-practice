from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from robot_practice.experiments.literature import w0_video_generation_eval as evaluation


def _synthetic_vbench_info() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    quality_dimensions = [
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
    ]
    for image_type in evaluation.VBENCH_SUBJECT_TYPES:
        rows.append(
            {
                "image_type": image_type,
                "image_name": f"subject-{image_type}.jpg",
                "prompt_en": f"a subject of type {image_type}",
                "dimension": ["i2v_subject", *quality_dimensions],
            }
        )

    background_counts = {
        "abstract": 3,
        "architecture": 3,
        "indoor": 3,
        "scenery": 3,
    }
    for image_type, count in background_counts.items():
        for index in range(count):
            rows.append(
                {
                    "image_type": image_type,
                    "image_name": f"background-{image_type}-{index}.jpg",
                    "prompt_en": f"a {image_type} background {index}",
                    "dimension": [
                        "i2v_background",
                        "background_consistency",
                        "aesthetic_quality",
                        "imaging_quality",
                    ],
                }
            )

    for image_index in range(3):
        camera_image = f"background-abstract-{image_index}.jpg"
        for instruction in evaluation.VBENCH_CAMERA_INSTRUCTIONS:
            rows.append(
                {
                    "image_type": "abstract",
                    "image_name": camera_image,
                    "prompt_en": f"a background, {instruction}",
                    "dimension": ["camera_motion"],
                }
            )
    return rows


def test_vbench_mini_selection_is_deterministic_and_disjoint() -> None:
    info = _synthetic_vbench_info()
    first = evaluation._select_vbench_entries(info, seed=20260824)
    second = evaluation._select_vbench_entries(info, seed=20260824)

    assert first == second
    assert len(first) == 21
    assert len({str(row["image_name"]) for row in first}) == 15
    assert sum("i2v_subject" in row["dimension"] for row in first) == 7
    assert sum("i2v_background" in row["dimension"] for row in first) == 7
    assert sum("camera_motion" in row["dimension"] for row in first) == 7
    assert {
        str(row["prompt_en"]).split(", ")[-1]
        for row in first
        if "camera_motion" in row["dimension"]
    } == set(evaluation.VBENCH_CAMERA_INSTRUCTIONS)


def test_selection_hash_changes_with_seed() -> None:
    entry = {"image_type": "plant", "image_name": "plant.jpg", "prompt_en": "a plant"}
    assert evaluation._selection_hash(1, entry) == evaluation._selection_hash(1, entry)
    assert evaluation._selection_hash(1, entry) != evaluation._selection_hash(2, entry)


def test_dummy_video_frames_are_deterministic_and_protocol_shaped() -> None:
    image = np.zeros((8, 16, 3), dtype=np.uint8)
    image[:, :8] = 32
    image[:, 8:] = 224
    protocol = {"width": 16, "height": 8, "frames": 5}

    first = evaluation._dummy_video_frames(
        input_image=image,
        protocol=protocol,
        seed=123,
    )
    second = evaluation._dummy_video_frames(
        input_image=image,
        protocol=protocol,
        seed=123,
    )

    assert len(first) == len(second) == 5
    assert all(frame.shape == (8, 16, 3) for frame in first)
    assert all(frame.dtype == np.uint8 for frame in first)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert any(not np.array_equal(first[0], frame) for frame in first[1:])


def test_git_probe_passes_safe_directory_for_bind_mounted_checkouts(monkeypatch, tmp_path) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="pinned\n", stderr="")

    monkeypatch.setattr(evaluation.subprocess, "run", fake_run)
    assert evaluation._git_head(tmp_path) == "pinned"
    assert evaluation._git_status(tmp_path) == "pinned"
    expected = f"safe.directory={tmp_path.resolve()}"
    assert all(command[1:3] == ["-c", expected] for command in calls)


def test_center_crop_resize_and_future_metrics() -> None:
    source = np.zeros((4, 8, 3), dtype=np.uint8)
    resized = evaluation._center_crop_resize(source, width=4, height=4)
    assert resized.shape == (4, 4, 3)

    frames = [np.full((4, 8, 3), frame_index, dtype=np.uint8) for frame_index in range(9)]
    metrics = evaluation._w0_sample_metrics(
        frames,
        frames,
        lpips_model=None,
        lpips_device="cpu",
    )
    assert metrics["psnr"] > 100.0
    assert metrics["ssim"] == 1.0
    assert metrics["lpips"] is None


def test_aggregate_and_delta_conventions() -> None:
    rows = [
        {"suite": "libero_goal", "psnr": 10.0, "ssim": 0.5, "lpips": 0.4},
        {"suite": "libero_goal", "psnr": 12.0, "ssim": 0.7, "lpips": 0.2},
        {"suite": "libero_spatial", "psnr": 20.0, "ssim": 0.9, "lpips": 0.1},
    ]
    summary = evaluation._aggregate_sample_metrics(rows)
    assert summary["per_suite"]["libero_goal"]["count"] == 2
    assert summary["macro"]["psnr"] == 14.0
    assert evaluation._metric_delta("psnr", 12.0, 10.0) == 2.0
    assert evaluation._metric_delta("lpips", 0.2, 0.4) == 0.2


def test_multi_gpu_manifest_shards_are_deterministic_and_cover_all_samples(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    payload = {
        "schema_version": 1,
        "dataset": "W0-Video-40",
        "protocol": {"action_conditioned": False},
        "samples": [
            {"sample_id": f"sample-{index}", "seed": index} for index in range(10)
        ],
    }
    evaluation._write_json(manifest_path, payload)

    first = evaluation._make_manifest_shards(
        manifest_path=manifest_path,
        manifest_payload=payload,
        requested_gpus=4,
        work_root=tmp_path / "first",
    )
    second = evaluation._make_manifest_shards(
        manifest_path=manifest_path,
        manifest_payload=payload,
        requested_gpus=4,
        work_root=tmp_path / "second",
    )

    first_ids = [
        [str(sample["sample_id"]) for sample in subset] for _path, subset in first
    ]
    second_ids = [
        [str(sample["sample_id"]) for sample in subset] for _path, subset in second
    ]
    assert first_ids == second_ids == [
        ["sample-0", "sample-4", "sample-8"],
        ["sample-1", "sample-5", "sample-9"],
        ["sample-2", "sample-6"],
        ["sample-3", "sample-7"],
    ]
    assert sorted(sample_id for shard in first_ids for sample_id in shard) == [
        f"sample-{index}" for index in range(10)
    ]
    assert evaluation._read_json(first[0][0] / "manifest.json")["multi_gpu_shard"]["count"] == 4


def test_parser_accepts_multi_gpu_option() -> None:
    args = evaluation.build_parser().parse_args(
        ["--mode=generate", "--num-gpus=4", "--dataset=w0-video-40"]
    )
    assert args.num_gpus == 4


def test_parser_accepts_dummy_model_without_checkpoint() -> None:
    args = evaluation.build_parser().parse_args(
        ["--mode=generate", "--dataset=vbench-mini-21", "--variant=fw_v", "--dummy-model"]
    )
    assert args.dummy_model is True


def _w0_test_manifest(sample_count: int = 4) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "W0-Video-40",
        "protocol": {
            "kind": "fastwam-native",
            "width": 448,
            "height": 224,
            "frames": 9,
            "fps": 16,
            "steps": 10,
            "action_conditioned": False,
        },
        "samples": [
            {
                "sample_id": f"libero_spatial_task{index:02d}",
                "suite": "libero_spatial",
                "task_id": index,
                "seed": index,
            }
            for index in range(sample_count)
        ],
    }


def test_multi_gpu_worker_processes_isolate_visible_devices(tmp_path) -> None:
    commands = [
        [sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
        [sys.executable, "-c", "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"],
    ]
    worker_dirs = [tmp_path / "worker-0", tmp_path / "worker-1"]
    evaluation._run_multi_gpu_workers(
        stage="test:multi-gpu",
        worker_commands=commands,
        worker_dirs=worker_dirs,
        device_tokens=["5", "7"],
        progress_root=tmp_path / "progress",
    )
    assert "5" in (worker_dirs[0] / "worker.stdout.log").read_text()
    assert "7" in (worker_dirs[1] / "worker.stdout.log").read_text()


def test_multi_gpu_generation_merges_worker_videos(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _w0_test_manifest()
    evaluation._write_json(manifest_path, manifest)

    monkeypatch.setattr(evaluation, "_require_multi_gpu_devices", lambda count: ["0", "1"])

    def fake_workers(**kwargs) -> None:
        for worker_dir in kwargs["worker_dirs"]:
            shard_manifest = evaluation._read_json(worker_dir.parent / "manifest.json")
            worker_dir.mkdir(parents=True, exist_ok=True)
            video_dir = worker_dir / "videos"
            video_dir.mkdir()
            for sample in shard_manifest["samples"]:
                (video_dir / f"{sample['sample_id']}.mp4").write_bytes(
                    str(sample["sample_id"]).encode()
                )
            evaluation._write_json(
                worker_dir / "run_config.json",
                {
                    "dataset": "w0-video-40",
                    "variant": "fw_v",
                    "generated": len(shard_manifest["samples"]),
                    "newly_generated": len(shard_manifest["samples"]),
                },
            )

    monkeypatch.setattr(evaluation, "_run_multi_gpu_workers", fake_workers)
    output_root = tmp_path / "fw_v"
    result = evaluation.generate_videos_multi_gpu(
        dataset="w0-video-40",
        variant="fw_v",
        checkpoint=tmp_path / "checkpoint.pt",
        manifest=manifest_path,
        output_root=output_root,
        vendor_root=tmp_path / "vendor",
        cache_root=tmp_path / "cache",
        wan_root=tmp_path / "wan",
        wan_model_root=tmp_path / "wan-model",
        device="cuda:0",
        overwrite=False,
        num_gpus=2,
        config=tmp_path / "config.json",
    )
    assert result["generated"] == 4
    assert result["multi_gpu"]["worker_count"] == 2
    assert len(list((output_root / "videos").glob("*.mp4"))) == 4


def test_multi_gpu_score_merges_in_manifest_order(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _w0_test_manifest()
    evaluation._write_json(manifest_path, manifest)
    videos = tmp_path / "videos"
    videos.mkdir()

    monkeypatch.setattr(evaluation, "_require_multi_gpu_devices", lambda count: ["0", "1"])

    def fake_workers(**kwargs) -> None:
        for worker_dir in kwargs["worker_dirs"]:
            shard_manifest = evaluation._read_json(worker_dir / "manifest.json")
            rows = [
                {
                    "sample_id": sample["sample_id"],
                    "suite": sample["suite"],
                    "task_id": sample["task_id"],
                    "psnr": float(sample["task_id"]),
                    "ssim": 0.5,
                    "lpips": 0.25,
                }
                for sample in shard_manifest["samples"]
            ]
            evaluation._write_json(
                worker_dir / "metrics.json",
                {
                    "schema_version": 1,
                    "dataset": "W0-Video-40",
                    "variant": "fw_v",
                    "metric_availability": {"psnr": True, "ssim": True, "lpips": True},
                    "samples": rows,
                    "summary": evaluation._aggregate_sample_metrics(rows),
                },
            )

    monkeypatch.setattr(evaluation, "_run_multi_gpu_workers", fake_workers)
    output = tmp_path / "metrics.json"
    result = evaluation.score_w0_multi_gpu(
        variant="fw_v",
        manifest=manifest_path,
        videos=videos,
        output=output,
        checkpoint=None,
        lpips_weights=None,
        device="cuda:0",
        allow_missing_lpips=True,
        num_gpus=2,
        config=tmp_path / "config.json",
    )
    assert [row["sample_id"] for row in result["samples"]] == [
        sample["sample_id"] for sample in manifest["samples"]
    ]
    assert result["summary"]["macro"]["count"] == 4
    assert json.loads(output.read_text())["multi_gpu"]["worker_count"] == 2


def test_protocol_requires_action_conditioning_disabled(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    evaluation._write_json(
        path,
        {
            "schema_version": 1,
            "kind": "fastwam-native",
            "width": 448,
            "height": 224,
            "frames": 9,
            "fps": 16,
            "steps": 10,
            "action_conditioned": False,
        },
    )
    assert evaluation._protocol_from_file(path)["kind"] == "fastwam-native"


class _FakeMoT:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def state_dict(self) -> dict[str, object]:
        return {key: object() for key in self._keys}


class _FakeModel:
    def __init__(self, keys: list[str]) -> None:
        self.mot = _FakeMoT(keys)


def test_checkpoint_load_audit_refuses_a_silently_partial_load(tmp_path) -> None:
    """The vendor loads `mot` with strict=False, so a key mismatch is otherwise invisible.

    This loader also sets `model.skip_dit_load_from_pretrain=true`, so an unmatched key leaves
    that tensor randomly initialised and the model still emits decodable frames.
    """
    keys = ["mixtures.video.blocks.0.weight", "mixtures.action.blocks.0.weight"]
    checkpoint = tmp_path / "step_000001.pt"
    checkpoint.write_bytes(b"unused by the audit")

    matching = evaluation._assert_checkpoint_actually_loaded(
        _FakeModel(keys), {"mot": dict.fromkeys(keys)}, checkpoint=checkpoint
    )
    assert matching["missing_count"] == 0
    assert matching["unexpected_count"] == 0
    assert matching["loaded"] == len(keys)

    # A DeepSpeed-style prefix makes every key miss; strict=False would accept it silently.
    prefixed = {"mot": {f"module.{key}": None for key in keys}}
    try:
        evaluation._assert_checkpoint_actually_loaded(
            _FakeModel(keys), prefixed, checkpoint=checkpoint
        )
    except RuntimeError as error:
        assert "did not fully load" in str(error)
        assert "strict=False" in str(error)
    else:  # pragma: no cover - the audit must not accept this
        raise AssertionError("prefixed checkpoint keys must be rejected")

    # A payload without `mot` cannot be verified at all and must not pass.
    try:
        evaluation._assert_checkpoint_actually_loaded(
            _FakeModel(keys), {"dit": {}}, checkpoint=checkpoint
        )
    except RuntimeError as error:
        assert "no 'mot' state dict" in str(error)
    else:  # pragma: no cover
        raise AssertionError("a payload without 'mot' must be rejected")


def test_single_camera_is_a_marked_diagnostic_not_the_protocol() -> None:
    """The frozen protocol is 448x224 two-camera; a single view must be self-identifying.

    A frozen video expert carries unmodified Wan2.2 weights, and no natural video contains a
    side-by-side camera pair, so the concatenated layout confounds "cannot predict this domain"
    with "cannot read this layout". The single-camera path separates them, but its outputs must
    never be mistakable for protocol results.
    """
    parser = evaluation.build_parser()
    default = parser.parse_args(["--mode=generate"])
    assert default.single_camera == "both"
    assert parser.parse_args(["--mode=generate", "--single-camera=front"]).single_camera == "front"

    sample = {
        "frame_indices": [0, 4, 8],
        "front_video": "front.mp4",
        "wrist_video": "wrist.mp4",
    }
    calls: list[str] = []

    def fake_read(path, indices=None):
        calls.append(str(path))
        return [np.zeros((512, 512, 3), np.uint8) for _ in (indices or [0])]

    original = evaluation._read_video_frames
    evaluation._read_video_frames = fake_read
    try:
        both = evaluation._w0_sample_input(sample, camera="both")
        calls.clear()
        front = evaluation._w0_sample_input(sample, camera="front")
        front_only_calls = list(calls)
        gt_front = evaluation._w0_ground_truth(sample, camera="front")
    finally:
        evaluation._read_video_frames = original

    assert both.shape == (224, 448, 3)
    assert front.shape == (224, 224, 3)
    # the wrist video must not even be opened when a single camera is requested
    assert all("wrist" not in item for item in front_only_calls)
    assert [frame.shape for frame in gt_front] == [(224, 224, 3)] * 3

    # the flag must be refused where it has no meaning, before any file is touched
    try:
        evaluation.generate_videos(
            dataset="vbench-mini-21",
            variant="fw_v",
            checkpoint=None,
            manifest="unused",
            output_root="unused",
            vendor_root="unused",
            cache_root="unused",
            wan_root="unused",
            wan_model_root="unused",
            device="cpu",
            overwrite=False,
            single_camera="front",
        )
    except ValueError as error:
        assert "only applies to the w0-video-40 dataset" in str(error)
    else:  # pragma: no cover
        raise AssertionError("--single-camera must be refused for the VBench dataset")
    try:
        evaluation._w0_sample_input(sample, camera="stereo")
    except ValueError as error:
        assert "unsupported camera" in str(error)
    else:  # pragma: no cover
        raise AssertionError("an unknown camera must be rejected")


def test_frame_comparison_tool_does_not_depend_on_opencv_codecs() -> None:
    """`cv2.imwrite` fails on OpenCV builds without the PNG writer, after all the work is done.

    That happened on a worker whose `cv2` reported version 5.0.0 yet raised
    "could not find a writer for the specified extension". Pillow is already a hard dependency
    of the evaluator module, so the tool must use it and must not reintroduce cv2.
    """
    import pathlib
    import re

    source = pathlib.Path("scripts/experiments/literature/compare_w0_video_frames.py").read_text(
        encoding="utf-8"
    )
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith(("#", '"""', "``"))
    )
    code = re.sub(r'""".*?"""', "", code, flags=re.DOTALL)
    assert "import cv2" not in code
    assert "cv2." not in code
    assert "from PIL import" in code


def test_single_camera_probe_script_is_the_pasteable_entry_point() -> None:
    """The direct generate invocation is long enough that pasting it inserts newlines.

    That failure is silent in a bad way: the first line runs without its arguments and the
    remaining lines execute as separate commands ("--single-camera=front: command not found").
    The probe script exists so the operator pastes one short line, so it must carry the long
    parts itself: both variants, the flag, and the derived 3-sample manifest.
    """
    import subprocess
    from pathlib import Path

    script = Path("scripts/experiments/literature/probe_w0_single_camera.sh")
    body = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in body
    assert '--single-camera="${camera}"' in body
    assert "for variant in fw_fz fw_v" in body
    # the probe manifest must be derived from the frozen 40-sample one, not selected afresh
    assert "w0_video_40/manifest.json" in body
    assert "libero_spatial_task00" in body
    # a missing checkpoint must stop the run rather than silently produce nothing
    assert "missing checkpoint" in body

    rejected = subprocess.run(
        ["bash", str(script), "stereo"], capture_output=True, text=True, check=False
    )
    assert rejected.returncode == 2
    assert "[front|wrist]" in rejected.stderr

    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_full_run_script_is_the_protocol_and_not_the_diagnostic() -> None:
    """Two entry points exist and must not be confusable.

    ``run_w0_video_40.sh`` is the frozen protocol: 40 samples, both cameras at 448x224.
    ``probe_w0_single_camera.sh`` is a diagnostic: 3 samples, one camera at 224x224. Mixing
    them would put diagnostic videos in the formal output tree, so the protocol script must
    never pass --single-camera and must score every variant it generates.
    """
    import subprocess
    from pathlib import Path

    script = Path("scripts/experiments/literature/run_w0_video_40.sh")
    body = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in body
    assert "--single-camera" not in body
    assert "--mode=generate" in body
    assert "--mode=score" in body
    # both stages must run for each variant, so a generated variant is always scored
    assert 'for variant in "${variants[@]}"' in body
    # the read-out must state that it is not --mode=compare
    assert "not a substitute for --mode=compare" in body
    # guards: a missing manifest, checkpoint or LPIPS weight must stop the run
    assert "run --mode=prepare-w0 first" in body
    assert "missing checkpoint" in body
    assert "copy it from the lpips package" in body

    for args, expected in ((["nope"], "[both|fw_v|fw_fz]"), (["both", "0"], "positive integer")):
        rejected = subprocess.run(
            ["bash", str(script), *args], capture_output=True, text=True, check=False
        )
        assert rejected.returncode == 2
        assert expected in rejected.stderr

    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_documentation_figures_are_tracked_and_not_gitignored() -> None:
    """A figure under a gitignored path renders locally and is a broken link everywhere else.

    That is exactly what happened when README embedded the montages from ``outputs/``: the
    conclusion depends on seeing that FW-FZ's frames are unrelated rather than merely blurrier,
    so the figures have to be committed. This asserts every embedded image in README and in the
    video-generation guide resolves to a real file that git will actually track.
    """
    import re
    import subprocess
    from pathlib import Path

    documents = [
        Path("README.md"),
        Path("reports/guides/literature/02_w0_wan_video_generation_preservation.md"),
    ]
    checked = 0
    for document in documents:
        body = document.read_text(encoding="utf-8")
        for reference in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", body):
            target = (document.parent / reference).resolve()
            assert target.is_file(), f"{document}: missing figure {reference}"
            relative = target.relative_to(Path.cwd())
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", str(relative)], check=False
            ).returncode
            assert ignored != 0, f"{document}: figure {relative} is gitignored"
            # keep committed figures small enough to belong in a source repository
            assert target.stat().st_size < 600_000, f"{relative} is {target.stat().st_size} bytes"
            checked += 1
    assert checked >= 2, f"expected at least two embedded figures, found {checked}"
