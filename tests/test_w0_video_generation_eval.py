from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

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
    smoke = evaluation.build_parser().parse_args(
        ["--mode=smoke-vbench", "--dataset=vbench-mini-21"]
    )
    assert smoke.mode == "smoke-vbench"
    full = evaluation.build_parser().parse_args(
        ["--mode=generate", "--dataset=vbench-i2v-full", "--variant=fw_v"]
    )
    assert full.dataset == "vbench-i2v-full"
    libero = evaluation.build_parser().parse_args(
        ["--mode=prepare-libero-wan", "--input-camera=front"]
    )
    assert libero.mode == "prepare-libero-wan"
    assert libero.input_camera == "front"


def test_parser_accepts_dummy_model_without_checkpoint() -> None:
    args = evaluation.build_parser().parse_args(
        ["--mode=generate", "--dataset=vbench-mini-21", "--variant=fw_v", "--dummy-model"]
    )
    assert args.dummy_model is True


def test_full_vbench_dimensions_include_temporal_flickering() -> None:
    assert evaluation._vbench_dimensions_for_manifest(
        {"dataset": "VBench-I2V-full"}
    ) == evaluation.VBENCH_FULL_DIMENSIONS
    assert evaluation._vbench_dimensions_for_manifest(
        {"dataset": "VBench-I2V-mini-21"}
    ) == evaluation.VBENCH_DIMENSIONS


def test_prepare_vbench_full_manifest_keeps_all_rows_and_reuses_images(tmp_path) -> None:
    image_root = tmp_path / "crop" / "16-9"
    image_root.mkdir(parents=True)
    for image_name, value in (("a.jpg", 32), ("b.jpg", 96)):
        Image.fromarray(np.full((8, 16, 3), value, dtype=np.uint8)).save(image_root / image_name)
    rows = [
        {
            "image_type": "abstract",
            "image_name": "a.jpg",
            "prompt_en": "a liquid",
            "dimension": [
                "i2v_background",
                "background_consistency",
                "aesthetic_quality",
                "imaging_quality",
                "temporal_flickering",
            ],
        },
        {
            "image_type": "abstract",
            "image_name": "a.jpg",
            "prompt_en": "a liquid, camera static",
            "dimension": ["camera_motion"],
        },
        {
            "image_type": "plant",
            "image_name": "b.jpg",
            "prompt_en": "a plant",
            "dimension": [
                "i2v_subject",
                "subject_consistency",
                "motion_smoothness",
                "dynamic_degree",
                "aesthetic_quality",
                "imaging_quality",
                "temporal_flickering",
            ],
        },
    ]
    manifest = evaluation.prepare_vbench_full_manifest(
        full_info=rows,
        assets={"full_info": {"path": "metadata.json"}},
        image_roots=[image_root.parent],
        protocol={"width": 16, "height": 8, "frames": 5, "steps": 10},
        output=tmp_path / "full" / "manifest.json",
        base_seed=52000,
    )
    assert manifest["dataset"] == "VBench-I2V-full"
    assert len(manifest["samples"]) == 3
    assert manifest["selected_image_count"] == 2
    assert manifest["evaluation_dimensions"] == list(evaluation.VBENCH_FULL_DIMENSIONS)
    assert len({sample["conditioning_image"] for sample in manifest["samples"]}) == 2
    assert all(Path(sample["conditioning_image"]).is_file() for sample in manifest["samples"])
    assert "camera static" in next(
        sample["video_filename"]
        for sample in manifest["samples"]
        if "camera_motion" in sample["dimensions"]
    )


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


def _vbench_test_manifest(sample_count: int = 4) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset": "VBench-I2V-mini-21",
        "protocol": {
            "kind": "wan22-ti2v-compatible",
            "width": 16,
            "height": 8,
            "frames": 5,
            "fps": 16,
            "steps": 10,
            "vbench_resolution": "16-9",
            "action_conditioned": False,
        },
        "samples": [
            {
                "sample_id": f"vbench-sample-{index}",
                "prompt_en": f"test prompt {index}",
                "dimensions": list(evaluation.VBENCH_DIMENSIONS),
                "video_filename": f"sample-{index}.mp4",
                "conditioning_image": f"images/sample-{index}.png",
                "seed": index,
            }
            for index in range(sample_count)
        ],
    }


def test_vbench_smoke_manifest_covers_all_dimensions_with_shared_samples() -> None:
    subject_dimensions = [
        "i2v_subject",
        "subject_consistency",
        "motion_smoothness",
        "dynamic_degree",
    ]
    background_dimensions = [
        "i2v_background",
        "background_consistency",
        "aesthetic_quality",
        "imaging_quality",
    ]
    manifest = {
        "dataset": "VBench-I2V-mini-21",
        "samples": [
            {"sample_id": "subject", "dimensions": subject_dimensions},
            {"sample_id": "background", "dimensions": background_dimensions},
            {"sample_id": "camera", "dimensions": ["camera_motion"]},
        ],
    }

    smoke, coverage = evaluation._make_vbench_smoke_manifest(manifest)

    assert [sample["sample_id"] for sample in smoke["samples"]] == [
        "subject",
        "background",
        "camera",
    ]
    assert set(coverage) == set(evaluation.VBENCH_DIMENSIONS)
    assert smoke["smoke"]["selected_sample_count"] == 3

    full_manifest = {
        "dataset": "VBench-I2V-full",
        "evaluation_dimensions": list(evaluation.VBENCH_FULL_DIMENSIONS),
        "samples": [
            {
                "sample_id": "full-subject",
                "dimensions": [*subject_dimensions, "temporal_flickering"],
            },
            {
                "sample_id": "full-background",
                "dimensions": [*background_dimensions, "temporal_flickering"],
            },
            {"sample_id": "full-camera", "dimensions": ["camera_motion"]},
        ],
    }
    full_smoke, full_coverage = evaluation._make_vbench_smoke_manifest(full_manifest)
    assert set(full_coverage) == set(evaluation.VBENCH_FULL_DIMENSIONS)
    assert full_smoke["smoke"]["selected_sample_count"] == 3


def test_vbench_dino_weight_is_staged_into_torch_hub_cache(tmp_path) -> None:
    cache_root = tmp_path / "vbench"
    source = cache_root / "dino_model/dino_vitbase16_pretrain.pth"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"dino")
    hub_dir = cache_root / "torch/hub"

    staged = evaluation._stage_vbench_torch_hub_weights(cache_root, hub_dir)

    target = hub_dir / "checkpoints/dino_vitbase16_pretrain.pth"
    assert target.read_bytes() == b"dino"
    assert staged == [evaluation._portable(target)]
    assert evaluation._stage_vbench_torch_hub_weights(cache_root, hub_dir) == []


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


def test_multi_gpu_generation_accepts_vbench_manifest(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _vbench_test_manifest()
    evaluation._write_json(manifest_path, manifest)

    monkeypatch.setattr(evaluation, "_require_multi_gpu_devices", lambda count: ["0", "1"])

    def fake_workers(**kwargs) -> None:
        for worker_dir in kwargs["worker_dirs"]:
            shard_manifest = evaluation._read_json(worker_dir.parent / "manifest.json")
            worker_dir.mkdir(parents=True, exist_ok=True)
            video_dir = worker_dir / "videos"
            video_dir.mkdir()
            for sample in shard_manifest["samples"]:
                (video_dir / sample["video_filename"]).write_bytes(b"video")
            evaluation._write_json(
                worker_dir / "run_config.json",
                {
                    "dataset": "vbench-mini-21",
                    "variant": "fw_v",
                    "generated": len(shard_manifest["samples"]),
                    "newly_generated": len(shard_manifest["samples"]),
                },
            )

    monkeypatch.setattr(evaluation, "_run_multi_gpu_workers", fake_workers)
    output_root = tmp_path / "fw_v"
    result = evaluation.generate_videos_multi_gpu(
        dataset="vbench-mini-21",
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
    assert result["dataset"] == "vbench-mini-21"
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


def test_multi_gpu_vbench_score_merges_weighted_dimensions(monkeypatch, tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = _vbench_test_manifest()
    evaluation._write_json(manifest_path, manifest)
    videos = tmp_path / "videos"
    videos.mkdir()
    for sample in manifest["samples"]:
        (videos / sample["video_filename"]).write_bytes(b"video")

    monkeypatch.setattr(evaluation, "_require_multi_gpu_devices", lambda count: ["0", "1"])

    def fake_workers(**kwargs) -> None:
        for worker_dir in kwargs["worker_dirs"]:
            shard_manifest = evaluation._read_json(worker_dir / "manifest.json")
            worker_index = int(worker_dir.name.split("-")[-1])
            score = 0.2 if worker_index == 0 else 0.6
            dimensions = {
                dimension: {
                    "score": score,
                    "sample_count": len(shard_manifest["samples"]),
                }
                for dimension in evaluation.VBENCH_DIMENSIONS
            }
            evaluation._write_json(
                worker_dir / "metrics.json",
                {
                    "schema_version": 1,
                    "dataset": "VBench-I2V-mini-21",
                    "variant": "fw_v",
                    "protocol": manifest["protocol"],
                    "checkpoint": {"path": "checkpoint.pt"},
                    "vbench": {
                        "root": "third_party/vbench",
                        "dimensions": dimensions,
                    },
                },
            )

    monkeypatch.setattr(evaluation, "_run_multi_gpu_workers", fake_workers)
    output = tmp_path / "fw_v" / "metrics.json"
    result = evaluation.score_vbench_multi_gpu(
        variant="fw_v",
        manifest=manifest_path,
        videos=videos,
        output=output,
        vbench_root=tmp_path / "vbench",
        cache_root=tmp_path / "cache",
        device="cuda:0",
        allow_vbench_downloads=False,
        num_gpus=2,
        config=tmp_path / "config.json",
    )
    assert result["vbench"]["dimensions"]["i2v_subject"]["score"] == 0.4
    assert result["vbench"]["dimensions"]["i2v_subject"]["sample_count"] == 4
    assert result["vbench"]["aggregation"] == "sample_count_weighted_mean_of_shards"
    assert result["multi_gpu"]["worker_count"] == 2
    assert (output.parent / "vbench_custom_info.json").is_file()


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


def test_image_layer_installs_into_both_conda_envs() -> None:
    """check-protocol runs in the model env, not the scoring env.

    An earlier version of the layer installed only into robot-practice, so the official Wan2.2
    strict-load check failed with ModuleNotFoundError for ftfy, the planned fallback fired, and
    a whole VBench-I2V run was produced at the FastWAM operating point instead of Wan2.2's. The
    fallback is silent by design, so the guard has to live here.
    """
    from pathlib import Path

    body = Path("docker/training/Dockerfile.w0-fastwam-video-eval").read_text(encoding="utf-8")

    scoring = "/opt/conda/envs/robot-practice/bin/python -m pip install"
    model = "/opt/conda/envs/fastwam-upstream/bin/python -m pip install"
    assert scoring in body, "the scoring env must still be installed into"
    assert model in body, "the model env must be installed into or check-protocol falls back"

    # the three packages the official `from wan import WanTI2V` chain needs, pinned
    model_stage = body.split(model, 1)[1]
    for requirement in ("ftfy==6.3.1", "decord==0.6.0", "librosa==0.11.0"):
        assert requirement in model_stage, f"{requirement} missing from the model-env stage"

    # --no-deps everywhere, or transformers/numpy move and the official path breaks differently
    assert body.count("--no-deps") >= 2
    # the model env's pinned versions must be asserted at build time
    assert 'transformers.__version__ == "4.49.0"' in body
    assert 'torch.__version__ == "2.7.1+cu128"' in body


def test_protocol_refuses_a_fallback_caused_by_a_missing_package(monkeypatch, tmp_path) -> None:
    """A missing package says nothing about the checkpoints, so it must not pick the protocol.

    This is the failure that already cost a full VBench-I2V run: one variant recorded
    ``ModuleNotFoundError: No module named 'ftfy'``, the planned fallback fired, and both
    variants generated at 448x224 / 9 frames instead of Wan2.2's 1280x720 / 81 frames.

    Driven through ``check_protocol`` rather than the guard alone, because the guard being
    correct is worthless if it is not wired into the decision.
    """
    fw_v = tmp_path / "fw_v.pt"
    fw_fz = tmp_path / "fw_fz.pt"
    for path in (fw_v, fw_fz):
        path.write_bytes(b"not read; _wan_strict_video_load is stubbed")
    monkeypatch.setattr(evaluation, "_require_pinned_checkout", lambda *a, **k: None)
    monkeypatch.setattr(evaluation, "_git_head", lambda *a, **k: "0" * 40)

    def run(reasons: dict[str, object]):
        monkeypatch.setattr(
            evaluation,
            "_wan_strict_video_load",
            lambda checkpoint, wan_root: dict(reasons[Path(checkpoint).stem]),
        )
        return evaluation.check_protocol(
            wan_root=tmp_path,
            fw_v=fw_v,
            fw_fz=fw_fz,
            model_root=tmp_path,
            device="cpu",
            output=tmp_path / "protocol.json",
        )

    loaded = {"strict_load": True, "video_state_keys": 825}
    for reason in (
        "ModuleNotFoundError: No module named 'ftfy'",
        "ImportError: cannot import name WanTI2V",
    ):
        monkeypatch.delenv("W0_ALLOW_ENVIRONMENT_FALLBACK", raising=False)
        try:
            run({"fw_v": {"strict_load": False, "reason": reason}, "fw_fz": loaded})
        except RuntimeError as error:
            message = str(error)
            assert "missing package" in message
            assert "fw_v" in message and reason.split(":")[0] in message
            assert "W0_ALLOW_ENVIRONMENT_FALLBACK" in message
        else:  # pragma: no cover - the guard must fire
            raise AssertionError(f"{reason!r} must be refused")

        # the escape hatch stays available for a deliberately degraded protocol
        monkeypatch.setenv("W0_ALLOW_ENVIRONMENT_FALLBACK", "1")
        protocol = run({"fw_v": {"strict_load": False, "reason": reason}, "fw_fz": loaded})
        assert protocol["kind"] == "fastwam-native"

    # a genuine weight mismatch is a protocol result and must still fall back silently
    monkeypatch.delenv("W0_ALLOW_ENVIRONMENT_FALLBACK", raising=False)
    protocol = run(
        {
            "fw_v": {"strict_load": False, "reason": "RuntimeError: Missing key(s) in state_dict"},
            "fw_fz": loaded,
        }
    )
    assert protocol["kind"] == "fastwam-native"
    assert protocol["wan22_candidate"]["reason"] == "at least one video expert failed strict load"


def test_protocol_fastwam_resolution_only_changes_only_spatial_resolution(
    monkeypatch, tmp_path
) -> None:
    fw_v = tmp_path / "fw_v.pt"
    fw_fz = tmp_path / "fw_fz.pt"
    for path in (fw_v, fw_fz):
        path.write_bytes(b"checkpoint stub")
    loaded = {"strict_load": True, "video_state_keys": 825}
    monkeypatch.setattr(evaluation, "_require_pinned_checkout", lambda *a, **k: None)
    monkeypatch.setattr(evaluation, "_git_head", lambda *a, **k: "0" * 40)
    monkeypatch.setattr(evaluation, "_wan_strict_video_load", lambda *a, **k: loaded)
    monkeypatch.setattr(evaluation, "_load_wan22_ti2v_model", lambda **kwargs: object())

    def fake_generate(**kwargs):
        protocol = kwargs["protocol"]
        return [np.zeros((int(protocol["height"]), int(protocol["width"]), 3), dtype=np.uint8)] * int(
            protocol["frames"]
        )

    monkeypatch.setattr(evaluation, "_generate_wan22_video", fake_generate)
    protocol = evaluation.check_protocol(
        wan_root=tmp_path,
        fw_v=fw_v,
        fw_fz=fw_fz,
        model_root=tmp_path,
        device="cpu",
        output=tmp_path / "protocol.json",
        operating_point="fastwam-resolution-only",
    )

    assert protocol["kind"] == "wan22-ti2v-compatible"
    assert protocol["operating_point"] == "fastwam-resolution-only"
    assert (protocol["width"], protocol["height"]) == (448, 224)
    assert protocol["frames"] == 81
    assert protocol["fps"] == 24
    assert protocol["steps"] == 50
    assert protocol["shift"] == 5.0
    assert protocol["sample_solver"] == "unipc"
    assert protocol["guide_scale"] == 5.0
    assert protocol["action_conditioned"] is False
    assert protocol["resolution_reference"]["official"]["width"] == 1248
    assert protocol["resolution_reference"]["official"]["frames"] == 81
    assert protocol["resolution_reference"]["fastwam"]["fps"] == 24
    assert protocol["wan22_candidate"]["smoke_frames"] == 81


def test_prepare_libero_wan_manifest_uses_same_command_and_protocol_shape(
    monkeypatch, tmp_path
) -> None:
    source_samples = [
        {
            "sample_id": f"libero_spatial_task{index:02d}",
            "suite": "libero_spatial",
            "task_id": index,
            "episode_index": index,
            "start_frame": 5,
            "frame_indices": [5, 9, 13],
            "front_video": "front.mp4",
            "wrist_video": "wrist.mp4",
            "language": f"pick object {index}",
            "motion_score": float(index),
        }
        for index in range(40)
    ]
    monkeypatch.setattr(
        evaluation,
        "prepare_w0_manifest",
        lambda **kwargs: {
            "dataset_fingerprint": {"libero_spatial": {"total_frames": 40}},
            "samples": source_samples,
        },
    )
    monkeypatch.setattr(
        evaluation,
        "_w0_sample_input",
        lambda sample, camera: np.full((4, 8, 3), int(sample["task_id"]), dtype=np.uint8),
    )
    # widening the ground-truth window is the only case that can read past a short episode, so the
    # builder looks up episode lengths; 200 frames is long enough for a window starting at 5
    monkeypatch.setattr(
        evaluation,
        "_load_w0_task_rows",
        lambda dataset: ({}, {index: {"length": 200} for index in range(40)}, {}),
    )
    protocol = {
        "schema_version": 1,
        "kind": "wan22-ti2v-compatible",
        "width": 448,
        "height": 224,
        "frames": 81,
        "fps": 24,
        "steps": 50,
        "shift": 5.0,
        "sample_solver": "unipc",
        "guide_scale": 5.0,
        "action_conditioned": False,
    }

    manifest = evaluation.prepare_libero_wan_manifest(
        dataset_root=tmp_path / "dataset",
        suites=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        selection="mini-21",
        raw_frames=81,
        video_stride=1,
        selection_seed=20260824,
        base_seed=52000,
        protocol=protocol,
        input_camera="both",
        output=tmp_path / "manifest.json",
    )

    assert manifest["dataset"] == "LIBERO-Wan-mini-21"
    assert len(manifest["samples"]) == 21
    assert manifest["protocol"]["input_source"] == "libero"
    assert manifest["protocol"]["input_camera"] == "both"
    assert all(sample["model_prompt"].endswith(sample["command"]) for sample in manifest["samples"])
    assert all(Path(sample["conditioning_image"]).is_file() for sample in manifest["samples"])
    assert [sample["seed"] for sample in manifest["samples"]] == list(range(52000, 52021))

    # The ground-truth window is widened to the protocol's 81 frames while the *selection* stays on
    # the frozen 33/4 window, so LIBERO-Wan is the same episodes and start frames as W0-Video-40.
    # Ranking candidates over a different window would silently pick different samples.
    for sample in manifest["samples"]:
        assert len(sample["frame_indices"]) == 81, sample["sample_id"]
        assert sample["frame_indices"][0] == sample["start_frame"] == 5
        assert sample["frame_indices"] == list(range(5, 86))
        assert sample["source_frame_indices"] == [5, 9, 13]
        assert sample["episode_length"] == 200
    import inspect

    source = inspect.getsource(evaluation.prepare_libero_wan_manifest)
    assert "raw_frames=W0_FROZEN_RAW_FRAMES" in source
    assert "video_stride=W0_FROZEN_VIDEO_STRIDE" in source

    # a window that runs off the end must be refused, not silently truncated
    monkeypatch.setattr(
        evaluation,
        "_load_w0_task_rows",
        lambda dataset: ({}, {index: {"length": 40} for index in range(40)}, {}),
    )
    with pytest.raises(ValueError, match="ground-truth window needs"):
        evaluation.prepare_libero_wan_manifest(
            dataset_root=tmp_path / "dataset",
            suites=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
            selection="mini-21",
            raw_frames=81,
            video_stride=1,
            selection_seed=20260824,
            base_seed=52000,
            protocol=protocol,
            input_camera="both",
            output=tmp_path / "short.json",
        )

    # an inconsistent (raw_frames, stride, protocol frames) triple must be refused up front
    with pytest.raises(ValueError, match="but the protocol asks for"):
        evaluation.prepare_libero_wan_manifest(
            dataset_root=tmp_path / "dataset",
            suites=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
            selection="mini-21",
            raw_frames=33,
            video_stride=4,
            selection_seed=20260824,
            base_seed=52000,
            protocol=protocol,
            input_camera="both",
            output=tmp_path / "mismatch.json",
        )


def test_vbench_entry_point_enforces_the_protocol_chain() -> None:
    """`generate` reads its protocol from the manifest, so the chain order is load-bearing.

    Skipping check-protocol or the manifest rebuild is what invalidated the 2026-08-25 VBench
    run: the manifest kept the old fastwam-native protocol, generate skipped the videos that
    already existed, and the whole thing silently re-scored 448x224 / 9-frame output.
    """
    import subprocess
    from pathlib import Path

    script = Path("scripts/experiments/literature/run_w0_vbench_native.sh")
    body = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in body
    # all four stages, in order. Matched on the stage banners rather than "--mode=", because
    # an error message also names prepare-vbench and would defeat a naive index comparison.
    banners = ("1/5 check-protocol", "2/5 prepare-vbench", "3/5 generate", "4/5 score")
    order = [body.index(banner) for banner in banners]
    assert order == sorted(order), "stages must run in check -> prepare -> generate -> score order"
    for stage in ("check-protocol", "prepare-vbench", "generate", "score"):
        assert f"--mode={stage}" in body
    # the manifest handed to generate must be the one prepare-vbench just wrote
    assert '--protocol="${protocol}"' in body
    assert '--manifest="${manifest}"' in body
    # a downgraded protocol must stop the run unless explicitly accepted
    assert 'if [[ "${kind}" != "wan22-ti2v-compatible" ]]' in body
    assert "W0_ACCEPT_FALLBACK_PROTOCOL" in body
    assert "Refusing to continue on the fastwam-native protocol" in body
    # protocol variants must not share an output tree
    assert 'echo native || echo fallback' in body
    # probe first, because the official protocol is 1280x720 x 81 frames x 50 steps
    assert 'scope="${1:-probe}"' in body
    # the two set sizes must be selectable and must not share an output tree
    assert 'probe | mini21 | vbench-full' in body
    assert 'selection="full"' in body and 'selection="mini-21"' in body
    assert 'dataset="vbench-i2v-full"' in body and 'dataset="vbench-mini-21"' in body
    assert '${vbench_root}/${suffix}-full' in body
    assert '--operating-point="${operating_point}"' in body
    assert 'fastwam-resolution-only' in body
    # a bare "full" is ambiguous between the 21-prompt subset and all 1118 prompts
    assert "ambiguous scope 'full'" in body
    # multi-GPU must reach both generate and score, and be validated before the slow stages
    assert body.count('"${gpu_args[@]}"') == 2
    assert 'gpu_args=(--num-gpus="${num_gpus}")' in body
    assert body.index("CUDA device(s) visible") < body.index("1/5 check-protocol")

    for args, expected in (
        (["nope"], "[probe|mini21|vbench-full]"),
        (["full"], "ambiguous scope"),
        (["probe", "0"], "positive integer"),
        (["probe", "1", "bad-point"], "unsupported operating point"),
    ):
        rejected = subprocess.run(
            ["bash", str(script), *args], capture_output=True, text=True, check=False
        )
        assert rejected.returncode == 2
        assert expected in rejected.stderr

    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_libero_wan_entry_point_uses_official_schedule_and_libero_inputs() -> None:
    script = Path("scripts/experiments/literature/run_w0_libero_wan.sh")
    body = script.read_text(encoding="utf-8")

    assert "set -euo pipefail" in body
    assert "prepare-libero-wan" in body
    assert "--dataset=\"${dataset}\"" in body
    assert "libero-wan-mini-21" in body
    assert "libero-wan-40" in body
    assert "--input-camera=\"${input_camera}\"" in body
    assert "--operating-point=\"${operating_point}\"" in body
    assert 'gpu_args=(--num-gpus="${num_gpus}")' in body
    assert "81 frames, 24 FPS, 50 UniPC steps" in body
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0


def test_strict_load_reads_metadata_only(tmp_path) -> None:
    """The protocol check needs key names, not tensor bytes, and the difference is not small.

    Measured on a 503 MB checkpoint: map_location="meta" alone read 503.3 MB, with mmap=True it
    read 0.1 MB, both returning identical keys. On the training host's vePFS the non-mmap path
    turned a 12 GB checkpoint into 903 GB of rchar -- roughly 75x, because the filesystem
    fetches a large stripe per small read -- and check-protocol never finished.
    """
    torch = pytest.importorskip("torch")

    checkpoint = tmp_path / "step_000001.pt"
    # large enough that reading the storages dwarfs any incidental I/O by the test runner
    state = {f"mixtures.video.blocks.{index}.weight": torch.zeros(1024, 1024) for index in range(8)}
    state["mixtures.action.blocks.0.weight"] = torch.zeros(8, 8)
    torch.save({"mot": state, "step": 1}, checkpoint)
    size = checkpoint.stat().st_size
    assert size > 16_000_000, size

    def rchar() -> int:
        with open("/proc/self/io", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("rchar"):
                    return int(line.split()[1])
        raise RuntimeError("/proc/self/io has no rchar")

    before = rchar()
    video_state = evaluation._load_video_state(checkpoint, map_location="meta")
    with_mmap = rchar() - before

    # the same load without mmap, for a relative comparison that does not depend on the
    # filesystem or on how much the test runner happens to read
    before = rchar()
    torch.load(checkpoint, map_location="meta", weights_only=False)
    without_mmap = rchar() - before

    # the action expert must be stripped and the video prefix removed
    assert sorted(video_state) == [f"blocks.{index}.weight" for index in range(8)]
    assert without_mmap > size // 2, f"baseline read only {without_mmap} of {size} bytes"
    assert with_mmap * 10 < without_mmap, (
        f"mmap path read {with_mmap} bytes against {without_mmap} without mmap; "
        "mmap=True is not in effect"
    )

    # the mmap attempt must be first, so a torch that supports it never takes the slow path
    import inspect

    source = inspect.getsource(evaluation._load_video_state)
    assert source.index('"mmap": True') < source.index('{"map_location": map_location}')


def test_failure_reasons_name_the_line_for_a_message_less_assert() -> None:
    """A bare ``assert`` must still be locatable from the persisted protocol reason.

    The first Wan2.2 smoke recorded ``"AssertionError: "`` and nothing else. The vendored Wan2.2
    generate path has about twenty message-less asserts, so that reason named all of them and
    identified none, which is not a diagnosis anyone can act on.
    """

    def inner() -> None:
        value = 1
        assert value == 2

    try:
        inner()
    except AssertionError as exc:
        reason = evaluation._exception_reason(exc)
    else:  # pragma: no cover - the assert above always fires
        raise AssertionError("expected an AssertionError")

    assert reason.startswith("AssertionError")
    # the file and line of the failing assert, not just the exception type
    assert f"{Path(__file__).name}:" in reason, reason
    assert "assert value == 2" in reason, reason

    # the environment-shaped-fallback guard keys off the leading type name, so rendering the
    # traceback must not push the type name out of position
    try:
        raise ModuleNotFoundError("No module named 'ftfy'")
    except ModuleNotFoundError as exc:
        env_reason = evaluation._exception_reason(exc)
    assert env_reason.startswith("ModuleNotFoundError")
    with pytest.raises(RuntimeError, match="missing package"):
        evaluation._refuse_environment_shaped_fallback(
            {"fw_v": {"strict_load": False, "reason": env_reason}}
        )


def test_sdpa_attention_backend_matches_the_reference_and_refuses_what_it_cannot_do(
    monkeypatch,
) -> None:
    """The official DiT calls flash_attention directly, so a missing flash-attn kills every forward.

    ``wan/modules/model.py:145`` calls ``flash_attention`` rather than the ``attention()``
    dispatcher, so without the ``flash_attn`` package the first forward pass dies on
    ``attention.py:112 assert FLASH_ATTN_2_AVAILABLE``. Substituting SDPA is sound because
    FastWAM's own DiT computes attention with ``F.scaled_dot_product_attention``, so these weights
    were trained under this kernel.

    What must not happen is a silent substitution that computes a *different* function. The
    vendored fallback does exactly that: it drops q_lens/k_lens padding and ignores q_scale and
    softmax_scale. This test pins the shim's numerics against a plain reference and pins the
    refusals.
    """
    torch = pytest.importorskip("torch")

    attention_module = types.ModuleType("wan.modules.attention")
    attention_module.FLASH_ATTN_2_AVAILABLE = False
    attention_module.FLASH_ATTN_3_AVAILABLE = False
    model_module = types.ModuleType("wan.modules.model")
    model_module.flash_attention = None
    package = types.ModuleType("wan")
    modules_package = types.ModuleType("wan.modules")
    for name, module in (
        ("wan", package),
        ("wan.modules", modules_package),
        ("wan.modules.attention", attention_module),
        ("wan.modules.model", model_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    backend = evaluation._install_sdpa_attention_backend(wan_root=Path("third_party/wan22"))
    assert backend == "torch-sdpa"
    shim = model_module.flash_attention
    assert callable(shim)

    # numerics: [B, L, N, C] layout, as the official flash_attention documents
    torch.manual_seed(0)
    q = torch.randn(2, 5, 4, 8, dtype=torch.float32)
    k = torch.randn(2, 5, 4, 8, dtype=torch.float32)
    v = torch.randn(2, 5, 4, 8, dtype=torch.float32)

    def reference(q_, k_, v_, *, q_scale=None, softmax_scale=None, causal=False):
        q_ = q_.to(torch.bfloat16)
        k_ = k_.to(torch.bfloat16)
        v_ = v_.to(torch.bfloat16)
        if q_scale is not None:
            q_ = q_ * q_scale
        out = torch.nn.functional.scaled_dot_product_attention(
            q_.transpose(1, 2),
            k_.transpose(1, 2),
            v_.transpose(1, 2),
            scale=softmax_scale,
            is_causal=causal,
        )
        return out.transpose(1, 2).contiguous().to(torch.float32)

    got = shim(q, k, v)
    torch.testing.assert_close(got, reference(q, k, v), rtol=2e-2, atol=2e-2)
    assert got.dtype == q.dtype, "the reference casts back to the input dtype"

    # q_scale and softmax_scale must be applied, unlike in the vendored fallback
    scaled = shim(q, k, v, q_scale=0.5, softmax_scale=0.25)
    torch.testing.assert_close(
        scaled, reference(q, k, v, q_scale=0.5, softmax_scale=0.25), rtol=2e-2, atol=2e-2
    )
    assert not torch.allclose(scaled, got, rtol=1e-3, atol=1e-3), "the scales changed nothing"

    # ragged lengths and sliding windows must raise, not silently compute something else
    with pytest.raises(RuntimeError, match="ragged"):
        shim(q, k, v, k_lens=torch.tensor([5, 3]))
    with pytest.raises(RuntimeError, match="sliding-window"):
        shim(q, k, v, window_size=(2, 2))
    # a full-length k_lens carries no padding, so it must be accepted
    shim(q, k, v, k_lens=torch.tensor([5, 5]))

    # when flash-attn is present the shim must not be installed at all
    attention_module.FLASH_ATTN_2_AVAILABLE = True
    model_module.flash_attention = "untouched"
    assert evaluation._install_sdpa_attention_backend(wan_root=Path("third_party/wan22")) == (
        "flash-attn-2"
    )
    assert model_module.flash_attention == "untouched"


def test_stage_timer_attributes_time_without_editing_vendored_code() -> None:
    """Per-video wall clock cannot separate compute from the offload shuffling, so attribute it.

    ``WanTI2V.generate`` does T5 encoding, the DiT sampling loop and VAE decoding inside one call.
    Measured on H20 the self-attention kernel alone accounts for about half the 300 s per video
    (3,000 calls x 49.8 ms), which is only knowable by timing submodules.

    The fakes here mirror the shapes the official pipeline actually relies on. The first version of
    this test used ``text_encoder = lambda ...``, which has no attributes, so it passed while the
    real run died on ``textimage2video.py:501 self.text_encoder.model.to(self.device)`` with
    ``AttributeError: 'function' object has no attribute 'model'``. A timing wrapper must stay
    transparent to attribute access, and the test has to be the thing that proves it.
    """

    class FakeInner:
        def to(self, device):
            self.device = device
            return self

    class FakeTextEncoder:
        def __init__(self) -> None:
            self.model = FakeInner()

        def __call__(self, prompts, device):
            return [len(prompt) for prompt in prompts]

    class FakeVae:
        def __init__(self) -> None:
            self.model = FakeInner()

        def decode(self, value):
            return value * 2

        def encode(self, value):
            return value

    class FakeDit:
        def __init__(self) -> None:
            self.calls = 0

        def to(self, device):
            self.device = device
            return self

        def forward(self, value):
            self.calls += 1
            return value + 1

        def __call__(self, value):
            # mirrors nn.Module.__call__ resolving self.forward
            return self.forward(value)

    class FakeWan:
        def __init__(self) -> None:
            self.model = FakeDit()
            self.vae = FakeVae()
            self.text_encoder = FakeTextEncoder()

    wan = FakeWan()
    original_class_forward = FakeDit.forward
    timer = evaluation._StageTimer().install(wan)

    # what the official pipeline does: reach through the text encoder, move the DiT, then call
    assert wan.text_encoder.model.to("cuda") is not None, "attribute reach-through must survive"
    assert wan.model.to("cuda") is not None
    assert wan.text_encoder(["abc"], "cuda") == [3]
    assert wan.model(1) == 2
    assert wan.model(1) == 2
    assert wan.vae.decode(3) == 6
    assert wan.vae.model.to("cpu") is not None, "the VAE's inner module must stay reachable"

    report = timer.report()
    assert report["calls"] == {"dit": 2, "t5": 1, "vae_decode": 1}, report
    assert set(report["seconds"]) == {"dit", "t5", "vae_decode"}, report
    assert all(value >= 0.0 for value in report["seconds"].values())
    # classes untouched, so a re-cloned third_party tree and other instances are unaffected
    assert FakeDit.forward is original_class_forward


def test_generate_exposes_offload_and_timing_switches() -> None:
    """Both knobs must reach generate_videos, and the default must not change behaviour."""
    import inspect

    signature = inspect.signature(evaluation.generate_videos)
    assert signature.parameters["offload_model"].default is True, "default must stay as it was"
    assert signature.parameters["time_stages"].default is False

    source = inspect.getsource(evaluation.generate_videos)
    # the switch has to be forwarded, not merely accepted
    assert "offload_model=offload_model" in source
    assert "_StageTimer().install(model)" in source
    assert '"offload_model": offload_model' in source

    body = Path("src/robot_practice/experiments/literature/w0_video_generation_eval.py").read_text(
        encoding="utf-8"
    )
    assert "offload_model=not args.no_offload_model" in body
    assert "time_stages=args.time_stages" in body
    # _generate_wan22_video must honour the flag instead of the old hardcoded True
    generate_source = inspect.getsource(evaluation._generate_wan22_video)
    assert "offload_model=offload_model" in generate_source
    assert "offload_model=True," not in generate_source


def test_sampler_settings_come_from_the_protocol_not_from_literals() -> None:
    """A manifest that declares a sampler setting must get that setting.

    ``shift``/``guide_scale``/``sample_solver`` used to be hardcoded in ``_generate_wan22_video``
    while ``prepare_libero_wan_manifest`` asserted them in the protocol, so the protocol fields
    were decorative: a manifest could declare one value and generation would use another. The
    resolution-only LIBERO-Wan ablation depends on the two operating points differing in exactly
    one field, which is only checkable if every declared field is the field that runs.

    The fallbacks must stay at the previously hardcoded values, because the check-protocol smoke
    passes a protocol with only width/height/frames/steps.
    """
    recorded: dict[str, object] = {}

    class FakeWan:
        def generate(self, **kwargs):
            recorded.update(kwargs)
            # channel-first, as _wan_video_to_frames expects
            return np.zeros((3, 2, 8, 8), dtype=np.float32)

    image = np.zeros((8, 8, 3), dtype=np.uint8)

    declared = {
        "width": 8,
        "height": 8,
        "frames": 2,
        "steps": 7,
        "shift": 3.5,
        "sample_solver": "dpm++",
        "guide_scale": 2.25,
    }
    evaluation._generate_wan22_video(
        model=FakeWan(), prompt="p", input_image=image, protocol=declared, seed=11
    )
    assert recorded["shift"] == 3.5
    assert recorded["sample_solver"] == "dpm++"
    assert recorded["guide_scale"] == 2.25
    assert recorded["sampling_steps"] == 7
    assert recorded["max_area"] == 64
    assert recorded["frame_num"] == 2
    assert recorded["offload_model"] is True

    recorded.clear()
    minimal = {"width": 8, "height": 8, "frames": 2, "steps": 1}
    evaluation._generate_wan22_video(
        model=FakeWan(),
        prompt="p",
        input_image=image,
        protocol=minimal,
        seed=11,
        offload_model=False,
    )
    # the smoke path: unchanged behaviour when the keys are absent
    assert recorded["shift"] == 5.0
    assert recorded["sample_solver"] == "unipc"
    assert recorded["guide_scale"] == 5.0
    assert recorded["offload_model"] is False


def test_metrics_take_the_horizon_from_the_data_not_from_a_literal() -> None:
    """LIBERO-Wan runs the same samples at the official 81 frames, so 9 cannot stay hardcoded.

    Frame 0 is the given conditioning frame in every protocol, so it is excluded regardless of the
    horizon; everything after it is scored.
    """

    def frames(count: int, value: int) -> list:
        return [np.full((8, 16, 3), value, dtype=np.uint8) for _ in range(count)]

    for count in (2, 9, 81):
        identical = evaluation._w0_sample_metrics(
            frames(count, 100), frames(count, 100), lpips_model=None, lpips_device="cpu"
        )
        assert identical["psnr"] > 90.0, (count, identical)
        assert identical["lpips"] is None

        degraded = evaluation._w0_sample_metrics(
            frames(count, 100), frames(count, 160), lpips_model=None, lpips_device="cpu"
        )
        assert degraded["psnr"] < identical["psnr"], (count, degraded)

    # a single frame carries no future to score
    with pytest.raises(TypeError, match="at least one future frame"):
        evaluation._w0_sample_metrics(
            frames(1, 100), frames(1, 100), lpips_model=None, lpips_device="cpu"
        )
    with pytest.raises(TypeError, match="frame count mismatch"):
        evaluation._w0_sample_metrics(
            frames(9, 100), frames(81, 100), lpips_model=None, lpips_device="cpu"
        )

    source = __import__("inspect").getsource(evaluation._w0_sample_metrics)
    assert "range(1, len(gt_frames))" in source, "the horizon must come from the data"
    assert "!= 9" not in source


def test_score_accepts_libero_wan_and_transforms_the_ground_truth_the_same_way() -> None:
    """At 1280x720 the conditioning image is cropped, so the ground truth must be cropped too.

    Shrinking the prediction instead would compare a 16:9 crop against an uncropped 2:1 frame,
    which is misaligned content rather than a resolution change. The scorer therefore records which
    transform it applied and warns that absolute values are not comparable across resolutions.
    """
    body = Path("src/robot_practice/experiments/literature/w0_video_generation_eval.py").read_text(
        encoding="utf-8"
    )
    assert 'scorable = {"W0-Video-40", *LIBERO_WAN_DATASETS.values()}' in body
    assert '"ground_truth_transform": gt_transform' in body
    assert '"cross_resolution_comparability"' in body
    # the 448x224 arm must stay untouched, since the conditioning image there is an identity
    assert 'gt_transform = "none" if (gt_width, gt_height) == (448, 224)' in body

    # the identity claim itself: a 448x224 source through the transform is byte-identical
    rng = np.random.default_rng(0)
    source = rng.integers(0, 256, size=(224, 448, 3), dtype=np.uint8)
    assert np.array_equal(
        evaluation._center_crop_resize(source, width=448, height=224), source
    ), "448x224 must be a no-op, otherwise the ablation is not resolution-only"
    cropped = evaluation._center_crop_resize(source, width=1280, height=720)
    assert cropped.shape == (720, 1280, 3)
    assert not np.array_equal(cropped[:, :224], source[:, :224])


def test_official_operating_point_needs_no_resampling() -> None:
    """The declared resolution must be what the official pipeline actually produces.

    ``WanTI2V.generate`` never forwards ``size`` to ``i2v``; the output comes from
    ``best_output_size(image_aspect, max_area)`` alone. Declaring 1280x720 therefore produced
    1280x704 and then a silent LANCZOS upscale in ``_wan_video_to_frames`` -- and 1280*720 is not
    in ``SUPPORTED_SIZES['ti2v-5B']`` at all, so the official entry point would have refused it.

    1248x704 is the fixed point for a 16-9 conditioning image: the official
    ``MAX_AREA_CONFIGS['1280*704'] = 901120`` yields it, and so does 1248*704 = 878592, which is
    what this evaluator derives from the protocol. Same pair, so the generation is identical.
    """

    def best_output_size(width, height, dw, dh, area):
        # mirrors third_party/wan22/wan/utils/utils.py:202
        ratio = width / height
        ow, oh = (area * ratio) ** 0.5, (area / ratio) ** 0.5
        ow1 = int(ow // dw * dw)
        oh1 = int(area / ow1 // dh * dh)
        ratio1 = ow1 / oh1
        oh2 = int(oh // dh * dh)
        ow2 = int(area / oh2 // dw * dw)
        ratio2 = ow2 / oh2
        if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
            return ow1, oh1
        return ow2, oh2

    protocol = evaluation._wan22_protocol_for_operating_point("wan22-official")
    width, height = int(protocol["width"]), int(protocol["height"])
    assert (width, height) == (1248, 704), protocol

    # the 16-9 conditioning image VBench-I2V ships, and what our own max_area yields from it
    derived = best_output_size(1280, 720, 32, 32, width * height)
    assert derived == (width, height), f"declared {(width, height)} but pipeline yields {derived}"
    # and the official max_area yields the same pair, so this is not a deviation
    official = best_output_size(1280, 720, 32, 32, 901120)
    assert official == derived, (official, derived)

    # frame count must be 4n+1 or the VAE temporal stride does not divide it
    assert (int(protocol["frames"]) - 1) % 4 == 0, protocol["frames"]
    # FPS must be Wan2.2's own, not Wan2.1's 16
    assert protocol["fps"] == 24, protocol
    # the single intended deviation has to be recorded, not implicit
    assert protocol["official_reference"]["frame_num"] == 121
    assert "frames=81" in protocol["official_reference"]["deviation"]

    # the resolution-only arm must keep every temporal setting identical
    fastwam = evaluation._wan22_protocol_for_operating_point("fastwam-resolution-only")
    for key in ("frames", "fps", "steps", "shift", "sample_solver", "guide_scale"):
        assert fastwam[key] == protocol[key], (key, fastwam[key], protocol[key])
    assert (fastwam["width"], fastwam["height"]) != (width, height)
    assert best_output_size(448, 224, 32, 32, 448 * 224) == (448, 224)


def test_w0_selection_is_cached_and_keyed_by_dataset_fingerprint(monkeypatch, tmp_path) -> None:
    """The motion-p75 scan decodes every episode twice; reusing it must be safe, not just fast.

    The scan decodes each episode's front and wrist video in full and resizes every frame to
    64x64 -- roughly 3,400 videos across the four suites, 10-25 minutes of pure CPU. It reruns for
    every operating point, every resolution and every scope, all of which leave the selection
    unchanged, so the decisions are cached.

    Safety comes from the key: it carries the dataset fingerprint, so different videos cannot hit a
    stale entry. And ``base_seed`` must stay outside the cache, since it changes the manifest but
    not the selection.
    """
    calls: list[int] = []

    def fake_motion_candidates(dataset, info, episode, *, raw_frames, video_stride):
        calls.append(int(episode["episode_index"]))
        return [
            {
                "episode_index": int(episode["episode_index"]),
                "start_frame": 5,
                "motion_score": 1.0 + int(episode["episode_index"]),
                "front_motion_score": 1.0,
                "wrist_motion_score": 1.0,
                "length": 200,
            }
        ]

    fingerprint = {"total_frames": 1234}

    def fake_task_rows(dataset):
        languages = {index: f"task {index}" for index in range(10)}
        episodes = {index: {"episode_index": index, "task_id": index, "length": 200}
                    for index in range(10)}
        return languages, episodes, fingerprint

    monkeypatch.setattr(evaluation, "_motion_candidates", fake_motion_candidates)
    monkeypatch.setattr(evaluation, "_load_w0_task_rows", fake_task_rows)
    real_read_json = evaluation._read_json
    monkeypatch.setattr(
        evaluation,
        "_read_json",
        lambda path: {"fake": "info"} if str(path).endswith("info.json") else real_read_json(path),
    )
    monkeypatch.setattr(
        evaluation, "_video_path_from_info", lambda dataset, info, key, index: tmp_path / "v.mp4"
    )
    (tmp_path / "v.mp4").write_bytes(b"x")
    monkeypatch.setattr(
        evaluation, "_file_info", lambda path, label: {"path": str(path), "label": label}
    )
    # REPO_ROOT is redirected rather than stubbing _w0_selection_cache_path, so the real key
    # construction -- including the dataset fingerprint -- is what the test exercises
    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path / "repo")

    def build(base_seed: int, output_name: str):
        return evaluation.prepare_w0_manifest(
            dataset_root=tmp_path / "dataset",
            suites=["libero_spatial"],
            selection="one-per-task-motion-p75",
            raw_frames=33,
            video_stride=4,
            base_seed=base_seed,
            output=tmp_path / output_name,
        )

    first = build(52000, "a.json")
    scanned = len(calls)
    assert scanned > 0, "the first build must actually scan"

    calls.clear()
    second = build(99000, "b.json")
    assert calls == [], "a cache hit must not rescan a single episode"

    # the selection is identical; only base_seed-derived fields move
    keys = ("sample_id", "episode_index", "start_frame", "frame_indices", "motion_score")
    for left, right in zip(first["samples"], second["samples"]):
        for key in keys:
            assert left[key] == right[key], key
    assert next(sample["seed"] for sample in second["samples"]) == 99000

    # a changed dataset must not hit the stale entry; the fingerprint is part of the real key
    fingerprint["total_frames"] = 4321
    calls.clear()
    build(52000, "c.json")
    assert len(calls) == scanned, "a different fingerprint must force a rescan"

    # and the cache file really is where the implementation says it is
    cache_files = sorted((tmp_path / "repo").rglob("w0_selection_*.json"))
    assert len(cache_files) == 2, [str(path) for path in cache_files]
    assert all("/.cache/" in path.as_posix() for path in cache_files)


def test_selection_cache_can_be_seeded_from_the_frozen_w0_manifest(monkeypatch, tmp_path) -> None:
    """The 10-25 minute scan has already been paid once; the frozen manifest records the result.

    W0-Video-40's manifest carries episode_index, start_frame, motion_score, motion_p75 and
    candidate_count for all 40 samples plus the per-suite dataset fingerprint -- exactly the cache
    contents. Seeding from it is not only faster, it is what makes LIBERO-Wan *provably* the same
    samples as W0-Video-40 rather than merely intended to be.
    """
    monkeypatch.setattr(evaluation, "REPO_ROOT", tmp_path / "repo")
    fingerprint = {"total_frames": 999}
    frozen = {
        "schema_version": 1,
        "dataset": "W0-Video-40",
        "selection": "one-per-task-motion-p75",
        "suites": ["libero_spatial"],
        "dataset_fingerprint": {"libero_spatial": fingerprint},
        "samples": [
            {
                "sample_id": f"libero_spatial_task{task:02d}",
                "suite": "libero_spatial",
                "task_id": task,
                "episode_index": 100 + task,
                "start_frame": 7 + task,
                "raw_frames": 33,
                "video_stride": 4,
                "motion_score": 1.5 + task,
                "motion_p75": 2.0,
                "candidate_count": 42,
            }
            for task in range(10)
        ],
    }
    manifest_path = tmp_path / "frozen.json"
    manifest_path.write_text(json.dumps(frozen), encoding="utf-8")

    result = evaluation.seed_w0_selection_cache(manifest=manifest_path)
    assert len(result["cache_files"]) == 1
    cache_file = sorted((tmp_path / "repo").rglob("w0_selection_*.json"))
    assert len(cache_file) == 1, [str(path) for path in cache_file]
    written = json.loads(cache_file[0].read_text(encoding="utf-8"))
    assert written["tasks"]["3"]["episode_index"] == 103
    assert written["tasks"]["3"]["start_frame"] == 10
    assert written["seeded_from"]["path"].endswith("frozen.json")

    # prepare must now hit that cache and not decode a single video
    calls: list[int] = []

    def fake_motion_candidates(*args, **kwargs):
        calls.append(1)
        raise AssertionError("prepare must not rescan after seeding")

    monkeypatch.setattr(evaluation, "_motion_candidates", fake_motion_candidates)
    monkeypatch.setattr(
        evaluation,
        "_load_w0_task_rows",
        lambda dataset: (
            {index: f"task {index}" for index in range(10)},
            {index: {"episode_index": index, "task_id": index, "length": 200} for index in range(10)},
            fingerprint,
        ),
    )
    real_read_json = evaluation._read_json
    monkeypatch.setattr(
        evaluation,
        "_read_json",
        lambda path: {"fake": "info"} if str(path).endswith("info.json") else real_read_json(path),
    )
    monkeypatch.setattr(
        evaluation, "_video_path_from_info", lambda dataset, info, key, index: tmp_path / "v.mp4"
    )
    (tmp_path / "v.mp4").write_bytes(b"x")
    monkeypatch.setattr(
        evaluation, "_file_info", lambda path, label: {"path": str(path), "label": label}
    )

    built = evaluation.prepare_w0_manifest(
        dataset_root=tmp_path / "dataset",
        suites=["libero_spatial"],
        selection="one-per-task-motion-p75",
        raw_frames=33,
        video_stride=4,
        base_seed=52000,
        output=tmp_path / "rebuilt.json",
    )
    assert calls == []
    # the seeded decisions are the ones that came out
    assert [sample["episode_index"] for sample in built["samples"]] == list(range(100, 110))
    assert [sample["start_frame"] for sample in built["samples"]] == list(range(7, 17))

    # a manifest selected over a different window must be refused, not silently trusted
    frozen["samples"][0]["raw_frames"] = 81
    frozen["samples"][0]["video_stride"] = 1
    other = tmp_path / "other.json"
    other.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="mixes sampling windows"):
        evaluation.seed_w0_selection_cache(manifest=other)


def test_frame_count_is_configurable_and_must_be_4n_plus_1() -> None:
    """The two arms need different clip lengths, and the constraint is the VAE's temporal stride.

    The VBench arm runs 81 frames (Wan2.2's own frame_num is 121; 81 halves the cost). The
    LIBERO-Wan arm runs 9, because the frozen W0 start frames sit at each episode's motion p75 and
    several land past the middle -- ``libero_spatial_task07`` starts at 61 in a 110-frame episode,
    so 81 frames of ground truth simply do not exist and the start cannot move without changing
    which samples are selected.

    9 frames also puts the ground-truth window back on the frozen 33/4 grid, which makes the
    448x224 arm directly comparable to W0-Video-40's 9.963 / 24.705 dB: same samples, same ground
    truth, same resolution, with only ``infer_joint`` vs ``WanTI2V.generate`` differing.
    """
    default = evaluation._wan22_protocol_for_operating_point("wan22-official")
    assert default["frames"] == 81, "the VBench arm's default must not move silently"

    for frames in (1, 5, 9, 33, 81, 121):
        for point in ("wan22-official", "fastwam-resolution-only"):
            protocol = evaluation._wan22_protocol_for_operating_point(point, frames=frames)
            assert protocol["frames"] == frames, (point, frames)
            # every temporal setting other than the clip length stays official
            assert protocol["fps"] == 24
            assert protocol["steps"] == 50
            assert protocol["shift"] == 5.0
            assert protocol["guide_scale"] == 5.0
            assert protocol["sample_solver"] == "unipc"

    for frames in (2, 8, 80, 120):
        with pytest.raises(ValueError, match="4n\\+1"):
            evaluation._wan22_protocol_for_operating_point("wan22-official", frames=frames)

    # the two operating points must still differ only in resolution at any given frame count
    official = evaluation._wan22_protocol_for_operating_point("wan22-official", frames=9)
    fastwam = evaluation._wan22_protocol_for_operating_point("fastwam-resolution-only", frames=9)
    differing = {
        key
        for key in set(official) | set(fastwam)
        if official.get(key) != fastwam.get(key)
    }
    assert differing <= {"width", "height", "input_crop_ratio", "official_reference"}, differing

    # and the entry point has to ask for 9, not inherit 81
    script = Path("scripts/experiments/literature/run_w0_libero_wan.sh").read_text(encoding="utf-8")
    assert "--frames=9" in script
    assert "--raw-frames=33" in script and "--video-stride=4" in script
    assert "--raw-frames=81" not in script


def test_libero_wan_entry_point_scores_and_pairs() -> None:
    """Generation alone cannot be compared to W0-Video-40's 9.963 / 24.705 dB.

    The official-code arm exists to isolate ``infer_joint`` vs ``WanTI2V.generate`` at the same
    samples, resolution and horizon, which is a numeric claim. The CLI used to refuse
    ``--mode=score`` for LIBERO-Wan with "no reference-video score mode" -- true when the scorer
    pinned 9 frames and one resolution, false now.
    """
    script = Path("scripts/experiments/literature/run_w0_libero_wan.sh").read_text(encoding="utf-8")
    # four stages, and score has to come after generate
    assert "=== 3/4 generate" in script
    assert "=== 4/4 score" in script
    assert script.index("=== 3/4 generate") < script.index("=== 4/4 score")
    assert "--mode=score" in script
    assert '--output="${work}/${variant}/metrics.json"' in script
    # the multi-GPU flag must reach score too, not just generate
    assert script.count('"${gpu_args[@]}"') >= 2
    # the read-out is a file, because a nested heredoc would terminate the outer one
    assert "read_w0_libero_wan_metrics.py" in script
    assert "<<'PY'" not in script

    body = Path("src/robot_practice/experiments/literature/w0_video_generation_eval.py").read_text(
        encoding="utf-8"
    )
    assert 'if args.dataset == "w0-video-40" or args.dataset in LIBERO_WAN_DATASETS:' in body
    assert "LIBERO-Wan generation has no reference-video score mode" not in body
    # multi-GPU score must forward the real dataset, not a hardcoded one
    assert '_option("dataset", cli_dataset[dataset_name])' in body
    assert '_option("dataset", "w0-video-40")' not in body


def test_paired_read_out_refuses_an_unpaired_comparison(tmp_path) -> None:
    """A delta between differently sized or differently sourced runs is not a comparison."""
    import subprocess

    def write(variant: str, *, count: int, sha: str) -> None:
        directory = tmp_path / variant
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "summary": {"count": count, "psnr": 10.0, "ssim": 0.5, "lpips": 0.4},
                    "manifest": {"sha256": sha},
                    "metric_definition": {
                        "future_frame_indices": list(range(1, 9)),
                        "ground_truth_resolution": [448, 224],
                        "ground_truth_transform": "none",
                    },
                }
            ),
            encoding="utf-8",
        )

    script = Path("scripts/experiments/literature/read_w0_libero_wan_metrics.py")

    def run() -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(script)],
            env={**os.environ, "W0_LIBERO_WAN_DIR": str(tmp_path)},
            capture_output=True,
            text=True,
            check=False,
        )

    write("fw_fz", count=40, sha="abc")
    write("fw_v", count=40, sha="abc")
    ok = run()
    assert ok.returncode == 0, ok.stderr
    assert "delta psnr=" in ok.stdout

    write("fw_v", count=21, sha="abc")
    unpaired = run()
    assert unpaired.returncode == 1
    assert "unpaired sample counts" in unpaired.stderr

    write("fw_v", count=40, sha="different")
    mismatched = run()
    assert mismatched.returncode == 1
    assert "different manifests" in mismatched.stderr


def test_vbench_operating_points_do_not_overwrite_each_other() -> None:
    """The resolution control needs its own tree, and both arms are wan22-ti2v-compatible.

    The output subtree is derived from ``kind``, which is the same string for both operating
    points, so without an explicit operating-point suffix the 448x224 control would overwrite the
    1248x704 videos it is supposed to be compared against. The protocol file has the same problem:
    a single shared ``protocol.json`` would describe whichever arm ran last.
    """
    script = Path("scripts/experiments/literature/run_w0_vbench_native.sh").read_text(
        encoding="utf-8"
    )
    # the operating point must reach check-protocol, or both arms generate at the same resolution
    assert '--operating-point="${operating_point}"' in script
    # ... and it must reach the output path, or the second arm overwrites the first
    assert 'suffix="${suffix}-fastwam-point"' in script
    assert 'protocol="${vbench_root}/protocol-${operating_point}.json"' in script
    # the default arm keeps its historical paths, so the completed run stays addressable
    assert 'protocol="${vbench_root}/protocol.json"' in script

    suffix_line = script.index('suffix="$([[ "${kind}"')
    work_line = script.index('work="${vbench_root}/${suffix}"')
    assert suffix_line < work_line, "the suffix must be decided before work is built"

    # both operating points must be accepted, and nothing else
    assert "wan22-official | fastwam-resolution-only)" in script
    assert "unsupported operating point" in script


def test_guide_records_the_four_official_runs() -> None:
    """The four-run matrix is the experiment; a guide that omits it leaves the plan in chat only.

    Two of the four exist purely to make a single variable move: A vs D differ only in resolution,
    B vs C only in the generation entry point. If the guide lists commands without saying which
    pair isolates what, the next reader cannot tell a control from a repeat.
    """
    guide = Path("reports/guides/literature/02_w0_wan_video_generation_preservation.md").read_text(
        encoding="utf-8"
    )

    # every run must be present as a runnable command, not just described
    for command in (
        "run_w0_vbench_native.sh mini21 8\n",
        "run_w0_libero_wan.sh all40 8 fastwam-resolution-only",
        "run_w0_video_40.sh fw_fz 8",
        "run_w0_vbench_native.sh mini21 8 fastwam-resolution-only",
    ):
        assert command in guide, command

    # the operating points and their output trees, so nobody overwrites a completed arm
    for token in ("1248x704", "448x224", "vbench/native/", "vbench/native-fastwam-point/",
                  "libero_wan/fastwam-resolution-only/both/all40/", "w0_video_40/"):
        assert token in guide, token

    # the two single-variable claims have to be stated, and the non-comparison flagged
    assert "只差分辨率" in guide
    assert "只差代码路径" in guide or "只差生成入口" in guide
    assert "不构成单变量" in guide

    # the commands in the guide must be the ones that exist
    for script in (
        "scripts/experiments/literature/run_w0_vbench_native.sh",
        "scripts/experiments/literature/run_w0_libero_wan.sh",
        "scripts/experiments/literature/run_w0_video_40.sh",
    ):
        assert Path(script).is_file(), script

    # fastwam-resolution-only must be a real operating point in both entry points. Asserted by
    # behaviour rather than by the literal case-arm text, which changed once a third point was added
    # and only one of the two entry points accepts it.
    for script in (
        "scripts/experiments/literature/run_w0_vbench_native.sh",
        "scripts/experiments/literature/run_w0_libero_wan.sh",
    ):
        body = Path(script).read_text(encoding="utf-8")
        assert "fastwam-resolution-only" in body, script
        assert "unsupported operating point" in body, script

    # the square high-resolution point belongs to the LIBERO front view only: VBench ships 16-9
    # crops, so a square target there would discard part of every conditioning image
    libero = Path("scripts/experiments/literature/run_w0_libero_wan.sh").read_text(encoding="utf-8")
    vbench = Path("scripts/experiments/literature/run_w0_vbench_native.sh").read_text(
        encoding="utf-8"
    )
    assert "front-square-hires) ;;" in libero.replace(" | ", " | ") or (
        "front-square-hires" in libero
    ), "the LIBERO entry point must accept front-square-hires"
    assert "not VBench's 16-9 images" in vbench, "VBench must refuse front-square-hires"


def test_single_camera_runs_are_scorable_and_not_split_in_half() -> None:
    """The single-camera diagnostic used to be generate-only, so its dB had to be computed by hand.

    ``score_w0`` built the ground truth with the default ``camera="both"``, i.e. 448x224, and a
    224x224 prediction could only raise a shape mismatch. The camera lives in the generation's
    ``run_config`` rather than the manifest, because the manifest is shared across variants while
    the camera is a per-run deviation, so the scorer has to read it back.

    And a 224x224 single view must not be split down the middle: two 112x224 halves are not two
    cameras, and averaging PSNR over halves is not the PSNR of the whole frame.
    """
    frame = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    assert len(evaluation._camera_metric_frames(frame, split=False)) == 1
    assert evaluation._camera_metric_frames(frame, split=False)[0].shape == (224, 224, 3)
    halves = evaluation._camera_metric_frames(frame, split=True)
    assert len(halves) == 2 and halves[0].shape == (224, 112, 3)

    # the whole-frame score must differ from the mean over halves, which is why split matters
    other = (frame.astype(np.int16) + 40).clip(0, 255).astype(np.uint8)
    whole = evaluation._w0_sample_metrics(
        [frame, frame], [frame, other], lpips_model=None, lpips_device="cpu", split_cameras=False
    )
    assert whole["psnr"] > 0

    body = Path("src/robot_practice/experiments/literature/w0_video_generation_eval.py").read_text(
        encoding="utf-8"
    )
    # the camera must come from run_config and drive both the ground truth and the split
    assert 'recorded = str(generation.get("single_camera", "both"))' in body
    assert "_w0_ground_truth(sample, camera=gt_camera)" in body
    assert "split_cameras=gt_camera == \"both\"" in body
    # and be recorded, so a single-camera metrics.json cannot be mistaken for the protocol
    assert '"ground_truth_camera": gt_camera' in body
    assert 'f"single {gt_camera}"' in body
    # the old unconditional default must be gone
    assert "gt_frames = _w0_ground_truth(sample)\n" not in body


def test_single_camera_reaches_every_multi_gpu_worker(monkeypatch, tmp_path) -> None:
    """Single-camera plus multi-GPU used to be refused, but the reason was a missing wire, not a
    real limit: the worker command omitted --single-camera, so a lifted guard alone would have had
    each worker silently generate 448x224 'both'. The fix is to forward the flag; this pins that a
    front-only multi-GPU run actually tells its workers front.
    """
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "w0-video-40",
                "protocol": {"kind": "fastwam-native", "width": 448, "height": 224, "frames": 9},
                "samples": [
                    {"sample_id": f"libero_spatial_task{i:02d}", "suite": "libero_spatial",
                     "task_id": i, "seed": i, "frame_indices": [0, 4, 8]}
                    for i in range(4)
                ],
            }
        ),
        encoding="utf-8",
    )

    captured: list[list[str]] = []

    class _Stop(Exception):
        pass

    def fake_run_workers(*, stage, worker_commands, worker_dirs, device_tokens, progress_root):
        captured.extend([list(command) for command in worker_commands])
        # stop before the merge stage, which would read run_config.json the stub never wrote
        raise _Stop

    monkeypatch.setattr(
        evaluation, "_require_multi_gpu_devices", lambda n: [str(i) for i in range(n)]
    )
    monkeypatch.setattr(evaluation, "_run_multi_gpu_workers", fake_run_workers)

    def run(**extra):
        try:
            evaluation.generate_videos_multi_gpu(
                dataset="w0-video-40", variant="fw_fz", checkpoint=tmp_path / "ck.pt",
                manifest=manifest, output_root=tmp_path / extra.pop("out"), vendor_root=tmp_path,
                cache_root=tmp_path, wan_root=tmp_path, wan_model_root=tmp_path,
                device="cuda", overwrite=True, num_gpus=2, config=tmp_path / "cfg.json",
                **extra,
            )
        except _Stop:
            pass

    run(out="out_front", single_camera="front")
    assert captured, "no worker commands were built"
    for command in captured:
        assert "--single-camera=front" in command, command

    # 'both' must not add the flag, so existing two-camera runs are unchanged
    captured.clear()
    run(out="out_both")
    assert captured
    for command in captured:
        assert not any(token.startswith("--single-camera") for token in command), command


def test_front_square_hires_isolates_resolution_alone() -> None:
    """A square view needs a square high-resolution partner, or the ablation is not single-variable.

    The existing cross-resolution evidence compares merged-camera 448x224 (2:1) against 1248x704
    (1.773), so fitting one into the other discards 11.4% of the width -- resolution and field of
    view move together. A single 224x224 front view scaled to 896x896 is a factor of 4 in each
    dimension with zero crop at either end.

    1248x704 cannot serve as that partner: fitting 1:1 into 1.773 discards 43.6% of the height, and
    the official ``best_output_size`` would return 928x928 rather than the declared size.
    """

    def best_output_size(width, height, dw, dh, area):
        ratio = width / height
        ow, oh = (area * ratio) ** 0.5, (area / ratio) ** 0.5
        ow1 = int(ow // dw * dw)
        oh1 = int(area / ow1 // dh * dh)
        oh2 = int(oh // dh * dh)
        ow2 = int(area / oh2 // dw * dw)
        r1, r2 = ow1 / oh1, ow2 / oh2
        if max(ratio / r1, r1 / ratio) < max(ratio / r2, r2 / ratio):
            return ow1, oh1
        return ow2, oh2

    assert "front-square-hires" in evaluation.WAN22_OPERATING_POINTS
    protocol = evaluation._wan22_protocol_for_operating_point("front-square-hires", frames=9)
    width, height = int(protocol["width"]), int(protocol["height"])
    assert (width, height) == (896, 896), protocol
    assert width % 32 == 0 and height % 32 == 0
    # exact: the pipeline generates what the protocol declares, so nothing is resampled
    assert best_output_size(224, 224, 32, 32, width * height) == (width, height)
    assert best_output_size(224, 224, 32, 32, 224 * 224) == (224, 224)
    # and the rejected alternative really is lossy for a square view
    assert best_output_size(224, 224, 32, 32, 1248 * 704) != (1248, 704)

    # zero crop at both ends, unlike the merged-camera pair
    source = np.random.default_rng(0).integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    assert np.array_equal(evaluation._center_crop_resize(source, width=224, height=224), source)
    upscaled = evaluation._center_crop_resize(source, width=896, height=896)
    assert upscaled.shape == (896, 896, 3)

    # every temporal setting must match the official arm, or more than resolution would move
    official = evaluation._wan22_protocol_for_operating_point("wan22-official", frames=9)
    for key in ("frames", "fps", "steps", "shift", "sample_solver", "guide_scale"):
        assert protocol[key] == official[key], key

    body = Path("src/robot_practice/experiments/literature/w0_video_generation_eval.py").read_text(
        encoding="utf-8"
    )
    # generation must keep a square protocol instead of forcing 224
    assert 'if int(protocol["width"]) != int(protocol["height"]):' in body
    # and scoring must read the size back rather than assuming 224
    assert 'gt_width = int(protocol_payload.get("width", 224))' in body
    assert "gt_width = gt_height = 224" not in body
    # a non-square single-camera run must be refused, not mis-scored
    assert "cannot fill it without cropping" in body


def test_score_finds_the_camera_in_either_recording_place(tmp_path, monkeypatch) -> None:
    """The camera is recorded in two places, and reading only one mis-scores the other silently.

    ``--single-camera`` records it in the generation block; ``prepare-libero-wan --input-camera``
    records it in the protocol, because there the camera is part of how the conditioning image was
    built. Reading only the first compared a front-only 896x896 prediction against a merged
    two-camera ground truth resized to 896x896 and split in half -- unrelated content that scored
    SSIM 0.04 instead of raising.
    """
    body = Path("src/robot_practice/experiments/literature/w0_video_generation_eval.py").read_text(
        encoding="utf-8"
    )
    # both lookups must be present, and the protocol one must not depend on run_config existing
    assert 'protocol_payload.get("input_camera", "both")' in body
    assert 'manifest_payload.get("input_camera", "both")' in body
    assert 'generation.get("single_camera", "both")' in body

    # the manifest lookup has to come before falling back to run_config, so a LIBERO-Wan run is
    # scored correctly even when its generation block says "both"
    protocol_index = body.index('protocol_payload.get("input_camera", "both")')
    generation_index = body.index('generation.get("single_camera", "both")')
    assert protocol_index < generation_index

    # and the size for a single-camera run must come from the protocol, not a literal
    assert 'gt_width = int(protocol_payload.get("width", 224))' in body
    assert "gt_width = gt_height = 224" not in body

    source = Path("src/robot_practice/experiments/literature/w0_video_generation_eval.py").read_text(
        encoding="utf-8"
    )
    marker = source.index("def score_w0(")
    section = source[marker : marker + 6000]
    # the ground truth must be built with the resolved camera, never the default
    assert "_w0_ground_truth(sample, camera=gt_camera)" in section
    assert "_w0_ground_truth(sample)\n" not in section
