"""Local DM05 LIBERO smoke/server using an isolated OpenDM installation.

Run with the OpenDM environment's Python and --help for all arguments.
No model downloads are performed. Smoke uses synthetic inputs, not a benchmark.
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opendm-dir", type=Path, default=Path("outputs/dm05/opendm"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/dm05/checkpoints/DM05-libero")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dm05/smoke"))
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=7891)
    args = parser.parse_args()
    if args.requests < 3:
        parser.error("--requests must be >= 3 (first two requests are warmup)")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)

    def status(message):
        print(message, flush=True)
        with (args.output / "progress.log").open("a") as log:
            log.write(message + "\n")

    for name in ("model.safetensors", "config.json", "norm_stats.json", "tokenizer.json"):
        if not (args.checkpoint / name).is_file():
            raise FileNotFoundError(f"Missing local checkpoint file: {name}")
    sys.path.insert(0, str(args.opendm_dir.resolve()))
    status("STATUS initialization=0/1")
    import numpy as np
    import torch
    from flask import Flask
    from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
    from PIL import Image
    from playground.dm05_libero import DM05Exp, DM05ModelConfig

    class LocalModelConfig(DM05ModelConfig):
        def _load_base_checkpoint_model(self):
            # Upstream applies runtime overrides after construction. The released
            # checkpoint requests FA2 during construction, so override it first.
            config = DM05Config.from_pretrained(self.model_name_or_path)
            config.chunk_size = self.chunk_size
            config.vlm_config._attn_implementation = "eager"
            config.vlm_config.text_config._attn_implementation = "eager"
            config.vlm_config.vision_config._attn_implementation = "sdpa"
            config.action_config._attn_implementation = "sdpa"
            return DM05ForConditionalGeneration.from_pretrained(
                self.model_name_or_path, config=config, dtype=self._torch_dtype()
            )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local validation")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    exp = DM05Exp()
    exp.model_config = LocalModelConfig()
    exp.model_config.model_name_or_path = str(args.checkpoint.resolve())
    exp.model_config.vision_attn_implementation = "sdpa"
    exp.model_config.liger_kernel = False
    exp._initialize_inference_runtime()
    app = Flask(__name__)
    app.add_url_rule("/v1/infer", "infer", exp.inference_config._infer, methods=["POST"])
    app.add_url_rule(
        "/process_frame", "legacy", exp.inference_config._infer_legacy, methods=["POST"]
    )
    status("DONE initialization=1/1")
    if args.serve:
        app.run(host="127.0.0.1", port=args.port, threaded=False)
        return

    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (100, 100, 100)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "observation": {
            "prompt": "pick up the black bowl and place it on the plate",
            "state": [0.0] * 8,
            "images": {"1": encoded, "2": encoded},
            "robot_type": "Franka",
        },
        "sampling": {"seed": args.seed},
    }
    latencies = []
    with app.test_client() as client:
        for index in range(args.requests):
            start = time.perf_counter()
            response = client.post("/v1/infer", json=payload)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if response.status_code != 200:
                raise RuntimeError(
                    f"Inference failed: HTTP {response.status_code}: "
                    f"{response.get_data(as_text=True)}"
                )
            actions = np.asarray(response.get_json()["actions"])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise RuntimeError(f"Invalid action chunk: {actions.shape}")
            latencies.append(elapsed)
            status(f"STATUS requests={index + 1}/{args.requests} latency_ms={elapsed:.1f}")
    result = {
        "category": "Smoke",
        "input": "synthetic gray images and zero state",
        "requests": args.requests,
        "warmup_requests": 2,
        "action_shape": [10, 7],
        "latency_ms": latencies,
        "warm_mean_latency_ms": float(np.mean(latencies[2:])),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "closed_loop_success_rate": None,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    status(f"DONE requests={args.requests}/{args.requests}")


if __name__ == "__main__":
    main()
