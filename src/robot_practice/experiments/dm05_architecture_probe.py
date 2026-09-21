"""Record DM05 tensor shapes, never tensor contents; no training or downloads.

Example (repository root):
  PYTHONNOUSERSITE=1 outputs/dm05/env/bin/python \
    src/robot_practice/experiments/dm05_architecture_probe.py \
    --opendm-dir outputs/dm05/opendm \
    --checkpoint outputs/dm05/checkpoints/DM05-libero \
    --output outputs/dm05/architecture
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opendm-dir", type=Path, default=Path("outputs/dm05/opendm"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/dm05/checkpoints/DM05-libero")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/dm05/architecture"))
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(args.opendm_dir.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)

    def status(message):
        print(message, flush=True)
        with (args.output / "progress.log").open("a") as f:
            f.write(message + "\n")

    import numpy as np
    import torch
    from opendm.data.collator import TrainingCollator
    from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
    from PIL import Image
    from playground.dm05_libero import DM05Exp, DM05ModelConfig

    class ProbeModelConfig(DM05ModelConfig):
        def _load_base_checkpoint_model(self):
            # Same pre-construction attention override as dm05_local.py.
            config = DM05Config.from_pretrained(self.model_name_or_path)
            config.chunk_size = self.chunk_size
            config.vlm_config._attn_implementation = "eager"
            config.vlm_config.text_config._attn_implementation = "eager"
            config.vlm_config.vision_config._attn_implementation = "sdpa"
            config.action_config._attn_implementation = "sdpa"
            return DM05ForConditionalGeneration.from_pretrained(
                self.model_name_or_path, config=config, dtype=self._torch_dtype()
            )

    def shape(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {k: shape(v) for k, v in value.items() if v is not None}
        if hasattr(value, "layers") and hasattr(value, "get_seq_length"):
            return {
                "layers": len(value.layers),
                "first_key": shape(value.layers[0].keys),
                "last_key": shape(value.layers[-1].keys),
            }
        if isinstance(value, (tuple, list)):
            return {"length": len(value), "first_two": [shape(v) for v in value[:2]]}
        return type(value).__name__

    torch.manual_seed(7)
    status("STATUS phases=0/3 loading")
    exp = DM05Exp(model_config=ProbeModelConfig())
    exp.model_config.model_name_or_path = str(args.checkpoint.resolve())
    exp.model_config.vision_attn_implementation = "sdpa"
    exp.model_config.liger_kernel = False
    exp._initialize_inference_runtime()
    model = exp.inference_config.model
    # Hooks inspect eager execution, never CUDA Graph capture or replay.
    model._can_use_suffix_graph = lambda _: False
    records = {"inference": {}, "training_forward_no_grad": {}}
    phase = "inference"
    handles = []

    def hook(name):
        def record(module, positional, keyword, output):
            records[phase].setdefault(
                name,
                {"input": shape(positional), "kwargs": shape(keyword), "output": shape(output)},
            )

        return record

    vlm = "model.vlm.model"
    selected = [
        f"{vlm}.vision_tower",
        f"{vlm}.vision_tower.vision_model.embeddings",
        f"{vlm}.vision_tower.vision_model.embeddings.patch_embedding",
        f"{vlm}.multi_modal_projector",
        f"{vlm}.multi_modal_projector.avg_pool",
        f"{vlm}.language_model",
        f"{vlm}.language_model.embed_tokens",
        "model.action_expert",
        "model.time_mlp_in",
        "model.time_mlp_out",
        "model.action_expert.input_time_modulators.0",
        "model.action_expert.final_time_modulator",
    ]
    for base in (f"{vlm}.language_model.layers.0", "model.action_expert.layers.0"):
        selected.extend(
            f"{base}.self_attn.{part}"
            for part in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm")
        )
        selected.extend(f"{base}.mlp.{part}" for part in ("gate_proj", "up_proj", "down_proj"))
    for name in selected:
        handles.append(
            model.get_submodule(name).register_forward_hook(hook(name), with_kwargs=True)
        )

    for name in (
        "_compute_prefix_cache",
        "_action_input_proj",
        "_action_output_proj",
        "_build_adarms_cond",
        "_build_suffix_position_ids",
    ):
        original = getattr(model, name)

        def wrapped(*positional, _name=name, _original=original, **keyword):
            output = _original(*positional, **keyword)
            records[phase].setdefault(
                _name,
                {"input": shape(positional), "kwargs": shape(keyword), "output": shape(output)},
            )
            return output

        setattr(model, name, wrapped)

    def inputs(state_value=0.0):
        sample = {
            "images": [Image.new("RGB", (256, 256), (100, 100, 100)) for _ in range(2)],
            "prompt": "pick up the black bowl and place it on the plate",
            "state": np.full(8, state_value, dtype=np.float32),
            "meta_data": {"robot_type": "Franka", "speed": "0.5"},
        }
        return exp.inference_config.input_transform(sample)

    data = inputs()
    alt = inputs(0.5)
    state_changes_tokens = not torch.equal(data["input_ids"], alt["input_ids"])
    parameters = Counter()
    for name, parameter in model.named_parameters():
        if "vision_tower" in name:
            key = "vision"
        elif "multi_modal_projector" in name:
            key = "multimodal_projector"
        elif "embed_tokens" in name:
            key = "text_embedding"
        elif "lm_head" in name:
            key = "lm_head"
        elif "language_model" in name:
            key = "text_layers_and_norm"
        elif "time_modulator" in name:
            key = "action_time_modulators"
        elif "action_expert" in name:
            key = "action_layers_and_norm"
        else:
            key = ".".join(name.split(".")[:2])
        parameters[key] += parameter.numel()
    inputs_summary = shape(data)
    inference = {k: v for k, v in data.items() if v is not None}
    mask = torch.zeros(1, 1, 32, device="cuda", dtype=torch.bfloat16)
    mask[..., :7] = 1
    with torch.no_grad():
        actions = model.inference_action(**inference, action_mask=mask, diffusion_steps=10)
    assert actions.shape == (1, 10, 32) and torch.isfinite(actions).all()
    status("STATUS phases=1/3 inference_shapes_complete")

    phase = "training_forward_no_grad"
    data["action"] = torch.zeros(1, 10, 32, device="cuda", dtype=torch.float32)
    data["action_mask"] = mask.expand(1, 10, 32).bool()
    collator = TrainingCollator(
        exp.inference_config.processor.tokenizer.pad_token_id,
        max_length=exp.trainer_config.model_max_length,
    )
    # The production collator runs before ToDevice, so collate on CPU first.
    cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    training = {k: v.cuda() for k, v in collator([cpu_data]).items()}
    model.train()
    with torch.no_grad():
        loss = model(**training).loss
    assert loss.ndim == 0 and torch.isfinite(loss)
    status("STATUS phases=2/3 training_forward_shapes_complete")
    embed = model.model.vlm.model.language_model.embed_tokens.weight
    head = model.model.vlm.lm_head.weight
    result = {
        "category": "Smoke",
        "backward_executed": False,
        "optimizer_step_executed": False,
        "parameter_counts": dict(parameters),
        "total_parameters": sum(parameters.values()),
        "embedding_and_head_share_storage": embed.data_ptr() == head.data_ptr(),
        "state_changes_tokens": state_changes_tokens,
        "inference_inputs": inputs_summary,
        "training_inputs": shape(training),
        "internal_action_output": shape(actions),
        "finite_scalar_loss": True,
        "records": records,
    }
    for handle in handles:
        handle.remove()
    (args.output / "shapes.json").write_text(json.dumps(result, indent=2) + "\n")
    status("DONE phases=3/3")


if __name__ == "__main__":
    main()
