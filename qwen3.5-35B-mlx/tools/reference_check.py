"""Opt-in real-weight validation against mlx-vlm; never used by the server.

Runs red/blue RGB counterfactuals, then compares the image features, multimodal
positions and first-token logits to an independently implemented reference.
"""

import argparse
import base64
import io
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from pear_server import Engine, language_positions
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--worker", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--nano-request", type=Path, help="Saved AISLE request with real RGB and history"
    )
    args = p.parse_args()
    engine = Engine(args.model_dir, args.worker, args.out, timeout=300)
    responses = []
    try:
        for color in ("red", "blue"):
            image = Image.new("RGB", (320, 240), "white")
            ImageDraw.Draw(image).rectangle((80, 50, 240, 210), fill=color)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            body = {
                "model": "local-checkpoint",
                "max_tokens": 16,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,"
                                    + base64.b64encode(buffer.getvalue()).decode()
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "What color is the large square? "
                                    "Answer with one color word only."
                                ),
                            },
                        ],
                    }
                ],
            }
            response = engine.infer(body, debug=True)
            responses.append(response)
            print(json.dumps({"expected": color, "response": response}), flush=True)
            answer = response["choices"][0]["message"]["content"].strip().lower().strip(".")
            assert answer == color, (color, answer)
        if args.nano_request:
            body = json.loads(args.nano_request.read_text())
            body.update(
                model="local-checkpoint",
                max_tokens=1,
                temperature=0,
                chat_template_kwargs={"enable_thinking": False},
            )
            response = engine.infer(body, debug=True)
            responses.append(response)
            assert response["choices"][0]["finish_reason"] == "length"
            print(json.dumps({"nano_probe": response}), flush=True)
    finally:
        engine.close()
    # Free the Rust process before loading the independent Python reference.
    from mlx_vlm.utils import load_model

    model = load_model(args.model_dir.resolve(), lazy=True)
    model.vision_tower.set_dtype(mx.float32)
    # Match Rust's FP32 compute with the checkpoint's BF16 embedding lookup.
    embedding_weights = model.language_model.model.embed_tokens.parameters()
    model.language_model.set_dtype(mx.float32)
    model.language_model.model.embed_tokens.update(embedding_weights)
    checks = []
    for file in sorted(args.out.glob("*/input.safetensors")):
        checks.append(check_reference(model, file.parent))
    report = {
        "ok": True,
        "model_dir": str(args.model_dir.resolve()),
        "reference": "mlx-vlm 0.8.0",
        "vision_compute_dtype": "float32",
        "checks": checks,
        "rgb_counterfactual_colors": ["red", "blue"],
        "responses": responses,
    }
    (args.out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("VALIDATION_PASS", flush=True)


def check_reference(model, directory):
    t = mx.load(str(directory / "input.safetensors"))
    actual = mx.load(str(directory / "debug.safetensors"))
    grid = mx.array(json.loads((directory / "grids.json").read_text()))
    pixels = mx.concatenate([t[f"pixels_{i}"] for i in range(len(grid))]).astype(
        model.vision_tower.patch_embed.proj.weight.dtype
    )
    features, _ = model.vision_tower(pixels, grid)
    mx.eval(features)
    rust_features = mx.concatenate([actual[f"features_{i}"] for i in range(len(grid))])
    diff = features.astype(mx.float32) - rust_features.astype(mx.float32)
    feature_metrics = {
        "max_abs": mx.max(mx.abs(diff)).item(),
        "rms": mx.sqrt(mx.mean(diff * diff)).item(),
        "reference_rms": mx.sqrt(mx.mean(features.astype(mx.float32) ** 2)).item(),
    }
    print(json.dumps({"vision": feature_metrics}), flush=True)
    assert feature_metrics["rms"] / feature_metrics["reference_rms"] < 0.001, feature_metrics
    req = json.loads((directory / "worker-request.json").read_text())
    positions, _ = model.language_model.get_rope_index(t["input_ids"], grid)
    local = language_positions(
        t["input_ids"].shape[1], req["image_ranges"], grid.tolist(), req["max_tokens"], 2
    )
    np.testing.assert_array_equal(np.array(positions[:, 0]), local[:, : positions.shape[-1]])
    inputs = model.get_input_embeddings(t["input_ids"], pixels, image_grid_thw=grid).to_dict()
    inputs["inputs_embeds"] = inputs["inputs_embeds"].astype(mx.float32)
    output = model.language_model(t["input_ids"], **inputs)
    expected = output.logits[:, -1, :].astype(mx.float32)
    mx.eval(expected)
    logits = actual["logits"].astype(mx.float32)
    diff = expected - logits
    logit_metrics = {
        "rms": mx.sqrt(mx.mean(diff * diff)).item(),
        "max_abs": mx.max(mx.abs(diff)).item(),
        "reference_top1": mx.argmax(expected, axis=-1).item(),
        "ominix_top1": mx.argmax(logits, axis=-1).item(),
    }
    expected_logp = expected - mx.logsumexp(expected, axis=-1, keepdims=True)
    actual_logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    logit_metrics["kl_divergence"] = mx.sum(
        mx.exp(expected_logp) * (expected_logp - actual_logp)
    ).item()
    logit_metrics["reference_choice_regret"] = (
        mx.max(expected) - expected[0, logit_metrics["ominix_top1"]]
    ).item()
    logit_metrics["top5_overlap"] = len(
        set(mx.argsort(expected[0])[-5:].tolist()) & set(mx.argsort(logits[0])[-5:].tolist())
    )
    print(json.dumps({"logits": logit_metrics}), flush=True)
    # Different fused kernels can round the final bits differently. Bound the
    # numerical/distribution error and retain any near-tie choice differences.
    assert logit_metrics["rms"] < 0.005, logit_metrics
    assert logit_metrics["max_abs"] < 0.05, logit_metrics
    assert logit_metrics["kl_divergence"] < 0.0001, logit_metrics
    assert logit_metrics["reference_choice_regret"] < 0.01, logit_metrics
    assert logit_metrics["top5_overlap"] >= 4, logit_metrics
    return {
        "directory": str(directory),
        "image_count": len(grid),
        "vision": feature_metrics,
        "logits": logit_metrics,
        "mrope_positions_equal": True,
    }


if __name__ == "__main__":
    main()
