"""Dump fixed-input I/O for Gemma 4 text decoder layer 0 (sliding) and the first
full-attention layer, using mlx-vlm as the trusted reference (loads the same 4-bit
weights, fits in 16GB).

Usage:
    python gemma4-mlx/scripts/dump_gemma4_layer_io.py <MODEL_DIR> <OUT_DIR>

Outputs (.npy, float32) into OUT_DIR:
    hidden_in.npy       [1, L, hidden]      fixed random input to both layers
    layer{i}_out.npy    [1, L, hidden]      decoder-layer output (i = sliding idx, full idx)
    meta.json           indices, layer_types, head dims, embed_scale, L

Reference: mlx_vlm/models/gemma4/language.py (DecoderLayer / Attention / ProportionalRoPE).
Deps: pip install mlx-vlm  (already provides mlx, mlx-lm).
"""
import json
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx

from mlx_vlm import load
from mlx_lm.models.base import create_causal_mask


def find_text_model(model):
    """Locate the Gemma4TextModel (has .layers, .embed_scale, .embed_tokens)."""
    for path in ("language_model.model", "language_model", "model"):
        obj = model
        ok = True
        for attr in path.split("."):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok and hasattr(obj, "layers") and hasattr(obj, "embed_scale"):
            return obj, path
    raise RuntimeError("could not locate Gemma4TextModel; inspect `model` tree manually")


def main():
    model_dir, out_dir = sys.argv[1], sys.argv[2]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, _ = load(model_dir)
    tm, tm_path = find_text_model(model)
    layers = tm.layers
    layer_types = [getattr(l, "layer_type", "?") for l in layers]
    print(f"text model at model.{tm_path}; {len(layers)} layers; embed_scale={float(tm.embed_scale)}")

    sliding_idx = layer_types.index("sliding_attention")
    full_idx = layer_types.index("full_attention")
    print(f"sliding layer idx={sliding_idx}, full layer idx={full_idx}")

    # Fixed input. L < sliding_window(1024) so sliding & full masks are both plain causal.
    B, L = 1, 6
    hidden = int(model.config.text_config.hidden_size)
    mx.random.seed(0)
    h_in = (mx.random.normal((B, L, hidden)) * 0.1).astype(mx.float32)
    np.save(out / "hidden_in.npy", np.array(h_in, dtype=np.float32))

    # Plain causal mask matching the dtype the attention expects.
    mask = create_causal_mask(L, 0).astype(mx.float32)

    meta = {
        "text_model_path": tm_path,
        "layer_types": layer_types,
        "sliding_idx": sliding_idx,
        "full_idx": full_idx,
        "hidden_size": hidden,
        "embed_scale": float(tm.embed_scale),
        "L": L,
        "head_dim": int(model.config.text_config.head_dim),
        "global_head_dim": int(getattr(model.config.text_config, "global_head_dim", 0)),
    }

    for idx in (sliding_idx, full_idx):
        layer = layers[idx]
        out_tuple = layer(h_in, mask=mask, cache=None, offset=0)
        h_out = out_tuple[0] if isinstance(out_tuple, tuple) else out_tuple
        mx.eval(h_out)
        arr = np.array(h_out, dtype=np.float32)
        np.save(out / f"layer{idx}_out.npy", arr)
        print(f"layer {idx} ({layer_types[idx]}): out shape {arr.shape}, "
              f"mean={arr.mean():.5f}, std={arr.std():.5f}, absmax={np.abs(arr).max():.5f}")

    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("dumped:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
