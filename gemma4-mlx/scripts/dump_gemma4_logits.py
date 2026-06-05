"""Full-model logits golden via mlx-vlm (same 4-bit weights; correct internal causal masks).

16GB-safe: `load(..., lazy=True)` keeps the unused vision/audio tower weights mmapped on
disk (never materialized), so only the text path is resident. Runs `language_model.model`
(embed×scale → 48 layers with proper masks → final RMSNorm), then projects ONLY the last
position via the official `logits_from_hidden` (tied embed_tokens.as_linear +
final_logit_softcapping) — exactly our Rust `Gemma4TextModel::forward(.., last_only=true)`.

Usage: python gemma4-mlx/scripts/dump_gemma4_logits.py <MODEL_DIR> <OUT_DIR>
Outputs:
    tokens.npy        [1, L] int32
    hidden_last.npy   [1, hidden] float32   (normed hidden at last position; fallback compare)
    logits_last.npy   [1, vocab] float32    (logits at last position)
    meta.json         {argmax, L, vocab, absmax}
Note: mlx arrays may be bf16 → cast via `.astype(mx.float32)` before numpy (direct np.array
of a bf16 mlx array raises a PEP-3118 buffer error).
"""
import json
import os
import sys

import numpy as np
import mlx.core as mx
from mlx_vlm import load

model_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

model, _ = load(model_dir, lazy=True)   # lazy => vision/audio towers stay on disk (fits 16GB)
lm = model.language_model

tokens = mx.array([[2, 1024, 2048, 4096, 8192, 16384]], dtype=mx.int32)  # [1, 6]

hidden = lm.model(tokens)                       # final-normed hidden [1, L, hidden]
mx.eval(hidden)
logits_last = lm.logits_from_hidden(hidden[:, -1:, :])   # [1, 1, vocab]
mx.eval(logits_last)

h32 = np.array(hidden.astype(mx.float32)[:, -1, :]).astype(np.float32)   # [1, hidden]
l32 = np.array(logits_last.astype(mx.float32)[:, 0, :]).astype(np.float32)  # [1, vocab]

np.save(os.path.join(out_dir, "tokens.npy"), np.array(tokens, dtype=np.int32))
np.save(os.path.join(out_dir, "hidden_last.npy"), h32)
np.save(os.path.join(out_dir, "logits_last.npy"), l32)
meta = {"argmax": int(l32.argmax()), "L": int(tokens.shape[1]),
        "vocab": int(l32.shape[-1]), "absmax": float(np.abs(l32).max())}
json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"))
print("DUMPED", meta, "top5:", np.argsort(-l32[0])[:5].tolist())
