"""Greedy-decode K tokens with mlx-vlm as the trusted reference (same 4-bit weights).

16GB-safe via `load(lazy=True)` (vision/audio towers stay mmapped). Each step re-forwards
the growing token sequence through the text model and takes argmax of the last-position
logits (no cache needed for small K — keeps the reference dead-simple and unambiguous).

Usage: python gemma4-mlx/scripts/dump_gemma4_greedy.py <MODEL_DIR> <OUT_DIR> [K]
Outputs: <OUT_DIR>/greedy_ids.json = {"prompt": [...], "greedy": [id0, id1, ...]}
Validates our Rust `generate_greedy` token-for-token.
"""
import json
import os
import sys

import numpy as np
import mlx.core as mx
from mlx_vlm import load

model_dir, out_dir = sys.argv[1], sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 8
os.makedirs(out_dir, exist_ok=True)

model, _ = load(model_dir, lazy=True)
lm = model.language_model

prompt = [2, 1024, 2048, 4096, 8192, 16384]
seq = list(prompt)
greedy = []
for _ in range(K):
    toks = mx.array([seq], dtype=mx.int32)
    hidden = lm.model(toks)
    logits_last = lm.logits_from_hidden(hidden[:, -1:, :])   # [1,1,vocab]
    mx.eval(logits_last)
    nxt = int(np.array(logits_last.astype(mx.float32)[0, 0, :]).argmax())
    greedy.append(nxt)
    seq.append(nxt)

json.dump({"prompt": prompt, "greedy": greedy},
          open(os.path.join(out_dir, "greedy_ids.json"), "w"))
print("prompt:", prompt)
print("greedy:", greedy)
