# Local PEAR vision inference for AISLE / Adora Nano

This adapter serves the official `mit-han-lab/Qwen3.8-27B-PEAR` checkpoint on
Apple Silicon through OminiX-MLX. The persistent Rust `pear_worker` runs the
Qwen vision tower and quantized hybrid language decoder. Python supplies the
checkpoint tokenizer, RGB image preprocessing, position tensors, and a loopback
Chat Completions endpoint. It does not perform neural inference in the server.

`mlx-vlm` is used separately to convert weights and to validate the Rust model.
PyTorch/torchvision are dependencies of the checkpoint's Transformers processor;
the server does not load a PyTorch model.

## Build and prepare the checkpoint

Run from the OminiX-MLX root. The setup used Python 3.13 on an M3 Max with
128 GiB unified memory. Minimum memory requirements have not been established.
The source weights occupy 55.6 GB; retain room for the converted model as well.

```sh
uv venv .venv-pear --python 3.13
uv pip install --python .venv-pear/bin/python \
  -r qwen3.5-35B-mlx/tools/requirements-pear.txt
cargo build --release -j 4 -p qwen3-5-35b-mlx --example pear_worker
```

The worker needs `mlx.metallib` beside its executable. This checkout's build
places it in `target/release`; if the examples directory lacks it, create the
relative link:

```sh
test -e target/release/examples/mlx.metallib || \
  ln -s ../mlx.metallib target/release/examples/mlx.metallib
```

Download the pinned PEAR model and its upstream blob metadata:

```sh
uv run --no-project --python .venv-pear/bin/python - <<'PY'
import json
import urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download
repo = "mit-han-lab/Qwen3.8-27B-PEAR"
revision = "65176e435e5fa80a059ebfafaf8215e6ad00a83e"
root = Path("models/pear-local")
root.mkdir(parents=True, exist_ok=True)
metadata = json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"))
(root / "upstream-model.json").write_text(json.dumps(metadata, indent=2))
snapshot_download(repo, revision=revision, local_dir=root / "source-model")
PY

uv run --no-project --python .venv-pear/bin/python \
  qwen3.5-35B-mlx/tools/convert_pear.py \
  --source models/pear-local/source-model \
  --metadata models/pear-local/upstream-model.json \
  --out models/pear-local/pear-4bit
```

Conversion verifies both source shards against their upstream SHA-256 values,
then writes affine INT4 language weights (group size 64), unquantized BF16
vision weights, and `provenance.json` with source and converted hashes.
Conversion refuses an existing output directory. At inference, the vision
tower expands the stored BF16 weights to float32 for stable attention math.
The language decoder uses the existing OminiX quantized implementation with
float32 activations and recurrent state. Text/image embeddings are merged in
the checkpoint embedding dtype before promotion. This keeps compute precision
consistent across the first and subsequent decoder layers.

## Start the local server

```sh
uv run --no-project --python .venv-pear/bin/python \
  qwen3.5-35B-mlx/tools/pear_server.py \
  --model-dir models/pear-local/pear-4bit \
  --worker target/release/examples/pear_worker \
  --trace-dir models/pear-local/server-traces \
  --model-name qwen3.8-27b-pear --port 8087
```

Endpoints: `http://127.0.0.1:8087/health`, `/v1/models`, and
`/v1/chat/completions`. Health includes checkpoint provenance. The adapter
accepts inline PNG images, greedy decoding (`temperature: 0`), at most 15
images / 8192 prompt tokens / 4096 generated tokens, and one inference at a
time. It rejects remote image URLs and streaming requests.

Set `chat_template_kwargs.enable_thinking` explicitly. AISLE's local smoke
test uses `false` and 512 output tokens to bound latency. Truncation remains
`finish_reason: length`, and AISLE must reject it. Reasoning is separated into
`reasoning_content`; only the final `content` is an action candidate. A worker
deadline terminates the worker and returns an error; restart the server after
such a timeout.

Every request saves the exact prompt, input PNGs, model tensors, worker request,
and response. Usage identifies `ominix-mlx`, checkpoint revision, image count,
and wall time. No API key, Codex process, or remote inference service is used.

## Validate the model implementation

```sh
uv run --no-project --python .venv-pear/bin/python -m pytest -q \
  qwen3.5-35B-mlx/tools/test_pear_server.py

uv run --no-project --python .venv-pear/bin/python \
  qwen3.5-35B-mlx/tools/reference_check.py \
  --model-dir models/pear-local/pear-4bit \
  --worker target/release/examples/pear_worker \
  --out models/pear-local/reference-check-001
```

Use a fresh output directory. Add `--nano-request /absolute/path/request.json`
to include a recorded AISLE RGB request with action history. The check changes
the image while keeping the color question fixed, compares vision features and
multimodal positions against independent `mlx-vlm`, and records logit RMS error,
distribution divergence, and top-choice differences. It does not assume that
different intermediate precision produces identical greedy token IDs.

## Connect AISLE's Nano fixture

From a neighboring AISLE checkout, using a Python environment with AISLE,
Nexus 0.2.1 and Dora 1.0.1 installed:

```sh
UV_PROJECT_ENVIRONMENT=runs/nano-pear-local/.venv uv run --no-sync \
  python -m aisle.harness.pear prepare \
  --out runs/nano-pear-001 --embodiment nano --sim-engine nexus \
  --nano-urdf "$PWD/../dorobot/data/adora_nano/adora_nano.urdf" \
  --python "$PWD/runs/nano-pear-local/.venv/bin/python" \
  --endpoint http://127.0.0.1:8087/v1 --model qwen3.8-27b-pear \
  --model-revision 65176e435e5fa80a059ebfafaf8215e6ad00a83e \
  --no-thinking --max-tokens 512 --duration-s 6 --mode lockstep-eval \
  --instruction 'Inspect the RGB images, name the cube colors, and select an arm for a non-contact probe toward the red cube. Move at most 0.3 cm per action, keep the other arm and grippers still, and reobserve before choosing the next move.'

uv tool run --from dora-rs-cli==1.0.1 dora run runs/nano-pear-001/dataflow.yaml
```

The model receives three RGB camera views and measured joint/FK state, with no
simulator object coordinates or depth. JSON pose deltas pass through Nano IK,
joint/workspace guards and measured motion feedback before the next observation.
`lockstep-eval` pauses physics during inference; it is not a real-time result.

This fixture fixes Nano's root and wheels. It validates bounded arm control and
RGB feedback, not room navigation, cube grasping/placement, or hardware use.
The Nano embodiment prompt is an adapter; this does not establish that PEAR
was trained on Nano. No benchmark skill/evalcard is registered by these tools.
