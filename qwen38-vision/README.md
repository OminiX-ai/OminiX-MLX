# Qwen3.8-27B local image chat

Serve a local multimodal MLX checkpoint on Apple silicon, including
`mlx-community/Qwen3.8-27B-8bit`. This Python entry point loads the
checkpoint's vision encoder and language decoder in one process using
MLX-VLM. Select this entry point for image chat. DFlash serves text.

From the repository root, in a Python 3.12 virtual environment:

```sh
python -m pip install -r qwen38-vision/requirements.txt
hf download mlx-community/Qwen3.8-27B-8bit --local-dir ./models/Qwen3.8-27B-8bit
python qwen38-vision/server.py --model ./models/Qwen3.8-27B-8bit --port 8080
```

Existing DFlash installations that already have
`scripts/serve_qwen38_multitenant.py` can also select image chat through that
launcher. With this directory installed at the same repository root, run:

```sh
git apply --check qwen38-vision/dflash-launcher.patch
git apply qwen38-vision/dflash-launcher.patch
python scripts/serve_qwen38_multitenant.py --vision --model ./models/Qwen3.8-27B-8bit
```

Use the directory basename as the model ID. The service binds to IPv4
loopback. It handles one local inference at a time. It supports text
answers, optional thinking, token usage, streaming, and images interleaved
with text across user turns. OpenAI-compatible clients can send
`reasoning_effort: "none"` for a direct answer or `"low"` for brief thinking.
The checkpoint defaults to `"xhigh"`. Model tools, URL fetching and file paths are
outside this review endpoint. Input limits are 32 MiB of JSON and 24 million
decoded image pixels per request.

```python
import base64
import json
import urllib.request
from pathlib import Path

image = base64.b64encode(Path("slide.png").read_bytes()).decode("ascii")
body = {"model": "Qwen3.8-27B-8bit", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Read the slide title."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}}
]}], "chat_template_kwargs": {"enable_thinking": False}}
req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
with urllib.request.urlopen(req) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

Run the protocol tests with `python -m unittest discover -s qwen38-vision`.

With the 8-bit model serving on port 8080, run
`QWEN38_TEST_PORT=8080 python -m unittest discover -s qwen38-vision` to
verify that swapping two real input images changes the answer in both the
JSON and SSE paths.
