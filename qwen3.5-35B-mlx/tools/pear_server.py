"""Loopback Chat Completions adapter for OminiX's Rust PEAR worker.

Python handles the checkpoint's tokenizer/image processor and position tensors.
Vision and language inference run exclusively in the persistent Rust worker.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import io
import json
import math
import queue
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np


def image_positions(grid, base, merge, head_dim):
    import mlx.core as mx

    t, h, w = map(int, grid)
    if t != 1 or h % merge or w % merge or head_dim % 4:
        raise ValueError("Only still images with complete merge blocks are supported")
    order = (
        np.arange(h * w)
        .reshape(h // merge, merge, w // merge, merge)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    row, col = np.meshgrid(
        np.linspace(0, base - 1, h, dtype=np.float32),
        np.linspace(0, base - 1, w, dtype=np.float32),
        indexing="ij",
    )
    r0, c0 = np.floor(row), np.floor(col)
    r1, c1 = np.minimum(r0 + 1, base - 1), np.minimum(c0 + 1, base - 1)
    dr, dc = row - r0, col - c0
    indices = (
        np.stack([r0 * base + c0, r0 * base + c1, r1 * base + c0, r1 * base + c1])
        .reshape(4, -1)[:, order]
        .astype(np.uint32)
    )
    weights = np.stack([(1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc]).reshape(4, -1)[
        :, order
    ]
    row, col = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # Use MLX float32 trig like the reference; NumPy trig differs near BF16
    # rounding boundaries and the discrepancy compounds through 27 blocks.
    inv = 1 / (10000 ** (mx.arange(0, head_dim // 2, 2, dtype=mx.float32) / (head_dim // 2)))
    angles = mx.concatenate(
        [
            mx.array(row.reshape(-1)[order, None]) * inv,
            mx.array(col.reshape(-1)[order, None]) * inv,
        ],
        axis=-1,
    )
    angles = mx.concatenate([angles, angles], axis=-1)[None, None]
    return indices, weights, np.array(mx.cos(angles)), np.array(mx.sin(angles))


def language_positions(length, ranges, grids, max_tokens, merge):
    if len(ranges) != len(grids):
        raise ValueError("Image/token count mismatch")
    result = np.empty((3, length + max_tokens), dtype=np.int32)
    cursor = position = 0
    for (offset, count), (t, h, w) in zip(ranges, grids, strict=True):
        h, w = int(h) // merge, int(w) // merge
        if t != 1 or offset < cursor or count != h * w or offset + count > length:
            raise ValueError("Invalid image placeholder range")
        n = offset - cursor
        result[:, cursor:offset] = np.arange(position, position + n)
        position += n
        result[:, offset : offset + count] = (
            np.stack(
                [np.zeros(count, dtype=int), np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]
            )
            + position
        )
        position += max(h, w)
        cursor = offset + count
    result[:, cursor:] = np.arange(position, position + length + max_tokens - cursor)
    return result


def language_rotary(pos, config):
    import mlx.core as mx

    rope = config["rope_parameters"]
    if not rope.get("mrope_interleaved") or rope.get("rope_type", "default") != "default":
        raise ValueError("This worker supports interleaved default mRoPE only")
    dim = int(config["head_dim"] * rope["partial_rotary_factor"])
    selector = np.zeros(dim // 2, dtype=int)
    for axis in (1, 2):
        selector[axis : min(rope["mrope_section"][axis] * 3, dim // 2) : 3] = axis
    inv = 1 / (rope["rope_theta"] ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    freqs = mx.array(pos[selector].T, dtype=mx.float32) * inv
    angles = mx.concatenate([freqs, freqs], axis=-1)[None, None]
    return np.array(mx.cos(angles)), np.array(mx.sin(angles))


def split_answer(text, thinking):
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "</think>" in text:
        reasoning, answer = text.split("</think>", 1)
        return answer.strip(), reasoning.removeprefix("<think>").strip()
    if thinking or "<think>" in text:
        return "", text.removeprefix("<think>").strip()
    return text, None


class Engine:
    def __init__(self, model_dir, worker, trace_dir, timeout=170):
        from transformers import AutoProcessor

        self.path, self.trace = Path(model_dir).resolve(), Path(trace_dir).resolve()
        self.trace.mkdir(parents=True, exist_ok=True)
        self.config = json.loads((self.path / "config.json").read_text())
        provenance = self.path / "provenance.json"
        self.provenance = (
            json.loads(provenance.read_text())
            if provenance.exists()
            else {"source_revision": "unverified"}
        )
        self.processor = AutoProcessor.from_pretrained(self.path, trust_remote_code=False)
        self.timeout = timeout
        self.lock = threading.Lock()
        worker = Path(worker).resolve()
        with worker.open("rb") as stream:
            self.worker_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.process = subprocess.Popen(
            [str(worker), str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.responses = queue.Queue()

        def read():
            for line in self.process.stdout:
                self.responses.put(line)
            self.responses.put(None)

        threading.Thread(target=read, daemon=True).start()
        ready = self.receive()
        if ready.get("ready") is not True:
            raise RuntimeError(f"Worker startup failed: {ready}")

    def receive(self):
        try:
            line = self.responses.get(timeout=self.timeout)
        except queue.Empty as error:
            self.process.kill()
            self.process.wait()
            raise TimeoutError("OminiX inference deadline exceeded; worker stopped") from error
        if line is None:
            raise RuntimeError(f"OminiX worker exited ({self.process.poll()})")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def prepare(self, body, directory):
        import mlx.core as mx
        from PIL import Image

        if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
            raise ValueError("Expected a JSON object with a messages list")
        maximum = body.get("max_tokens", 512)
        if type(maximum) is not int or not 1 <= maximum <= 4096:
            raise ValueError("max_tokens must be between 1 and 4096")
        if body.get("stream") or body.get("temperature", 0) != 0:
            raise ValueError("Only non-streaming greedy decoding is supported")
        messages = copy.deepcopy(body["messages"])
        images = []
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Unsupported message role")
            if isinstance(message["content"], str):
                continue
            for part in message["content"]:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/png;base64,"):
                        raise ValueError("Only inline PNG RGB observations are supported")
                    image = Image.open(
                        io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))
                    )
                    if image.width * image.height > 1048576:
                        raise ValueError("Observation exceeds one megapixel")
                    images.append(image.convert("RGB"))
                    part.clear()
                    part["type"] = "image"
                elif part.get("type") != "text":
                    raise ValueError("Unsupported content type")
        if len(images) > 15:
            raise ValueError("At most 15 observation images are supported")
        template_args = body.get("chat_template_kwargs", {})
        allowed = {"enable_thinking", "preserve_thinking", "reasoning_effort"}
        if set(template_args) - allowed:
            raise ValueError("Unsupported chat template argument")
        thinking = template_args.get("enable_thinking", True)
        if type(thinking) is not bool:
            raise ValueError("enable_thinking must be a boolean")
        if type(template_args.get("preserve_thinking", False)) is not bool:
            raise ValueError("preserve_thinking must be a boolean")
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_args
        )
        inputs = self.processor(text=prompt, images=images or None, return_tensors="np")
        ids = np.asarray(inputs["input_ids"], dtype=np.uint32)
        if np.any(ids == self.config["video_token_id"]):
            raise ValueError("Video tokens are unsupported")
        if ids.shape[1] > 8192:
            raise ValueError("Prompt exceeds 8192 tokens")
        image_id = self.config["image_token_id"]
        positions = np.flatnonzero(ids[0] == image_id)
        groups = (
            np.split(positions, np.flatnonzero(np.diff(positions) != 1) + 1)
            if positions.size
            else []
        )
        ranges = [(int(group[0]), len(group)) for group in groups]
        grids = inputs.get("image_grid_thw", [])
        vc = self.config["vision_config"]
        pos = language_positions(ids.shape[1], ranges, grids, maximum, vc["spatial_merge_size"])
        cos, sin = language_rotary(pos, self.config["text_config"])
        tensors = {"input_ids": ids, "rotary_cos": cos, "rotary_sin": sin}
        cursor = 0
        for i, grid in enumerate(grids):
            n = int(np.prod(grid))
            if n > 4096:
                raise ValueError("Image exceeds 4096 patches")
            tensors[f"pixels_{i}"] = inputs["pixel_values"][cursor : cursor + n]
            cursor += n
            values = image_positions(
                grid,
                math.isqrt(vc["num_position_embeddings"]),
                vc["spatial_merge_size"],
                vc["hidden_size"] // vc["num_heads"],
            )
            for key, value in zip(
                ("pos_indices", "pos_weights", "vision_cos", "vision_sin"), values, strict=True
            ):
                tensors[f"{key}_{i}"] = value
            images[i].save(directory / f"image-{i}.png")
        mx.save_safetensors(
            str(directory / "input.safetensors"), {k: mx.array(v) for k, v in tensors.items()}
        )
        (directory / "prompt.txt").write_text(prompt)
        (directory / "grids.json").write_text(json.dumps(np.asarray(grids).tolist()))
        request = {
            "tensors": str(directory / "input.safetensors"),
            "max_tokens": maximum,
            "image_ranges": ranges,
            "eos_token_ids": [248046, 248044],
        }
        (directory / "worker-request.json").write_text(json.dumps(request, indent=2))
        return request, thinking

    def infer(self, body, debug=False):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("OminiX worker is busy")
        try:
            started = time.monotonic()
            directory = self.trace / str(time.time_ns())
            directory.mkdir()
            request, thinking = self.prepare(body, directory)
            if debug:
                request["debug_output"] = str(directory / "debug.safetensors")
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            result = self.receive()
            (directory / "worker-response.json").write_text(json.dumps(result, indent=2))
            text = self.processor.tokenizer.decode(result["token_ids"], skip_special_tokens=False)
            content, reasoning = split_answer(text, thinking)
            message = {"role": "assistant", "content": content}
            if reasoning is not None:
                message["reasoning_content"] = reasoning
            response = {
                "id": f"ominix-{directory.name}",
                "object": "chat.completion",
                "model": body["model"],
                "choices": [
                    {"index": 0, "message": message, "finish_reason": result["finish_reason"]}
                ],
                "usage": {
                    "prompt_tokens": result["prompt_tokens"],
                    "completion_tokens": len(result["token_ids"]),
                    "total_tokens": result["prompt_tokens"] + len(result["token_ids"]),
                    "backend": "ominix-mlx",
                    "worker_sha256": self.worker_sha256,
                    "source_repo": self.provenance.get("source_repo"),
                    "vision_compute_dtype": "float32",
                    "language_compute_dtype": "float32",
                    "source_revision": self.provenance["source_revision"],
                    "wall_time_s": time.monotonic() - started,
                    "image_count": result["image_count"],
                    "request_id": directory.name,
                },
                "ominix": {k: v for k, v in result.items() if k != "token_ids"},
            }
            (directory / "response.json").write_text(json.dumps(response, indent=2))
            return response
        finally:
            self.lock.release()

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="qwen3.8-27b-pear")
    parser.add_argument("--port", default=8087, type=int)
    parser.add_argument("--timeout", default=170, type=float)
    args = parser.parse_args()
    engine = Engine(args.model_dir, args.worker, args.trace_dir, args.timeout)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": args.model_name, "object": "model", "owned_by": "local"}],
                    },
                )
            elif self.path == "/health":
                ready = engine.process.poll() is None
                self.reply(
                    200 if ready else 503,
                    {
                        "ok": ready,
                        "backend": "ominix-mlx",
                        "model": args.model_name,
                        "busy": engine.lock.locked(),
                        "provenance": engine.provenance,
                        "worker_sha256": engine.worker_sha256,
                    },
                )
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 20 * 1024 * 1024:
                    raise ValueError("Request body exceeds 20 MiB or is empty")
                self.connection.settimeout(15)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if body.get("model") != args.model_name:
                    raise ValueError("Unknown model")
                self.reply(200, engine.infer(body))
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {"error": str(error)})
            except (RuntimeError, TimeoutError, BrokenPipeError) as error:
                self.reply(503, {"error": str(error)})

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    server = None
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
        print(
            json.dumps(
                {
                    "ready": True,
                    "endpoint": f"http://127.0.0.1:{args.port}/v1",
                    "backend": "ominix-mlx",
                    "model": args.model_name,
                }
            ),
            flush=True,
        )
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        engine.close()


if __name__ == "__main__":
    main()
