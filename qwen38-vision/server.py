"""Local Qwen3.8 image chat on Apple silicon, using the model's MLX vision tower."""

import argparse
import base64
import binascii
import io
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image, ImageOps

MAX_BODY = 32 << 20
MAX_PIXELS = 24_000_000
FORMATS = {
    "data:image/png;base64": "PNG",
    "data:image/jpeg;base64": "JPEG",
    "data:image/webp;base64": "WEBP",
}


def decode_image(url, pixels_left):
    if not isinstance(url, str):
        raise ValueError("image_url.url must be an inline image data URL")
    header, _, encoded = url.partition(",")
    if header not in FORMATS:
        raise ValueError("Use an inline PNG, JPEG or WebP data URL")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw), formats=[FORMATS[header]]) as image:
            if image.width * image.height > pixels_left:
                raise ValueError("Request image pixel budget exceeded")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (binascii.Error, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Invalid inline image") from exc


def prepare_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    chat, images = [], []
    pixels_left = MAX_PIXELS
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") not in (
            "system",
            "user",
            "assistant",
        ):
            raise ValueError("Expected system, user or assistant message")
        content = msg.get("content")
        if isinstance(content, str):
            chat.append({"role": msg["role"], "content": content})
            continue
        if not isinstance(content, list) or not content:
            raise ValueError("Expected text or a nonempty list of message parts")
        parts = []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("Invalid message part")
            if part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append({"type": "text", "text": part["text"]})
            elif msg["role"] == "user" and part.get("type") == "image_url":
                ref = part.get("image_url")
                image = decode_image(
                    ref.get("url") if isinstance(ref, dict) else None, pixels_left
                )
                pixels_left -= image.width * image.height
                images.append(image)
                parts.append({"type": "image"})
            else:
                raise ValueError("Unsupported message part")
        chat.append({"role": msg["role"], "content": parts})
    return chat, images


class VisionModel:
    def __init__(self, path):
        from mlx_vlm import load

        config = json.loads((path / "config.json").read_text())
        if not config.get("vision_config") or config.get("language_model_only"):
            raise ValueError("Checkpoint must include a vision tower")
        self.name = path.name
        self.model, self.processor = load(str(path), trust_remote_code=False)

    def generate(self, messages, images, max_tokens, temperature, top_p, thinking):
        from mlx_vlm import stream_generate

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        yield from stream_generate(
            self.model,
            self.processor,
            prompt,
            image=images or None,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=thinking,
        )


def number(body, key, default, low, high):
    value = body.get(key, default)
    if type(value) not in (int, float) or not low <= value <= high:
        raise ValueError(f"{key} must be between {low} and {high}")
    return value


class Handler(BaseHTTPRequestHandler):
    timeout = 60

    def json_response(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def error(self, status, message):
        self.json_response(status, {"error": {"message": message}})

    def do_GET(self):
        if self.path == "/v1/models":
            self.json_response(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.engine.name,
                            "object": "model",
                            "owned_by": "local",
                        }
                    ],
                },
            )
        else:
            self.error(404, "Unknown route")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.error(404, "Unknown route")
            return
        if self.headers.get_content_type() != "application/json":
            self.error(415, "Expected application/json")
            return
        try:
            size = int(self.headers.get("Content-Length", "-1"))
            if not 0 < size <= MAX_BODY:
                self.error(
                    413 if size > MAX_BODY else 411, "Expected bounded Content-Length"
                )
                return
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            if body.get("model") != self.server.engine.name:
                self.error(404, "Unknown model")
                return
            chat, images = prepare_messages(body.get("messages"))
            max_tokens = body.get("max_completion_tokens", body.get("max_tokens", 1024))
            if type(max_tokens) is not int or not 1 <= max_tokens <= 65536:
                raise ValueError("Invalid completion token budget")
            temperature = number(body, "temperature", 0.0, 0, 2)
            top_p = number(body, "top_p", 1.0, 0, 1)
            stream = body.get("stream", False)
            options = body.get("stream_options") or {}
            template = body.get("chat_template_kwargs", {})
            if not isinstance(template, dict):
                raise ValueError("Invalid chat template options")
            thinking = template.get("enable_thinking", True)
            if type(thinking) is not bool:
                raise ValueError("enable_thinking must be boolean")
            if type(stream) is not bool or not isinstance(options, dict):
                raise ValueError("Invalid streaming options")
            if body.get("tool_choice") not in (None, "none", "auto"):
                raise ValueError("This visual reviewer returns text answers only")
        except (ValueError, RecursionError) as exc:
            self.error(400, str(exc))
            return
        engine = self.server.engine
        from mlx_vlm.server.responses_state import ThinkingStreamState

        result = engine.generate(chat, images, max_tokens, temperature, top_p, thinking)
        splitter = ThinkingStreamState(enable_thinking=thinking)
        meta = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "model": engine.name,
            "created": int(time.time()),
            "object": "chat.completion.chunk" if stream else "chat.completion",
        }
        started = False

        def event(value):
            self.wfile.write(
                b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"
            )
            self.wfile.flush()

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                started = True
                event(
                    {
                        **meta,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            content, reasoning = [], []
            last = None
            for last in result:
                split = splitter.feed(last.text)
                delta = {}
                if split.content:
                    delta["content"] = split.content
                if split.reasoning:
                    delta["reasoning_content"] = split.reasoning
                if stream and delta:
                    event(
                        {
                            **meta,
                            "choices": [
                                {"index": 0, "delta": delta, "finish_reason": None}
                            ],
                        }
                    )
                elif not stream:
                    content.append(split.content or "")
                    reasoning.append(split.reasoning or "")
            if last is None:
                raise RuntimeError("No inference result")
            if splitter.buffer:
                field = "reasoning_content" if splitter.in_thinking else "content"
                if stream:
                    event(
                        {
                            **meta,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {field: splitter.buffer},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                else:
                    (reasoning if splitter.in_thinking else content).append(
                        splitter.buffer
                    )
            finish = last.finish_reason or "stop"
            usage = {
                "prompt_tokens": last.prompt_tokens,
                "completion_tokens": last.generation_tokens,
                "total_tokens": last.total_tokens,
            }
            if stream:
                event(
                    {
                        **meta,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                    }
                )
                if options.get("include_usage"):
                    event({**meta, "choices": [], "usage": usage})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            else:
                message = {"role": "assistant", "content": "".join(content)}
                if any(reasoning):
                    message["reasoning_content"] = "".join(reasoning)
                self.json_response(
                    200,
                    {
                        **meta,
                        "choices": [
                            {"index": 0, "message": message, "finish_reason": finish}
                        ],
                        "usage": usage,
                    },
                )
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self.log_error("Inference failed (%s)", type(exc).__name__)
            if started:
                event({"error": {"message": "Vision inference failed"}})
            else:
                self.error(500, "Vision inference failed")
        finally:
            result.close()


class LocalServer(HTTPServer):
    # The model and its Metal stream stay on their loading thread.
    request_queue_size = 32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Local multimodal MLX checkpoint"
    )
    parser.add_argument(
        "--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1"
    )
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    with LocalServer((args.host, args.port), Handler) as server:
        server.engine = VisionModel(args.model.expanduser().resolve())
        print(
            f"Serving {server.engine.name} on 127.0.0.1:{server.server_port}",
            flush=True,
        )
        server.serve_forever()


if __name__ == "__main__":
    main()
