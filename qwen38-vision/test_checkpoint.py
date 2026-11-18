"""Optional integration test against a running local Qwen3.8-27B-8bit server."""

import json
import os
import unittest
import urllib.request

from test_server import picture


@unittest.skipUnless(
    os.getenv("QWEN38_TEST_PORT"), "set QWEN38_TEST_PORT for real-weight inference"
)
class Checkpoint(unittest.TestCase):
    def test_image_order_changes_answer_in_json_and_sse(self):
        prompt = {
            "type": "text",
            "text": "Name the colors of the first and second images in order. Reply with just the two English color names.",
        }
        for first, second, stream in [("red", "blue", False), ("blue", "red", True)]:
            with self.subTest(first=first, stream=stream):
                req = urllib.request.Request(
                    f"http://127.0.0.1:{int(os.environ['QWEN38_TEST_PORT'])}/v1/chat/completions",
                    json.dumps(
                        {
                            "model": "Qwen3.8-27B-8bit",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        prompt,
                                        picture(first),
                                        picture(second),
                                    ],
                                }
                            ],
                            "temperature": 0,
                            "max_tokens": 32,
                            "stream": stream,
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                    ).encode(),
                    {"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=180) as response:
                    if stream:
                        lines = [
                            line[6:].strip()
                            for line in response
                            if line.startswith(b"data: ")
                        ]
                        self.assertEqual(lines.pop(), b"[DONE]")
                        chunks = [json.loads(line) for line in lines]
                        answer = "".join(
                            c["choices"][0]["delta"].get("content", "") for c in chunks
                        )
                    else:
                        answer = json.load(response)["choices"][0]["message"]["content"]
                self.assertRegex(answer.lower(), rf"\b{first}\b.*\b{second}\b")
