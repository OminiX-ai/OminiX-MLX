import base64
import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace

from PIL import Image

from server import Handler, LocalServer, prepare_messages


def picture(color):
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}}


class Probe:
    name = "test-model"

    def generate(self, messages, images, max_tokens, temperature, top_p, thinking):
        for text in (["check", "</think>"] if thinking else []) + [
            str([image.getpixel((0, 0)) for image in images]) if images else "text"
        ]:
            yield SimpleNamespace(text=text, prompt_tokens=8, generation_tokens=4,
                                  total_tokens=12, finish_reason="stop")


class Protocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = LocalServer(("127.0.0.1", 0), Handler)
        cls.server.engine = Probe()
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1/"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def post(self, content, **kwargs):
        body = {"model": Probe.name, "messages": [{"role": "user", "content": content}], **kwargs}
        req = urllib.request.Request(self.url + "chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read()

    def test_text_and_image_order(self):
        with urllib.request.urlopen(self.url + "models") as r:
            self.assertEqual(json.load(r)["data"][0]["id"], Probe.name)
        self.assertEqual(json.loads(self.post("hello"))["choices"][0]["message"]["content"], "text")
        red, blue = picture("red"), picture("blue")
        first = json.loads(self.post([red, blue]))["choices"][0]["message"]
        self.assertEqual(first["content"], "[(255, 0, 0), (0, 0, 255)]")
        self.assertEqual(first["reasoning_content"], "check")
        reverse = json.loads(self.post([blue, red]))["choices"][0]["message"]
        self.assertEqual(reverse["content"], "[(0, 0, 255), (255, 0, 0)]")

    def test_interleaved_history(self):
        t = {"type": "text", "text": "here"}
        chat, images = prepare_messages([{"role": "user", "content": [t, picture("red"), t]},
                                         {"role": "assistant", "content": "ok"},
                                         {"role": "user", "content": [picture("blue"), t]}])
        self.assertEqual(chat[0]["content"], [t, {"type": "image"}, t])
        self.assertEqual(chat[2]["content"], [{"type": "image"}, t])
        self.assertEqual([img.getpixel((0, 0)) for img in images], [(255, 0, 0), (0, 0, 255)])

    def test_stream_completion_and_usage(self):
        data = self.post([picture("red")], stream=True, stream_options={"include_usage": True},
                         chat_template_kwargs={"enable_thinking": False})
        rows = [line[6:] for line in data.splitlines() if line.startswith(b"data: ")]
        self.assertEqual(rows.pop(), b"[DONE]")
        chunks = [json.loads(row) for row in rows]
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "[(255, 0, 0)]"})
        self.assertEqual(chunks[-2]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(chunks[-1]["usage"]["total_tokens"], 12)

    def test_invalid_input_returns_client_error(self):
        for url in ("file:///missing", "https://example.invalid/image.png", "data:image/png;base64,AAAA"):
            with self.subTest(url=url), self.assertRaises(urllib.error.HTTPError) as ctx:
                self.post([{"type": "image_url", "image_url": {"url": url}}], stream=True)
            self.assertEqual(ctx.exception.code, 400)
            ctx.exception.close()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post("hello", model="different")
        self.assertEqual(ctx.exception.code, 404)
        ctx.exception.close()


if __name__ == "__main__":
    unittest.main()
