"""The SillyTavern quietImage payload may arrive in arbitrarily small HTTP chunks."""
import asyncio
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_test_http_json", ROOT / "http_json.py")
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)


class FakeChunkedReader:
    def __init__(self, raw, chunk_sizes=(13, 3, 4096, 1, 8192)):
        self.raw = raw
        self.offset = 0
        self.sizes = iter(chunk_sizes)
        self.calls = 0

    async def read(self, n):
        self.calls += 1
        if self.offset == len(self.raw):
            return b""
        # aiohttp read(n) may return MUCH fewer than n bytes.
        try:
            chunk_size = next(self.sizes)
        except StopIteration:
            chunk_size = 137
        end = min(len(self.raw), self.offset + min(n, chunk_size))
        value = self.raw[self.offset:end]
        self.offset = end
        return value


class TestHttpJSON(unittest.IsolatedAsyncioTestCase):
    async def test_large_multimodal_payload_split_across_network_chunks(self):
        encoded = "QUJD" * 200000
        data = {
            "model": "qwen3.8_27b_uncensored_w4a8_convrotMTP.safetensors",
            "stream": False,
            "messages": [{"role": "system", "content": [
                {"type": "text", "text": "Keep the opening frame exact."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + encoded, "detail": "high",
                }},
            ]}],
        }
        raw = json.dumps(data).encode("utf8")
        stream = FakeChunkedReader(raw)
        result = await parser.read_json_body(stream, 48 * 1024 * 1024)
        self.assertEqual(result, data)
        self.assertGreater(stream.calls, 30)

    async def test_multipart_utf8_and_escaped_json(self):
        message = "Café 🌍 ' a quote: \\\" and \n new line"
        raw = json.dumps({"messages": [{"role": "user", "content": message}]}).encode()
        got = await parser.read_json_body(FakeChunkedReader(raw, chunk_sizes=[1]*1000), 1000)
        self.assertEqual(got["messages"][0]["content"], message)

    async def test_size_limit_checked_across_chunks(self):
        raw = b'{"data":"' + b"A" * 300 + b'"}'
        with self.assertRaises(parser.RequestTooLarge):
            await parser.read_json_body(FakeChunkedReader(raw), 100)

    async def test_broken_json_is_rejected_only_after_full_read(self):
        raw = b'{"messages": ["cut off"'
        reader = FakeChunkedReader(raw, chunk_sizes=[5, 5])
        with self.assertRaises(parser.InvalidRequestJSON):
            await parser.read_json_body(reader, 100)
        self.assertEqual(reader.offset, len(raw))

    async def test_exact_limit_succeeds(self):
        raw = b'{"ok":true}'
        self.assertEqual(await parser.read_json_body(
            FakeChunkedReader(raw, chunk_sizes=[1]*40), len(raw)), {"ok": True})


if __name__ == "__main__":
    unittest.main()
