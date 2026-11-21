"""Safety and payload tests for SillyTavern quietImage transport."""
import asyncio
import base64
import io
import unittest
from PIL import Image

from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_vision_tests", ROOT / "image_context.py")
vision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vision)


def example_image(fmt="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (8, 4), (40, 80, 120)).save(output, format=fmt)
    return output.getvalue()


def data_url(payload, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(payload).decode("ascii")


class ImageInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_quiet_image_data_url(self):
        payload = example_image()
        source = data_url(payload)
        result = await vision.resolve_images([source])
        self.assertEqual(result, [payload])
        self.assertEqual(vision.parse_image_url({
            "type": "image_url", "image_url": {"url": source, "detail": "high"}
        }), source)

    async def test_png_jpeg_webp_image_format_validation(self):
        for fmt, mime in (("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
            with self.subTest(fmt=fmt):
                payload = example_image(fmt)
                self.assertEqual(await vision.resolve_images([data_url(payload, mime)]),
                                 [payload])

    async def test_image_order(self):
        p1 = example_image("PNG")
        p2 = example_image("JPEG")
        self.assertEqual(await vision.resolve_images([
            data_url(p1), data_url(p2, "image/jpeg"),
        ]), [p1, p2])

    async def test_size_and_mime_validation(self):
        for source in (
            "data:text/plain;base64,YWJj",
            "data:image/png,hello",
            "data:image/png;base64,not-base64!",
            "data:image/png;base64," + base64.b64encode(example_image("JPEG")).decode("ascii"),
            "data:image/png;base64," + base64.b64encode(b"broken").decode("ascii"),
        ):
            with self.subTest(source=source[:50]), self.assertRaises(vision.ImageContextError):
                await vision.resolve_images([source])

    async def test_too_many_images_are_rejected(self):
        source = data_url(example_image())
        with self.assertRaisesRegex(vision.ImageContextError, "Maximum eight"):
            await vision.resolve_images([source] * 9)

    async def test_ssrf_private_and_local_urls_rejected(self):
        for url in (
            "http://example.com/first.png", "http://localhost:8188/view",
            "https://localhost/image.png", "https://127.0.0.1/image.png",
            "https://192.168.1.100/picture", "https://169.254.169.254/latest/meta-data",
            "https://10.0.0.2/x.png", "https://[::1]/file", "https://public.example:9443/file",
            "https://alice:secret@example.org/private.png",
        ):
            with self.subTest(url=url), self.assertRaises(vision.ImageContextError):
                vision._check_https_url(url)
        self.assertEqual(vision._check_https_url("https://example.com/image.jpg"),
                         "https://example.com/image.jpg")

    async def test_reject_animated_first_frame(self):
        buf = io.BytesIO()
        image = Image.new("RGB", (4, 4), "red")
        image.save(buf, format="WEBP", save_all=True,
                   append_images=[Image.new("RGB", (4, 4), "blue")], duration=120)
        # Pillow builds without animated WebP can collapse frames, so the test
        # asserts the protection for multi-frame encodings when supported.
        with Image.open(io.BytesIO(buf.getvalue())) as check:
            if getattr(check, "n_frames", 1) > 1:
                with self.assertRaisesRegex(vision.ImageContextError, "Animated"):
                    await vision.resolve_images([data_url(buf.getvalue(), "image/webp")])


if __name__ == "__main__":
    unittest.main()
