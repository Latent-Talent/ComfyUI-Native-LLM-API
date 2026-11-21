"""Bounded private vision payloads for OpenAI/SillyTavern chat messages.

Images are decoded from data URLs (the SillyTavern quietImage path) or securely
downloaded from public HTTPS URLs. Raw bytes and image tensors never enter a
ComfyUI prompt graph, outputs or history.
"""
import asyncio
import base64
import binascii
import ipaddress
import io
import socket
from urllib.parse import urlsplit

import aiohttp

MAX_IMAGES = 8
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_TOTAL_PIXELS = 16_000_000
MAX_REQUEST_BYTES = 48 * 1024 * 1024
ACCEPTED_TYPES = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}

class ImageContextError(ValueError):
    pass


def parse_image_url(item):
    """Accept standard OpenAI image_url, used by SillyTavern's quietImage."""
    if not isinstance(item, dict):
        raise ImageContextError("Image part must be an object")
    kind = item.get("type")
    if kind not in ("image_url", "input_image"):
        raise ImageContextError("Unsupported image content part")
    if kind == "image_url":
        value = item.get("image_url")
        if isinstance(value, dict):
            value = value.get("url")
    else:
        value = item.get("image_url") or item.get("image")
    if not isinstance(value, str) or not value:
        raise ImageContextError("Image part must contain an image URL or base64 data URL")
    return value


def _image_from_data_url(value):
    header, sep, encoded = value.partition(",")
    if not sep or not header.startswith("data:"):
        raise ImageContextError("Invalid image data URL")
    mime = header[5:].lower()
    if not mime.endswith(";base64"):
        raise ImageContextError("Image data URL must use base64 encoding")
    mime = mime[:-7]
    if mime not in ACCEPTED_TYPES:
        raise ImageContextError("Only JPEG, PNG and WebP images are supported")
    if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4 + 10:
        raise ImageContextError("Encoded image exceeds 12 MiB limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ImageContextError("Invalid base64 image payload") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageContextError("Image exceeds 12 MiB limit")
    return data, mime


def _check_https_url(value):
    try:
        parts = urlsplit(value)
        hostname = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise ImageContextError("Malformed image URL") from exc
    if parts.scheme.lower() != "https" or not hostname or port not in (None, 443):
        raise ImageContextError("Remote image URLs must use HTTPS on port 443")
    if parts.username or parts.password or parts.fragment:
        raise ImageContextError("Image URL must not contain credentials or fragments")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        if len(hostname) > 253 or hostname.lower() in ("localhost",) or "." not in hostname:
            raise ImageContextError("Image URL must use a public DNS name")
    else:
        if not address.is_global:
            raise ImageContextError("Private or local image URL addresses are not allowed")
    return value


class _PublicDNSResolver(aiohttp.abc.AbstractResolver):
    """Resolve a public address once and pin the address for the TCP connection."""
    async def resolve(self, host, port=0, family=socket.AF_INET):
        entries = await asyncio.get_running_loop().getaddrinfo(
            host, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
        )
        if not entries:
            raise ImageContextError("Image URL host has no addresses")
        answers = []
        for af, kind, proto, canon, addr in entries:
            ip = ipaddress.ip_address(addr[0])
            if not ip.is_global:
                raise ImageContextError("Image URL resolves to a private or local address")
            answers.append({
                "hostname": host, "host": addr[0], "port": port,
                "family": af, "proto": proto, "flags": 0,
            })
        return answers

    async def close(self):
        return None


async def _fetch_image_url(url):
    _check_https_url(url)
    timeout = aiohttp.ClientTimeout(total=20, connect=10)
    connector = aiohttp.TCPConnector(
        resolver=_PublicDNSResolver(), use_dns_cache=False,
        limit=2, force_close=True
    )
    async with aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=False) as client:
        try:
            async with client.get(url, allow_redirects=False) as response:
                if response.status != 200:
                    raise ImageContextError("Remote image did not return HTTP 200")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower().strip()
                if content_type not in ACCEPTED_TYPES:
                    raise ImageContextError("Remote URL did not return a supported image")
                if response.content_length and response.content_length > MAX_IMAGE_BYTES:
                    raise ImageContextError("Remote image exceeds 12 MiB limit")
                data = await response.content.read(MAX_IMAGE_BYTES + 1)
                if len(data) > MAX_IMAGE_BYTES:
                    raise ImageContextError("Remote image exceeds 12 MiB limit")
                return data, content_type
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise ImageContextError("Unable to retrieve remote image securely") from exc


def _inspect_image(data, mime):
    """Validate format and decompression size without loading a full tensor."""
    from PIL import Image, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != ACCEPTED_TYPES[mime]:
                raise ImageContextError("Image format does not match declared MIME type")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_TOTAL_PIXELS:
                raise ImageContextError("Image dimensions exceed 16 megapixels")
            if getattr(image, "n_frames", 1) > 1:
                raise ImageContextError("Animated images are unsupported; supply a still first frame")
            image.verify()
            return width * height
    except (OSError, UnidentifiedImageError, ValueError, SyntaxError) as exc:
        raise ImageContextError("Invalid or corrupted image data") from exc


async def resolve_images(image_urls):
    """Return private immutable compressed image data, preserving content-part order."""
    if not image_urls:
        return []
    if len(image_urls) > MAX_IMAGES:
        raise ImageContextError("Maximum eight images per request")
    data = []
    total_bytes = 0
    total_pixels = 0
    for source in image_urls:
        if source.startswith("data:"):
            payload, mime = _image_from_data_url(source)
        else:
            payload, mime = await _fetch_image_url(source)
        total_bytes += len(payload)
        if total_bytes > MAX_TOTAL_BYTES:
            raise ImageContextError("Images exceed 32 MiB combined limit")
        total_pixels += await asyncio.to_thread(_inspect_image, payload, mime)
        if total_pixels > MAX_TOTAL_PIXELS:
            raise ImageContextError("Images exceed 16 megapixels combined limit")
        data.append(payload)
    return data


def image_tensors(image_bytes):
    """Decode ComfyUI IMAGE tensors [B,H,W,C] on the native execution worker."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    tensors = {}
    for index, payload in enumerate(image_bytes, start=1):
        with Image.open(io.BytesIO(payload)) as raw:
            rgb = ImageOps.exif_transpose(raw).convert("RGB")
            pixels = np.asarray(rgb, dtype=np.uint8).copy()
        tensors[f"image{index}"] = torch.from_numpy(pixels).unsqueeze(0).float().mul_(1.0 / 255.0)
    return tensors
