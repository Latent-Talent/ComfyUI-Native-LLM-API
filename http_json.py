"""Bounded JSON body reads for large OpenAI multimodal requests.

aiohttp StreamReader.read(n) returns UP TO n bytes, not necessarily n bytes
or the entire request. A single call can return an incomplete SillyTavern
base64 image payload. Accumulate chunks until EOF before parsing JSON.
No body content, image data, or chat messages are logged or persisted.
"""
import json


class RequestTooLarge(ValueError):
    pass


class InvalidRequestJSON(ValueError):
    pass


async def read_json_body(reader, max_bytes):
    """Consume arbitrary-sized StreamReader chunks through EOF, with a hard cap."""
    chunks = []
    size = 0
    while True:
        # Never request or buffer more than one byte beyond the hard limit.
        part = await reader.read(min(65536, max_bytes - size + 1))
        if not part:
            break
        size += len(part)
        if size > max_bytes:
            raise RequestTooLarge("Request exceeds the configured body limit")
        chunks.append(part)
    try:
        return json.loads(b"".join(chunks))
    except (ValueError, TypeError, UnicodeDecodeError) as exc:
        raise InvalidRequestJSON("Invalid JSON request body") from exc
