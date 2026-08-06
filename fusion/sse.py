"""Server-sent-event plumbing for OpenAI-style streaming responses.

The orchestrator does not merely relay upstream bytes - when it chains two
models it has to splice their output into one well-formed stream. That means
parsing frames on the way in and minting them on the way out.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

DONE = b"data: [DONE]\n\n"


def new_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def encode(payload: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def delta_chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any],
    *,
    created: int | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }


def text_chunk(completion_id: str, model: str, text: str, **kwargs) -> bytes:
    return encode(delta_chunk(completion_id, model, {"content": text}, **kwargs))


def role_chunk(completion_id: str, model: str, **kwargs) -> bytes:
    return encode(delta_chunk(completion_id, model, {"role": "assistant"}, **kwargs))


def finish_chunk(
    completion_id: str, model: str, reason: str = "stop", **kwargs
) -> bytes:
    return encode(
        delta_chunk(completion_id, model, {}, finish_reason=reason, **kwargs)
    )


def error_chunk(completion_id: str, model: str, message: str) -> bytes:
    """Surface a mid-stream failure inside the stream.

    Once bytes are on the wire the status line is long gone, so a failure can
    only be reported as content. Better a visible message than a silent
    truncation the client mistakes for a complete answer.
    """
    return encode(
        delta_chunk(
            completion_id,
            model,
            {"content": f"\n\n[fusion] upstream error: {message}"},
            finish_reason="stop",
        )
    )


async def iter_data(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield the payload of each `data:` frame, skipping the terminal [DONE]."""
    async for line in lines:
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        yield payload


def parse_chunk(payload: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def chunk_text(chunk: dict[str, Any]) -> str:
    """Extract assistant text from a streaming chunk, ignoring reasoning traces.

    Several NIM models emit `reasoning_content` alongside `content`. That is
    thinking, not answer - it must not be counted as the plan or relayed as if
    it were the model's output.
    """
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str):
            return content
    return ""


def chunk_finish_reason(chunk: dict[str, Any]) -> str | None:
    for choice in chunk.get("choices") or []:
        reason = choice.get("finish_reason")
        if reason:
            return reason
    return None


def message_text(response: dict[str, Any]) -> str:
    for choice in response.get("choices") or []:
        content = (choice.get("message") or {}).get("content")
        if isinstance(content, str):
            return content
    return ""
