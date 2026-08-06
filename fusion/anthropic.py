"""Anthropic Messages API <-> OpenAI chat completions.

Claude Code talks `/v1/messages`, not `/v1/chat/completions`. The two describe
the same conversation but disagree on nearly every detail: where the system
prompt lives, whether tool results are messages or content blocks, how images
are carried, and - most of all - what a stream looks like. OpenAI streams one
flat sequence of deltas; Anthropic streams *typed blocks* that must be opened
and closed in order.

Translating here rather than teaching the router two dialects keeps the
coordinator, governor and orchestrator working in exactly one shape.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Iterable

from . import sse

# Anthropic requires max_tokens; OpenAI treats it as optional. A request that
# arrives without one still has to go upstream with something sane.
DEFAULT_MAX_TOKENS = 4096

_STOP_REASONS = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


# -- request: Anthropic -> OpenAI -------------------------------------------


def _system_text(system: Any) -> str:
    """`system` is a string or a list of text blocks, depending on the client."""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "\n\n".join(
            block.get("text", "")
            for block in system
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _image_part(block: dict) -> dict | None:
    """Anthropic carries image bytes inline; OpenAI wants a URL or data: URI."""
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    if source.get("type") == "base64":
        media = source.get("media_type") or "image/png"
        data = source.get("data") or ""
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{data}"}}
    if source.get("type") == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result(block: dict) -> tuple[str, list[dict]]:
    """Flatten a tool result to text, plus any images it carried.

    OpenAI's `tool` messages are text-only, but Claude Code's tools return
    screenshots, so the images are handed back separately for the caller to
    re-attach rather than thrown away.
    """
    content = block.get("content")
    images: list[dict] = []

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "image":
                image = _image_part(item)
                if image:
                    images.append(image)
                    parts.append("[image returned by the tool, attached below]")
        text = "\n".join(parts)
    elif content is None:
        text = ""
    else:
        text = json.dumps(content)

    if block.get("is_error"):
        # Without this the model reads a stack trace as a successful result and
        # carries on, instead of recovering - which is the whole point of an
        # agentic loop being told a tool failed.
        text = f"[tool error] {text}" if text.strip() else "[tool error]"
    elif not text.strip():
        # A silent success still has to be visible; an empty message is easy
        # for a model to skip over entirely.
        text = "(no output)"

    return text, images


def _convert_message(message: dict) -> list[dict]:
    """One Anthropic message can become several OpenAI ones.

    A `user` turn carrying tool results is the reason: Anthropic nests them as
    blocks inside the user message, while OpenAI expects each to be its own
    `role: "tool"` message, before any user text.
    """
    role = message.get("role")
    content = message.get("content")

    if isinstance(content, str):
        return [{"role": role, "content": content}]
    if not isinstance(content, list):
        return []

    parts: list[dict] = []
    tool_calls: list[dict] = []
    tool_messages: list[dict] = []

    for block in content:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif kind == "image":
            part = _image_part(block)
            if part:
                parts.append(part)
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
        elif kind == "tool_result":
            text, images = _tool_result(block)
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id"),
                    "content": text,
                }
            )
            # Re-attached after the tool message, since `tool` cannot carry one.
            parts.extend(images)

    if role == "assistant":
        out: dict[str, Any] = {"role": "assistant"}
        text = "".join(p["text"] for p in parts if p.get("type") == "text")
        out["content"] = text or None
        if tool_calls:
            out["tool_calls"] = tool_calls
        return [out]

    # Tool results first: they answer the assistant's previous turn, so putting
    # new user text ahead of them would reorder the conversation.
    messages = list(tool_messages)
    if parts:
        only_text = all(p.get("type") == "text" for p in parts)
        messages.append(
            {
                "role": role,
                "content": "".join(p["text"] for p in parts) if only_text else parts,
            }
        )
    return messages


def _convert_tools(tools: Any) -> list[dict]:
    converted = []
    for tool in tools or []:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description") or "",
                    "parameters": tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def _convert_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def to_openai_request(body: dict) -> dict:
    """An Anthropic Messages request in the shape the rest of Fusion speaks."""
    messages: list[dict] = []
    system = _system_text(body.get("system"))
    if system.strip():
        messages.append({"role": "system", "content": system})

    for message in body.get("messages") or []:
        if isinstance(message, dict):
            messages.extend(_convert_message(message))

    converted: dict[str, Any] = {
        "model": body.get("model") or "auto",
        "messages": messages,
        "max_tokens": body.get("max_tokens") or DEFAULT_MAX_TOKENS,
        "stream": bool(body.get("stream")),
    }

    tools = _convert_tools(body.get("tools"))
    if tools:
        converted["tools"] = tools
    tool_choice = _convert_tool_choice(body.get("tool_choice"))
    if tool_choice is not None:
        converted["tool_choice"] = tool_choice

    for src, dst in (("temperature", "temperature"), ("top_p", "top_p")):
        if body.get(src) is not None:
            converted[dst] = body[src]
    if body.get("stop_sequences"):
        converted["stop"] = body["stop_sequences"]

    return converted


# -- response: OpenAI -> Anthropic ------------------------------------------


def _blocks_from_message(message: dict) -> list[dict]:
    blocks: list[dict] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        blocks.append({"type": "text", "text": text})
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            parsed = json.loads(raw)
        except ValueError:
            # A model that emitted malformed JSON should not take the whole
            # response down; hand the tool what it actually said.
            parsed = {"__raw_arguments": raw}
        blocks.append(
            {
                "type": "tool_use",
                "id": call.get("id") or f"toolu_{len(blocks)}",
                "name": function.get("name") or "",
                "input": parsed if isinstance(parsed, dict) else {"value": parsed},
            }
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def to_anthropic_response(response: dict, model: str) -> dict:
    choices = response.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    usage = response.get("usage") or {}

    return {
        "id": _message_id(response.get("id")),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": _blocks_from_message(message),
        "stop_reason": _STOP_REASONS.get(choice.get("finish_reason") or "stop", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _message_id(raw: Any) -> str:
    text = str(raw or "")
    if text.startswith("msg_"):
        return text
    return "msg_" + (text.replace("chatcmpl-", "") or "fusion")


# -- streaming: OpenAI deltas -> Anthropic typed blocks ----------------------


def _event(name: str, payload: dict) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode()


class _BlockWriter:
    """Opens and closes Anthropic content blocks around a flat delta stream.

    OpenAI says "here is more text" and "here are more tool arguments" with no
    structure around either. Anthropic requires every span to be announced with
    `content_block_start`, extended with deltas, and terminated with
    `content_block_stop` - and a client that receives an unterminated block, or
    a delta for a block that was never opened, will reject the stream.
    """

    def __init__(self) -> None:
        self.index = -1
        self.kind: str | None = None

    def open(self, kind: str, block: dict) -> list[bytes]:
        out = self.close()
        self.index += 1
        self.kind = kind
        out.append(
            _event(
                "content_block_start",
                {"type": "content_block_start", "index": self.index,
                 "content_block": block},
            )
        )
        return out

    def close(self) -> list[bytes]:
        if self.kind is None:
            return []
        self.kind = None
        return [
            _event("content_block_stop",
                   {"type": "content_block_stop", "index": self.index})
        ]


async def translate_stream(
    chunks: AsyncIterator[bytes], model: str
) -> AsyncIterator[bytes]:
    """Re-emit Fusion's OpenAI SSE output as an Anthropic message stream."""
    writer = _BlockWriter()
    message_id = "msg_fusion"
    started = False
    stop_reason = "end_turn"
    output_tokens = 0

    # Tool calls are accumulated rather than streamed straight through.
    # OpenAI may chunk one call's arguments across several deltas and alternate
    # between calls while doing it; Anthropic allows exactly one open block at a
    # time and no reopening. Streaming them naively therefore split a single
    # tool across several blocks - two of them nameless - the moment a second
    # call interleaved. Buffering costs nothing real: a client cannot invoke a
    # tool until its JSON is complete anyway.
    tools: dict[int, dict[str, Any]] = {}

    async for payload in sse.iter_data(_lines(chunks)):
        chunk = sse.parse_chunk(payload)
        if chunk is None:
            continue

        if not started:
            started = True
            message_id = _message_id(chunk.get("id"))
            yield _event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )

        for choice in chunk.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}

            text = delta.get("content")
            if isinstance(text, str) and text:
                if writer.kind != "text":
                    for event in writer.open("text", {"type": "text", "text": ""}):
                        yield event

                output_tokens += 1
                yield _event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": writer.index,
                     "delta": {"type": "text_delta", "text": text}},
                )

            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                slot = call.get("index", 0)
                function = call.get("function") or {}
                pending = tools.setdefault(
                    slot, {"id": None, "name": "", "arguments": ""}
                )
                if call.get("id"):
                    pending["id"] = call["id"]
                if function.get("name"):
                    pending["name"] = function["name"]
                if function.get("arguments"):
                    pending["arguments"] += function["arguments"]
                    output_tokens += 1

            finish = choice.get("finish_reason")
            if finish:
                stop_reason = _STOP_REASONS.get(finish, "end_turn")

    if not started:
        # Nothing arrived at all. Still owe the client a well-formed message.
        yield _event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message_id, "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    # Tool blocks last, each one whole: start, its own arguments, stop.
    for slot in sorted(tools):
        call = tools[slot]
        for event in writer.open(
            "tool",
            {
                "type": "tool_use",
                "id": call["id"] or f"toolu_{slot}",
                "name": call["name"],
                "input": {},
            },
        ):
            yield event
        if call["arguments"]:
            yield _event(
                "content_block_delta",
                {"type": "content_block_delta", "index": writer.index,
                 "delta": {"type": "input_json_delta",
                           "partial_json": call["arguments"]}},
            )

    for event in writer.close():
        yield event
    yield _event(
        "message_delta",
        {"type": "message_delta",
         "delta": {"stop_reason": stop_reason, "stop_sequence": None},
         "usage": {"output_tokens": output_tokens}},
    )
    yield _event("message_stop", {"type": "message_stop"})


async def _lines(chunks: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Re-split the orchestrator's encoded SSE back into lines.

    Encoding and immediately re-parsing looks wasteful, and is - but it means
    the Anthropic endpoint inherits hedging, failover and the plan-then-code
    splice exactly as the OpenAI one has them, rather than reimplementing any
    of it a second way.
    """
    buffer = ""
    async for raw in chunks:
        buffer += raw.decode("utf-8", "replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
    if buffer:
        yield buffer


def estimate_tokens(body: dict) -> int:
    """Rough count for /v1/messages/count_tokens - chars/4, same as routing."""
    total = len(_system_text(body.get("system")))
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    total += len(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    total += len(_tool_result(block)[0])
                elif block.get("type") == "tool_use":
                    total += len(json.dumps(block.get("input") or {}))
    total += len(json.dumps(body.get("tools") or []))
    return max(1, total // 4)


def iter_text(blocks: Iterable[dict]) -> str:
    return "".join(
        b.get("text", "") for b in blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )
