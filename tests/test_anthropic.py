"""Anthropic Messages translation, in both directions.

Claude Code will not tolerate an approximately-correct stream: every content
block has to be opened and closed, and a delta for a block that was never
started is a protocol error, not a cosmetic one. These tests pin the shape.
"""

from __future__ import annotations

import json

from fusion import anthropic


def sse_bytes(*chunks: dict) -> list[bytes]:
    out = [f"data: {json.dumps(c)}\n\n".encode() for c in chunks]
    out.append(b"data: [DONE]\n\n")
    return out


async def as_stream(chunks: list[bytes]):
    for chunk in chunks:
        yield chunk


async def translate(chunks: list[bytes], model: str = "fusion/auto") -> list[dict]:
    """Collect the translated stream as parsed events, in order."""
    raw = b"".join(
        [c async for c in anthropic.translate_stream(as_stream(chunks), model)]
    ).decode()
    events = []
    for block in raw.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def delta_chunk(**delta) -> dict:
    return {"id": "chatcmpl-abc", "model": "up",
            "choices": [{"index": 0, "delta": delta}]}


def finish_chunk(reason: str = "stop") -> dict:
    return {"id": "chatcmpl-abc", "model": "up",
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


class TestRequestTranslation:
    def test_system_becomes_a_message(self):
        out = anthropic.to_openai_request(
            {"system": "be brief", "messages": [{"role": "user", "content": "hi"}],
             "max_tokens": 100}
        )

        assert out["messages"][0] == {"role": "system", "content": "be brief"}
        assert out["messages"][1] == {"role": "user", "content": "hi"}

    def test_system_as_blocks_is_flattened(self):
        out = anthropic.to_openai_request(
            {"system": [{"type": "text", "text": "one"},
                        {"type": "text", "text": "two"}],
             "messages": [{"role": "user", "content": "hi"}]}
        )

        assert out["messages"][0]["content"] == "one\n\ntwo"

    def test_max_tokens_is_defaulted_when_absent(self):
        """Anthropic makes it mandatory, so a client may rely on us having one."""
        out = anthropic.to_openai_request({"messages": [{"role": "user", "content": "hi"}]})

        assert out["max_tokens"] == anthropic.DEFAULT_MAX_TOKENS

    def test_tools_are_reshaped(self):
        out = anthropic.to_openai_request({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "edit", "description": "edit a file",
                       "input_schema": {"type": "object",
                                        "properties": {"path": {"type": "string"}}}}],
        })

        assert out["tools"] == [{
            "type": "function",
            "function": {"name": "edit", "description": "edit a file",
                         "parameters": {"type": "object",
                                        "properties": {"path": {"type": "string"}}}},
        }]

    def test_tool_choice_any_means_required(self):
        out = anthropic.to_openai_request({
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "any"},
        })

        assert out["tool_choice"] == "required"

    def test_a_named_tool_choice_is_carried_over(self):
        out = anthropic.to_openai_request({
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "edit"},
        })

        assert out["tool_choice"] == {"type": "function",
                                      "function": {"name": "edit"}}

    def test_assistant_tool_use_becomes_tool_calls(self):
        out = anthropic.to_openai_request({"messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "read",
                 "input": {"path": "a.py"}}]},
        ]})

        message = out["messages"][0]
        assert message["role"] == "assistant"
        assert message["tool_calls"][0]["id"] == "toolu_1"
        assert message["tool_calls"][0]["function"]["name"] == "read"
        assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {
            "path": "a.py"
        }

    def test_tool_results_become_their_own_messages_and_come_first(self):
        """Anthropic nests results in the *user* turn; OpenAI wants separate
        `tool` messages, and putting new user text ahead of them would reorder
        the conversation the model sees."""
        out = anthropic.to_openai_request({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": "file contents"},
                {"type": "text", "text": "now fix it"},
            ]},
        ]})

        assert out["messages"][0] == {"role": "tool", "tool_call_id": "toolu_1",
                                      "content": "file contents"}
        assert out["messages"][1] == {"role": "user", "content": "now fix it"}

    def test_tool_result_blocks_are_flattened_to_text(self):
        out = anthropic.to_openai_request({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": "line one"},
                             {"type": "text", "text": "line two"}]}]},
        ]})

        assert out["messages"][0]["content"] == "line one\nline two"

    def test_a_failed_tool_says_so(self):
        """`is_error` is how a tool reports failure. Dropping it leaves the
        model reading a stack trace as if it were a successful result, so it
        cheerfully continues instead of recovering - the exact opposite of what
        an agentic loop needs."""
        out = anthropic.to_openai_request({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                 "content": "ENOENT: no such file"}]},
        ]})

        content = out["messages"][0]["content"]
        assert "ENOENT: no such file" in content
        assert "error" in content.lower()

    def test_an_empty_tool_result_is_still_visible(self):
        """A silent success (`rm`, a no-match grep) must not become an empty
        message the model may skip over entirely."""
        out = anthropic.to_openai_request({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": ""}]},
        ]})

        assert out["messages"][0]["content"].strip() != ""

    def test_an_image_returned_by_a_tool_reaches_the_model(self):
        """Screenshots come back *inside* tool results, but OpenAI tool messages
        are text-only. Carry the image on as a user part rather than dropping
        it, or a vision request round-trips blind."""
        out = anthropic.to_openai_request({"messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [
                    {"type": "text", "text": "screenshot taken"},
                    {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/png",
                                                 "data": "QUJD"}}]}]},
        ]})

        assert out["messages"][0]["role"] == "tool"
        images = [
            part
            for message in out["messages"]
            for part in (message.get("content") or [])
            if isinstance(part, dict) and part.get("type") == "image_url"
        ]
        assert len(images) == 1
        assert images[0]["image_url"]["url"] == "data:image/png;base64,QUJD"

    def test_images_become_data_uris(self):
        out = anthropic.to_openai_request({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image", "source": {"type": "base64",
                                             "media_type": "image/png",
                                             "data": "QUJD"}}]},
        ]})

        parts = out["messages"][0]["content"]
        assert parts[0] == {"type": "text", "text": "what is this?"}
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,QUJD"

    def test_stop_sequences_are_renamed(self):
        out = anthropic.to_openai_request({
            "messages": [{"role": "user", "content": "hi"}],
            "stop_sequences": ["END"],
        })

        assert out["stop"] == ["END"]


class TestResponseTranslation:
    def test_text_becomes_a_content_block(self):
        out = anthropic.to_anthropic_response({
            "id": "chatcmpl-1", "choices": [
                {"message": {"role": "assistant", "content": "hello"},
                 "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }, "fusion/auto")

        assert out["type"] == "message"
        assert out["role"] == "assistant"
        assert out["content"] == [{"type": "text", "text": "hello"}]
        assert out["stop_reason"] == "end_turn"
        assert out["usage"] == {"input_tokens": 7, "output_tokens": 3}
        assert out["id"].startswith("msg_")

    def test_tool_calls_become_tool_use_blocks(self):
        out = anthropic.to_anthropic_response({
            "choices": [{"message": {"role": "assistant", "content": None,
                                     "tool_calls": [{"id": "call_1", "type": "function",
                                                     "function": {"name": "edit",
                                                                  "arguments": '{"path":"a.py"}'}}]},
                         "finish_reason": "tool_calls"}],
        }, "fusion/code")

        assert out["content"] == [{"type": "tool_use", "id": "call_1",
                                   "name": "edit", "input": {"path": "a.py"}}]
        assert out["stop_reason"] == "tool_use"

    def test_length_maps_to_max_tokens(self):
        out = anthropic.to_anthropic_response(
            {"choices": [{"message": {"content": "cut"}, "finish_reason": "length"}]},
            "fusion/auto")

        assert out["stop_reason"] == "max_tokens"

    def test_malformed_tool_arguments_do_not_break_the_response(self):
        """A model that emits broken JSON should degrade, not 500."""
        out = anthropic.to_anthropic_response({
            "choices": [{"message": {"tool_calls": [
                {"id": "c1", "function": {"name": "edit", "arguments": "{oops"}}]},
                "finish_reason": "tool_calls"}]}, "fusion/code")

        assert out["content"][0]["input"] == {"__raw_arguments": "{oops"}

    def test_an_empty_answer_still_has_a_content_block(self):
        """Anthropic clients index into content[0]; an empty list crashes them."""
        out = anthropic.to_anthropic_response(
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            "fusion/auto")

        assert out["content"] == [{"type": "text", "text": ""}]


class TestStreamTranslation:
    async def test_text_is_wrapped_in_an_opened_and_closed_block(self):
        events = await translate(sse_bytes(
            delta_chunk(role="assistant"),
            delta_chunk(content="Hel"),
            delta_chunk(content="lo"),
            finish_chunk("stop"),
        ))
        kinds = [e["type"] for e in events]

        assert kinds == [
            "message_start", "content_block_start",
            "content_block_delta", "content_block_delta",
            "content_block_stop", "message_delta", "message_stop",
        ]
        text = "".join(e["delta"]["text"] for e in events
                       if e["type"] == "content_block_delta")
        assert text == "Hello"

    async def test_stop_reason_is_reported_on_message_delta(self):
        events = await translate(sse_bytes(
            delta_chunk(content="hi"), finish_chunk("length")))

        message_delta = [e for e in events if e["type"] == "message_delta"][0]
        assert message_delta["delta"]["stop_reason"] == "max_tokens"

    async def test_tool_calls_stream_as_input_json_deltas(self):
        events = await translate(sse_bytes(
            delta_chunk(content="I'll edit it."),
            delta_chunk(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": "edit", "arguments": ""}}]),
            delta_chunk(tool_calls=[{"index": 0,
                                     "function": {"arguments": '{"path":'}}]),
            delta_chunk(tool_calls=[{"index": 0,
                                     "function": {"arguments": '"a.py"}'}}]),
            finish_chunk("tool_calls"),
        ))
        kinds = [e["type"] for e in events]

        # The text block must be closed before the tool block opens.
        assert kinds.index("content_block_stop") < kinds.index("content_block_start", 2)

        starts = [e for e in events if e["type"] == "content_block_start"]
        assert starts[1]["content_block"] == {"type": "tool_use", "id": "call_1",
                                              "name": "edit", "input": {}}
        partial = "".join(e["delta"]["partial_json"] for e in events
                          if e["type"] == "content_block_delta"
                          and e["delta"]["type"] == "input_json_delta")
        assert json.loads(partial) == {"path": "a.py"}
        assert [e for e in events if e["type"] == "message_delta"][0][
            "delta"]["stop_reason"] == "tool_use"

    async def test_two_parallel_tool_calls_become_two_blocks(self):
        """Claude Code batches reads constantly, so this is the common case."""
        events = await translate(sse_bytes(
            delta_chunk(tool_calls=[{"index": 0, "id": "c0", "type": "function",
                                     "function": {"name": "read",
                                                  "arguments": '{"path":"a.py"}'}}]),
            delta_chunk(tool_calls=[{"index": 1, "id": "c1", "type": "function",
                                     "function": {"name": "read",
                                                  "arguments": '{"path":"b.py"}'}}]),
            finish_chunk("tool_calls"),
        ))

        starts = [e for e in events if e["type"] == "content_block_start"]
        assert [s["index"] for s in starts] == [0, 1]
        assert [s["content_block"]["id"] for s in starts] == ["c0", "c1"]
        assert len([e for e in events if e["type"] == "content_block_stop"]) == 2

    async def test_arguments_split_across_interleaved_slots(self):
        """A provider may chunk arguments and alternate between tool indices.
        Anthropic allows only one open block at a time, so each tool still has
        to arrive as exactly one block carrying exactly its own arguments."""
        events = await translate(sse_bytes(
            delta_chunk(tool_calls=[{"index": 0, "id": "c0", "type": "function",
                                     "function": {"name": "read", "arguments": '{"pa'}}]),
            delta_chunk(tool_calls=[{"index": 1, "id": "c1", "type": "function",
                                     "function": {"name": "grep", "arguments": '{"q"'}}]),
            delta_chunk(tool_calls=[{"index": 0,
                                     "function": {"arguments": 'th":"a.py"}'}}]),
            delta_chunk(tool_calls=[{"index": 1,
                                     "function": {"arguments": ':"todo"}'}}]),
            finish_chunk("tool_calls"),
        ))

        starts = [e for e in events if e["type"] == "content_block_start"]
        assert len(starts) == 2, "each tool must be exactly one block"
        assert [s["content_block"]["name"] for s in starts] == ["read", "grep"]

        by_index: dict[int, str] = {}
        for event in events:
            if (event["type"] == "content_block_delta"
                    and event["delta"]["type"] == "input_json_delta"):
                by_index[event["index"]] = (
                    by_index.get(event["index"], "") + event["delta"]["partial_json"]
                )
        assert json.loads(by_index[0]) == {"path": "a.py"}
        assert json.loads(by_index[1]) == {"q": "todo"}

    async def test_every_opened_block_is_closed(self):
        events = await translate(sse_bytes(
            delta_chunk(content="text"),
            delta_chunk(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "edit",
                                                  "arguments": "{}"}}]),
            finish_chunk("tool_calls"),
        ))

        opened = len([e for e in events if e["type"] == "content_block_start"])
        closed = len([e for e in events if e["type"] == "content_block_stop"])
        assert opened == closed == 2

    async def test_indices_are_sequential(self):
        """A client tracks blocks by index; gaps or repeats corrupt the message."""
        events = await translate(sse_bytes(
            delta_chunk(content="a"),
            delta_chunk(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "x", "arguments": "{}"}}]),
            finish_chunk("tool_calls"),
        ))

        starts = [e["index"] for e in events if e["type"] == "content_block_start"]
        assert starts == [0, 1]

    async def test_an_empty_upstream_still_produces_a_valid_message(self):
        """Claude Code hangs on a stream that never terminates properly."""
        events = await translate([b"data: [DONE]\n\n"])

        assert [e["type"] for e in events] == [
            "message_start", "message_delta", "message_stop"]

    async def test_the_requested_model_name_is_echoed(self):
        events = await translate(sse_bytes(delta_chunk(content="hi"),
                                           finish_chunk()), model="claude-opus-4")

        assert events[0]["message"]["model"] == "claude-opus-4"


class TestTokenCounting:
    def test_it_counts_system_messages_and_tools(self):
        small = anthropic.estimate_tokens(
            {"messages": [{"role": "user", "content": "hi"}]})
        large = anthropic.estimate_tokens({
            "system": "x" * 4000,
            "messages": [{"role": "user", "content": "y" * 4000}],
            "tools": [{"name": "edit", "input_schema": {"type": "object"}}],
        })

        assert large > small
        assert large >= 2000

    def test_it_never_returns_zero(self):
        assert anthropic.estimate_tokens({"messages": []}) >= 1
