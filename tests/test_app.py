"""End-to-end: an OpenAI-compatible request in, the right specialist out."""

from __future__ import annotations

import json

import httpx
import pytest
import respx
import yaml

from fusion.app import create_app, verify_catalog
from fusion.config import Settings

from .conftest import make_config
from .test_orchestrator import UPSTREAM, json_response, stream_response

CATALOG = "https://nim.test/v1/models"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NIM_API_KEY_1", "key-a")
    monkeypatch.setenv("NIM_API_KEY_2", "key-b")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(make_config().model_dump()))
    return create_app(Settings(config_path=config_file))


@pytest.fixture
def client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fusion"
    )


def tool(name):
    return {"type": "function", "function": {"name": name, "parameters": {}}}


CODING_TOOLS = [tool("read"), tool("edit"), tool("write"), tool("bash")]


def sent_models(route) -> list[str]:
    return [json.loads(call.request.content)["model"] for call in route.calls]


class TestMetaEndpoints:
    async def test_healthz(self, client):
        async with client as http:
            response = await http.get("/healthz")

        assert response.status_code == 200
        assert response.json()["keys"] == 2

    async def test_models_advertises_every_lane(self, client):
        async with client as http:
            response = await http.get("/v1/models")

        ids = {entry["id"] for entry in response.json()["data"]}
        assert "fusion/auto" in ids
        assert "fusion/code" in ids
        assert "fusion/vision" in ids
        assert "deepseek" in ids  # real models remain addressable


class TestRequestValidation:
    async def test_rejects_a_body_without_messages(self, client):
        async with client as http:
            response = await http.post("/v1/chat/completions", json={"model": "auto"})

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"


class TestRouting:
    @respx.mock
    async def test_a_coding_request_reaches_deepseek_after_planning(self, client):
        route = respx.post(UPSTREAM).mock(
            side_effect=[stream_response("the plan"), stream_response("the code")]
        )

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/auto",
                    "stream": True,
                    "tools": CODING_TOOLS,
                    "messages": [{"role": "user", "content": "fix the parser bug"}],
                },
            )

        assert response.headers["x-fusion-role"] == "worker"
        assert sent_models(route) == ["glm", "deepseek"]
        assert "the plan" in response.text
        assert "the code" in response.text

    @respx.mock
    async def test_a_question_goes_straight_to_the_thinker(self, client):
        route = respx.post(UPSTREAM).mock(return_value=stream_response("because..."))

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/auto",
                    "stream": True,
                    "messages": [{"role": "user", "content": "why is the sky blue?"}],
                },
            )

        assert response.headers["x-fusion-role"] == "thinker"
        assert sent_models(route) == ["glm"]  # one call: no planning pass

    @respx.mock
    async def test_an_image_reaches_the_vision_model(self, client):
        route = respx.post(UPSTREAM).mock(return_value=stream_response("nice ui"))

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/auto",
                    "stream": True,
                    "tools": CODING_TOOLS,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "build this screen"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,AA"},
                                },
                            ],
                        }
                    ],
                },
            )

        assert response.headers["x-fusion-role"] == "eyes"
        assert sent_models(route) == ["minimax"]

    @respx.mock
    async def test_mid_tool_loop_never_replans(self, client):
        """The rule the agentic loop depends on, verified through the real app."""
        route = respx.post(UPSTREAM).mock(return_value=stream_response("continuing"))
        conversation = [
            {"role": "user", "content": "implement the retry logic"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "file body"},
        ]

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/auto",
                    "stream": True,
                    "tools": CODING_TOOLS,
                    "messages": conversation,
                },
            )

        assert len(route.calls) == 1  # no planning pass
        assert "Plan" not in response.text
        assert sent_models(route) == ["deepseek"]

    @respx.mock
    async def test_a_pinned_lane_overrides_the_heuristics(self, client):
        route = respx.post(UPSTREAM).mock(return_value=stream_response("answer"))

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/context",
                    "stream": True,
                    "messages": [{"role": "user", "content": "why is the sky blue?"}],
                },
            )

        assert response.headers["x-fusion-role"] == "librarian"
        assert sent_models(route) == ["kimi"]

    @respx.mock
    async def test_a_real_model_name_is_used_verbatim(self, client):
        route = respx.post(UPSTREAM).mock(return_value=stream_response("direct"))

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "kimi",
                    "stream": True,
                    "messages": [{"role": "user", "content": "fix the parser"}],
                },
            )

        assert response.headers["x-fusion-role"] == "direct"
        assert sent_models(route) == ["kimi"]

    @respx.mock
    async def test_an_unknown_model_falls_back_to_auto(self, client):
        respx.post(UPSTREAM).mock(return_value=stream_response("ok"))

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "some/unknown-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "why is the sky blue?"}],
                },
            )

        assert response.status_code == 200
        assert response.headers["x-fusion-role"] == "thinker"


class TestNonStreamingEndpoint:
    @respx.mock
    async def test_returns_a_completion_object(self, client):
        respx.post(UPSTREAM).mock(return_value=json_response("hello"))

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/auto",
                    "messages": [{"role": "user", "content": "why is the sky blue?"}],
                },
            )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "hello"

    @respx.mock
    async def test_upstream_failure_surfaces_as_an_error(self, client):
        respx.post(UPSTREAM).mock(return_value=httpx.Response(500, text="boom"))

        async with client as http:
            response = await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/auto",
                    "messages": [{"role": "user", "content": "why is the sky blue?"}],
                },
            )

        assert response.status_code >= 500
        assert response.json()["error"]["type"] == "upstream_error"


class TestStats:
    @respx.mock
    async def test_records_the_decision_and_the_signals(self, client):
        respx.post(UPSTREAM).mock(return_value=stream_response("hi"))

        async with client as http:
            await http.post(
                "/v1/chat/completions",
                json={
                    "model": "fusion/auto",
                    "stream": True,
                    "messages": [{"role": "user", "content": "why is the sky blue?"}],
                },
            )
            stats = (await http.get("/stats")).json()

        recent = stats["routing"]["recent"][0]
        assert recent["role"] == "thinker"
        assert recent["served_by"] == "glm"
        assert recent["status"] == "ok"
        assert stats["capacity"]["models"]["glm"]["succeeded"] == 1


class TestCatalogVerification:
    @respx.mock
    async def test_flags_models_missing_from_the_catalog(self, app):
        respx.get(CATALOG).mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": "glm"}, {"id": "kimi"}, {"id": "minimax"}]},
            )
        )
        runtime = app.state.runtime

        await verify_catalog(runtime)

        assert any("deepseek" in problem for problem in runtime.catalog_problems)

    @respx.mock
    async def test_silent_when_everything_matches(self, app):
        runtime = app.state.runtime
        respx.get(CATALOG).mock(
            return_value=httpx.Response(
                200,
                json={"data": [{"id": m} for m in runtime.config.all_models()]},
            )
        )

        await verify_catalog(runtime)

        assert runtime.catalog_problems == []

    @respx.mock
    async def test_an_unreachable_catalog_is_a_warning_not_a_crash(self, app):
        respx.get(CATALOG).mock(return_value=httpx.Response(401, text="bad key"))
        runtime = app.state.runtime

        await verify_catalog(runtime)

        assert runtime.catalog_problems
        assert "could not fetch" in runtime.catalog_problems[0]


class TestAnthropicEndpoint:
    """Claude Code's dialect, end to end through the real router."""

    @respx.mock
    async def test_non_streaming_returns_an_anthropic_message(self, client):
        respx.get(CATALOG).mock(return_value=httpx.Response(200, json={"data": []}))
        respx.post(UPSTREAM).mock(return_value=json_response("hello there"))

        async with client as http:
            response = await http.post("/v1/messages", json={
                "model": "claude-opus-4",
                "max_tokens": 100,
                "system": "be brief",
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert response.status_code == 200
        body = response.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["model"] == "claude-opus-4"   # echoed, not our upstream name
        assert body["content"] == [{"type": "text", "text": "hello there"}]
        assert body["stop_reason"] == "end_turn"

    @respx.mock
    async def test_streaming_emits_a_well_formed_event_sequence(self, client):
        respx.get(CATALOG).mock(return_value=httpx.Response(200, json={"data": []}))
        respx.post(UPSTREAM).mock(return_value=stream_response("stream ", "me"))

        async with client as http:
            response = await http.post("/v1/messages", json={
                "model": "claude-opus-4", "max_tokens": 100, "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            })
            text = response.text

        assert response.status_code == 200
        for name in ("message_start", "content_block_start", "content_block_delta",
                     "content_block_stop", "message_delta", "message_stop"):
            assert f"event: {name}" in text
        assert text.index("event: message_start") < text.index("event: message_stop")
        assert "stream " in text and "me" in text

    @respx.mock
    async def test_it_routes_like_any_other_request(self, client):
        """The point of translating rather than proxying: Claude Code gets the
        same coordinator, so a coding request still reaches the Worker."""
        respx.get(CATALOG).mock(return_value=httpx.Response(200, json={"data": []}))
        route = respx.post(UPSTREAM).mock(
            side_effect=[json_response("plan"), json_response("code")])

        async with client as http:
            response = await http.post("/v1/messages", json={
                "model": "claude-opus-4", "max_tokens": 100,
                "messages": [{"role": "user", "content": "implement the retry logic"}],
                "tools": [{"name": "edit", "description": "edit a file",
                           "input_schema": {"type": "object", "properties": {}}}],
            })

        assert response.status_code == 200
        assert response.headers["x-fusion-role"] == "worker"
        assert sent_models(route)[-1] == "deepseek"   # the Worker's primary

    @respx.mock
    async def test_a_pinned_lane_still_works(self, client):
        respx.get(CATALOG).mock(return_value=httpx.Response(200, json={"data": []}))
        route = respx.post(UPSTREAM).mock(return_value=json_response("ok"))

        async with client as http:
            response = await http.post("/v1/messages", json={
                "model": "fusion/think", "max_tokens": 50,
                "messages": [{"role": "user", "content": "hi"}],
            })

        assert response.headers["x-fusion-role"] == "thinker"
        assert sent_models(route)[0] == "glm"

    @respx.mock
    async def test_count_tokens_answers(self, client):
        respx.get(CATALOG).mock(return_value=httpx.Response(200, json={"data": []}))

        async with client as http:
            response = await http.post("/v1/messages/count_tokens", json={
                "model": "claude-opus-4",
                "messages": [{"role": "user", "content": "x" * 400}],
            })

        assert response.status_code == 200
        assert response.json()["input_tokens"] >= 100

    @respx.mock
    async def test_a_bad_request_uses_anthropic_error_shape(self, client):
        respx.get(CATALOG).mock(return_value=httpx.Response(200, json={"data": []}))

        async with client as http:
            response = await http.post("/v1/messages", json={"model": "x"})

        assert response.status_code == 400
        body = response.json()
        assert body["type"] == "error"           # Anthropic tags the envelope
        assert body["error"]["type"] == "invalid_request_error"
