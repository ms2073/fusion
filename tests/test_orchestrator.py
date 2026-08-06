"""Orchestrator: failover before first byte, and two models spliced into one stream."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import respx

from fusion.coordinator import RouteDecision
from fusion.governor import Governor
from fusion.orchestrator import Orchestrator
from fusion.stats import Stats
from fusion.upstream import UpstreamClient, UpstreamError

from .conftest import make_config

UPSTREAM = "https://nim.test/v1/chat/completions"


def sse_body(*deltas: dict, finish: str | None = "stop") -> str:
    frames = [
        f"data: {json.dumps({'id': 'up', 'model': 'upstream', 'choices': [{'index': 0, 'delta': d}]})}\n\n"
        for d in deltas
    ]
    if finish:
        frames.append(
            "data: "
            + json.dumps(
                {
                    "id": "up",
                    "model": "upstream",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                }
            )
            + "\n\n"
        )
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def stream_response(*texts: str, finish: str | None = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        text=sse_body(*({"content": t} for t in texts), finish=finish),
        headers={"content-type": "text/event-stream"},
    )


def json_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "up",
            "object": "chat.completion",
            "model": "upstream",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


@pytest.fixture
def harness(keys):
    config = make_config()
    governor = Governor(config, keys)
    upstream = UpstreamClient(config)
    stats = Stats()
    orchestrator = Orchestrator(config, governor, upstream, stats)
    return config, governor, upstream, stats, orchestrator


def decision(role="worker", candidates=("deepseek", "glm", "kimi"), chain=False):
    return RouteDecision(
        role=role,
        candidates=candidates,
        chain=chain,
        signals=("test",),
        fingerprint="abc123",
    )


def new_record(stats: Stats, dec: RouteDecision):
    return stats.start(
        lane="auto",
        role=dec.role,
        planned=list(dec.candidates),
        signals=list(dec.signals),
        chain=dec.chain,
        fingerprint=dec.fingerprint,
    )


async def collect(orchestrator, body, dec, record) -> str:
    return b"".join([c async for c in orchestrator.stream(body, dec, record)]).decode()


BODY = {"messages": [{"role": "user", "content": "fix the parser"}], "stream": True}


class TestSingleShotStream:
    @respx.mock
    async def test_relays_content_and_terminates_once(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(return_value=stream_response("hello ", "world"))
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "hello " in out and "world" in out
        assert out.count("data: [DONE]") == 1

    @respx.mock
    async def test_rewrites_id_and_model_consistently(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(return_value=stream_response("hi"))
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        ids = {
            json.loads(line[6:])["id"]
            for line in out.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        }
        assert len(ids) == 1
        assert "upstream" not in out  # upstream's own model name never leaks

    @respx.mock
    async def test_preserves_tool_calls(self, harness):
        """opencode's whole loop depends on tool_calls surviving the relay."""
        _, _, upstream, stats, orch = harness
        tool_delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "edit", "arguments": '{"path":"a.py"}'},
                }
            ]
        }
        respx.post(UPSTREAM).mock(
            return_value=httpx.Response(
                200,
                text=sse_body(tool_delta, finish="tool_calls"),
                headers={"content-type": "text/event-stream"},
            )
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "call_1" in out
        assert '"name": "edit"' in out or '"name":"edit"' in out
        assert "tool_calls" in out

    @respx.mock
    async def test_synthesises_a_finish_frame_when_upstream_omits_one(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(return_value=stream_response("hi", finish=None))
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert '"finish_reason": "stop"' in out or '"finish_reason":"stop"' in out


class TestFailover:
    @respx.mock
    async def test_429_retries_the_same_model_on_the_other_key(self, harness):
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                httpx.Response(429, json={"error": "rate limited"}),
                stream_response("recovered"),
            ]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "recovered" in out
        assert record.attempts == ["deepseek#key0", "deepseek#key1"]
        assert record.served_by == "deepseek"

    @respx.mock
    async def test_falls_back_to_another_model_when_both_keys_are_limited(
        self, harness
    ):
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                httpx.Response(429),
                httpx.Response(429),
                stream_response("from glm"),
            ]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "from glm" in out
        assert record.served_by == "glm"
        assert governor.snapshot()["failovers"]["deepseek->glm"] == 1

    @respx.mock
    async def test_honours_retry_after(self, harness):
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "30"}),
                stream_response("ok"),
            ]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        cooling = governor.snapshot()["cooling_down"]
        assert any(c["seconds_remaining"] > 25 for c in cooling)

    @respx.mock
    async def test_a_bad_request_is_not_retried_across_models(self, harness):
        """A malformed request fails identically everywhere; retrying wastes quota."""
        _, _, upstream, stats, orch = harness
        route = respx.post(UPSTREAM).mock(
            return_value=httpx.Response(400, json={"error": "bad schema"})
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert route.call_count == 1
        assert "upstream error" in out
        assert out.count("data: [DONE]") == 1

    @respx.mock
    async def test_server_errors_do_fail_over(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[httpx.Response(503), stream_response("second try")]
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "second try" in out


class TestChaining:
    @respx.mock
    async def test_plan_streams_before_the_code(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                stream_response("1. parse ", "2. emit"),
                stream_response("def parse(): ..."),
            ]
        )
        dec = decision(chain=True)
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert out.index("Plan") < out.index("1. parse ")
        assert out.index("1. parse ") < out.index("def parse")
        assert out.count("data: [DONE]") == 1
        assert record.plan_model == "glm"
        assert record.served_by == "deepseek"

    @respx.mock
    async def test_the_plan_is_handed_to_the_worker(self, harness):
        _, _, upstream, stats, orch = harness
        route = respx.post(UPSTREAM).mock(
            side_effect=[
                stream_response("step one then step two"),
                stream_response("done"),
            ]
        )
        dec = decision(chain=True)

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        worker_payload = json.loads(route.calls[1].request.content)
        trailing = worker_payload["messages"][-1]
        assert trailing["role"] == "system"
        assert "step one then step two" in trailing["content"]

    @respx.mock
    async def test_the_planner_is_given_no_tools(self, harness):
        """A Thinker holding tools will call them, and we cannot run them."""
        _, _, upstream, stats, orch = harness
        route = respx.post(UPSTREAM).mock(
            side_effect=[stream_response("a plan"), stream_response("code")]
        )
        body = {**BODY, "tools": [{"type": "function", "function": {"name": "edit"}}]}
        dec = decision(chain=True)

        await collect(orch, body, dec, new_record(stats, dec))
        await upstream.aclose()

        plan_payload = json.loads(route.calls[0].request.content)
        assert "tools" not in plan_payload
        assert plan_payload["max_tokens"] == 900
        assert json.loads(route.calls[1].request.content)["tools"]  # worker keeps them

    @respx.mock
    async def test_the_plan_does_not_end_the_stream_early(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                stream_response("plan text", finish="stop"),
                stream_response("code text", finish="stop"),
            ]
        )
        dec = decision(chain=True)

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        finishes = [
            line
            for line in out.splitlines()
            if line.startswith("data: ")
            and "[DONE]" not in line
            and json.loads(line[6:])["choices"][0].get("finish_reason")
        ]
        assert len(finishes) == 1

    @respx.mock
    async def test_a_failed_plan_still_produces_code(self, harness):
        """Planning is an enhancement; losing it must not lose the request."""
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(500),
                httpx.Response(500),
                stream_response("code anyway"),
            ]
        )
        dec = decision(chain=True)

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "code anyway" in out
        assert out.count("data: [DONE]") == 1


class TestNonStreaming:
    @respx.mock
    async def test_returns_a_plain_completion(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(return_value=json_response("the answer"))
        dec = decision()

        result = await orch.complete({"messages": []}, dec, new_record(stats, dec))
        await upstream.aclose()

        assert result["choices"][0]["message"]["content"] == "the answer"
        assert result["model"] == "deepseek"

    @respx.mock
    async def test_chained_plan_is_prepended(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[json_response("the plan"), json_response("the code")]
        )
        dec = decision(chain=True)

        result = await orch.complete({"messages": []}, dec, new_record(stats, dec))
        await upstream.aclose()

        content = result["choices"][0]["message"]["content"]
        assert content.index("the plan") < content.index("the code")

    @respx.mock
    async def test_a_queued_model_is_overtaken_by_a_hedge(self, hedging):
        """A client that does not stream is behind the same contended pool as
        one that does. Waiting out each queued model in turn took three
        minutes against NIM while the streaming path answered in six seconds."""
        _, _, upstream, stats, orch = hedging

        async def queued(_request):
            await asyncio.sleep(5.0)
            return json_response("too late")

        respx.post(UPSTREAM).mock(
            side_effect=[queued, json_response("hedge won")]
        )
        dec = decision()
        record = new_record(stats, dec)

        result = await orch.complete({"messages": []}, dec, record)
        await upstream.aclose()

        assert result["choices"][0]["message"]["content"] == "hedge won"
        assert record.served_by == "glm"

    @respx.mock
    async def test_raises_once_every_attempt_is_spent(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(return_value=httpx.Response(500))
        dec = decision()

        with pytest.raises(UpstreamError):
            await orch.complete({"messages": []}, dec, new_record(stats, dec))
        await upstream.aclose()


class TestCapacitySignals:
    """NIM answers 529 when a model's shared pool is saturated. On the free
    tier this is far more common than a real 429, and must not be mistaken for
    the model being broken."""

    @respx.mock
    async def test_529_fails_over_and_cools_down_rather_than_breaking(self, harness):
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                httpx.Response(529, json={"message": "Service temporarily overloaded"}),
                stream_response("from glm"),
            ]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "from glm" in out
        # One 529 is enough to rule out the model: the sibling key would only
        # queue behind the same saturated pool.
        assert record.attempts == ["deepseek#key0", "glm#key0"]
        snap = governor.snapshot()
        assert snap["models"]["deepseek"]["rate_limited"] == 1
        assert snap["models"]["deepseek"]["failed"] == 0  # busy, not broken
        assert snap["breakers_open"] == []
        assert {c["key_index"] for c in snap["cooling_down"]
                if c["model"] == "deepseek"} == {0, 1}

    @respx.mock
    async def test_503_is_treated_the_same_way(self, harness):
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[httpx.Response(503), stream_response("ok")]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert governor.snapshot()["models"]["deepseek"]["rate_limited"] == 1

    @respx.mock
    async def test_a_genuine_500_still_counts_against_the_breaker(self, harness):
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[httpx.Response(500), stream_response("ok")]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert governor.snapshot()["models"]["deepseek"]["failed"] == 1

    @respx.mock
    async def test_a_stalled_model_fails_over_instead_of_hanging(self, harness):
        """A model that accepts the request then goes silent must not block."""
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[httpx.ReadTimeout("upstream went quiet"),
                         stream_response("second model answered")]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "second model answered" in out
        assert len(record.attempts) == 2


class TestStallHandling:
    """A queued model answers with silence. It must be remembered, or every
    request re-pays the full stall timeout rediscovering the same busy model."""

    @respx.mock
    async def test_a_stall_cools_the_model_down_for_longer_than_a_429(self, harness):
        config, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[httpx.ReadTimeout("silent"), stream_response("ok")]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        cooling = {c["model"]: c["seconds_remaining"] for c in
                   governor.snapshot()["cooling_down"]}
        assert cooling["deepseek"] > config.rate_limit.cooldown_base_seconds
        assert governor.snapshot()["models"]["deepseek"]["failed"] == 0

    @respx.mock
    async def test_the_next_request_skips_the_stalled_model(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                httpx.ReadTimeout("silent"),
                stream_response("first"),
                stream_response("second"),
            ]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        second = new_record(stats, dec)
        await collect(orch, BODY, dec, second)
        await upstream.aclose()

        # deepseek is not retried at all; which glm key serves is up to headroom.
        assert [a.split("#")[0] for a in second.attempts] == ["glm"]


class SlowStream(httpx.AsyncByteStream):
    """A response whose first byte arrives late - a queued NIM model."""

    def __init__(self, text: str, delay: float) -> None:
        self._body = text.encode()
        self._delay = delay

    async def __aiter__(self):
        await asyncio.sleep(self._delay)
        yield self._body


def slow_response(*texts: str, delay: float) -> httpx.Response:
    return httpx.Response(
        200,
        stream=SlowStream(sse_body(*({"content": t} for t in texts)), delay),
        headers={"content-type": "text/event-stream"},
    )


class BreaksMidStream(httpx.AsyncByteStream):
    """Sends real content, then the connection dies - the one failure that
    cannot be re-routed, because the client already holds bytes."""

    def __init__(self, text: str, exc: Exception) -> None:
        self._body = text.encode()
        self._exc = exc

    async def __aiter__(self):
        yield self._body
        raise self._exc


def breaks_mid_stream(*texts: str, exc: Exception) -> httpx.Response:
    # Deliberately no [DONE] and no finish frame: the connection dies while the
    # answer is still coming, which is the case that cannot be re-routed.
    frames = "".join(
        "data: "
        + json.dumps(
            {"id": "up", "model": "upstream",
             "choices": [{"index": 0, "delta": {"content": t}}]}
        )
        + "\n\n"
        for t in texts
    )
    return httpx.Response(
        200,
        stream=BreaksMidStream(frames, exc),
        headers={"content-type": "text/event-stream"},
    )


class PreambleThenStall(httpx.AsyncByteStream):
    """Answers the handshake instantly, then queues - what glm-5.2 really does."""

    def __init__(self, text: str, delay: float) -> None:
        self._body = text.encode()
        self._delay = delay

    async def __aiter__(self):
        yield (
            "data: "
            + json.dumps(
                {
                    "id": "up",
                    "model": "upstream",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                }
            )
            + "\n\n"
        ).encode()
        await asyncio.sleep(self._delay)
        yield self._body


def preamble_then_stall(*texts: str, delay: float) -> httpx.Response:
    return httpx.Response(
        200,
        stream=PreambleThenStall(sse_body(*({"content": t} for t in texts)), delay),
        headers={"content-type": "text/event-stream"},
    )


@pytest.fixture
def hedging(keys):
    config = make_config(hedge={"enabled": True, "delay_seconds": 0.05,
                                "max_parallel": 3})
    governor = Governor(config, keys)
    upstream = UpstreamClient(config)
    stats = Stats()
    return config, governor, upstream, stats, Orchestrator(
        config, governor, upstream, stats
    )


class TestHedging:
    """Which models are queued flips within seconds, so the fastest cannot be
    known ahead of time - only discovered by asking several at once."""

    @respx.mock
    async def test_a_healthy_primary_costs_exactly_one_request(self, hedging):
        _, _, upstream, stats, orch = hedging
        route = respx.post(UPSTREAM).mock(return_value=stream_response("instant"))
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "instant" in out
        assert route.call_count == 1  # no hedge fired
        assert record.attempts == ["deepseek#key0"]

    @respx.mock
    async def test_a_stalled_primary_is_overtaken_by_a_hedge(self, hedging):
        _, _, upstream, stats, orch = hedging
        respx.post(UPSTREAM).mock(
            side_effect=[
                slow_response("too late", delay=5.0),   # deepseek, queued
                stream_response("hedge won"),           # glm, ready
            ]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "hedge won" in out
        assert "too late" not in out
        assert record.served_by == "glm"
        assert len(record.attempts) == 2

    @respx.mock
    async def test_the_race_is_to_the_first_byte_not_the_status_line(self, hedging):
        """NIM returns 200 immediately and then leaves you in the queue, so a
        header-only race would crown the slowest model."""
        _, _, upstream, stats, orch = hedging
        respx.post(UPSTREAM).mock(
            side_effect=[
                slow_response("slow", delay=5.0),  # 200 headers, no body
                stream_response("fast"),
            ]
        )
        dec = decision()
        record = new_record(stats, dec)

        await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert record.served_by == "glm"

    @respx.mock
    async def test_an_empty_role_frame_does_not_win_the_race(self, hedging):
        """Models emit a role-only frame the instant they accept a request and
        *then* queue. Treating that as an answer crowns the queued model and,
        worse, does it before the commit point - so the stall lands mid-stream
        where failover is no longer possible."""
        _, _, upstream, stats, orch = hedging
        respx.post(UPSTREAM).mock(
            side_effect=[
                preamble_then_stall("too late", delay=5.0),  # deepseek: handshake, then queue
                stream_response("hedge won"),                # glm: actually generating
            ]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "hedge won" in out
        assert "too late" not in out
        assert record.served_by == "glm"

    @respx.mock
    async def test_the_winners_preamble_is_replayed_not_dropped(self, hedging):
        """The role frame is part of a well-formed OpenAI stream, so frames
        pulled off the wire while judging the race must still reach the client."""
        _, _, upstream, stats, orch = hedging
        respx.post(UPSTREAM).mock(
            return_value=preamble_then_stall("hello", delay=0.0)
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert '"role": "assistant"' in out or '"role":"assistant"' in out
        assert "hello" in out

    @respx.mock
    async def test_a_model_that_loses_is_parked_for_the_next_request(self, hedging):
        """Contention is the one failure that never raises. The queued model
        just gets cancelled, so unless losing is recorded the governor learns
        nothing and sends the next request to the same queue."""
        _, governor, upstream, stats, orch = hedging
        respx.post(UPSTREAM).mock(
            side_effect=[slow_response("queued", delay=5.0), stream_response("won")]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert governor.snapshot()["models"]["deepseek"]["outraced"] == 1
        assert not governor.is_available("deepseek", 0)
        assert governor.is_available("glm", 0)  # the winner is untouched

    @respx.mock
    async def test_the_next_request_leads_with_the_model_that_answered(
        self, hedging
    ):
        _, governor, upstream, stats, orch = hedging
        respx.post(UPSTREAM).mock(
            side_effect=[
                slow_response("queued", delay=5.0),
                stream_response("won"),
                stream_response("second request"),
            ]
        )
        dec = decision()
        await collect(orch, BODY, dec, new_record(stats, dec))

        second = new_record(stats, dec)
        out = await collect(orch, BODY, dec, second)
        await upstream.aclose()

        assert "second request" in out
        assert second.attempts[0].startswith("glm")  # no longer led by deepseek

    @respx.mock
    async def test_a_winning_primary_blames_nobody(self, hedging):
        """Only a model given *more* time than the winner is evidence of
        anything. A hedge launched later that simply had less time to run must
        never be punished for it."""
        _, governor, upstream, stats, orch = hedging
        respx.post(UPSTREAM).mock(return_value=stream_response("instant"))
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert governor.snapshot()["models"]["deepseek"]["outraced"] == 0
        assert governor.is_available("deepseek", 0)

    @respx.mock
    async def test_hedging_can_be_switched_off(self, keys):
        config = make_config(hedge={"enabled": False, "delay_seconds": 0.05})
        governor = Governor(config, keys)
        upstream = UpstreamClient(config)
        stats = Stats()
        orch = Orchestrator(config, governor, upstream, stats)
        route = respx.post(UPSTREAM).mock(
            side_effect=[slow_response("slow but alone", delay=0.3),
                         stream_response("never reached")]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "slow but alone" in out
        assert route.call_count == 1

    @respx.mock
    async def test_never_exceeds_max_parallel(self, keys):
        config = make_config(hedge={"enabled": True, "delay_seconds": 0.02,
                                    "max_parallel": 2})
        governor = Governor(config, keys)
        upstream = UpstreamClient(config)
        stats = Stats()
        orch = Orchestrator(config, governor, upstream, stats)
        route = respx.post(UPSTREAM).mock(
            side_effect=[slow_response("a", delay=0.4), slow_response("b", delay=0.4),
                         stream_response("c"), stream_response("d")]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert route.call_count <= 2


class PausesMidAnswer(httpx.AsyncByteStream):
    """Generates, thinks for a while, then finishes - a working model."""

    def __init__(self, first: str, rest: str, pause: float) -> None:
        self._first, self._rest, self._pause = first.encode(), rest.encode(), pause

    async def __aiter__(self):
        yield self._first
        await asyncio.sleep(self._pause)
        yield self._rest


def pauses_mid_answer(pause: float) -> httpx.Response:
    head = ("data: " + json.dumps(
        {"id": "up", "model": "upstream",
         "choices": [{"index": 0, "delta": {"content": "thinking"}}]}) + "\n\n")
    tail = ("data: " + json.dumps(
        {"id": "up", "model": "upstream",
         "choices": [{"index": 0, "delta": {"content": " done"}, "finish_reason": "stop"}]})
        + "\n\ndata: [DONE]\n\n")
    return httpx.Response(
        200,
        stream=PausesMidAnswer(head, tail, pause),
        headers={"content-type": "text/event-stream"},
    )


class TestStallBudgetAppliesBeforeFirstOutput:
    """The stall budget exists to spot a *queued* model - one that accepted the
    request and never started. Enforcing it as httpx's read timeout applied it
    to the whole stream instead, so a model that was generating fine but paused
    to think got killed at exactly the point failover is impossible, and the
    user got a truncated answer plus an error."""

    @respx.mock
    async def test_a_pause_while_generating_is_allowed(self, keys):
        config = make_config(upstream={
            "base_url": "https://nim.test/v1",
            "stall_timeout_seconds": 0.05,   # tight: for the race only
            "idle_timeout_seconds": 5.0,     # generous: once it is answering
        })
        governor = Governor(config, keys)
        upstream = UpstreamClient(config)
        stats = Stats()
        orch = Orchestrator(config, governor, upstream, stats)
        respx.post(UPSTREAM).mock(return_value=pauses_mid_answer(pause=0.4))
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "thinking" in out and " done" in out
        assert "ReadTimeout" not in out
        assert out.count("data: [DONE]") == 1

    @respx.mock
    async def test_silence_before_any_output_still_fails_over(self, keys):
        """The aggressive half must survive: a model that never starts is
        abandoned quickly, because before the first byte that is still free."""
        config = make_config(
            upstream={
                "base_url": "https://nim.test/v1",
                "stall_timeout_seconds": 0.05,
                "idle_timeout_seconds": 5.0,
            },
            hedge={"enabled": False, "delay_seconds": 0.05},
        )
        governor = Governor(config, keys)
        upstream = UpstreamClient(config)
        stats = Stats()
        orch = Orchestrator(config, governor, upstream, stats)
        respx.post(UPSTREAM).mock(
            side_effect=[slow_response("too late", delay=3.0),
                         stream_response("failed over")]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "failed over" in out
        assert "too late" not in out


def error_frame_response(message: str, code: int | None = 503) -> httpx.Response:
    """NIM answers 200, then puts the failure *inside* the stream."""
    body = {"error": {"message": message, "type": "Service Unavailable"}}
    if code is not None:
        body["error"]["code"] = code
    return httpx.Response(
        200,
        text=f"data: {json.dumps(body)}\n\n",
        headers={"content-type": "text/event-stream"},
    )


EXHAUSTED = "ResourceExhausted: Worker local total request limit reached (33/32)"


class TestErrorDeliveredInsideTheStream:
    """A 200 whose body is an error is still a failure.

    Treating an unrecognised frame as output made the error envelope *win the
    race*, and the relay then forwarded it verbatim - so NIM's internal capacity
    message was delivered to the user as the model's answer, with no warning
    logged anywhere, because as far as the proxy knew the request had succeeded.
    """

    @respx.mock
    async def test_it_loses_the_race_instead_of_winning_it(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[error_frame_response(EXHAUSTED), stream_response("real answer")]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "real answer" in out
        assert "ResourceExhausted" not in out
        assert record.served_by == "glm"

    @respx.mock
    async def test_a_capacity_code_parks_the_model(self, harness):
        """503 inside the body means the same thing as 503 in the status line."""
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[error_frame_response(EXHAUSTED), stream_response("ok")]
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert not governor.is_available("deepseek", 0)
        assert governor.snapshot()["models"]["deepseek"]["failed"] == 0  # not a fault

    @respx.mock
    async def test_an_error_with_no_code_is_still_caught(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[error_frame_response("something broke", code=None),
                         stream_response("real answer")]
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "real answer" in out
        assert "something broke" not in out

    @respx.mock
    async def test_an_error_after_real_content_is_reported_not_relayed(
        self, harness
    ):
        """Too late to re-route, but it must still be labelled as a failure
        rather than appended to the answer as if the model had said it."""
        _, governor, upstream, stats, orch = harness
        body = (
            "data: " + json.dumps(
                {"id": "up", "model": "upstream",
                 "choices": [{"index": 0, "delta": {"content": "partial"}}]}) + "\n\n"
            + "data: " + json.dumps({"error": {"message": EXHAUSTED, "code": 503}})
            + "\n\n"
        )
        respx.post(UPSTREAM).mock(
            return_value=httpx.Response(
                200, text=body, headers={"content-type": "text/event-stream"}
            )
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "partial" in out                  # what arrived is kept
        assert "[fusion] upstream error" in out  # and it is labelled
        assert "deepseek" in out                 # naming the model that failed
        assert not governor.is_available("deepseek", 0)
        assert out.count("data: [DONE]") == 1

    @respx.mock
    async def test_a_usage_frame_is_not_mistaken_for_an_error(self, harness):
        """Only an `error` key is a failure; other unfamiliar frames pass."""
        _, _, upstream, stats, orch = harness
        usage = 'data: {"id":"up","model":"upstream","usage":{"total_tokens":5}}\n\n'
        respx.post(UPSTREAM).mock(
            return_value=httpx.Response(
                200,
                text=usage + sse_body({"content": "hello"}),
                headers={"content-type": "text/event-stream"},
            )
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "hello" in out


class TestMidStreamFailure:
    """The client already holds bytes, so all we can do is say what happened -
    which means the message has to actually say something."""

    @respx.mock
    async def test_a_nameless_transport_error_is_still_named(self, harness):
        """Several httpx errors stringify to "" when raised without a message,
        and the report was passing that straight through: the user saw
        "[fusion] upstream error:" with nothing after it, and the log line was
        equally blank. The one thing a mid-stream failure owes you is a name."""
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            return_value=breaks_mid_stream("partial", exc=httpx.ReadError(""))
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "partial" in out            # what did arrive is kept
        assert "ReadError" in out          # and the failure is diagnosable
        assert "upstream error: \\n" not in out
        assert out.count("data: [DONE]") == 1

    @respx.mock
    async def test_a_model_that_dies_mid_answer_is_parked(self, harness):
        """This is the one failure the governor was never told about. It cannot
        be re-routed for *this* request, but it is still the best evidence we
        have about the model, and without it the next request walks straight
        back into the same stalled stream."""
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            return_value=breaks_mid_stream("partial", exc=httpx.ReadTimeout(""))
        )
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert not governor.is_available("deepseek", 0)
        assert not governor.is_available("deepseek", 1)  # queue is per-model

    @respx.mock
    async def test_the_failing_model_is_identified(self, harness):
        """With four models behind one lane, "it broke" is not actionable."""
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            return_value=breaks_mid_stream("partial", exc=httpx.ReadError(""))
        )
        dec = decision()

        out = await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert "deepseek" in out


class TestModelUnavailable:
    """NIM answers 404 for a model your key cannot call, and sometimes for one
    that is briefly unrouteable. That names the model, not the request."""

    @respx.mock
    async def test_404_moves_to_another_model_instead_of_failing(self, harness):
        _, _, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[
                httpx.Response(404, json={"detail": "Not found for account"}),
                stream_response("a working model"),
            ]
        )
        dec = decision()
        record = new_record(stats, dec)

        out = await collect(orch, BODY, dec, record)
        await upstream.aclose()

        assert "a working model" in out
        assert record.served_by == "glm"

    @respx.mock
    async def test_404_drops_the_model_for_subsequent_requests(self, harness):
        _, governor, upstream, stats, orch = harness
        respx.post(UPSTREAM).mock(
            side_effect=[httpx.Response(404), stream_response("first"),
                         stream_response("second")]
        )
        dec = decision()
        await collect(orch, BODY, dec, new_record(stats, dec))
        second = new_record(stats, dec)
        await collect(orch, BODY, dec, second)
        await upstream.aclose()

        assert "deepseek" in governor.snapshot()["breakers_open"]
        assert all("deepseek" not in a for a in second.attempts)

    @respx.mock
    async def test_a_400_still_stops_everything(self, harness):
        """Unlike 404, a malformed request fails the same way everywhere."""
        _, _, upstream, stats, orch = harness
        route = respx.post(UPSTREAM).mock(return_value=httpx.Response(400))
        dec = decision()

        await collect(orch, BODY, dec, new_record(stats, dec))
        await upstream.aclose()

        assert route.call_count == 1
