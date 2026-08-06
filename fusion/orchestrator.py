"""Orchestrator: turn a routing decision into an actual answer.

Two jobs. First, dispatch with failover - walk the governor's attempt list until
one opens, which is only safe *before* the first byte reaches the client.
Second, run the Thinker->Worker chain, splicing two upstream calls into one
well-formed SSE stream so opencode sees an ordinary single-model response.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from . import sse
from .config import FusionConfig
from .coordinator import RouteDecision
from .governor import Attempt, Governor
from .stats import RequestRecord, Stats
from .upstream import UpstreamClient, UpstreamError

log = logging.getLogger("fusion.orchestrator")

PLAN_INSTRUCTION = (
    "Before any code is written, produce a short implementation plan for the "
    "user's request: the approach, the files or components involved, and the "
    "order of work. Be concrete and brief - at most a dozen lines. Do not write "
    "the implementation itself and do not call any tools; another model will "
    "execute this plan."
)

WORKER_INSTRUCTION = (
    "A planning pass has already been run for this request and produced the "
    "plan below. Execute it. Deviate only where it is wrong, and say so if you "
    "do.\n\n<plan>\n{plan}\n</plan>"
)

PLAN_HEADER = "**Plan**\n\n"
PLAN_FOOTER = "\n\n---\n\n"

# Fields that only make sense for the executing call, stripped from the
# planning call. `tools` above all: a Thinker that can call tools will try to,
# and we have no way to run them.
_PLAN_STRIPPED = frozenset(
    {
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "response_format",
        "n",
    }
)


class Orchestrator:
    def __init__(
        self,
        config: FusionConfig,
        governor: Governor,
        upstream: UpstreamClient,
        stats: Stats,
    ) -> None:
        self._config = config
        self._governor = governor
        self._upstream = upstream
        self._stats = stats

    # -- streaming --------------------------------------------------------

    async def stream(
        self, body: dict[str, Any], decision: RouteDecision, record: RequestRecord
    ) -> AsyncIterator[bytes]:
        completion_id = sse.new_completion_id()
        created = int(time.time())
        label = decision.primary

        yield sse.role_chunk(completion_id, label, created=created)

        plan = ""
        if decision.chain:
            async for output in self._stream_plan(
                body, completion_id, label, created, record
            ):
                if isinstance(output, str):
                    plan = output
                else:
                    yield output

        worker_body = self._with_plan(body, plan) if plan else body
        saw_finish = False
        live_attempt: Attempt | None = None

        try:
            async with self._open(
                worker_body, decision.candidates, record
            ) as live:
                live_attempt = live.attempt
                record.served_by = live.attempt.model
                async for chunk in self._relay(live, completion_id, label):
                    if chunk.finish_reason:
                        saw_finish = True
                    yield chunk.data
                self._governor.record_success(live.attempt)
        except (UpstreamError, httpx.HTTPError) as exc:
            # A transport failure part-way through a stream cannot be re-routed:
            # the client already holds bytes from this model. Report it inside
            # the stream so a truncated answer is never mistaken for a whole one.
            # Several httpx errors carry no message at all - ReadError,
            # RemoteProtocolError and friends all stringify to "" when raised
            # bare - so passing the text straight through reported a failure
            # with nothing in it, to the log and to the user alike. Fall back to
            # the class name, and say which model broke: with four behind one
            # lane, "upstream error" on its own is not actionable.
            raw = exc.message if isinstance(exc, UpstreamError) else str(exc)
            message = raw.strip() or type(exc).__name__
            served = record.served_by or "unknown model"
            message = f"{served}: {message}"
            log.warning("stream failed mid-answer | %s", message)
            # This request is beyond saving, but the evidence is not: a model
            # that went quiet mid-answer is exactly the kind the next request
            # should route around. Without this the governor never hears about
            # the one failure it cannot see for itself.
            if live_attempt is not None:
                self._governor.record_overloaded(
                    live_attempt,
                    base_seconds=self._config.rate_limit.stall_cooldown_seconds,
                )
            self._stats.finish(record, status="error", detail=message)
            yield sse.error_chunk(completion_id, label, message)
            yield sse.DONE
            return

        if not saw_finish:
            # Some models close without a terminal frame; opencode waits for one.
            yield sse.finish_chunk(completion_id, label, created=created)
        self._stats.finish(record, status="ok")
        yield sse.DONE

    async def _stream_plan(
        self,
        body: dict[str, Any],
        completion_id: str,
        label: str,
        created: int,
        record: RequestRecord,
    ) -> AsyncIterator[Any]:
        """Stream the planning pass, then yield the accumulated plan text.

        Yields bytes for the wire and finally a single str: the plan. A failed
        planning pass is not fatal - we simply proceed without a plan.
        """
        plan_body = self._plan_request(body)
        thinkers = self._config.models_for_role("thinker")
        parts: list[str] = []

        try:
            async with self._open(plan_body, thinkers, record) as live:
                record.plan_model = live.attempt.model
                yield sse.text_chunk(completion_id, label, PLAN_HEADER, created=created)
                async for chunk in self._relay(
                    live, completion_id, label, suppress_finish=True
                ):
                    parts.append(chunk.text)
                    yield chunk.data
                self._governor.record_success(live.attempt)
        except (UpstreamError, httpx.HTTPError) as exc:
            log.warning("planning pass failed (%s); going straight to the worker", exc)
            yield ""
            return

        plan = "".join(parts).strip()
        if plan:
            yield sse.encode(
                sse.delta_chunk(completion_id, label, {"content": PLAN_FOOTER},
                                created=created)
            )
        yield plan

    # -- non-streaming ----------------------------------------------------

    async def complete(
        self, body: dict[str, Any], decision: RouteDecision, record: RequestRecord
    ) -> dict[str, Any]:
        plan = ""
        if decision.chain:
            plan = await self._plan_once(body, record)

        worker_body = self._with_plan(body, plan) if plan else body

        try:
            attempt, response = await self._race(
                worker_body, decision.candidates, record,
                self._upstream.complete, _drop_body,
            )
        except UpstreamError as exc:
            self._stats.finish(record, status="error", detail=exc.message)
            raise

        self._governor.record_success(attempt)
        record.served_by = attempt.model
        if plan:
            _prepend_content(response, PLAN_HEADER + plan + PLAN_FOOTER)
        response["model"] = decision.primary
        self._stats.finish(record, status="ok")
        return response

    async def _plan_once(self, body: dict[str, Any], record: RequestRecord) -> str:
        plan_body = self._plan_request(body)
        thinkers = self._config.models_for_role("thinker")
        try:
            attempt, response = await self._race(
                plan_body, thinkers, record, self._upstream.complete, _drop_body
            )
        except UpstreamError:
            log.warning("planning pass failed; going straight to the worker")
            return ""
        self._governor.record_success(attempt)
        record.plan_model = attempt.model
        return sse.message_text(response).strip()

    # -- dispatch ---------------------------------------------------------

    async def _prime(
        self, attempt: Attempt, body: dict[str, Any]
    ) -> "LiveStream":
        """Open a stream and wait until it proves the model is generating.

        Racing the HTTP status would pick the wrong winner: NIM returns 200
        headers immediately and *then* leaves you in the queue. Racing the
        first *frame* is barely better - models answer the handshake with an
        empty role delta the moment they accept the request, and queue after
        it. Only a frame carrying generated output settles the race.

        The distinction is worth more than the latency it saves. The race is
        the last moment failover is possible, so a preamble frame that ends it
        early strands the stall on the far side of the commit point, where
        nothing can be done about it.
        """
        response = await self._upstream.open_stream(attempt, body)
        lines = response.aiter_lines()
        preamble: list[str] = []

        async def first_output() -> "LiveStream":
            async for line in lines:
                if not line.strip():
                    continue
                problem = _frame_error(line)
                if problem is not None:
                    # A 200 whose body is an error is still a failure. Raising
                    # here keeps the race going; letting it through would crown
                    # it and relay NIM's message as the model's answer.
                    text, code = problem
                    raise UpstreamError(code, text)
                preamble.append(line)
                if _carries_output(line) or len(preamble) > _MAX_PREAMBLE_FRAMES:
                    return LiveStream(attempt, response, lines, preamble)
            raise UpstreamError(None, "upstream closed without producing output")

        try:
            # The strict deadline belongs here and nowhere else. Until a model
            # produces something, dropping it costs nothing - another is already
            # racing. The moment it does, this budget stops applying and the far
            # looser idle timeout takes over, because from then on there is no
            # failover left and cutting it off would throw away a real answer.
            return await asyncio.wait_for(
                first_output(), timeout=self._upstream.stall_budget(body)
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            await response.aclose()
            raise UpstreamError(
                None, "silent past the stall budget", timed_out=True
            ) from exc
        except BaseException:
            await response.aclose()
            raise

    async def _race(
        self,
        body: dict[str, Any],
        models: tuple[str, ...] | list[str],
        record: RequestRecord,
        run: "Callable[[Attempt, dict[str, Any]], Awaitable[Any]]",
        discard: "Callable[[Any], Awaitable[None]]",
    ) -> tuple[Attempt, Any]:
        """Race the top candidates, first to actually answer wins.

        Sequential failover pays the stall timeout for every queued model
        before reaching a live one - measured at 68s against NIM's free tier.
        Hedging pays min() instead of sum(): the same pool answered in 0.35s.

        The hedges are staggered rather than fired at once, so a healthy
        primary still costs exactly one request. A second is only launched if
        the leader has not answered within `hedge.delay_seconds`.

        `run` decides what "answered" means - a primed stream or a parsed
        completion body - and `discard` releases a runner-up that arrives
        anyway. Both dispatch paths share this loop because a queued model
        costs the same either way.
        """
        hedge = self._config.hedge
        tried: set[tuple[str, int]] = set()
        running: dict[asyncio.Task, Attempt] = {}
        errors: list[UpstreamError] = []
        winner: Any = None
        winning_attempt: Attempt | None = None
        # A malformed request fails identically on every model, so one such
        # answer ends the race rather than starting more of it.
        fatal: UpstreamError | None = None

        started: dict[asyncio.Task, float] = {}
        won_at = 0.0

        def launch() -> bool:
            attempt = self._next_attempt(
                models, tried, frozenset(a.model for a in running.values())
            )
            if attempt is None:
                return False
            tried.add((attempt.model, attempt.key_index))
            record.attempts.append(attempt.label)
            self._governor.record_dispatch(attempt)
            task = asyncio.ensure_future(run(attempt, body))
            running[task] = attempt
            started[task] = time.monotonic()
            return True

        try:
            launch()
            while running:
                room = hedge.enabled and len(running) < hedge.max_parallel
                done, _ = await asyncio.wait(
                    running,
                    timeout=hedge.delay_seconds if room else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    attempt = running.pop(task)
                    try:
                        result = task.result()
                    except UpstreamError as exc:
                        errors.append(exc)
                        self._note(attempt, exc)
                        if not exc.is_retryable:
                            fatal = exc
                        continue
                    except asyncio.CancelledError:
                        continue
                    except httpx.HTTPError as exc:
                        wrapped = UpstreamError(None, f"{type(exc).__name__}: {exc}")
                        errors.append(wrapped)
                        self._note(attempt, wrapped)
                        continue
                    if winner is None:
                        winner, winning_attempt = result, attempt
                        won_at = started[task]
                    else:
                        await discard(result)  # arrived second; let it go
                if winner is not None or fatal is not None:
                    break
                if not done and room:
                    launch()  # leader is slow, add a hedge
                elif not running:
                    launch()  # everything failed, keep walking the list
        finally:
            # Contention is the one failure mode that never raises: a queued
            # model accepts the request and goes quiet, so losing is the only
            # signal it ever gives. Record it against anything that started no
            # later than the winner - it had at least as long and produced
            # nothing - and leave later hedges alone, since having less time is
            # not evidence of anything.
            if winner is not None:
                stall_base = self._config.rate_limit.stall_cooldown_seconds
                for task, attempt in running.items():
                    if started[task] <= won_at:
                        self._governor.record_outraced(
                            attempt, base_seconds=stall_base
                        )
            # Drop the losers as soon as the race is settled rather than
            # holding upstream connections open for the length of the answer.
            await _abandon(running, discard)

        if winner is None or winning_attempt is None:
            raise fatal or (
                errors[-1]
                if errors
                else UpstreamError(None, "no upstream attempt could be made")
            )

        if winning_attempt.model != (models[0] if models else None):
            self._governor.record_failover(
                str(models[0]), winning_attempt.model
            )
        record.hedges = len(record.attempts)
        return winning_attempt, winner

    @contextlib.asynccontextmanager
    async def _open(
        self,
        body: dict[str, Any],
        models: tuple[str, ...] | list[str],
        record: RequestRecord,
    ):
        """Race for a live stream. Nothing has reached the client when this
        yields, so picking a winner here is still safe."""
        _, winner = await self._race(
            body, models, record, self._prime, _close_stream
        )
        try:
            yield winner
        finally:
            await winner.aclose()

    def _next_attempt(
        self,
        models: tuple[str, ...] | list[str],
        tried: set[tuple[str, int]],
        avoid_models: frozenset[str] = frozenset(),
    ) -> Attempt | None:
        """Re-ask the governor, so each launch reflects what the last one taught.

        Recomputing rather than walking a fixed list matters: a 529 cools the
        model across every key, and only a fresh plan will skip the sibling key
        instead of queueing behind the same saturated pool a second time.

        `avoid_models` keeps a hedge off a model already in flight. The governor
        orders by model preference then key, so without this the second hedge
        would be the same model on the other key - which buys nothing against
        contention, since a queue is per-model and ignores whose key is asking.
        A sibling key is still returned as a last resort if no fresh model is
        left, because that does help against a 429.
        """
        spare: Attempt | None = None
        for attempt in self._governor.plan_attempts(models):
            if (attempt.model, attempt.key_index) in tried:
                continue
            if attempt.model in avoid_models:
                spare = spare or attempt
                continue
            return attempt
        return spare

    async def _relay(
        self,
        live: "LiveStream",
        completion_id: str,
        label: str,
        *,
        suppress_finish: bool = False,
    ) -> AsyncIterator["_Relayed"]:
        """Re-emit upstream chunks under one stable id and model name."""
        async for payload in sse.iter_data(live.replay()):
            problem = _payload_error(payload)
            if problem is not None:
                # The same envelope can arrive after real content. Too late to
                # re-route, but it must not be relayed as if the model said it.
                text, code = problem
                raise UpstreamError(code, text)
            chunk = sse.parse_chunk(payload)
            if chunk is None:
                continue
            text = sse.chunk_text(chunk)
            finish = sse.chunk_finish_reason(chunk)
            chunk["id"] = completion_id
            chunk["model"] = label
            if suppress_finish and finish:
                # The plan ending is not the response ending.
                for choice in chunk.get("choices") or []:
                    choice["finish_reason"] = None
                finish = None
            yield _Relayed(sse.encode(chunk), text, finish)

    def _note(self, attempt: Attempt, exc: UpstreamError) -> None:
        if exc.is_model_unavailable:
            log.warning("%s is not callable (%s); dropping it", attempt.model, exc.message[:80])
            self._governor.record_unavailable(attempt)
        elif not exc.is_capacity:
            self._governor.record_failure(attempt)
        elif exc.is_rate_limit:
            # Our key's quota - another key on the same model is still viable.
            self._governor.record_rate_limited(attempt, exc.retry_after)
        else:
            # Upstream is saturated or silent: no key gets in. Park the model.
            # A stall is our own inference, so it seeds the backoff rather than
            # dictating it - a model that stalls every time must back off, not
            # draw the same short cooldown forever.
            base = (
                self._config.rate_limit.stall_cooldown_seconds
                if exc.timed_out
                else None
            )
            self._governor.record_overloaded(
                attempt, exc.retry_after, base_seconds=base
            )

    # -- body shaping -----------------------------------------------------

    def _plan_request(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = {k: v for k, v in body.items() if k not in _PLAN_STRIPPED}
        payload["messages"] = list(body.get("messages") or []) + [
            {"role": "system", "content": PLAN_INSTRUCTION}
        ]
        payload["max_tokens"] = self._config.routing.chain_plan_max_tokens
        return payload

    def _with_plan(self, body: dict[str, Any], plan: str) -> dict[str, Any]:
        # Appended rather than merged into the leading system prompt: trailing
        # instructions sit closer to the user's request and get followed more
        # reliably.
        payload = dict(body)
        payload["messages"] = list(body.get("messages") or []) + [
            {"role": "system", "content": WORKER_INSTRUCTION.format(plan=plan)}
        ]
        return payload


class _Relayed:
    __slots__ = ("data", "text", "finish_reason")

    def __init__(self, data: bytes, text: str, finish_reason: str | None) -> None:
        self.data = data
        self.text = text
        self.finish_reason = finish_reason


def _prepend_content(response: dict[str, Any], prefix: str) -> None:
    for choice in response.get("choices") or []:
        message = choice.get("message")
        if isinstance(message, dict):
            message["content"] = prefix + (message.get("content") or "")


class LiveStream:
    """An upstream response that has already proven it is generating output.

    Holds every frame pulled off the wire while judging the race - the empty
    role delta is part of a well-formed OpenAI stream - so the relay can
    replay them in order before continuing with the rest of the body.
    """

    __slots__ = ("attempt", "response", "_lines", "_preamble")

    def __init__(self, attempt: Attempt, response, lines, preamble: list[str]) -> None:
        self.attempt = attempt
        self.response = response
        self._lines = lines
        self._preamble = preamble

    async def replay(self) -> AsyncIterator[str]:
        for line in self._preamble:
            yield line
        async for line in self._lines:
            yield line

    async def aclose(self) -> None:
        await self.response.aclose()


#: Delta fields that represent something the model actually generated.
_OUTPUT_FIELDS = ("content", "reasoning_content", "tool_calls", "refusal")

#: Give up distinguishing preamble from output after this many frames. Nothing
#: legitimate needs more, and the stall timeout only catches *silence* - a model
#: dribbling empty deltas forever would otherwise buffer without bound.
_MAX_PREAMBLE_FRAMES = 32


def _payload_error(payload: str) -> tuple[str, int | None] | None:
    """A failure NIM delivered *inside* a 200 stream, if this frame is one.

    Not a corner case: the free tier reports its own capacity limits this way,
    sending `200 OK` headers and then an error object where a delta should be.
    Nothing upstream of here can tell - the status line said success - so unless
    this frame is recognised the error wins the race and the relay hands NIM's
    internal message to the user as if the model had said it.
    """
    payload = payload.strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        frame = json.loads(payload)
    except ValueError:
        return None
    if not isinstance(frame, dict):
        return None
    problem = frame.get("error")
    if problem is None:
        return None
    if isinstance(problem, dict):
        code = problem.get("code")
        return (
            str(problem.get("message") or problem)[:200],
            code if isinstance(code, int) else None,
        )
    return str(problem)[:200], None


def _frame_error(line: str) -> tuple[str, int | None] | None:
    if not line.startswith("data:"):
        return None
    return _payload_error(line[len("data:"):])


def _carries_output(line: str) -> bool:
    """Does this frame prove the model is generating, or is it just handshake?

    Deliberately permissive: anything unparseable or unfamiliar counts as
    output. A frame we cannot read is a bad reason to keep waiting.
    """
    if not line.startswith("data:"):
        return False  # SSE comment or event name - not the model talking
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return True
    try:
        frame = json.loads(payload)
    except ValueError:
        return True
    if not isinstance(frame, dict) or "choices" not in frame:
        return True  # a usage frame or similar; errors are caught before this
    for choice in frame.get("choices") or ():
        if not isinstance(choice, dict):
            return True
        if choice.get("finish_reason"):
            return True
        delta = choice.get("delta") or choice.get("message") or {}
        if any(delta.get(field) for field in _OUTPUT_FIELDS):
            return True
    return False


async def _close_stream(live: "LiveStream") -> None:
    await live.aclose()


async def _drop_body(_response: dict[str, Any]) -> None:
    """A completion is already fully read; there is nothing left to release."""


async def _abandon(running: dict, discard) -> None:
    """Cancel the losing hedges and release anything they managed to open."""
    if not running:
        return
    for task in running:
        task.cancel()
    results = await asyncio.gather(*running, return_exceptions=True)
    for result in results:
        if not isinstance(result, BaseException):
            await discard(result)
    running.clear()
