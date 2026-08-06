"""FastAPI surface: one router behind two dialects.

`/v1/chat/completions` for opencode and anything else OpenAI-compatible;
`/v1/messages` for Claude Code, which speaks Anthropic. Both land on the same
coordinator, governor and orchestrator - only the wire format differs.
"""

from __future__ import annotations

import difflib
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__, anthropic
from .classifier import classify
from .config import (
    ApiKey,
    FusionConfig,
    Settings,
    load_api_key_entries,
    load_config,
)
from .coordinator import Coordinator, RouteDecision, extract_facts, fingerprint
from .governor import Governor
from .orchestrator import Orchestrator
from .stats import Stats
from .upstream import UpstreamClient, UpstreamError

log = logging.getLogger("fusion")

VIRTUAL_PREFIX = "fusion/"


@dataclass
class Runtime:
    config: FusionConfig
    settings: Settings
    governor: Governor
    coordinator: Coordinator
    upstream: UpstreamClient
    orchestrator: Orchestrator
    stats: Stats
    catalog: list[str]
    catalog_problems: list[str]
    api_keys: list[ApiKey]


def build_runtime(settings: Settings | None = None) -> Runtime:
    settings = settings or Settings()
    config = load_config(settings.config_path)
    entries = load_api_key_entries()
    governor = Governor(config, [entry.value for entry in entries])
    upstream = UpstreamClient(config)
    stats = Stats()
    return Runtime(
        config=config,
        settings=settings,
        governor=governor,
        coordinator=Coordinator(config),
        upstream=upstream,
        orchestrator=Orchestrator(config, governor, upstream, stats),
        stats=stats,
        catalog=[],
        catalog_problems=[],
        api_keys=entries,
    )


async def verify_catalog(runtime: Runtime) -> None:
    """Check every configured model against the live NIM catalog.

    Model IDs move between releases and published listings disagree with each
    other, so a mismatch here is likely. Failing loudly at startup beats a 404
    on the first real request.
    """
    try:
        runtime.catalog = await runtime.upstream.list_models(
            runtime.api_keys[0].value
        )
    except UpstreamError as exc:
        message = f"could not fetch the NIM catalog: {exc}"
        runtime.catalog_problems.append(message)
        log.warning("%s - model IDs are unverified", message)
        return

    available = set(runtime.catalog)
    for model in runtime.config.all_models():
        if model in available:
            continue
        suggestion = difflib.get_close_matches(model, runtime.catalog, n=1, cutoff=0.4)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        problem = f"configured model {model!r} is not in the NIM catalog.{hint}"
        runtime.catalog_problems.append(problem)
        log.warning("%s", problem)

    if not runtime.catalog_problems:
        log.info("all %d configured models verified against the NIM catalog",
                 len(runtime.config.all_models()))


def create_app(settings: Settings | None = None) -> FastAPI:
    runtime = build_runtime(settings)
    logging.basicConfig(
        level=runtime.settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        log.info(
            "fusion %s | %d API key(s) | ~%d req/min per model | lanes: %s",
            __version__,
            runtime.governor.key_count,
            runtime.config.rate_limit.default_rpm * runtime.governor.key_count,
            ", ".join(sorted(runtime.config.lanes)),
        )
        for index, entry in enumerate(runtime.api_keys):
            log.info("  key[%d] %s from %s", index, entry.masked, entry.origin)
        await verify_catalog(runtime)
        yield
        await runtime.upstream.aclose()

    app = FastAPI(title="Fusion", version=__version__, lifespan=lifespan)
    app.state.runtime = runtime

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "keys": runtime.governor.key_count,
            "key_sources": [
                {"index": i, "key": e.masked, "from": e.origin}
                for i, e in enumerate(runtime.api_keys)
            ],
            "rpm_ceiling_per_model": (
                runtime.config.rate_limit.default_rpm * runtime.governor.key_count
            ),
            "catalog_problems": runtime.catalog_problems,
        }

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return {
            "routing": runtime.stats.snapshot(),
            "capacity": runtime.governor.snapshot(),
            "catalog_problems": runtime.catalog_problems,
        }

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        created = int(time.time())
        entries = [
            {
                "id": f"{VIRTUAL_PREFIX}{lane}",
                "object": "model",
                "created": created,
                "owned_by": "fusion",
            }
            for lane in sorted(runtime.config.lanes)
        ]
        entries += [
            {
                "id": model,
                "object": "model",
                "created": created,
                "owned_by": "nvidia-nim",
            }
            for model in sorted(runtime.config.all_models())
        ]
        return {"object": "list", "data": entries}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return _bad_request("request body is not valid JSON")
        if not isinstance(body, dict) or not body.get("messages"):
            return _bad_request("`messages` is required")

        lane, decision = await _route(runtime, body)
        record = runtime.stats.start(
            lane=lane,
            role=decision.role,
            planned=list(decision.candidates),
            signals=list(decision.signals),
            chain=decision.chain,
            fingerprint=decision.fingerprint,
        )
        log.info(
            "route lane=%s role=%s chain=%s -> %s [%s]",
            lane,
            decision.role,
            decision.chain,
            decision.primary,
            ", ".join(decision.signals),
        )

        if body.get("stream"):
            return StreamingResponse(
                runtime.orchestrator.stream(body, decision, record),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    **_route_headers(decision),
                },
            )

        try:
            response = await runtime.orchestrator.complete(body, decision, record)
        except UpstreamError as exc:
            return _upstream_error(exc)
        return JSONResponse(response, headers=_route_headers(decision))

    @app.post("/v1/messages")
    async def messages(request: Request):
        """Anthropic Messages, for Claude Code.

        Translated to the OpenAI shape on the way in and back on the way out, so
        every routing and failover behaviour is shared rather than duplicated.
        """
        try:
            body = await request.json()
        except ValueError:
            return _bad_request("request body is not valid JSON", anthropic_style=True)
        if not isinstance(body, dict) or not body.get("messages"):
            return _bad_request("`messages` is required", anthropic_style=True)

        requested = str(body.get("model") or "auto")
        converted = anthropic.to_openai_request(body)

        lane, decision = await _route(runtime, converted)
        record = runtime.stats.start(
            lane=lane,
            role=decision.role,
            planned=list(decision.candidates),
            signals=list(decision.signals),
            chain=decision.chain,
            fingerprint=decision.fingerprint,
        )
        log.info(
            "route (anthropic) lane=%s role=%s chain=%s -> %s [%s]",
            lane, decision.role, decision.chain, decision.primary,
            ", ".join(decision.signals),
        )

        if body.get("stream"):
            return StreamingResponse(
                anthropic.translate_stream(
                    runtime.orchestrator.stream(converted, decision, record),
                    requested,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    **_route_headers(decision),
                },
            )

        try:
            response = await runtime.orchestrator.complete(converted, decision, record)
        except UpstreamError as exc:
            return _upstream_error(exc, anthropic_style=True)
        return JSONResponse(
            anthropic.to_anthropic_response(response, requested),
            headers=_route_headers(decision),
        )

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        """Claude Code asks before it trims. An estimate is enough: it only ever
        decides *when* to compact, and being slightly high is the safe side."""
        try:
            body = await request.json()
        except ValueError:
            return _bad_request("request body is not valid JSON", anthropic_style=True)
        return JSONResponse({"input_tokens": anthropic.estimate_tokens(body)})

    return app


async def _route(runtime: Runtime, body: dict) -> tuple[str, RouteDecision]:
    """Resolve the requested virtual model, then classify if the rules ask."""
    requested = str(body.get("model") or "auto")
    lane = requested[len(VIRTUAL_PREFIX):] if requested.startswith(VIRTUAL_PREFIX) else requested

    # Escape hatch: name a real NIM model and it is used verbatim, still with
    # key rotation and failover, just no routing.
    if lane not in runtime.config.lanes and lane in set(runtime.config.all_models()):
        return lane, RouteDecision(
            role="direct",
            candidates=(lane,),
            chain=False,
            signals=("direct_model",),
            fingerprint=fingerprint(body),
        )

    if lane not in runtime.config.lanes:
        log.debug("unknown model %r; falling back to the auto lane", requested)
        lane = "auto"

    decision = runtime.coordinator.decide(body, lane=lane)
    if decision.needs_classification:
        role, signal = await classify(
            runtime.config, runtime.governor, runtime.upstream, extract_facts(body)
        )
        if role is not None:
            decision = runtime.coordinator.with_role(decision, role, signal)
        else:
            decision = runtime.coordinator.with_role(
                decision, decision.role, signal
            )
    return lane, decision


def _route_headers(decision: RouteDecision) -> dict[str, str]:
    """Which role and model served this, on every response.

    Previously only streaming carried these, so a non-streaming request was
    unexplainable without cross-referencing /stats by timestamp.
    """
    return {"X-Fusion-Role": decision.role, "X-Fusion-Model": decision.primary}


def _error_body(message: str, kind: str, *, anthropic_style: bool) -> dict:
    """Both dialects nest under `error`, but Anthropic tags the envelope too and
    uses its own type names - a client that validates will reject the other."""
    if anthropic_style:
        return {"type": "error", "error": {"type": kind, "message": message}}
    return {"error": {"message": message, "type": kind}}


def _bad_request(message: str, *, anthropic_style: bool = False) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=_error_body(message, "invalid_request_error",
                            anthropic_style=anthropic_style),
    )


def _upstream_error(
    exc: UpstreamError, *, anthropic_style: bool = False
) -> JSONResponse:
    status = exc.status if exc.status and exc.status >= 400 else 502
    kind = "api_error" if anthropic_style else "upstream_error"
    return JSONResponse(
        status_code=status,
        content=_error_body(
            f"fusion: all upstream attempts failed - {exc.message}",
            kind,
            anthropic_style=anthropic_style,
        ),
    )
