"""Thin async client for the NVIDIA NIM OpenAI-compatible endpoint."""

from __future__ import annotations

import contextlib
import logging
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx

from .config import FusionConfig
from .governor import Attempt

log = logging.getLogger("fusion.upstream")


class UpstreamError(Exception):
    """A failed upstream call, classified enough for the governor to react."""

    def __init__(
        self,
        status: int | None,
        message: str,
        *,
        retry_after: float | None = None,
        timed_out: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after
        self.timed_out = timed_out

    @property
    def is_rate_limit(self) -> bool:
        return self.status == 429

    @property
    def is_capacity(self) -> bool:
        """Upstream has no room right now - back off, don't blame the model.

        NIM answers 529 "Service temporarily overloaded" when a model's shared
        pool is saturated, which is far more common on the free tier than a
        real 429. Treating it as a generic failure would trip the circuit
        breaker on a model that is merely busy; a cooldown is the right
        response, exactly as for a rate limit.

        A stall counts too. A model queued behind other users accepts the
        request and then says nothing - the same condition, reported by
        silence instead of a status code.
        """
        return self.timed_out or self.status in (429, 503, 529)

    @property
    def is_model_unavailable(self) -> bool:
        """404 names the *model*, not the request.

        NIM answers 404 "Not found for account" for models that are listed in
        the catalog but not callable with your key, and occasionally for a
        model whose backend is momentarily unrouteable. Either way the fix is a
        different model, so this must not be lumped in with a malformed request.
        """
        return self.status == 404

    @property
    def is_retryable(self) -> bool:
        # Other 4xx means the request itself is wrong; resending it to a
        # different model would just fail again more slowly.
        if self.status is None:
            return True  # transport error
        if self.status in (404, 408, 429):
            return True
        return self.status >= 500


def parse_retry_after(value: str | None) -> float | None:
    """Retry-After is either delta-seconds or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _prompt_chars(body: dict[str, Any]) -> int:
    """Size of the prompt, cheaply. Serialising a 200k-token body to measure it
    would cost more than the check is worth."""
    total = 0
    for message in body.get("messages") or ():
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"])
    return total


class UpstreamClient:
    def __init__(self, config: FusionConfig) -> None:
        self._config = config
        self._base = config.upstream.base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                config.upstream.timeout_seconds,
                connect=config.upstream.connect_timeout_seconds,
                read=config.upstream.stall_timeout_seconds,
            ),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def stall_budget(self, body: dict[str, Any]) -> float:
        """How long silence is allowed before we judge a model queued.

        Time-to-first-token grows with the prompt, because the model reads all
        of it before it can emit anything. A flat ceiling tuned on one-line
        probes therefore condemns every large request: a 200k-token payload
        cannot prefill in 30s, so all three racers time out and the request
        fails outright rather than merely slowly.
        """
        limits = self._config.upstream
        tokens = _prompt_chars(body) / 4
        return limits.stall_timeout_seconds + (
            tokens / 100_000 * limits.stall_timeout_per_100k_tokens
        )

    def _timeout_for(self, body: dict[str, Any]) -> httpx.Timeout:
        """httpx's read timeout is max silence between *any* two bytes, so it
        cannot express "strict until the model starts, patient afterwards".

        Set it to the looser of the two and let the orchestrator enforce the
        strict half around the first frame, where it belongs. Using the stall
        budget here instead applied a race deadline to an answer already in
        flight and cut working models off mid-sentence.
        """
        limits = self._config.upstream
        return httpx.Timeout(
            limits.timeout_seconds,
            connect=limits.connect_timeout_seconds,
            read=max(limits.idle_timeout_seconds, self.stall_budget(body)),
        )

    @staticmethod
    def payload_for(attempt: Attempt, body: dict[str, Any], *, stream: bool) -> dict:
        """Rewrite the virtual model name to the real upstream one."""
        payload = dict(body)
        payload["model"] = attempt.model
        payload["stream"] = stream
        payload.pop("stream_options", None)
        return payload

    async def complete(self, attempt: Attempt, body: dict[str, Any]) -> dict[str, Any]:
        payload = self.payload_for(attempt, body, stream=False)
        try:
            response = await self._client.post(
                f"{self._base}/chat/completions",
                json=payload,
                headers=self._headers(attempt.api_key),
                timeout=self._timeout_for(body),
            )
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                None, f"no response: {type(exc).__name__}", timed_out=True
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(None, f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise self._error_from(response, await _safe_text(response))
        return response.json()

    async def open_stream(
        self, attempt: Attempt, body: dict[str, Any]
    ) -> httpx.Response:
        """Send and return the open response. The caller must close it.

        Separate from `stream()` so several of these can be raced against each
        other, with the losers closed explicitly.
        """
        payload = self.payload_for(attempt, body, stream=True)
        request = self._client.build_request(
            "POST",
            f"{self._base}/chat/completions",
            json=payload,
            headers={**self._headers(attempt.api_key), "Accept": "text/event-stream"},
            timeout=self._timeout_for(body),
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                None, f"no response: {type(exc).__name__}", timed_out=True
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(None, f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            text = await _safe_text(response)
            await response.aclose()
            raise self._error_from(response, text)
        return response

    @contextlib.asynccontextmanager
    async def stream(
        self, attempt: Attempt, body: dict[str, Any]
    ) -> AsyncIterator[httpx.Response]:
        response = await self.open_stream(attempt, body)
        try:
            yield response
        finally:
            await response.aclose()

    async def list_models(self, api_key: str) -> list[str]:
        try:
            response = await self._client.get(
                f"{self._base}/models",
                headers=self._headers(api_key),
                timeout=httpx.Timeout(30.0, connect=10.0),
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise UpstreamError(None, f"catalog fetch failed: {exc}") from exc

        data = response.json().get("data") or []
        return sorted(
            entry["id"] for entry in data if isinstance(entry, dict) and entry.get("id")
        )

    @staticmethod
    def _error_from(response: httpx.Response, text: str) -> UpstreamError:
        return UpstreamError(
            response.status_code,
            f"HTTP {response.status_code}: {text[:400]}",
            retry_after=parse_retry_after(response.headers.get("retry-after")),
        )


async def _safe_text(response: httpx.Response) -> str:
    try:
        if response.is_closed:
            return response.text
        return (await response.aread()).decode(errors="replace")
    except Exception:  # noqa: BLE001 - diagnostics must never mask the real error
        return "<unreadable body>"
