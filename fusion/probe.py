"""Measure what your key can actually call: `python -m fusion.probe`.

The startup catalog check asks NIM what models *exist*. That is not the same
question as what your account may *invoke* - a model can sit in /v1/models and
still answer 404 "Not found for account" - nor whether it will answer in a
useful amount of time. Free-tier capacity for the frontier models is shared and
varies hour to hour.

This sends one 1-token request per configured model and reports the truth.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from .config import Settings, load_api_key_entries, load_config

PROBE_TIMEOUT = 45.0
CONCURRENCY = 4


@dataclass
class Result:
    model: str
    status: int | None
    ttft: float | None
    note: str

    @property
    def verdict(self) -> str:
        if self.status == 404:
            return "NOT ENTITLED"
        if self.status in (429, 503, 529):
            return "BUSY"
        if self.status is None:
            return "QUEUED"
        if self.status >= 400:
            return f"HTTP {self.status}"
        if self.ttft is None:
            return "EMPTY"
        if self.ttft < 3:
            return "FAST"
        return "OK" if self.ttft < 20 else "SLOW"

    @property
    def usable(self) -> bool:
        return self.verdict in {"FAST", "OK", "SLOW"}


async def _probe(client: httpx.AsyncClient, base: str, key: str, model: str) -> Result:
    started = time.monotonic()
    payload = {
        "model": model,
        "stream": True,
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    try:
        async with client.stream(
            "POST", f"{base}/chat/completions", json=payload, headers=headers
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode(errors="replace")
                return Result(model, response.status_code, None, body[:90])
            async for line in response.aiter_lines():
                if line.strip():
                    return Result(model, 200, time.monotonic() - started, "")
            return Result(model, 200, None, "stream closed with no data")
    except httpx.TimeoutException:
        return Result(model, None, None, f"silent for {PROBE_TIMEOUT:.0f}s")
    except httpx.HTTPError as exc:
        return Result(model, None, None, f"{type(exc).__name__}: {exc}")


async def _run() -> int:
    settings = Settings()
    config = load_config(settings.config_path)
    entries = load_api_key_entries()
    if not entries:
        print("No API keys found. Run `python -m fusion.keys` for help.")
        return 1

    models = config.all_models()
    print(f"Probing {len(models)} models with 1 token each...\n")

    limiter = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(PROBE_TIMEOUT, connect=10.0)
    ) as client:

        async def guarded(model: str) -> Result:
            async with limiter:
                return await _probe(
                    client, config.upstream.base_url.rstrip("/"),
                    entries[0].value, model,
                )

        results = await asyncio.gather(*(guarded(m) for m in models))

    width = max(len(r.model) for r in results)
    for result in sorted(results, key=lambda r: (not r.usable, r.ttft or 999)):
        timing = f"{result.ttft:6.1f}s" if result.ttft is not None else "     -"
        print(f"  {result.model:<{width}}  {result.verdict:<13} {timing}  {result.note}")

    print()
    _advise(config, results)
    return 0 if any(r.usable for r in results) else 1


def _advise(config, results: list[Result]) -> None:
    by_model = {r.model: r for r in results}
    broken = [r.model for r in results if r.verdict == "NOT ENTITLED"]
    if broken:
        print("Not available on your account - remove these from config.yaml:")
        for model in broken:
            print(f"  {model}")
        print()

    for role in config.roles:
        candidates = config.models_for_role(role)
        ranked = [m for m in candidates if by_model.get(m) and by_model[m].usable]
        if not ranked:
            print(f"  {role:<10} NOTHING USABLE right now ({', '.join(candidates)})")
            continue
        ranked.sort(key=lambda m: by_model[m].ttft or 999)
        current = candidates[0]
        if ranked[0] != current:
            print(
                f"  {role:<10} consider promoting {ranked[0]} "
                f"({by_model[ranked[0]].ttft:.1f}s) over {current} "
                f"({by_model[current].verdict})"
            )
        else:
            print(f"  {role:<10} {current} is the best available")


def main() -> None:
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
