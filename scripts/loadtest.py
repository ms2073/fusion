"""Measure what the router actually does under concurrent load.

    python scripts/loadtest.py                 # 14 requests, 14 at once
    python scripts/loadtest.py -n 30 -c 10     # 30 requests, 10 in flight
    python scripts/loadtest.py --json          # machine-readable

Reports time-to-first-token and total duration as percentiles, how the work
spread across keys and models, and how many 429s were absorbed without the
client ever seeing one.

Two things this measures that a stopwatch cannot:

  absorbed 429s   the point of the governor is that upstream rate limiting
                  never reaches the caller. A run where the client saw zero
                  errors while `/stats` counted several 429s is the governor
                  working, and the only way to see it is to diff the counters
                  either side of the run.

  cost of hedging every model still sees roughly one dispatch per request,
                  because hedges fan out *across* models rather than retrying
                  one. Dispatches-per-request is printed for exactly this: if
                  it climbs toward the parallelism setting, the hedge delay is
                  below the pool's normal time-to-first-token and every request
                  is paying for racers it does not need.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter

import httpx

PROMPT = "In one sentence, what is a semaphore?"


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


async def one(client: httpx.AsyncClient, base: str, model: str,
              gate: asyncio.Semaphore) -> dict:
    """One streamed request. Times the first content byte separately.

    Total duration mixes routing with generation length, so it says little
    about the router. Time-to-first-token is the number the design moves.
    """
    async with gate:
        started = time.monotonic()
        first: float | None = None
        try:
            async with client.stream(
                "POST", f"{base}/chat/completions",
                json={"model": model, "stream": True, "max_tokens": 40,
                      "messages": [{"role": "user", "content": PROMPT}]},
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    return {"ok": False, "status": response.status_code}
                async for line in response.aiter_lines():
                    if not line.startswith("data:") or "[DONE]" in line:
                        continue
                    if first is None and '"content":"' in line.replace(
                            '"reasoning_content":"', '"content":"'):
                        if '"content":""' not in line:
                            first = time.monotonic() - started
                    # Deliberately not read from the stream: the router rewrites
                    # `model` to the lane's identity so the client sees one stable
                    # name, which means the field says "fusion/auto", not which
                    # upstream actually answered. That comes from /stats instead.
        except httpx.HTTPError as exc:
            return {"ok": False, "status": type(exc).__name__}
        return {"ok": True, "ttft": first, "total": time.monotonic() - started}


async def counters(base: str) -> dict:
    """Per-model tallies from /stats, so the run can be diffed against them."""
    stats_url = base.removesuffix("/v1") + "/stats"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            body = (await client.get(stats_url)).json()
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for model, row in ((body.get("capacity") or {}).get("models") or {}).items():
        out[model] = {k: row.get(k, 0) for k in
                      ("dispatched", "succeeded", "rate_limited", "outraced")}
    keys = {}
    for key, row in ((body.get("capacity") or {}).get("keys") or {}).items():
        keys[key] = row.get("dispatched", 0)
    return {"models": out, "keys": keys}


def delta(before: dict, after: dict, field: str) -> Counter:
    out: Counter = Counter()
    for model, row in (after.get("models") or {}).items():
        was = ((before.get("models") or {}).get(model) or {}).get(field, 0)
        if row.get(field, 0) - was:
            out[model] = row[field] - was
    return out


async def main() -> int:
    parser = argparse.ArgumentParser(description="Load-test the router.")
    parser.add_argument("-n", "--requests", type=int, default=14)
    parser.add_argument("-c", "--concurrency", type=int, default=0,
                        help="in flight at once (default: all)")
    parser.add_argument("-m", "--model", default="fusion/auto")
    parser.add_argument("-u", "--url", default="http://localhost:8787/v1")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    gate = asyncio.Semaphore(args.concurrency or args.requests)
    before = await counters(base)

    started = time.monotonic()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(600, connect=15),
        headers={"Authorization": "Bearer loadtest"},
    ) as client:
        results = await asyncio.gather(
            *(one(client, base, args.model, gate) for _ in range(args.requests))
        )
    wall = time.monotonic() - started
    after = await counters(base)

    ok = [r for r in results if r.get("ok")]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    totals = [r["total"] for r in ok]
    dispatched = delta(before, after, "dispatched")
    limited = delta(before, after, "rate_limited")
    succeeded = delta(before, after, "succeeded")
    keys = Counter({
        k: v - (before.get("keys") or {}).get(k, 0)
        for k, v in (after.get("keys") or {}).items()
        if v - (before.get("keys") or {}).get(k, 0)
    })

    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency or args.requests,
        "succeeded": len(ok),
        "wall_seconds": round(wall, 2),
        "ttft": {p: round(percentile(ttfts, p), 2) for p in (50, 95, 99)} if ttfts else {},
        "total": {p: round(percentile(totals, p), 2) for p in (50, 95, 99)} if totals else {},
        "dispatches_per_request": round(sum(dispatched.values()) / max(args.requests, 1), 2),
        "rate_limits_absorbed": sum(limited.values()),
        "rate_limits_surfaced": sum(1 for r in results if r.get("status") == 429),
        "dispatched_by_model": dict(dispatched),
        "answered_by_model": dict(succeeded),
        "dispatched_by_key": dict(keys),
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0 if len(ok) == args.requests else 1

    print(f"\n  {args.requests} requests, {summary['concurrency']} in flight, "
          f"model {args.model}")
    print(f"  {len(ok)}/{args.requests} succeeded in {wall:.1f}s wall\n")
    if ttfts:
        print(f"  time to first token   p50 {summary['ttft'][50]:>6.2f}s   "
              f"p95 {summary['ttft'][95]:>6.2f}s   p99 {summary['ttft'][99]:>6.2f}s")
    if totals:
        print(f"  total duration        p50 {summary['total'][50]:>6.2f}s   "
              f"p95 {summary['total'][95]:>6.2f}s   p99 {summary['total'][99]:>6.2f}s")

    print(f"\n  429s absorbed by the governor : {summary['rate_limits_absorbed']}")
    print(f"  429s that reached the client  : {summary['rate_limits_surfaced']}")
    print(f"  dispatches per request        : {summary['dispatches_per_request']} "
          f"(1.0 = no hedging cost)")

    if dispatched:
        print("\n  dispatched → answered, by model:")
        for model, count in dispatched.most_common():
            print(f"    {model:42} {count:>3} → {succeeded.get(model, 0)}")
    if keys:
        print("\n  spread across keys: "
              + ", ".join(f"key{k} {v}" for k, v in sorted(keys.items())))
    print()
    return 0 if len(ok) == args.requests else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
