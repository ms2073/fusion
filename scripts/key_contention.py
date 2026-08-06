"""Is NIM's queue per-key or global? `python scripts/key_contention.py`

This decides how load balancing must work, and the two answers are opposites:

  per-key  - two keys are two independent queue slots. Racing the *same* model
             on both keys is then a real win against contention, and the
             orchestrator's rule of hedging only across models is leaving
             throughput on the table.
  global   - keys share one saturated pool. Extra keys buy 429 headroom and
             nothing else, and hedging across models stays the only answer.

Method: hit one contended model with every key simultaneously, repeatedly. If
outcomes within a round disagree (one answers, another stalls), the queue is
per-key. If they move together, it is global.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

import httpx

sys.path.insert(0, ".")
from fusion.config import load_api_key_entries, load_config  # noqa: E402

CAP = 25.0
ROUNDS = 5
GAP = 12
# Contended enough to queue, but not hopeless - glm-5.2 never answers at all,
# which would tell us nothing about whether keys are independent.
TARGET = "deepseek-ai/deepseek-v4-pro"


async def probe(client, base, key, label):
    started = time.monotonic()
    try:
        async with client.stream(
            "POST",
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": TARGET,
                "stream": True,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "count to three"}],
            },
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                return label, None, f"HTTP {response.status_code}"
            async for line in response.aiter_lines():
                if line.strip():
                    return label, time.monotonic() - started, "ok"
            return label, None, "empty"
    except httpx.TimeoutException:
        return label, None, f"silent>{CAP:.0f}s"
    except httpx.HTTPError as exc:
        return label, None, type(exc).__name__


async def main() -> None:
    entries = load_api_key_entries()
    if len(entries) < 2:
        print(f"Need 2+ keys, found {len(entries)}. Add NIM_API_KEY_2 to .env.")
        raise SystemExit(1)

    base = load_config("config.yaml").upstream.base_url.rstrip("/")
    print(f"Target: {TARGET}\nKeys:   {len(entries)}\n")

    disagreements = 0
    latencies: dict[str, list[float]] = {f"key{i}": [] for i in range(len(entries))}

    async with httpx.AsyncClient(timeout=httpx.Timeout(CAP, connect=10)) as client:
        for rnd in range(1, ROUNDS + 1):
            t0 = time.monotonic()
            results = await asyncio.gather(
                *(
                    probe(client, base, e.value, f"key{i}")
                    for i, e in enumerate(entries)
                )
            )
            answered = [r for r in results if r[1] is not None]
            print(f"--- round {rnd} ---")
            for label, ttft, note in results:
                print(f"   {label}  {f'{ttft:6.2f}s' if ttft else f'  {note}'}")
                if ttft:
                    latencies[label].append(ttft)
            if 0 < len(answered) < len(results):
                disagreements += 1
                print("   => DISAGREEMENT: keys are not sharing a queue")
            if rnd < ROUNDS:
                await asyncio.sleep(max(0, GAP - (time.monotonic() - t0)))

    print("\n=== per key ===")
    for label, times in latencies.items():
        if times:
            print(f"   {label}  {len(times)}/{ROUNDS} answered, "
                  f"median {statistics.median(times):.2f}s")
        else:
            print(f"   {label}  0/{ROUNDS} answered")

    print("\n=== verdict ===")
    if disagreements:
        print(f"   PER-KEY queues ({disagreements}/{ROUNDS} rounds disagreed).")
        print("   Two keys are two queue slots: racing the same model across")
        print("   keys is worth doing, so allow same-model hedges.")
    else:
        print("   GLOBAL queue - every key rose and fell together.")
        print("   Extra keys buy 429 headroom only. Keep hedging across models.")


asyncio.run(main())
