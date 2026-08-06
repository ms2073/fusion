"""Does racing a model on two keys help, or do the keys fight each other?

The previous test showed the keys diverging ~2x within every round, with the
winner alternating - which is either (a) independent queue slots, in which case
racing both is a real win, or (b) one account-wide queue in which two
simultaneous requests serialize and you simply wait behind yourself.

Those demand opposite designs, so measure directly. Each round runs three
trials back to back:

    solo0    key0 alone
    solo1    key1 alone
    paired   both at once, take the faster

If paired-min beats both solos, the keys are independent - race them.
If paired-min is no better than the best solo, they interfere - don't.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time

import httpx

sys.path.insert(0, ".")
from fusion.config import load_api_key_entries, load_config  # noqa: E402

CAP = 30.0
ROUNDS = 5
SETTLE = 4          # let the pool relax between trials
TARGET = "deepseek-ai/deepseek-v4-pro"


async def probe(client, base, key):
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
                return None
            async for line in response.aiter_lines():
                if line.strip():
                    return time.monotonic() - started
            return None
    except (httpx.TimeoutException, httpx.HTTPError):
        return None


def summarise(name, values):
    got = [v for v in values if v is not None]
    if not got:
        return f"   {name:<8} never answered"
    return (f"   {name:<8} median {statistics.median(got):6.2f}s   "
            f"best {min(got):5.2f}s   {len(got)}/{len(values)} answered")


async def main() -> None:
    entries = load_api_key_entries()
    if len(entries) < 2:
        print(f"Need 2+ keys, found {len(entries)}.")
        raise SystemExit(1)
    base = load_config("config.yaml").upstream.base_url.rstrip("/")
    k0, k1 = entries[0].value, entries[1].value

    solo0: list[float | None] = []
    solo1: list[float | None] = []
    paired: list[float | None] = []

    print(f"Target: {TARGET}\n")
    async with httpx.AsyncClient(timeout=httpx.Timeout(CAP, connect=10)) as client:
        for rnd in range(1, ROUNDS + 1):
            a = await probe(client, base, k0)
            await asyncio.sleep(SETTLE)
            b = await probe(client, base, k1)
            await asyncio.sleep(SETTLE)
            both = await asyncio.gather(probe(client, base, k0), probe(client, base, k1))
            got = [t for t in both if t is not None]
            m = min(got) if got else None
            solo0.append(a)
            solo1.append(b)
            paired.append(m)

            def fmt(v):
                return f"{v:6.2f}s" if v is not None else "   ---"
            print(f"  round {rnd}:  solo0 {fmt(a)}   solo1 {fmt(b)}   "
                  f"paired {fmt(m)}   (both: "
                  f"{', '.join(fmt(t) for t in both)})")
            if rnd < ROUNDS:
                await asyncio.sleep(SETTLE)

    print()
    print(summarise("solo0", solo0))
    print(summarise("solo1", solo1))
    print(summarise("paired", paired))

    solos = [v for v in solo0 + solo1 if v is not None]
    pairs = [v for v in paired if v is not None]
    print("\n=== verdict ===")
    if not pairs or not solos:
        print("   inconclusive - not enough answers")
        return
    best_solo = statistics.median(solos)
    pair_med = statistics.median(pairs)
    gain = best_solo / pair_med if pair_med else 1.0
    print(f"   median solo {best_solo:.2f}s  vs  median paired {pair_med:.2f}s"
          f"   ({gain:.2f}x)")
    reliability = len(pairs) / len(paired)
    solo_reliability = len([v for v in solo0 if v is not None]) / len(solo0)
    print(f"   answered: solo0 {solo_reliability:.0%}  paired {reliability:.0%}")
    if gain >= 1.25 or reliability > solo_reliability:
        print("   RACE THE KEYS - pairing is faster and/or more reliable.")
    elif gain <= 0.9:
        print("   DO NOT RACE - the keys interfere; you queue behind yourself.")
    else:
        print("   NEUTRAL - pairing neither helps nor hurts latency materially.")


asyncio.run(main())
