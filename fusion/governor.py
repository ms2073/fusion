"""Rate-limit governor: paces requests, rotates keys, and plans failover.

NIM allows roughly 40 requests per minute per model on a best-effort basis. The
governor's job is to bounce off our own limiter instead of collecting 429s, and
when that is not possible, to already know where to send the request next.

Every method here is synchronous and non-awaiting, so under FastAPI's single
event loop the state mutations are atomic. No locking required.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable

from .config import FusionConfig

DEFAULT_ATTEMPT_LIMIT = 6

_COUNTERS = ("dispatched", "succeeded", "rate_limited", "failed", "outraced")


@dataclass(frozen=True, slots=True)
class Attempt:
    """One dispatch target: a model paired with the key to send it under."""

    model: str
    key_index: int
    api_key: str

    @property
    def label(self) -> str:
        return f"{self.model}#key{self.key_index}"


class _Bucket:
    """Token bucket. Tokens may go negative - that just means a longer wait."""

    __slots__ = ("capacity", "rate", "_tokens", "_updated")

    def __init__(self, rpm: int, now: float) -> None:
        self.capacity = float(rpm)
        self.rate = rpm / 60.0
        self._tokens = float(rpm)
        self._updated = now

    def peek(self, now: float) -> float:
        elapsed = max(0.0, now - self._updated)
        return min(self.capacity, self._tokens + elapsed * self.rate)

    def consume(self, now: float, amount: float = 1.0) -> None:
        self._tokens = self.peek(now) - amount
        self._updated = now

    def drain(self, now: float) -> None:
        """Upstream said we are over budget; believe it over our own accounting."""
        self._tokens = min(self.peek(now), 0.0)
        self._updated = now


def _new_counters() -> dict[str, int]:
    return {name: 0 for name in _COUNTERS}


class Governor:
    def __init__(
        self,
        config: FusionConfig,
        api_keys: list[str],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_keys:
            raise ValueError(
                "no NIM API keys configured. Add at least one to your .env - any "
                "of: NIM_API_KEY=..., NIM_API_KEY_1=... (numbered, any count), "
                "NIM_API_KEYS=key1,key2, or NIM_API_KEYS_FILE=/path. "
                "Run `python -m fusion.keys` to see what is being picked up."
            )
        self._config = config
        self._keys = list(api_keys)
        self._clock = clock

        self._buckets: dict[tuple[str, int], _Bucket] = {}
        self._cooldown_until: dict[tuple[str, int], float] = {}
        self._cooldown_streak: dict[tuple[str, int], int] = {}
        self._breaker_failures: dict[str, int] = {}
        self._breaker_open_until: dict[str, float] = {}

        self._model_stats: dict[str, dict[str, int]] = defaultdict(_new_counters)
        self._key_stats: dict[int, dict[str, int]] = defaultdict(_new_counters)
        self._failovers: dict[str, int] = defaultdict(int)

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def attempt(self, model: str, key_index: int) -> Attempt:
        return Attempt(model, key_index, self._keys[key_index])

    # -- planning ---------------------------------------------------------

    def plan_attempts(
        self, models: Iterable[str], limit: int = DEFAULT_ATTEMPT_LIMIT
    ) -> list[Attempt]:
        """Ordered dispatch targets, best first. Never empty.

        Three tiers, in order:
          0. usable now, with headroom     - the normal path
          1. usable, but over our own pace - our pacing is self-imposed, and
                                             exceeding it beats failing outright
          2. cooling down from a 429, or behind an open breaker - last resort,
             soonest-to-recover first

        Within tiers 0 and 1, the caller's model preference order wins first and
        key headroom breaks the tie - which alternates keys on its own, and
        degrades more gracefully than round-robin when one key is already hot.
        """
        now = self._clock()
        ordered_models = list(dict.fromkeys(models))

        tier0: list[tuple[int, float, int, str]] = []
        tier1: list[tuple[int, float, int, str]] = []
        tier2: list[tuple[float, int, int, str]] = []

        for rank, model in enumerate(ordered_models):
            breaker_open = self._breaker_open(model, now)
            for key_index in range(len(self._keys)):
                cool_until = self._cooldown_until.get((model, key_index), 0.0)
                tokens = self._bucket(model, key_index).peek(now)
                if breaker_open or cool_until > now:
                    tier2.append((cool_until, rank, key_index, model))
                elif tokens >= 1.0:
                    tier0.append((rank, -tokens, key_index, model))
                else:
                    tier1.append((rank, -tokens, key_index, model))

        tier0.sort()
        tier1.sort()
        tier2.sort()

        pairs = [(m, k) for _, _, k, m in tier0]
        pairs += [(m, k) for _, _, k, m in tier1]
        pairs += [(m, k) for _, _, k, m in tier2]

        # Truncating a flat preference list budgets models and keys out of one
        # pot, so adding a key evicts a model: three keys across two models is
        # already six attempts, which left the third model in a role - the
        # reliable tail every role ends in - unreachable. Keep the head in
        # preference order, then re-admit any model that got cut entirely.
        chosen = pairs[:limit]
        covered = {model for model, _ in chosen}
        for model, key_index in pairs[limit:]:
            if model not in covered:
                covered.add(model)
                chosen.append((model, key_index))
        return [self.attempt(m, k) for m, k in chosen]

    def is_available(self, model: str, key_index: int) -> bool:
        """True when nothing is actively blocking this pair (pacing aside)."""
        now = self._clock()
        if self._cooldown_until.get((model, key_index), 0.0) > now:
            return False
        return not self._breaker_open(model, now)

    def headroom(self, model: str, key_index: int) -> float:
        return self._bucket(model, key_index).peek(self._clock())

    # -- outcome recording ------------------------------------------------

    def record_dispatch(self, attempt: Attempt) -> None:
        self._bucket(attempt.model, attempt.key_index).consume(self._clock())
        self._bump(attempt, "dispatched")

    def record_success(self, attempt: Attempt) -> None:
        key = (attempt.model, attempt.key_index)
        self._cooldown_streak.pop(key, None)
        self._breaker_failures.pop(attempt.model, None)
        self._breaker_open_until.pop(attempt.model, None)
        self._bump(attempt, "succeeded")

    def record_rate_limited(
        self, attempt: Attempt, retry_after: float | None = None
    ) -> None:
        """A 429: *this key* is over quota. Other keys are still good."""
        self._cool(attempt.model, attempt.key_index, retry_after)
        self._bump(attempt, "rate_limited")

    def record_overloaded(
        self,
        attempt: Attempt,
        retry_after: float | None = None,
        base_seconds: float | None = None,
    ) -> None:
        """A 529 or a stall: the *model's* shared pool is saturated.

        Unlike a 429, this has nothing to do with which key was used, so
        cooling only one key would send the next attempt straight back into the
        same queue under a different name. Park the model across every key and
        let the caller move on to a different one.

        `base_seconds` starts the backoff somewhere other than the default -
        a stall says less than a 529 does, so it starts longer. It is still a
        starting point that escalates, unlike `retry_after`.
        """
        for key_index in range(len(self._keys)):
            self._cool(attempt.model, key_index, retry_after, base_seconds)
        self._bump(attempt, "rate_limited")

    def record_outraced(
        self, attempt: Attempt, base_seconds: float | None = None
    ) -> None:
        """This model was still silent when a different one answered.

        A contended model never errors - it accepts the request and queues, so
        the only signal it ever emits is losing. Discarding that (the loser is
        cancelled, and a cancelled attempt reports nothing) left the governor
        unable to learn from the single most common outcome on a busy tier, and
        the same queued model led the preference list into every request.

        Park it across every key, exactly like a stall: the queue is per-model
        and does not care whose key is asking. The backoff escalates while it
        keeps losing and resets the moment it answers, so a model that is
        merely busy right now returns on its own.
        """
        for key_index in range(len(self._keys)):
            self._cool(attempt.model, key_index, None, base_seconds)
        self._bump(attempt, "outraced")

    def _cool(
        self,
        model: str,
        key_index: int,
        retry_after: float | None,
        base_seconds: float | None = None,
    ) -> None:
        """Park one (model, key). `retry_after` is the server's instruction and
        is obeyed verbatim; everything else escalates on a streak.

        Keeping those apart is the whole point. Feeding an inferred delay in as
        a `Retry-After` makes it authoritative, and a model that fails the same
        way every time then draws the same short cooldown forever instead of
        backing off.
        """
        now = self._clock()
        key = (model, key_index)
        streak = self._cooldown_streak.get(key, 0) + 1
        self._cooldown_streak[key] = streak

        limits = self._config.rate_limit
        if retry_after is not None and retry_after > 0:
            delay = retry_after
        else:
            base = base_seconds or limits.cooldown_base_seconds
            delay = min(
                base * (2 ** (streak - 1)),
                max(limits.cooldown_max_seconds, base),
            )
        self._cooldown_until[key] = now + delay
        self._bucket(*key).drain(now)

    def record_unavailable(self, attempt: Attempt) -> None:
        """The model cannot be called at all. Cut it out without blaming the key.

        Distinct from a failure count: one 404 is already conclusive, so there
        is nothing to be learned from four more.
        """
        self._breaker_open_until[attempt.model] = (
            self._clock() + self._config.rate_limit.unavailable_seconds
        )
        self._bump(attempt, "failed")

    def record_failure(self, attempt: Attempt) -> None:
        """A non-429 upstream error. Enough in a row and we stop trying."""
        now = self._clock()
        limits = self._config.rate_limit
        failures = self._breaker_failures.get(attempt.model, 0) + 1
        self._breaker_failures[attempt.model] = failures
        if failures >= limits.breaker_failure_threshold:
            self._breaker_open_until[attempt.model] = now + limits.breaker_reset_seconds
        self._bump(attempt, "failed")

    def record_failover(self, from_model: str, to_model: str) -> None:
        self._failovers[f"{from_model}->{to_model}"] += 1

    # -- reporting --------------------------------------------------------

    def snapshot(self) -> dict:
        now = self._clock()
        cooling = [
            {
                "model": model,
                "key_index": key_index,
                "seconds_remaining": round(until - now, 1),
            }
            for (model, key_index), until in sorted(self._cooldown_until.items())
            if until > now
        ]
        return {
            "keys_configured": len(self._keys),
            "models": {m: dict(s) for m, s in sorted(self._model_stats.items())},
            "keys": {str(k): dict(s) for k, s in sorted(self._key_stats.items())},
            "failovers": dict(sorted(self._failovers.items())),
            "cooling_down": cooling,
            "breakers_open": [
                m for m in sorted(self._breaker_open_until) if self._breaker_open(m, now)
            ],
            "headroom": {
                f"{model}#key{key_index}": round(bucket.peek(now), 1)
                for (model, key_index), bucket in sorted(self._buckets.items())
            },
        }

    # -- internals --------------------------------------------------------

    def _bucket(self, model: str, key_index: int) -> _Bucket:
        key = (model, key_index)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(self._config.rpm_for(model), self._clock())
            self._buckets[key] = bucket
        return bucket

    def _breaker_open(self, model: str, now: float) -> bool:
        until = self._breaker_open_until.get(model)
        if until is None:
            return False
        if until > now:
            return True
        # Window elapsed: close it and forget the failures that opened it.
        self._breaker_open_until.pop(model, None)
        self._breaker_failures.pop(model, None)
        return False

    def _bump(self, attempt: Attempt, counter: str) -> None:
        self._model_stats[attempt.model][counter] += 1
        self._key_stats[attempt.key_index][counter] += 1
