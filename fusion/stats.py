"""In-memory observability. Enough to tune the routing rules, nothing more."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RequestRecord:
    started_at: float
    lane: str
    role: str
    planned: list[str]
    signals: list[str]
    chain: bool
    fingerprint: str
    served_by: str | None = None
    plan_model: str | None = None
    status: str = "in_flight"
    detail: str | None = None
    latency_ms: int | None = None
    attempts: list[str] = field(default_factory=list)
    hedges: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "at": round(self.started_at, 3),
            "lane": self.lane,
            "role": self.role,
            "signals": self.signals,
            "chain": self.chain,
            "planned": self.planned,
            "plan_model": self.plan_model,
            "served_by": self.served_by,
            "attempts": self.attempts,
            "hedges": self.hedges,
            "status": self.status,
            "detail": self.detail,
            "latency_ms": self.latency_ms,
            "session": self.fingerprint,
        }


class Stats:
    def __init__(self, capacity: int = 100) -> None:
        self._recent: deque[RequestRecord] = deque(maxlen=capacity)
        self._latencies: deque[int] = deque(maxlen=500)
        self._roles: dict[str, int] = {}
        self._signals: dict[str, int] = {}
        self._totals = {"requests": 0, "chained": 0, "succeeded": 0, "failed": 0}

    def start(
        self,
        *,
        lane: str,
        role: str,
        planned: list[str],
        signals: list[str],
        chain: bool,
        fingerprint: str,
    ) -> RequestRecord:
        record = RequestRecord(
            started_at=time.time(),
            lane=lane,
            role=role,
            planned=planned,
            signals=signals,
            chain=chain,
            fingerprint=fingerprint,
        )
        self._recent.append(record)
        self._totals["requests"] += 1
        if chain:
            self._totals["chained"] += 1
        self._roles[role] = self._roles.get(role, 0) + 1
        for signal in signals:
            key = signal.split(":", 1)[0]
            self._signals[key] = self._signals.get(key, 0) + 1
        return record

    def finish(
        self, record: RequestRecord, *, status: str, detail: str | None = None
    ) -> None:
        record.status = status
        record.detail = detail
        record.latency_ms = int((time.time() - record.started_at) * 1000)
        self._latencies.append(record.latency_ms)
        if status == "ok":
            self._totals["succeeded"] += 1
        else:
            self._totals["failed"] += 1

    def snapshot(self) -> dict[str, Any]:
        latencies = sorted(self._latencies)
        return {
            "totals": dict(self._totals),
            "chain_rate": _ratio(self._totals["chained"], self._totals["requests"]),
            "roles": dict(sorted(self._roles.items())),
            "signals": dict(sorted(self._signals.items())),
            "latency_ms": {
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
                "samples": len(latencies),
            },
            "recent": [r.as_dict() for r in reversed(self._recent)],
        }


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 3) if whole else 0.0


def _percentile(ordered: list[int], fraction: float) -> int | None:
    if not ordered:
        return None
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]
