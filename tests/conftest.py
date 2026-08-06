from __future__ import annotations

import pytest

from fusion.config import FusionConfig


class FakeClock:
    """Monotonic clock we control, so cooldown tests never sleep."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def make_config(**overrides) -> FusionConfig:
    """A miniature version of the shipped config.yaml.

    RPM is deliberately tiny so exhaustion tests stay readable.
    """
    base = {
        "upstream": {"base_url": "https://nim.test/v1"},
        "rate_limit": {
            "default_rpm": 3,
            "cooldown_base_seconds": 10.0,
            "cooldown_max_seconds": 80.0,
            "breaker_failure_threshold": 3,
            "breaker_reset_seconds": 60.0,
        },
        "routing": {
            "default_role": "thinker",
            "long_context_tokens": 200_000,
            "chain_enabled": True,
            "chain_plan_max_tokens": 900,
            "verifier_enabled": False,
            "ambiguous_escalation": False,
        },
        "classifier": {"model": "small/classifier", "max_tokens": 4},
        "session": {"ttl_seconds": 3600.0, "max_entries": 8},
        "roles": {
            "thinker": {"models": ["glm", "kimi", "deepseek"]},
            "worker": {"models": ["deepseek", "glm", "kimi"]},
            "librarian": {"models": ["kimi", "glm"]},
            "eyes": {"models": ["minimax"]},
            "verifier": {"models": ["glm"]},
        },
        "models": {
            "glm": {"context": 200_000},
            "kimi": {"context": 256_000},
            "deepseek": {"context": 1_000_000},
            "minimax": {"context": 200_000, "vision": True},
        },
        "lanes": {
            "auto": None,
            "think": "thinker",
            "code": "worker",
            "context": "librarian",
            "vision": "eyes",
        },
    }
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(base.get(section), dict):
            base[section] = {**base[section], **values}
        else:
            base[section] = values
    return FusionConfig.model_validate(base)


@pytest.fixture
def config() -> FusionConfig:
    return make_config()


@pytest.fixture
def keys() -> list[str]:
    return ["key-a", "key-b"]
