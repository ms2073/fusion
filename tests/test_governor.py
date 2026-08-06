"""Governor: rate-limit pacing, key rotation, and failover.

These are the guarantees the whole proxy exists to provide, so they are tested
against a simulated clock rather than by sleeping.
"""

from __future__ import annotations

from fusion.governor import Governor

from .conftest import make_config


def drain(gov: Governor, model: str, key_index: int, times: int) -> None:
    """Spend `times` tokens from one (model, key) bucket."""
    for _ in range(times):
        gov.record_dispatch(gov.attempt(model, key_index))


class TestModelPreference:
    def test_primary_model_wins_when_it_has_headroom(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        attempts = gov.plan_attempts(config.models_for_role("worker"))

        assert attempts[0].model == "deepseek"

    def test_every_model_stays_reachable_as_keys_are_added(self, config, clock):
        """A flat attempt limit budgets models and keys out of one pot, so
        adding a key silently evicts a model: at three keys the first two
        models filled all six slots and the third - the reliable tail every
        role ends in - became undispatchable."""
        gov = Governor(config, ["key-a", "key-b", "key-c"], clock=clock)

        attempts = gov.plan_attempts(config.models_for_role("worker"))

        assert {a.model for a in attempts} == {"deepseek", "glm", "kimi"}
        assert attempts[0].model == "deepseek"  # preference order still leads

    def test_falls_back_to_next_model_when_primary_is_spent(
        self, config, keys, clock
    ):
        gov = Governor(config, keys, clock=clock)
        # rpm is 3 per key, and there are two keys.
        drain(gov, "deepseek", 0, 3)
        drain(gov, "deepseek", 1, 3)

        attempts = gov.plan_attempts(config.models_for_role("worker"))

        assert attempts[0].model == "glm"

    def test_spent_primary_is_still_offered_as_a_last_resort(
        self, config, keys, clock
    ):
        """Our pacing is self-imposed; exceeding it beats failing the request."""
        gov = Governor(config, keys, clock=clock)
        for model in ("deepseek", "glm", "kimi"):
            drain(gov, model, 0, 3)
            drain(gov, model, 1, 3)

        attempts = gov.plan_attempts(config.models_for_role("worker"))

        assert attempts, "governor must never return zero attempts"
        assert attempts[0].model == "deepseek"

    def test_buckets_refill_over_time(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)
        drain(gov, "deepseek", 0, 3)
        drain(gov, "deepseek", 1, 3)
        assert gov.plan_attempts(["deepseek", "glm"])[0].model == "glm"

        clock.advance(60)

        assert gov.plan_attempts(["deepseek", "glm"])[0].model == "deepseek"


class TestKeyRotation:
    def test_alternates_to_the_less_used_key(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        first = gov.plan_attempts(["deepseek"])[0]
        gov.record_dispatch(first)
        second = gov.plan_attempts(["deepseek"])[0]

        assert second.key_index != first.key_index

    def test_attempt_carries_the_matching_secret(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        attempt = gov.plan_attempts(["deepseek"])[0]

        assert attempt.api_key == keys[attempt.key_index]

    def test_single_key_setup_still_works(self, config, clock):
        gov = Governor(config, ["only-key"], clock=clock)

        attempts = gov.plan_attempts(["deepseek", "glm"])

        assert attempts[0].key_index == 0
        assert {a.key_index for a in attempts} == {0}


class TestRateLimitResponses:
    def test_429_moves_to_the_other_key_before_changing_model(
        self, config, keys, clock
    ):
        gov = Governor(config, keys, clock=clock)

        gov.record_rate_limited(gov.attempt("deepseek", 0))
        attempts = gov.plan_attempts(config.models_for_role("worker"))

        assert attempts[0].model == "deepseek"
        assert attempts[0].key_index == 1

    def test_changes_model_once_every_key_is_cooling_down(
        self, config, keys, clock
    ):
        gov = Governor(config, keys, clock=clock)

        gov.record_rate_limited(gov.attempt("deepseek", 0))
        gov.record_rate_limited(gov.attempt("deepseek", 1))
        attempts = gov.plan_attempts(config.models_for_role("worker"))

        assert attempts[0].model == "glm"

    def test_retry_after_header_overrides_the_default_backoff(
        self, config, keys, clock
    ):
        gov = Governor(config, keys, clock=clock)

        gov.record_rate_limited(gov.attempt("deepseek", 0), retry_after=45.0)

        clock.advance(44)
        assert not gov.is_available("deepseek", 0)
        clock.advance(2)
        assert gov.is_available("deepseek", 0)

    def test_cooldown_expires_and_the_key_returns(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)
        gov.record_rate_limited(gov.attempt("deepseek", 0))
        assert not gov.is_available("deepseek", 0)

        clock.advance(11)  # cooldown_base_seconds is 10

        assert gov.is_available("deepseek", 0)

    def test_a_429_does_not_touch_the_other_key(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_rate_limited(gov.attempt("deepseek", 0))

        assert gov.is_available("deepseek", 1)

    def test_repeated_429s_back_off_exponentially(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_rate_limited(gov.attempt("deepseek", 0))  # 10s
        clock.advance(11)
        gov.record_rate_limited(gov.attempt("deepseek", 0))  # now 20s, not 10s

        clock.advance(11)
        assert not gov.is_available("deepseek", 0)
        clock.advance(10)
        assert gov.is_available("deepseek", 0)

    def test_backoff_is_capped(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        for _ in range(10):  # unbounded doubling would be ~85 minutes
            gov.record_rate_limited(gov.attempt("deepseek", 0))

        clock.advance(81)  # cooldown_max_seconds is 80
        assert gov.is_available("deepseek", 0)

    def test_success_clears_the_backoff_streak(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)
        gov.record_rate_limited(gov.attempt("deepseek", 0))
        clock.advance(11)
        gov.record_success(gov.attempt("deepseek", 0))

        gov.record_rate_limited(gov.attempt("deepseek", 0))

        clock.advance(11)  # back to the 10s base, not an escalated 20s
        assert gov.is_available("deepseek", 0)


class TestStalls:
    """A stall is *our* inference that a model is queued, not a server
    instruction. Passing it through as a Retry-After made it authoritative,
    which short-circuited the streak: glm-5.2 stalled 131 times in one session
    and drew the same flat cooldown every time, so it was re-tried, and paid
    for, on essentially every request."""

    def test_repeated_stalls_escalate(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_overloaded(gov.attempt("deepseek", 0), base_seconds=20)
        clock.advance(21)
        gov.record_overloaded(gov.attempt("deepseek", 0), base_seconds=20)

        clock.advance(21)
        assert not gov.is_available("deepseek", 0)  # 40s now, not another 20s
        clock.advance(20)
        assert gov.is_available("deepseek", 0)

    def test_a_stall_parks_every_key(self, config, keys, clock):
        """The queue is per-model; another key rejoins the same one."""
        gov = Governor(config, keys, clock=clock)

        gov.record_overloaded(gov.attempt("deepseek", 0), base_seconds=20)

        assert not gov.is_available("deepseek", 1)

    def test_a_real_retry_after_is_still_obeyed_verbatim(
        self, config, keys, clock
    ):
        """A server-supplied delay is authoritative - never escalated past it."""
        gov = Governor(config, keys, clock=clock)

        gov.record_overloaded(gov.attempt("deepseek", 0), retry_after=5.0)
        gov.record_overloaded(gov.attempt("deepseek", 0), retry_after=5.0)

        clock.advance(6)
        assert gov.is_available("deepseek", 0)


class TestOutraced:
    """Losing a race is evidence, and it was being thrown away.

    A contended model is not broken, so it never errors - it just sits in the
    queue until a hedge beats it and gets cancelled. Cancelled attempts record
    nothing, so the governor could not learn: glm-5.2 lost 131 races in one
    session and led the preference list into every one of them.
    """

    def test_being_outraced_parks_the_model(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_outraced(gov.attempt("deepseek", 0), base_seconds=20)

        assert not gov.is_available("deepseek", 0)
        assert not gov.is_available("deepseek", 1)  # the queue is per-model

    def test_losing_repeatedly_backs_off_further(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_outraced(gov.attempt("deepseek", 0), base_seconds=20)
        clock.advance(21)
        gov.record_outraced(gov.attempt("deepseek", 0), base_seconds=20)

        clock.advance(21)
        assert not gov.is_available("deepseek", 0)

    def test_it_always_comes_back(self, config, keys, clock):
        """Parked, never blacklisted: contention is temporary, so the model has
        to be re-offered or it can never prove it recovered."""
        gov = Governor(config, keys, clock=clock)

        for _ in range(20):
            gov.record_outraced(gov.attempt("deepseek", 0), base_seconds=20)

        clock.advance(config.rate_limit.cooldown_max_seconds + 1)
        assert gov.is_available("deepseek", 0)

    def test_answering_again_restores_it_immediately(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)
        for _ in range(5):
            gov.record_outraced(gov.attempt("deepseek", 0), base_seconds=20)
        clock.advance(config.rate_limit.cooldown_max_seconds + 1)

        gov.record_success(gov.attempt("deepseek", 0))
        gov.record_outraced(gov.attempt("deepseek", 0), base_seconds=20)

        clock.advance(21)  # back to the base, not the escalated ceiling
        assert gov.is_available("deepseek", 0)

    def test_it_is_counted_apart_from_rate_limiting(self, config, keys, clock):
        """Conflating the two would hide which problem you actually have."""
        gov = Governor(config, keys, clock=clock)

        gov.record_outraced(gov.attempt("deepseek", 0), base_seconds=20)

        stats = gov.snapshot()["models"]["deepseek"]
        assert stats["outraced"] == 1
        assert stats["rate_limited"] == 0
        assert stats["failed"] == 0


class TestCircuitBreaker:
    def test_opens_after_consecutive_failures(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        for _ in range(3):  # breaker_failure_threshold
            gov.record_failure(gov.attempt("deepseek", 0))

        assert gov.plan_attempts(["deepseek", "glm"])[0].model == "glm"

    def test_a_success_resets_the_failure_count(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_failure(gov.attempt("deepseek", 0))
        gov.record_failure(gov.attempt("deepseek", 0))
        gov.record_success(gov.attempt("deepseek", 0))
        gov.record_failure(gov.attempt("deepseek", 0))

        assert gov.plan_attempts(["deepseek", "glm"])[0].model == "deepseek"

    def test_closes_again_after_the_reset_window(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)
        for _ in range(3):
            gov.record_failure(gov.attempt("deepseek", 0))

        clock.advance(61)  # breaker_reset_seconds is 60

        assert gov.plan_attempts(["deepseek", "glm"])[0].model == "deepseek"


class TestAttemptPlan:
    def test_attempts_are_unique_and_bounded(self, config, keys, clock):
        """The limit bounds key fan-out, not model coverage: it caps the
        preference head, then every model still gets one way in."""
        gov = Governor(config, keys, clock=clock)

        attempts = gov.plan_attempts(["deepseek", "glm", "kimi"], limit=4)

        pairs = [(a.model, a.key_index) for a in attempts]
        assert len(set(pairs)) == len(pairs)
        assert len(attempts) == 5  # 4 preferred + kimi, which the cap had cut
        assert [a.model for a in attempts[:4]] == ["deepseek", "deepseek",
                                                   "glm", "glm"]
        assert attempts[4].model == "kimi"

    def test_unknown_models_are_skipped_not_fatal(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        attempts = gov.plan_attempts(["ghost", "glm"])

        assert attempts[0].model in {"ghost", "glm"}
        assert any(a.model == "glm" for a in attempts)


class TestSnapshot:
    def test_counts_are_reported_per_model_and_key(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_dispatch(gov.attempt("deepseek", 0))
        gov.record_success(gov.attempt("deepseek", 0))
        gov.record_rate_limited(gov.attempt("deepseek", 1))
        gov.record_failover("deepseek", "glm")

        snap = gov.snapshot()

        assert snap["models"]["deepseek"]["dispatched"] == 1
        assert snap["models"]["deepseek"]["succeeded"] == 1
        assert snap["models"]["deepseek"]["rate_limited"] == 1
        assert snap["keys"]["1"]["rate_limited"] == 1
        assert snap["failovers"]["deepseek->glm"] == 1

    def test_cooldowns_are_visible_while_active(self, config, keys, clock):
        gov = Governor(config, keys, clock=clock)

        gov.record_rate_limited(gov.attempt("deepseek", 0))

        cooling = gov.snapshot()["cooling_down"]
        assert any(c["model"] == "deepseek" and c["key_index"] == 0 for c in cooling)


class TestZeroKeys:
    def test_missing_keys_raise_at_construction(self, config, clock):
        import pytest

        with pytest.raises(ValueError, match="no NIM API keys"):
            Governor(config, [], clock=clock)


class TestPerModelRpm:
    def test_model_specific_rpm_is_respected(self, keys, clock):
        config = make_config(models={"deepseek": {"rpm": 1}, "glm": {}})
        gov = Governor(config, keys, clock=clock)

        drain(gov, "deepseek", 0, 1)
        drain(gov, "deepseek", 1, 1)

        assert gov.plan_attempts(["deepseek", "glm"])[0].model == "glm"
