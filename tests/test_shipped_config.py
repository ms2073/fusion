"""Invariants for the real config.yaml.

Every value in the shipped config was wrong on the first pass - contexts guessed
rather than looked up, and a vision-capable model left out of the vision role.
These assertions make that class of mistake fail at test time instead of at
2am under load.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fusion.config import load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@pytest.fixture(scope="module")
def shipped():
    return load_config(CONFIG_PATH)


def test_it_loads(shipped):
    assert shipped.roles
    assert shipped.lanes


def test_every_role_model_has_an_explicit_entry(shipped):
    """A missing entry silently defaults the context, which is how you overflow."""
    declared = set(shipped.models)
    for role, cfg in shipped.roles.items():
        for model in cfg.models:
            assert model in declared, f"{role!r} uses {model!r} with no models: entry"


def test_every_eyes_model_can_actually_see(shipped):
    for model in shipped.models_for_role("eyes"):
        assert shipped.supports_vision(model), f"{model} cannot accept images"


def test_vision_requests_have_somewhere_to_fail_over_to(shipped):
    """A single-model role has no failover, which defeats the whole point."""
    assert len(shipped.models_for_role("eyes")) >= 2


def test_the_librarian_can_hold_what_gets_routed_to_it(shipped):
    """The long-context rule must not send a payload past every candidate."""
    biggest = max(shipped.context_for(m) for m in shipped.models_for_role("librarian"))
    assert biggest >= shipped.routing.long_context_tokens


def test_every_role_has_somewhere_to_fail_over_to(shipped):
    """A single-model role cannot fail over, which defeats the whole design."""
    for role, cfg in shipped.roles.items():
        if role == "verifier":  # disabled by default
            continue
        assert len(cfg.models) >= 2, f"{role} has no fallback"


def test_every_role_ends_in_a_reliably_available_model(shipped):
    """The free tier queues the frontier models; each role needs a real tail.

    `make probe` has seen all four primaries QUEUED or 529 at once. Without a
    dependable last entry, a busy hour turns into an error rather than a
    slightly weaker answer.
    """
    # Measured, not assumed: each of these answered a live probe while the
    # frontier models were queued. Re-run `make probe` before adding one.
    dependable = {
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        "nvidia/nemotron-nano-12b-v2-vl",
        "meta/llama-3.2-90b-vision-instruct",
    }
    for role, cfg in shipped.roles.items():
        assert cfg.models[-1] in dependable, f"{role} ends in a contended model"


def test_the_tail_survives_a_large_payload(shipped):
    """A reliability tail that the context filter removes is not a tail.

    Every text role ended in nemotron-3-super at 131k, so past that size each
    role silently collapsed to exactly the two most contended models in the
    pool - which between them answered 0 of 264 dispatches in one measured
    session. A long agentic conversation therefore did not degrade as its
    context grew; it stopped working altogether.
    """
    threshold = shipped.routing.long_context_tokens
    for role in ("thinker", "worker", "librarian"):
        models = shipped.models_for_role(role)
        usable = [m for m in models if shipped.context_for(m) >= threshold]
        assert len(usable) >= 3, f"{role} keeps only {usable} at {threshold} tokens"
        # The sharpest form: an understated context does not merely deprioritise
        # the tail, it deletes it. This is the assertion that catches the actual
        # mistake - nemotron-3-super was declared at 131072 against a real 1M.
        assert shipped.context_for(models[-1]) >= threshold, (
            f"{role}'s reliability tail {models[-1]} is filtered out at "
            f"{threshold} tokens - exactly when it is needed"
        )


def test_the_unavailable_model_stays_out(shipped):
    """kimi-k2.6 is in NIM's catalog but 404s on inference for this account."""
    for cfg in shipped.roles.values():
        assert "moonshotai/kimi-k2.6" not in cfg.models


def test_the_long_context_threshold_fits_the_primary_librarian(shipped):
    """Route to Kimi only inside the band Kimi can serve."""
    primary = shipped.models_for_role("librarian")[0]
    assert shipped.routing.long_context_tokens <= shipped.context_for(primary)


def test_roles_referenced_by_the_coordinator_all_exist(shipped):
    for role in ("thinker", "worker", "librarian", "eyes"):
        assert role in shipped.roles


def test_no_role_lists_a_model_twice(shipped):
    for role, cfg in shipped.roles.items():
        assert len(cfg.models) == len(set(cfg.models)), f"{role} repeats a model"


def test_contexts_are_plausible(shipped):
    """Guard against an off-by-1000 typo in either direction."""
    for model, cfg in shipped.models.items():
        assert 32_000 <= cfg.context <= 2_000_000, f"{model} context {cfg.context}"
