"""Time-to-first-token grows with the prompt, so the silence budget must too."""

from __future__ import annotations

from fusion.upstream import UpstreamClient

from .conftest import make_config


def _body(chars: int) -> dict:
    return {"messages": [{"role": "user", "content": "x" * chars}]}


class TestStallBudget:
    def test_a_short_prompt_uses_the_base_budget(self):
        client = UpstreamClient(make_config())

        assert 30.0 <= client.stall_budget(_body(100)) < 31.0

    def test_a_large_prompt_is_given_longer_to_prefill(self):
        """A 200k-token payload cannot be read in 30s. Holding it to the
        short-probe ceiling marked every large request as stalled, so the whole
        race timed out and the request failed outright - which is exactly how a
        long agentic session died."""
        client = UpstreamClient(make_config())

        budget = client.stall_budget(_body(200_000 * 4))

        assert budget > 100.0

    def test_the_budget_grows_with_the_prompt(self):
        client = UpstreamClient(make_config())

        assert client.stall_budget(_body(400_000)) > client.stall_budget(_body(4_000))

    def test_multipart_content_is_counted(self):
        client = UpstreamClient(make_config())
        multipart = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "y" * 400_000}]}
            ]
        }

        assert client.stall_budget(multipart) > client.stall_budget(_body(100))
