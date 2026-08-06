"""Coordinator: request body in, specialist role out.

Bodies here are synthetic but shaped like the OpenAI chat-completions payloads
opencode sends. Step 2 of the plan replaces the important ones with captured
real traffic - especially the plan-mode system prompt, whose exact wording
opencode does not document.
"""

from __future__ import annotations

from fusion.coordinator import Coordinator, extract_facts, fingerprint

from .conftest import FakeClock, make_config


def tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "parameters": {}}}


CODING_TOOLS = [tool("read"), tool("edit"), tool("write"), tool("bash")]
READONLY_TOOLS = [tool("read"), tool("grep"), tool("glob")]


def body(
    user: str = "hello",
    *,
    system: str = "You are opencode.",
    tools: list | None = None,
    messages: list | None = None,
) -> dict:
    payload: dict = {
        "messages": messages
        or [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    }
    if tools is not None:
        payload["tools"] = tools
    return payload


def image_body(user: str = "make this look better") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are opencode."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
                ],
            },
        ],
        "tools": CODING_TOOLS,
    }


def tool_loop_body(user: str = "implement the parser") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are opencode."},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
        ],
        "tools": CODING_TOOLS,
    }


def resumed_body(user: str = "now add error handling") -> dict:
    """A finished tool loop, then the user speaks again - a new task."""
    payload = tool_loop_body()
    payload["messages"] = payload["messages"] + [
        {"role": "assistant", "content": "Done - the parser is in place."},
        {"role": "user", "content": user},
    ]
    return payload


class TestFactExtraction:
    def test_detects_images_in_multipart_content(self):
        assert extract_facts(image_body()).has_images

    def test_plain_text_request_has_no_images(self):
        assert not extract_facts(body()).has_images

    def test_detects_tool_result_messages(self):
        assert extract_facts(tool_loop_body()).mid_tool_loop

    def test_detects_assistant_tool_calls(self):
        payload = {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "c", "function": {"name": "read", "arguments": "{}"}}
                    ],
                },
            ]
        }
        assert extract_facts(payload).mid_tool_loop

    def test_first_turn_is_not_a_tool_loop(self):
        assert not extract_facts(body(tools=CODING_TOOLS)).mid_tool_loop

    def test_a_new_user_turn_closes_the_loop(self):
        """Not "has a tool ever been called". Every request after the first
        tool call carries that history forever, so the ever-used reading
        pinned a whole session to whatever its opening turn happened to be."""
        assert not extract_facts(resumed_body()).mid_tool_loop

    def test_write_tools_are_recognised(self):
        assert extract_facts(body(tools=CODING_TOOLS)).has_write_tools
        assert not extract_facts(body(tools=READONLY_TOOLS)).has_write_tools

    def test_last_user_message_wins(self):
        facts = extract_facts(
            body(
                messages=[
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "second"},
                ]
            )
        )
        assert facts.last_user_text == "second"
        assert facts.user_turns == 2

    def test_tool_schemas_count_toward_the_estimate(self):
        without = extract_facts(body()).estimated_tokens
        with_tools = extract_facts(body(tools=CODING_TOOLS)).estimated_tokens
        assert with_tools > without


class TestFingerprint:
    def test_stable_as_the_conversation_grows(self):
        first = body("implement the parser", tools=CODING_TOOLS)
        later = tool_loop_body("implement the parser")
        assert fingerprint(first) == fingerprint(later)

    def test_differs_between_conversations(self):
        assert fingerprint(body("task A")) != fingerprint(body("task B"))


class TestRoleSelection:
    def test_images_route_to_the_vision_specialist(self, config):
        decision = Coordinator(config).decide(image_body(), lane="auto")

        assert decision.role == "eyes"
        assert decision.primary == "minimax"
        assert "images_present" in decision.signals

    def test_coding_directive_routes_to_the_worker(self, config):
        decision = Coordinator(config).decide(
            body("fix the off-by-one in the tokenizer", tools=CODING_TOOLS),
            lane="auto",
        )

        assert decision.role == "worker"
        assert decision.primary == "deepseek"

    def test_planning_language_routes_to_the_thinker(self, config):
        decision = Coordinator(config).decide(
            body("what's the best approach to structure this module?",
                 tools=READONLY_TOOLS),
            lane="auto",
        )

        assert decision.role == "thinker"
        assert decision.primary == "glm"

    def test_oversized_payload_routes_to_the_librarian(self):
        config = make_config(routing={"long_context_tokens": 100})
        decision = Coordinator(config).decide(
            body("fix this bug " + "x" * 2000, tools=CODING_TOOLS), lane="auto"
        )

        assert decision.role == "librarian"
        assert decision.primary == "kimi"

    def test_a_bare_question_without_write_tools_is_not_coding(self, config):
        decision = Coordinator(config).decide(
            body("why does this function return None?", tools=READONLY_TOOLS),
            lane="auto",
        )

        assert decision.role == "thinker"


class TestExplicitLanes:
    def test_pinned_lane_is_honoured(self, config):
        decision = Coordinator(config).decide(
            body("what's the best approach here?", tools=READONLY_TOOLS), lane="code"
        )

        assert decision.role == "worker"
        assert "lane:code" in decision.signals

    def test_auto_lane_does_not_pin(self, config):
        decision = Coordinator(config).decide(
            body("fix the parser", tools=CODING_TOOLS), lane="auto"
        )

        assert not any(s.startswith("lane:") for s in decision.signals)


class TestChaining:
    def test_fresh_coding_task_plans_first(self, config):
        decision = Coordinator(config).decide(
            body("implement the retry logic", tools=CODING_TOOLS), lane="auto"
        )

        assert decision.chain is True

    def test_never_chains_mid_tool_loop(self, config):
        decision = Coordinator(config).decide(tool_loop_body(), lane="auto")

        assert decision.chain is False

    def test_questions_do_not_chain(self, config):
        decision = Coordinator(config).decide(
            body("how does this work?", tools=READONLY_TOOLS), lane="auto"
        )

        assert decision.chain is False

    def test_chaining_can_be_switched_off(self):
        config = make_config(routing={"chain_enabled": False})
        decision = Coordinator(config).decide(
            body("implement the retry logic", tools=CODING_TOOLS), lane="auto"
        )

        assert decision.role == "worker"
        assert decision.chain is False


class TestStickiness:
    def test_tool_loop_reuses_the_role_from_the_opening_turn(self, config):
        coordinator = Coordinator(config)
        first = coordinator.decide(
            body("implement the parser", tools=CODING_TOOLS), lane="auto"
        )

        later = coordinator.decide(tool_loop_body("implement the parser"), lane="auto")

        assert first.role == later.role == "worker"
        assert "mid_tool_loop:sticky" in later.signals

    def test_the_next_user_request_is_routed_afresh(self, config):
        """Stickiness exists to stop re-planning *during* a loop, not to freeze
        a session. A three-hour hermes run stayed on the opening turn's role
        with chain=False throughout, so no coding request ever reached the
        Worker or got a plan."""
        coordinator = Coordinator(config)
        coordinator.decide(body("what does this repo do?"), lane="auto")

        later = coordinator.decide(
            resumed_body("now implement the retry logic"), lane="auto"
        )

        assert later.role == "worker"
        assert later.chain is True
        assert not any(s.startswith("mid_tool_loop") for s in later.signals)

    def test_a_cold_proxy_still_answers_mid_loop(self, config):
        """Restarting the proxy mid-session must not break the loop."""
        decision = Coordinator(config).decide(tool_loop_body(), lane="auto")

        assert decision.role == "worker"  # inferred from write tools
        assert "mid_tool_loop:no_session" in decision.signals
        assert decision.chain is False

    def test_sessions_expire(self, config):
        clock = FakeClock()
        coordinator = Coordinator(config, clock=clock)
        coordinator.decide(body("implement the parser", tools=CODING_TOOLS), lane="auto")

        clock.advance(config.session.ttl_seconds + 1)
        later = coordinator.decide(tool_loop_body("implement the parser"), lane="auto")

        assert "mid_tool_loop:no_session" in later.signals

    def test_cache_stays_bounded(self, config):
        coordinator = Coordinator(config)
        for i in range(config.session.max_entries * 3):
            coordinator.decide(body(f"task {i}", tools=CODING_TOOLS), lane="auto")

        assert len(coordinator._sticky) <= config.session.max_entries


class TestFeasibilityOverrides:
    def test_an_image_never_goes_to_a_text_only_model(self, config):
        """A pinned lane must not be able to blind the request."""
        decision = Coordinator(config).decide(image_body(), lane="code")

        assert decision.role == "worker"
        assert config.supports_vision(decision.primary)
        assert "override:vision_required" in decision.signals

    def test_candidates_that_cannot_hold_the_context_are_dropped(self):
        config = make_config(
            models={
                "deepseek": {"context": 100_000},
                "glm": {"context": 10},
                "kimi": {"context": 100_000},
            },
            routing={"long_context_tokens": 10_000_000},  # keep librarian out of it
        )
        decision = Coordinator(config).decide(
            body("fix this " + "x" * 4000, tools=CODING_TOOLS), lane="auto"
        )

        assert "glm" not in decision.candidates
        assert "override:context_fit" in decision.signals

    def test_warns_when_nothing_can_hold_the_context(self):
        config = make_config(
            models={
                "deepseek": {"context": 10},
                "glm": {"context": 20},
                "kimi": {"context": 30},
            },
            routing={"long_context_tokens": 10_000_000},
        )
        decision = Coordinator(config).decide(
            body("fix this " + "x" * 4000, tools=CODING_TOOLS), lane="auto"
        )

        assert "warn:context_exceeds_all" in decision.signals
        assert decision.primary == "kimi"  # the roomiest of a bad set


class TestClassifierEscalation:
    def test_vague_first_turn_asks_for_help(self):
        config = make_config(routing={"ambiguous_escalation": True})
        decision = Coordinator(config).decide(body("the thing again"), lane="auto")

        assert decision.needs_classification
        assert "ambiguous" in decision.signals

    def test_escalation_can_be_switched_off(self, config):
        decision = Coordinator(config).decide(body("the thing again"), lane="auto")

        assert not decision.needs_classification
        assert decision.role == config.routing.default_role

    def test_applying_a_verdict_reroutes_and_records_it(self):
        config = make_config(routing={"ambiguous_escalation": True})
        coordinator = Coordinator(config)
        decision = coordinator.decide(body("the thing again", tools=CODING_TOOLS),
                                      lane="auto")

        resolved = coordinator.with_role(decision, "worker", "classifier:worker")

        assert resolved.role == "worker"
        assert resolved.primary == "deepseek"
        assert resolved.chain is True
        assert not resolved.needs_classification
        assert "classifier:worker" in resolved.signals

    def test_a_nonsense_verdict_is_ignored(self):
        config = make_config(routing={"ambiguous_escalation": True})
        coordinator = Coordinator(config)
        decision = coordinator.decide(body("the thing again"), lane="auto")

        resolved = coordinator.with_role(decision, "wizard", "classifier:wizard")

        assert resolved.role == decision.role
        assert not resolved.needs_classification
