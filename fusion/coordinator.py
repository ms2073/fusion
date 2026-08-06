"""Coordinator: read a request, decide which specialist should answer it.

This is Fusion's version of Fugu's role assignment, with hand-written rules
standing in for an evolved coordinator. `decide()` performs no I/O so it can be
replayed exhaustively against captured opencode request bodies.

Ambiguous first turns are the one case rules cannot settle. Rather than making
`decide()` async for that minority, it flags `needs_classification` and the
caller may resolve it with `with_role()`.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Callable

from .config import FusionConfig

# opencode's write-capable tools. Their presence means the session *can* edit,
# which separates a build turn from a read-only question.
WRITE_TOOLS = frozenset(
    {"edit", "write", "apply_patch", "patch", "multiedit", "str_replace_editor"}
)

_IMAGE_PART_TYPES = frozenset({"image_url", "image", "input_image"})

_CODING = re.compile(
    r"\b(implement|refactor|fix|debug|repair|add|create|build|write|migrate|"
    r"rename|delete|remove|update|patch|optimi[sz]e|install|configure|wire|"
    r"integrate|scaffold|generate|port|upgrade|rewrite|extract|inline)\b",
    re.IGNORECASE,
)

_PLANNING = re.compile(
    r"\b(plan|design|architect|architecture|approach|strategy|trade[- ]?offs?|"
    r"compare|evaluate|options|should i|how would|what.s the best|think through|"
    r"brainstorm|outline|propose)\b",
    re.IGNORECASE,
)

_QUESTION = re.compile(
    r"\b(what|why|how|when|where|which|who|does|do|is|are|can|could|explain|"
    r"describe|tell me)\b",
    re.IGNORECASE,
)

# Markers that a system prompt belongs to a read-only / planning agent.
# CONFIRM AGAINST CAPTURED FIXTURES - opencode's exact wording is not documented.
_PLAN_MODE_MARKERS = (
    "plan mode",
    "do not make any edits",
    "must not make any edits",
    "read-only",
    "you are in planning mode",
    "without making changes",
)


@dataclass(frozen=True, slots=True)
class RequestFacts:
    """Everything the rules look at, extracted from the body exactly once."""

    has_images: bool
    mid_tool_loop: bool
    estimated_tokens: int
    tool_names: frozenset[str]
    has_write_tools: bool
    last_user_text: str
    system_text: str
    user_turns: int


@dataclass(frozen=True, slots=True)
class RouteDecision:
    role: str
    candidates: tuple[str, ...]
    chain: bool
    signals: tuple[str, ...]
    fingerprint: str
    needs_classification: bool = False

    @property
    def primary(self) -> str:
        return self.candidates[0]


def _iter_parts(content: Any):
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                yield part


def _still_in_tool_loop(messages: list) -> bool:
    """Is the assistant mid-loop *right now*?

    Not "has a tool ever been called here". An agentic client carries the whole
    transcript on every request, so once the first tool ran, the ever-used
    reading was true forever after - it pinned a three-hour session to whatever
    role its opening turn happened to get, with planning suppressed throughout.

    The loop is open only while the last thing said was the assistant calling a
    tool, or a tool answering one. A new user turn closes it: that is a fresh
    task and deserves fresh routing.
    """
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "tool":
            return True
        if role == "assistant":
            return bool(message.get("tool_calls"))
        if role == "user":
            return False
    return False


def extract_facts(body: dict) -> RequestFacts:
    messages = body.get("messages") or []

    has_images = False
    mid_tool_loop = _still_in_tool_loop(messages)
    text_chars = 0
    system_chunks: list[str] = []
    last_user_text = ""
    user_turns = 0

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")

        content = message.get("content")
        if isinstance(content, str):
            text = content
        else:
            fragments = []
            for part in _iter_parts(content):
                if part.get("type") in _IMAGE_PART_TYPES:
                    has_images = True
                elif isinstance(part.get("text"), str):
                    fragments.append(part["text"])
            text = "\n".join(fragments)

        text_chars += len(text)
        if role == "system":
            system_chunks.append(text)
        elif role == "user":
            user_turns += 1
            last_user_text = text

    tools = body.get("tools") or []
    tool_names = frozenset(
        name
        for tool in tools
        if isinstance(tool, dict)
        for name in [(tool.get("function") or {}).get("name") or tool.get("name")]
        if isinstance(name, str)
    )
    # Tool schemas are a real share of the prompt; ignoring them underestimates
    # badly on agentic requests.
    text_chars += len(json.dumps(tools)) if tools else 0

    return RequestFacts(
        has_images=has_images,
        mid_tool_loop=mid_tool_loop,
        estimated_tokens=text_chars // 4,
        tool_names=tool_names,
        has_write_tools=bool(tool_names & WRITE_TOOLS),
        last_user_text=last_user_text.strip(),
        system_text="\n".join(system_chunks),
        user_turns=user_turns,
    )


def fingerprint(body: dict) -> str:
    """Stable per-conversation id: the opening system + first user message.

    Later turns append to the history but never rewrite the opening, so this
    survives an entire session while staying distinct between sessions.
    """
    messages = body.get("messages") or []
    seed: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in {"system", "user"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            seed.append(f"{role}:{content}")
        else:
            seed.extend(
                f"{role}:{part['text']}"
                for part in _iter_parts(content)
                if isinstance(part.get("text"), str)
            )
        if len(seed) >= 2:
            break
    return hashlib.sha256("\n".join(seed).encode()).hexdigest()[:16]


class _StickyCache:
    """Bounded TTL map so a conversation keeps its model across a tool loop."""

    def __init__(self, max_entries: int, ttl: float, clock: Callable[[], float]):
        self._max = max_entries
        self._ttl = ttl
        self._clock = clock
        self._entries: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> str | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        role, expires = entry
        if expires <= self._clock():
            self._entries.pop(key, None)
            return None
        return role

    def put(self, key: str, role: str) -> None:
        now = self._clock()
        self._entries[key] = (role, now + self._ttl)
        if len(self._entries) > self._max:
            for stale in [k for k, (_, exp) in self._entries.items() if exp <= now]:
                self._entries.pop(stale, None)
        while len(self._entries) > self._max:
            self._entries.pop(next(iter(self._entries)))

    def __len__(self) -> int:
        return len(self._entries)


class Coordinator:
    def __init__(
        self, config: FusionConfig, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._config = config
        self._sticky = _StickyCache(
            config.session.max_entries, config.session.ttl_seconds, clock
        )

    def decide(self, body: dict, lane: str | None = None) -> RouteDecision:
        config = self._config
        facts = extract_facts(body)
        print_id = fingerprint(body)
        signals: list[str] = []
        chain = False
        needs_classification = False

        lane_role = config.lanes.get(lane) if lane else None

        # 1. Mid tool loop. Load-bearing: opencode's agentic loop means most
        #    requests are continuations, and injecting a planning stage into
        #    turn seven would derail it.
        if facts.mid_tool_loop:
            remembered = self._sticky.get(print_id)
            if remembered is not None:
                signals.append("mid_tool_loop:sticky")
                return self._build(remembered, facts, chain=False,
                                   signals=signals, print_id=print_id)
            signals.append("mid_tool_loop:no_session")
            role = lane_role or ("worker" if facts.has_write_tools
                                 else config.routing.default_role)
            return self._build(role, facts, chain=False,
                               signals=signals, print_id=print_id)

        # 2. An explicitly pinned lane is an instruction, not a hint. Honored
        #    here; the feasibility pass below still stops it doing the
        #    impossible, like sending an image to a text-only model.
        if lane_role is not None:
            signals.append(f"lane:{lane}")
            role = lane_role
            chain = (
                role == "worker"
                and config.routing.chain_enabled
                and _looks_like_coding(facts)
            )

        # 3. Images can only go to a model with eyes.
        elif facts.has_images:
            signals.append("images_present")
            role = "eyes"

        # 4. Oversized payloads go to the long-context specialist.
        elif facts.estimated_tokens > config.routing.long_context_tokens:
            signals.append(f"long_context:~{facts.estimated_tokens}tok")
            role = "librarian"

        # 5. Write tools available and the user issued a directive: code it,
        #    but plan first.
        elif facts.has_write_tools and _looks_like_coding(facts):
            signals.append("coding_intent")
            role = "worker"
            chain = config.routing.chain_enabled

        # 6. Planning language, or a read-only agent driving.
        elif _looks_like_planning(facts):
            signals.append("planning_intent")
            role = "thinker"

        # 7. Genuinely unclear. Let the caller spend ~300ms on a small model.
        else:
            role = config.routing.default_role
            if config.routing.ambiguous_escalation and facts.user_turns > 0:
                signals.append("ambiguous")
                needs_classification = True
            else:
                signals.append("default")

        return self._build(
            role,
            facts,
            chain=chain,
            signals=signals,
            print_id=print_id,
            needs_classification=needs_classification,
        )

    def with_role(
        self, decision: RouteDecision, role: str, signal: str
    ) -> RouteDecision:
        """Apply a classifier verdict to a decision that asked for one."""
        if role not in self._config.roles:
            return replace(decision, needs_classification=False)
        candidates = tuple(self._config.models_for_role(role))
        chain = decision.chain or (
            role == "worker" and self._config.routing.chain_enabled
        )
        self._sticky.put(decision.fingerprint, role)
        return replace(
            decision,
            role=role,
            candidates=candidates,
            chain=chain,
            signals=decision.signals + (signal,),
            needs_classification=False,
        )

    def _build(
        self,
        role: str,
        facts: RequestFacts,
        *,
        chain: bool,
        signals: list[str],
        print_id: str,
        needs_classification: bool = False,
    ) -> RouteDecision:
        candidates = self._feasible(role, facts, signals)
        self._sticky.put(print_id, role)
        return RouteDecision(
            role=role,
            candidates=candidates,
            chain=chain,
            signals=tuple(signals),
            fingerprint=print_id,
            needs_classification=needs_classification,
        )

    def _feasible(
        self, role: str, facts: RequestFacts, signals: list[str]
    ) -> tuple[str, ...]:
        """Hard constraints the role map cannot express.

        A pinned lane or a sticky session can otherwise select a model that
        physically cannot serve the request - a text-only model handed an
        image, or a 200k model handed 400k of context.
        """
        config = self._config
        candidates = config.models_for_role(role)

        if facts.has_images:
            with_eyes = [m for m in candidates if config.supports_vision(m)]
            if not with_eyes:
                with_eyes = config.vision_models()
                if with_eyes:
                    signals.append("override:vision_required")
            candidates = with_eyes or candidates

        fits = [m for m in candidates if config.context_for(m) >= facts.estimated_tokens]
        if fits and len(fits) != len(candidates):
            signals.append("override:context_fit")
            candidates = fits
        elif not fits:
            # Nothing fits; lead with whichever holds the most.
            candidates = sorted(candidates, key=config.context_for, reverse=True)
            signals.append("warn:context_exceeds_all")

        return tuple(candidates)


def _looks_like_coding(facts: RequestFacts) -> bool:
    text = facts.last_user_text
    if not text:
        return False
    return bool(_CODING.search(text))


def _looks_like_planning(facts: RequestFacts) -> bool:
    system = facts.system_text.lower()
    if any(marker in system for marker in _PLAN_MODE_MARKERS):
        return True
    text = facts.last_user_text
    if not text:
        return False
    if _PLANNING.search(text):
        return True
    return bool(_QUESTION.search(text)) and not facts.has_write_tools
