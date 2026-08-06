"""Small-model fallback for requests the rules genuinely cannot place.

Deliberately cheap: a handful of output tokens, a short timeout, and a single
attempt. If it is slow or unavailable we take the default role rather than make
the user wait - a misroute costs less than a stall.
"""

from __future__ import annotations

import asyncio
import logging

from .config import FusionConfig
from .coordinator import RequestFacts
from .governor import Governor
from .upstream import UpstreamClient, UpstreamError

log = logging.getLogger("fusion.classifier")

_LABEL_TO_ROLE = {
    "code": "worker",
    "plan": "thinker",
    "ask": "thinker",
    "search": "librarian",
    "look": "eyes",
}

_PROMPT = """Classify the developer request into exactly one label.

code   - write, change, fix, or refactor code
plan   - design, architect, compare approaches, decide strategy
ask    - answer a question or explain something
search - find or cross-reference things across a large codebase
look   - anything involving an image, screenshot, or visual design

Reply with the label only, nothing else."""


async def classify(
    config: FusionConfig,
    governor: Governor,
    upstream: UpstreamClient,
    facts: RequestFacts,
) -> tuple[str | None, str]:
    """Return (role, signal). Role is None when classification is inconclusive."""
    excerpt = facts.last_user_text[:1500]
    if not excerpt:
        return None, "classifier:empty"

    request = {
        "messages": [
            {"role": "system", "content": _PROMPT},
            {
                "role": "user",
                "content": (
                    f"Tools available: "
                    f"{', '.join(sorted(facts.tool_names)) or 'none'}\n"
                    f"Request:\n{excerpt}"
                ),
            },
        ],
        "max_tokens": config.classifier.max_tokens,
        "temperature": 0,
    }

    attempts = governor.plan_attempts([config.classifier.model], limit=1)
    attempt = attempts[0]
    governor.record_dispatch(attempt)
    try:
        response = await asyncio.wait_for(
            upstream.complete(attempt, request),
            timeout=config.classifier.timeout_seconds,
        )
    except (UpstreamError, asyncio.TimeoutError) as exc:
        governor.record_failure(attempt)
        log.debug("classifier unavailable (%s); using the default role", exc)
        return None, "classifier:unavailable"

    governor.record_success(attempt)

    from . import sse  # local import keeps the module dependency-light

    label = sse.message_text(response).strip().lower().split()[:1]
    if not label:
        return None, "classifier:empty_reply"

    role = _LABEL_TO_ROLE.get(label[0].strip(".,:"))
    if role is None:
        return None, f"classifier:unknown({label[0][:12]})"
    return role, f"classifier:{label[0]}"
