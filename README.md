# Fusion

[![test](https://github.com/ms2073/fusion/actions/workflows/test.yml/badge.svg)](https://github.com/ms2073/fusion/actions/workflows/test.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests: 201](https://img.shields.io/badge/tests-201%20passing-brightgreen)](tests/)

A model router for [opencode](https://opencode.ai) and
[Claude Code](https://claude.com/claude-code) that sits in front of the free
[NVIDIA NIM](https://build.nvidia.com) API. It reads each request, sends it to whichever
model is actually good at that kind of work, and rotates two API keys so you stop
collecting 429s.

## Where the idea came from

[Sakana AI's Fugu](https://sakana.ai/fugu/) makes an argument worth taking seriously: that
picking one frontier model for a whole task is the wrong unit of choice. Fugu instead puts a
**coordinator** in front of a pool of models and assigns specialist roles per task —
**Thinker** for analysis and planning, **Worker** to execute, **Verifier** to validate —
and exposes the lot through a single API, so the caller never manages endpoints. The
research behind it, TRINITY (ICLR 2026), uses a *lightweight evolved coordinator* to
delegate across coding, maths, reasoning and knowledge tasks. It scores 59.0 on SWE-Bench
Pro (73.7 for Fugu Ultra) and 92.9 on LiveCodeBench.

Fusion borrows the shape and diverges on two points, both deliberate.

**The coordinator is hand-written, not evolved.** Fugu's is trained; this one is ordered
rules you can read in `coordinator.py` and change in a text editor. That is a real
capability gap — a learned router will find delegations these rules never will — bought in
exchange for being able to explain, in one line, why any given request went where it did.
Every decision and the signals behind it land in `/stats`.

**Fugu does not have to survive a free tier.** It runs on infrastructure it controls. Fusion
runs on a pool where four frontier models can be simultaneously queued, where a `200` means
"accepted, now wait", and where availability flips within seconds. So most of what is here —
hedged racing, the governor, escalating backoff, a tail model in every role — exists for a
problem Fugu never had. The role assignment is Fugu's idea. Everything that keeps it
answering is not.

```
opencode     ──► /v1/chat/completions ──┐
                                        ├─► coordinator ──► governor ──► NVIDIA NIM
Claude Code  ──► /v1/messages ──────────┘   which role?     which key?
```

## The pool

| Role | Leads with | Because | Ends in |
|---|---|---|---|
| **Thinker** | `z-ai/glm-5.2` | strongest at planning and one-shot generation; 62.1% SWE-bench Pro, 81.0% Terminal-Bench 2.1 | `nemotron-3-super` |
| **Worker** | `deepseek-v4-pro` | top SWE-bench Verified here, 80.6% | `nemotron-3-super` |
| **Librarian** | `z-ai/glm-5.2` | the pick past ~200k tokens: repo-scale refactoring, long-horizon engineering | `nemotron-3-super` |
| **Eyes** | `minimaxai/minimax-m3` | alone in open weights for UI automation and screenshot-to-code; drives a desktop from screenshots | `nemotron-nano-12b-v2-vl` |

Order is by **what each model is best at**, never by who is fast right now. Availability on
this tier flips within seconds and the governor already routes around it, so pinning a
snapshot of it into the config would only be wrong by tomorrow.

Three things about this pool that are easy to get backwards:

- **Every text role holds a million tokens all the way down.** Not decoration: a model whose
  declared context is too small is *deleted* from big requests, so an understated fallback
  vanishes exactly when it is needed. `nemotron-3-super` was declared at 131072 against a real
  1M, which is how a long session went from slow to broken.
- **The strongest model in the pool cannot see.** `nemotron-3-ultra-550b` answers an image
  request with `multimodal processing is not enabled`. Every model in Eyes was verified by
  sending it a solid-colour PNG and checking the colour came back.
- **Kimi K2.6 is absent on purpose.** It is in `/v1/models` and 404s on inference for this
  account. Catalog membership is not entitlement — `make probe` is the check that matters.

## Quick start

Needs Docker and a free key from [build.nvidia.com](https://build.nvidia.com). Nothing else —
Python, deps and the venv all live inside the container.

```bash
git clone https://github.com/ms2073/fusion.git && cd fusion
cp .env.example .env         # then put your NIM key(s) in it
make up
```

`make up` builds the image, starts the proxy on `127.0.0.1:8787`, and prints a health check.
It ends with `"status": "ok"` and `"catalog_problems": []` when everything is right.

Confirm it works:

```bash
curl -s localhost:8787/healthz | python3 -m json.tool     # keys loaded, RPM ceiling
make probe                                               # which models can you ACTUALLY call?
```

**Run `make probe` before trusting the shipped config.** `/v1/models` lists models your key
cannot invoke — Kimi K2.6 is in the catalog and 404s on inference — so probe measures what
really answers, and how fast. `make catalog` checks existence only; the server runs that
check at startup too, so a wrong model ID is a loud warning rather than a silent 404.

### Point a client at it

Fusion speaks **two dialects over one router**, so the client you use doesn't change any of
the routing, hedging or failover behaviour:

| Dialect | Endpoint | For |
|---|---|---|
| OpenAI | `/v1/chat/completions` | opencode, and anything OpenAI-compatible |
| Anthropic | `/v1/messages` | Claude Code |

Every client needs a non-empty API key and none of them need a real one — the proxy has no
auth of its own, because it holds the real keys and binds to loopback. Use
`host.docker.internal` instead of `localhost` if the client is itself in a container.

**opencode** — merge `opencode.example.json` into `~/.config/opencode/opencode.json`, then
pick `fusion/auto`. Keep the model keys bare (`auto`, not `fusion/auto`): opencode displays
them as `<provider>/<key>`, so a prefixed key becomes `fusion/fusion/auto`.

**Claude Code** — point it at the base URL, *without* `/v1`; the Anthropic SDK appends the
path itself.

```bash
ANTHROPIC_BASE_URL=http://localhost:8787 \
ANTHROPIC_AUTH_TOKEN=fusion-local \
ANTHROPIC_MODEL='fusion/think[1m]' \
claude
```

Verified end to end: Claude Code answers through the router, and `/stats` shows
`lane=think role=thinker served_by=nvidia/nemotron-3-ultra-550b-a55b`.

The `[1m]` suffix matters. Claude Code does not recognise `fusion/*` as a model,
so it assumes a 200k context window and starts auto-compacting the conversation
long before it needs to — every model behind these lanes holds 1M. The suffix
tells it the real window. `CLAUDE_CODE_MAX_CONTEXT_TOKENS` does the same job.

Or make it permanent in `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:8787",
    "ANTHROPIC_AUTH_TOKEN": "fusion-local",
    "ANTHROPIC_MODEL": "fusion/auto"
  }
}
```

Any model name works — an unrecognised one falls through to the `auto` lane — so leaving
Claude Code's own default in place routes fine. Name a lane (`fusion/code`, `fusion/think`)
to pin one instead.

The translation covers system prompts, tool definitions and results, images, tool-use
streaming, and `/v1/messages/count_tokens`. The stream is the fiddly half: OpenAI sends one
flat run of deltas while Anthropic sends *typed blocks* that must be opened and closed in
order, and a client that receives an unterminated block — or a delta for a block that was
never started — rejects the whole message rather than degrading.

### Everyday commands

| | |
|---|---|
| `make up` | build and start — needed after any change under `fusion/` |
| `make restart` | reload `config.yaml` only (it's bind-mounted, no rebuild) |
| `make logs` | follow routing decisions live |
| `make stats` | what each model is doing: dispatched, succeeded, outraced, 429s |
| `make keys` | which keys loaded, masked, and the resulting RPM ceiling |
| `make probe` | measure real availability and latency per model |
| `make test` | the suite, no network needed |
| `make down` | stop |

`make` on its own lists them.

### More keys, more headroom

Every key is an independent 429 bucket. Add them in whichever form suits you — mixable,
duplicates collapse — then `make restart`:

```bash
NIM_API_KEY_1=nvapi-...
NIM_API_KEY_2=nvapi-...
NIM_API_KEY_3=nvapi-...
```

Three keys took the ceiling from 36 to ~108 requests/min per model. Note this buys 429
headroom only, not queue relief — see [Not hitting rate limits](#not-hitting-rate-limits).

## How routing works

Rules run in order; first match wins. Every decision and the signals behind it are logged
and visible at `/stats`.

| # | Rule | Goes to |
|---|---|---|
| 1 | **Mid tool loop** (a `tool` message or `tool_calls` present) | the session's existing model, never re-planned |
| 2 | Pinned lane (`fusion/code`, `fusion/think`, …) | that role |
| 3 | Request contains an image | Eyes |
| 4 | Estimated tokens over the threshold | Librarian |
| 5 | Write tools available **and** the user gave a directive | Worker, **plan first** |
| 6 | Planning or question language | Thinker |
| 7 | Genuinely ambiguous | one small-model call to classify |
| 8 | Nothing matched | default role |

Rule 1 is the load-bearing one. opencode runs an agentic loop, so most requests are
continuations — injecting a planning stage into turn seven would derail it.

Two hard constraints run after the rules and can override them: an image never goes to a
text-only model, and a payload never goes to a model too small to hold it.

### Plan before code

When rule 5 fires, one request produces two upstream calls:

```
GLM      ──► short plan  ──┐ streamed to you as it generates
                           ├─ one SSE response, one [DONE]
DeepSeek ──► the code    ──┘ with the plan in its context
```

The plan is streamed rather than hidden, so opencode stores it in the conversation and it
stays in context for the rest of the task at no cost. The planner is deliberately handed
**no tools** — a Thinker holding tools would try to call them, and nothing here can run
them.

Turn it off with `routing.chain_enabled: false`.

## Adding API keys

Every key is a separate rate-limit bucket, so keys are the cheapest way to buy 429 headroom
— but **only** 429 headroom. NIM's queue is a shared per-model pool: racing one model on two
keys measured 6.60s vs 7.31s solo (1.11×, inside the noise). Keys raise the ceiling; only
hedging across *different* models beats contention.

Add them in whichever form suits you — these are interchangeable and mixable, and
duplicates collapse:

```bash
NIM_API_KEY=nvapi-a                  # a single key
NIM_API_KEY_1=nvapi-a                # numbered, any count
NIM_API_KEY_2=nvapi-b
NIM_API_KEY_3=nvapi-c
NIM_API_KEYS=nvapi-a,nvapi-b         # a list in one variable
NIM_API_KEYS_FILE=/run/secrets/nim   # one per line, for docker secrets
```

Then `make restart`. To confirm they were picked up:

```bash
make keys
#   3 API key(s) loaded:
#     [0] nvapi-abc1...wxyz  from NIM_API_KEY_1
#     [1] nvapi-def2...stuv  from NIM_API_KEY_2
#     [2] nvapi-ghi3...mnop  from NIM_API_KEYS
#   Effective ceiling: ~108 requests/min per model.
```

Keys are masked everywhere they're shown — `make keys`, the startup log, and `/healthz`.
Blank values and the shipped `nvapi-...` placeholder are ignored, so a half-filled `.env`
runs on however many real keys it holds instead of failing requests against a fake one.

## Not hitting rate limits

NIM allows roughly 40 RPM per model, best-effort. The governor:

- keeps a **token bucket per (key, model)**, paced slightly under the ceiling so requests
  bounce off our own limiter instead of collecting 429s
- picks the key with the **most headroom**, which alternates keys on its own and degrades
  better than round-robin when one key is already hot
- on a 429, honours `Retry-After`, puts that (key, model) into exponential-backoff
  cooldown, and **retries the same model on the other key**
- only when every key for a model is cooling down does it **drop to the next model** in
  that role's list
- opens a **circuit breaker** on a model after repeated non-429 failures

Failover only happens **before the first streamed byte**. Once opencode holds bytes from a
model, switching would splice two models' output together; a mid-stream failure is
reported inside the stream instead of silently truncating.

## Tuning

`config.yaml` is mounted into the container, so edits need only a restart — no rebuild.
Watch `/stats` to see what the rules are actually doing:

```bash
curl -s localhost:8787/stats | jq '.routing.recent[0]'
curl -s localhost:8787/stats | jq '.capacity.headroom'
```

Each entry shows the role, the signals that fired, which model was planned, and which one
actually served it after failover.

## Verification

```bash
uv run pytest                # 196 tests, no network
```

The real gate is a live session. With the container up and opencode pointed at it:

| Try this | Expect in `/stats` |
|---|---|
| Ask a plain question | `thinker`, `chain: false`, one call |
| Attach a screenshot | `eyes` → minimax |
| "implement X" in build mode | plan streams first, `chain: true`, `plan_model` set |
| Let the tool loop run | same `served_by` each turn, no re-planning |
| Several requests at once | both keys used, `failovers` non-zero, no 429 reaches opencode |

## Hedging: the answer to contention

The free tier's frontier models are heavily contended, and **which ones are queued flips
within seconds**. Measured over four rounds, 15s apart:

| Model | Answered | Avg when it did |
|---|---|---|
| `nemotron-3-super-120b` | 4/4 | 3.4s |
| `minimax-m3` | 4/4 | 17.2s |
| `deepseek-v4-pro` | **3/4** | **3.5s** |
| `deepseek-v4-flash` | 2/4 | 1.0s |
| `glm-5.2` | 0/4 | never |

Two things follow, and they killed the original design:

**You cannot know the fastest model in advance — only discover it.** Sequential failover
waits out each queued model in turn, paying `stall_timeout` every time: **68s** on this pool.
Firing at several at once and taking the first to answer: **0.35s**. So Fusion races.

**A single bad sample must not blacklist a model.** deepseek-v4-pro is usable 3 times in 4,
yet a 120s cooldown from one stall was excluding it entirely — throwing away the good model
to avoid a 3-second wait. Cooldowns are now short (20s); *hedging*, not exclusion, is what
handles a slow model.

```
t=0.0s   fire at deepseek-v4-pro         the preferred model, alone
t=2.5s   still silent -> also fire glm-5.2
t=5.0s   still silent -> also fire nemotron
         first one to send a byte wins; the rest are cancelled
```

Staggered rather than simultaneous, so a healthy primary still costs exactly **one** request.
Tunable under `hedge:` in `config.yaml`; set `enabled: false` for strict sequential.

### Losing a race is evidence

Contention is the one failure that never raises. A queued model returns `200`, goes quiet,
and eventually gets cancelled when a hedge beats it — and a cancelled attempt reports
nothing. So the governor kept re-learning nothing: `glm-5.2` lost **131 races in one
session** and led the preference list into every single one.

Now a racer that started no later than the winner and still said nothing is parked across
every key, with the same escalating backoff a stall gets. Later hedges are never blamed —
having had less time is not evidence of anything.

It is a park, never a blacklist: the backoff caps out, so a busy model is re-offered every
couple of minutes, and one success resets it to primary. Measured over eight consecutive
requests as the pool was found:

```
req 1  7.4s   glm-5.2            leads, loses, parked
req 2  3.2s   deepseek-v4-pro    leads, loses, parked
req 3  0.8s   deepseek-v4-flash  leads, loses, parked
req 4+ 0.3s   nemotron-3-super   leads and answers
```

Watch it happen in `curl -s localhost:8787/stats | jq '.capacity.models'` — `outraced` is
counted separately from `rate_limited`, because conflating them hides which problem you have.

Two details that matter:

- **The race is to the first *generated* byte** — not the HTTP status, and not the first
  frame either. NIM returns `200` headers immediately and then leaves you in the queue, and
  models answer the handshake with an empty `{"role":"assistant"}` delta before queuing too.
  Both crown the slowest model. Only a frame carrying `content`, `reasoning_content` or
  `tool_calls` ends the race. Measured cost of getting this wrong: **33s** on a
  one-sentence question, because the race is the last moment failover is possible — a
  preamble frame ends it early and strands the stall where nothing can be done about it.
- **Hedges go to different models, never to the same model on another key.** A queue is
  per-model and does not care whose key is asking. A sibling key is only used as a last
  resort, where it genuinely helps — against a 429.
- **Non-streaming requests hedge identically.** A client that doesn't stream sits behind the
  same contended pool. Waiting each queued model out in turn took **3 minutes** where the
  streaming path answered in 6s; both now share one race loop.

Measured on one real coding request, same prompt: **4:00 (incomplete) → 68s → 34s**, and
steady-state questions land in **~3.6s**.

## Free-tier reality

The four frontier models are heavily contended. Measured on a live key:

```
$ make probe
  nvidia/nemotron-3-super-120b-a12b  FAST             0.4s
  minimaxai/minimax-m3               SLOW            24.8s
  z-ai/glm-5.2                       QUEUED             -   silent for 45s
  deepseek-ai/deepseek-v4-pro        QUEUED             -   silent for 45s
  deepseek-ai/deepseek-v4-flash      BUSY               -   529 Overloaded
  moonshotai/kimi-k2.6               NOT ENTITLED       -   404 for this account
```

Three things follow, all of which the router handles:

- **`/v1/models` lists more than your key can call.** Kimi K2.6 appears in the catalog and
  404s on inference. `make catalog` checks existence; **`make probe` checks reality.** Run
  probe before trusting a config.
- **Probe the *whole* free tier occasionally, not just what you configured.** The four
  headline models are the most contended things on it, and the quiet ones are not:
  `nemotron-3-ultra-550b` (550B, 1M context, agentic) answered in **0.46s** while both
  frontier leads were queued, and now serves most traffic. Measured the same minute:

  ```
  nvidia/nemotron-3-ultra-550b-a55b   0.46s   ok
  nvidia/nemotron-3-nano-30b-a3b      0.32s   ok
  stepfun-ai/step-3.7-flash           0.52s   ok
  nvidia/nemotron-3-super-120b-a12b   2.95s   ok
  mistralai/mistral-medium-3.5-128b    --     silent >45s
  google/gemma-4-31b-it                --     silent >45s
  openai/gpt-oss-120b                  --     silent >45s
  moonshotai/kimi-k2.6                 --     404 for this account
  ```
- **A queued model answers with silence, not an error.** `stall_timeout_seconds` (30s) caps
  how long we wait for the first byte, then fails over. The model is then parked for
  `stall_cooldown_seconds` (120s) across *every* key — unlike a 429, a saturated pool
  doesn't care which key you use, so rotating keys would just rejoin the same queue.
- **Every role ends in a model that reliably answers — at every size.** A tail the context
  filter removes is not a tail. Each role used to end in a 131k model, so past that size a
  role silently collapsed to its two most contended entries, and a long session stopped
  working rather than degrading. `test_the_tail_survives_a_large_payload` now pins this.

Measured effect of the last two, same request: **4:00 and incomplete → 1:08 and complete.**

Re-run `make probe` when things feel slow; it recommends which model to promote.

## Measuring it

The numbers above are reproducible, not remembered:

```bash
python scripts/loadtest.py            # 14 requests, all at once
python scripts/loadtest.py -n 30 -c 10
```

A real run against three keys:

```
  14 requests, 14 in flight, model fusion/auto
  14/14 succeeded in 12.1s wall

  time to first token   p50   0.68s   p95   4.30s   p99   4.32s
  total duration        p50   2.25s   p95   6.19s   p99  12.05s

  429s absorbed by the governor : 0
  429s that reached the client  : 0
  dispatches per request        : 1.29 (1.0 = no hedging cost)

  dispatched → answered, by model:
    nvidia/nemotron-3-ultra-550b-a55b           15 → 14
    deepseek-ai/deepseek-v4-pro                  3 → 0

  spread across keys: key0 6, key1 6, key2 6
```

Two of those lines matter more than the percentiles.

**Dispatches per request** is the cost of hedging. At 1.29 the pool is healthy
and most requests are answered by the first model tried. If it climbs toward
`hedge.max_parallel`, the hedge delay has fallen below the pool's real
time-to-first-token and every request is paying for racers it does not need.

**429s absorbed versus surfaced** is the governor's whole job, and it is only
visible by diffing `/stats` either side of the run — the client sees success
either way, which is the point.

Attribution comes from `/stats`, not from the stream: the router rewrites the
`model` field to the lane's identity so callers see one stable name, which means
the stream cannot tell you which upstream actually answered.

## Limitations

- **The coordinator is hand-written**, not learned. Fugu's is evolved/RL-trained; this one
  is rules you can read and edit. Misroutes are expected — that's what `/stats` is for.
- **No Verifier role.** Real verification means running tests and reading files, which
  only opencode can do. A proxy-side reviewer could only read the diff, at 2–3× the
  rate-limit cost. It's stubbed in `config.yaml` and off.
- **Plan-mode detection is unconfirmed.** opencode doesn't document its plan-agent system
  prompt, so `_PLAN_MODE_MARKERS` in `coordinator.py` is a guess. Pin `plan` to
  `fusion/think` in opencode's config (as the example does) and it doesn't matter.
- **Single instance.** State is in memory; running two replicas would double your
  effective rate limit and cause 429s.
- **No auth.** Bound to loopback because it holds your keys. Don't expose it.
