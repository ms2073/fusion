# Contributing to Fusion

Thanks for your interest in contributing.

## Reporting a bug

Open an issue with:

1. The output of `make catalog` (so the model pool is visible)
2. A redacted `curl` against `/v1/chat/completions` that reproduces the problem
3. The output of `make stats | python3 -m json.tool` after the bad request

## Requesting a feature

Open an issue before writing code. The design surface of fusion is deliberately
small — new routing rules go in `fusion/coordinator.py` and new load or failover
behaviour goes in `fusion/governor.py`. Changes elsewhere usually mean the
feature belongs in a different layer, not a different file.

## Running the tests

```bash
make test
```

Everything runs offline via `respx` — no live NVIDIA calls, no keys needed.

## Style

Python 3.12, standard-library only where possible. Dependencies stay tiny.
No typing annotations that fight the type checker; simple ones only.
