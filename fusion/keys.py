"""Show which API keys are being picked up: `python -m fusion.keys`.

Keys are masked. This exists so "why isn't my third key being used?" is one
command to answer instead of a guess.
"""

from __future__ import annotations

import sys

from .config import Settings, load_api_key_entries, load_config

HELP = """
Add keys to .env in whichever form suits you - they can be mixed, and
duplicates are collapsed:

  NIM_API_KEY=nvapi-a                  a single key
  NIM_API_KEY_1=nvapi-a                numbered, any count
  NIM_API_KEY_2=nvapi-b
  NIM_API_KEY_3=nvapi-c
  NIM_API_KEYS=nvapi-a,nvapi-b         a list in one variable
  NIM_API_KEYS_FILE=/run/secrets/nim   one per line, for docker secrets

Each key is an independent rate-limit bucket, so more keys means more
headroom before anything has to fall back to a different model.
""".rstrip()


def main() -> None:
    entries = load_api_key_entries()
    if not entries:
        print("No API keys found.\n")
        print(HELP)
        raise SystemExit(1)

    print(f"{len(entries)} API key(s) loaded:\n")
    width = max(len(entry.masked) for entry in entries)
    for index, entry in enumerate(entries):
        print(f"  [{index}] {entry.masked:<{width}}  from {entry.origin}")

    suspect = [e for e in entries if not e.value.startswith("nvapi-")]
    if suspect:
        print(
            f"\nWarning: {len(suspect)} key(s) lack the usual 'nvapi-' prefix. "
            "They will still be tried."
        )
    try:
        rpm = load_config(Settings().config_path).rate_limit.default_rpm
    except OSError:
        rpm = 36
    print(f"\nEffective ceiling: ~{rpm * len(entries)} requests/min per model.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
