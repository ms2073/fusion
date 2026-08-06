"""Probe the live NIM catalog: `python -m fusion.catalog [filter]`.

Published model IDs disagree between sources and change between releases, so
this is the authority for what goes in config.yaml.
"""

from __future__ import annotations

import asyncio
import sys

from .config import Settings, load_api_keys, load_config
from .upstream import UpstreamClient, UpstreamError


async def _run(needle: str) -> int:
    settings = Settings()
    config = load_config(settings.config_path)
    keys = load_api_keys()
    if not keys:
        print("No API keys found. Set NIM_API_KEY_1 in your .env.", file=sys.stderr)
        return 1

    client = UpstreamClient(config)
    try:
        catalog = await client.list_models(keys[0])
    except UpstreamError as exc:
        print(f"Could not reach the catalog: {exc}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()

    matches = [m for m in catalog if needle in m.lower()]
    print(f"{len(matches)} of {len(catalog)} models"
          + (f" matching {needle!r}" if needle else ""))
    for model in matches:
        print(f"  {model}")

    available = set(catalog)
    missing = [m for m in config.all_models() if m not in available]
    print()
    if missing:
        print("Configured but NOT in the catalog - fix these in config.yaml:")
        for model in missing:
            import difflib

            close = difflib.get_close_matches(model, catalog, n=3, cutoff=0.4)
            hint = f"  ->  try: {', '.join(close)}" if close else ""
            print(f"  {model}{hint}")
        return 1

    print(f"All {len(config.all_models())} configured models exist upstream.")
    return 0


def main() -> None:
    needle = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    raise SystemExit(asyncio.run(_run(needle)))


if __name__ == "__main__":
    main()
