"""Minimal .env loader (zero dependencies).

Reads simple ``KEY=VALUE`` lines from a local ``.env`` file and puts them into
``os.environ`` (without overwriting variables already set in the real
environment). This is what makes the pipeline "plug and go": drop your API keys
into ``.env`` once and every run picks them up automatically — no need to export
them each session.

The ``.env`` file is gitignored and must never be committed; it holds secrets.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Load KEY=VALUE lines from `path` into os.environ. Returns keys set."""
    p = Path(path)
    if not p.exists():
        return []
    set_keys: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        # Real environment variables win over the file.
        if key not in os.environ:
            os.environ[key] = value
            set_keys.append(key)
    return set_keys
