"""Step 4 — Update logs.

Everything the pipeline produces is appended to durable JSONL logs so history
accumulates across runs and stays greppable / loadable by any tool:

* ``responses.jsonl``   — one line per (query, model) raw answer + timing.
* ``comparisons.jsonl`` — one line per query verdict from the judge.

Append-only keeps the "update logs" step simple and audit-friendly; nothing is
ever rewritten in place.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .comparator import Comparison
from .models import ModelResponse


def _append(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    stamp = datetime.now(timezone.utc).isoformat()
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            row = {"logged_at": stamp, **row}
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def log_responses(path: str | Path, responses: list[ModelResponse]) -> int:
    return _append(path, (r.to_dict() for r in responses))


def log_comparisons(path: str | Path, comparisons: list[Comparison]) -> int:
    return _append(path, (c.to_dict() for c in comparisons))


def load_comparisons(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
