"""Step 1 — Record queries.

Queries live in a JSONL file, one object per line::

    {"id": "q-0001", "prompt": "How do I center a div?", "tags": ["css"]}

``add_query`` appends one (assigning an id if absent); ``load_queries`` reads
them back. Keeping this a plain append-only file makes the "record" step usable
from a CLI, a script, or by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


@dataclass
class Query:
    id: str
    prompt: str
    tags: list[str] = field(default_factory=list)
    recorded_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "tags": self.tags,
            "recorded_at": self.recorded_at,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_queries(path: str | Path) -> list[Query]:
    p = Path(path)
    if not p.exists():
        return []
    queries: list[Query] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        queries.append(
            Query(
                id=obj["id"],
                prompt=obj["prompt"],
                tags=obj.get("tags", []),
                recorded_at=obj.get("recorded_at", ""),
            )
        )
    return queries


def _next_id(existing: Iterable[Query]) -> str:
    n = sum(1 for _ in existing) + 1
    return f"q-{n:04d}"


def add_query(
    path: str | Path,
    prompt: str,
    tags: list[str] | None = None,
    query_id: str | None = None,
) -> Query:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = load_queries(p)
    query = Query(
        id=query_id or _next_id(existing),
        prompt=prompt,
        tags=tags or [],
        recorded_at=_now(),
    )
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(query.to_dict()) + "\n")
    return query
