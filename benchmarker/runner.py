"""Step 2 — Give each query to all models.

Fans a batch of queries out across every configured adapter, gathering a
:class:`ModelResponse` per (query, model) pair. Adapters run concurrently across
models (each backend is independent I/O); a failure in one is captured on its
response, never raised, so one broken model can't sink the run.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

from .models import ModelAdapter, ModelResponse
from .recorder import Query


def run_batch(
    queries: Iterable[Query],
    adapters: list[ModelAdapter],
    max_workers: int = 8,
) -> list[ModelResponse]:
    """Return one ModelResponse per (query, adapter) pair."""
    jobs: list[tuple[ModelAdapter, Query]] = [
        (adapter, q) for q in queries for adapter in adapters
    ]
    if not jobs:
        return []

    results: list[ModelResponse] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(adapter.ask, q.id, q.prompt) for adapter, q in jobs
        ]
        for fut in futures:
            results.append(fut.result())
    return results


def group_by_query(
    responses: list[ModelResponse],
) -> dict[str, list[ModelResponse]]:
    grouped: dict[str, list[ModelResponse]] = {}
    for r in responses:
        grouped.setdefault(r.query_id, []).append(r)
    return grouped
