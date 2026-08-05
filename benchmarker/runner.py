"""Step 2 — Give each query to all models.

Fans a batch of queries out across every configured adapter, gathering a
:class:`ModelResponse` per (query, model, repeat). Two correctness details:

* **Parallelism is per-adapter, not per-job.** Each adapter instance is driven by
  exactly one worker thread, which runs all of that adapter's queries serially.
  This is what keeps the Playwright-based Fenzo adapter safe — Playwright's sync
  API must be used from the thread that created the browser, so scattering one
  adapter's calls across threads would crash it. Different adapters still run
  concurrently, which is where the real latency win is anyway.
* **Repeats + run_id.** Every response is stamped with the run's ``run_id`` and a
  ``repeat`` index, so multiple samples can be collected for variance and so a
  later run never blends into an earlier one in the logs.

A failure in one adapter is captured on its response, never raised.
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Iterable

from .models import ModelAdapter, ModelResponse
from .recorder import Query


def new_run_id() -> str:
    """A unique id for one `run` invocation.

    Microsecond-precision timestamp plus a random suffix for uniqueness. Do not
    infer recency from a string comparison of two ids — two runs can share a
    timestamp and the suffix is not temporal. Selection of the "latest" run is
    done by append order in the log (see cli report), which is always correct.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"run-{stamp}-{uuid.uuid4().hex[:6]}"


def run_batch(
    queries: Iterable[Query],
    adapters: list[ModelAdapter],
    repeats: int = 1,
    run_id: str | None = None,
    max_workers: int = 8,
) -> list[ModelResponse]:
    """Return one ModelResponse per (query, adapter, repeat)."""
    queries = list(queries)
    if not queries or not adapters:
        return []
    run_id = run_id or new_run_id()
    repeats = max(1, repeats)

    def run_adapter(adapter: ModelAdapter) -> list[ModelResponse]:
        # All of this adapter's work happens on one thread (Playwright-safe).
        out: list[ModelResponse] = []
        for rep in range(repeats):
            for q in queries:
                resp = adapter.ask(q.id, q.prompt)
                resp.run_id = run_id
                resp.repeat = rep
                out.append(resp)
        return out

    results: list[ModelResponse] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(adapters))) as pool:
        for adapter_results in pool.map(run_adapter, adapters):
            results.extend(adapter_results)
    return results


def group_by_query(
    responses: list[ModelResponse],
) -> dict[str, list[ModelResponse]]:
    grouped: dict[str, list[ModelResponse]] = {}
    for r in responses:
        grouped.setdefault(r.query_id, []).append(r)
    return grouped
