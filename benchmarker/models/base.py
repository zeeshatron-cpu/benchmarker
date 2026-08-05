"""Model adapter interface.

Every backend the pipeline can query — Fenzo AI, Claude, GPT, a local mock —
implements :class:`ModelAdapter`. The rest of the pipeline only ever talks to
this interface, so adding a new model is a matter of writing one small class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelResponse:
    """The outcome of asking one model one query."""

    model: str
    query_id: str
    text: str
    latency_s: float
    ok: bool = True
    error: str | None = None
    # Correlates a response to one `run` invocation and one sample within it.
    # Set by the runner, so a re-run of the same query never mixes with old data.
    run_id: str = ""
    repeat: int = 0
    # Free-form extras an adapter may fill in (token counts, finish reason, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "query_id": self.query_id,
            "text": self.text,
            "latency_s": round(self.latency_s, 3),
            "ok": self.ok,
            "error": self.error,
            "run_id": self.run_id,
            "repeat": self.repeat,
            "meta": self.meta,
        }


class ModelAdapter:
    """Base class for anything the pipeline can send a query to.

    Subclasses implement :meth:`_ask`. The public :meth:`ask` wraps it with
    timing and uniform error handling so a single flaky backend never aborts a
    whole benchmark run.
    """

    #: Human-facing name, e.g. "fenzo" or "claude-opus-5". Set per instance.
    name: str = "base"

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name
        self.options = kwargs

    def ask(self, query_id: str, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        try:
            text, meta = self._ask(prompt)
            return ModelResponse(
                model=self.name,
                query_id=query_id,
                text=text,
                latency_s=time.perf_counter() - start,
                meta=meta,
            )
        except Exception as exc:  # noqa: BLE001 - isolate one backend's failure
            return ModelResponse(
                model=self.name,
                query_id=query_id,
                text="",
                latency_s=time.perf_counter() - start,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        """Return (response_text, meta). Subclasses must implement this."""
        raise NotImplementedError

    def close(self) -> None:
        """Release any resources (browsers, sessions). Safe to call twice."""
