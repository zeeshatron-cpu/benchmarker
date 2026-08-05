"""Deterministic mock adapter.

Lets the whole pipeline run end-to-end with no API keys and no network — useful
for development, CI, and demoing the flow. Each mock "model" has a persona that
shapes how it answers, so the comparison step has something real to chew on.
"""

from __future__ import annotations

import hashlib
import textwrap
from typing import Any

from .base import ModelAdapter

_PERSONAS = {
    "concise": "Answers in one tight paragraph, no fluff.",
    "verbose": "Answers thoroughly with headers and caveats.",
    "balanced": "Answers directly, then adds one useful note.",
}


class MockAdapter(ModelAdapter):
    """Produces stable, persona-flavoured pseudo-answers from the prompt hash."""

    def __init__(self, name: str, persona: str = "balanced", **kwargs: Any) -> None:
        super().__init__(name, **kwargs)
        self.persona = persona if persona in _PERSONAS else "balanced"

    def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        digest = hashlib.sha256(f"{self.name}:{prompt}".encode()).hexdigest()[:8]
        topic = prompt.strip().rstrip("?.").strip() or "your question"

        if self.persona == "concise":
            text = (
                f"Short answer to '{topic}': it depends on context, but the key "
                f"factor is usually the trade-off between speed and accuracy. "
                f"[mock:{self.name}:{digest}]"
            )
        elif self.persona == "verbose":
            text = textwrap.dedent(
                f"""\
                ## On '{topic}'

                There are several angles worth considering here.

                1. **Definition** — first we should be clear about terms.
                2. **Trade-offs** — most real answers involve a balance.
                3. **Recommendation** — in the common case, start simple.

                Caveat: the right choice depends on your specific constraints,
                so treat this as a starting point rather than a verdict.
                [mock:{self.name}:{digest}]"""
            )
        else:  # balanced
            text = (
                f"For '{topic}': the direct answer is to weigh cost against "
                f"benefit and pick the option that fits your constraints. "
                f"One useful note: measure before optimizing. "
                f"[mock:{self.name}:{digest}]"
            )

        return text, {"persona": self.persona, "digest": digest, "mock": True}
