"""Anthropic (Claude) adapter.

Uses the official ``anthropic`` SDK. Credentials resolve from the environment
(``ANTHROPIC_API_KEY``, or an ``ant auth login`` profile) — we never take a key
in config. Import and client construction are lazy so the package imports fine
on machines that don't have the SDK installed or don't intend to hit Claude.
"""

from __future__ import annotations

from typing import Any

from .base import ModelAdapter


class AnthropicAdapter(ModelAdapter):
    def __init__(
        self,
        name: str,
        model: str = "claude-opus-5",
        max_tokens: int = 2048,
        system: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic  # imported lazily; optional dependency

            self._client = anthropic.Anthropic()
        return self._client

    def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.system:
            kwargs["system"] = self.system

        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        meta = {
            "model_id": resp.model,
            "stop_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return text, meta
