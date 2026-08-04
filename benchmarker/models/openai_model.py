"""OpenAI (GPT) adapter — a second real comparison point.

Kept deliberately thin and dependency-lazy, mirroring the Anthropic adapter.
Credentials come from ``OPENAI_API_KEY`` in the environment.
"""

from __future__ import annotations

from typing import Any

from .base import ModelAdapter


class OpenAIAdapter(ModelAdapter):
    def __init__(
        self,
        name: str,
        model: str = "gpt-4o",
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
            import openai  # imported lazily; optional dependency

            self._client = openai.OpenAI()
        return self._client

    def _ask(self, prompt: str) -> tuple[str, dict[str, Any]]:
        client = self._get_client()
        messages = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""
        meta = {
            "model_id": resp.model,
            "finish_reason": choice.finish_reason,
            "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
            "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
        }
        return text, meta
