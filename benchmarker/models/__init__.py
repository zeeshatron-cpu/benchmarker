"""Model adapter registry.

Maps the ``type`` field in config to an adapter class and builds instances.
"""

from __future__ import annotations

from typing import Any

from .anthropic_model import AnthropicAdapter
from .base import ModelAdapter, ModelResponse
from .fenzo_web import FenzoWebAdapter
from .mock import MockAdapter
from .openai_model import OpenAIAdapter

_REGISTRY: dict[str, type[ModelAdapter]] = {
    "mock": MockAdapter,
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "fenzo_web": FenzoWebAdapter,
}


def build_adapter(spec: dict[str, Any]) -> ModelAdapter:
    """Build one adapter from a config entry like::

        {"name": "fenzo", "type": "fenzo_web", "url": "...", ...}
    """
    spec = dict(spec)
    name = spec.pop("name")
    kind = spec.pop("type")
    if kind not in _REGISTRY:
        raise ValueError(
            f"unknown adapter type {kind!r}; known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[kind](name=name, **spec)


__all__ = [
    "ModelAdapter",
    "ModelResponse",
    "build_adapter",
    "MockAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    "FenzoWebAdapter",
]
