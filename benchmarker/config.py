"""Config loading.

The pipeline is driven by a small YAML file that lists the models to compare and
where logs/queries live. YAML is optional — a JSON config works too, and if
neither PyYAML is present nor a file is given, a sensible all-mock default is
used so the prototype always runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Which model in the line-up is the one under evaluation. Everything is compared
# relative to this one ("is Fenzo better or worse than X here?").
DEFAULT_SUBJECT = "fenzo"

DEFAULT_CONFIG: dict[str, Any] = {
    "subject": DEFAULT_SUBJECT,
    "judge": {"name": "judge", "type": "anthropic", "model": "claude-opus-5"},
    "paths": {
        "queries": "queries/queries.jsonl",
        "responses_log": "logs/responses.jsonl",
        "comparisons_log": "logs/comparisons.jsonl",
        "reports_dir": "reports",
    },
    "models": [
        {"name": "fenzo", "type": "mock", "persona": "balanced"},
        {"name": "claude-opus-5", "type": "mock", "persona": "verbose"},
        {"name": "gpt-4o", "type": "mock", "persona": "concise"},
    ],
}


@dataclass
class Config:
    subject: str
    judge: dict[str, Any]
    paths: dict[str, str]
    models: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict)

    def path(self, key: str) -> Path:
        return Path(self.paths[key])


def _load_raw(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_CONFIG)
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to read YAML config; `pip install pyyaml` "
                "or use a .json config"
            ) from exc
        return yaml.safe_load(text)
    return json.loads(text)


def load_config(path: str | Path | None = None) -> Config:
    raw = _load_raw(path)
    merged = {**DEFAULT_CONFIG, **raw}
    # Shallow-merge the nested dicts we care about.
    merged["paths"] = {**DEFAULT_CONFIG["paths"], **raw.get("paths", {})}
    return Config(
        subject=merged["subject"],
        judge=merged["judge"],
        paths=merged["paths"],
        models=merged["models"],
        raw=merged,
    )
